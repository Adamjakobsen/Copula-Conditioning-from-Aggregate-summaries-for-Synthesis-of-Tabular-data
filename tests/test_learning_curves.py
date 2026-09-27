"""Learning-curve design invariants, using artificial records only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cocast import learning_curves as curves, schema, utility
from cocast.cli import parser
from test_utility import questionnaire_fixture


class SamplingTests(unittest.TestCase):
    def test_nested_anchor_reproduces_original_and_survives_larger_pool(self):
        small, anchor = curves.nested_positions(24, 12, 6, (.5, 1, 2), 42, 0, 0)
        big, _ = curves.nested_positions(40, 12, 6, (.5, 1, 2, 5), 42, 0, 0)
        expected = np.random.default_rng(np.random.SeedSequence([42, 0, 0])).choice(12, size=6, replace=False)
        np.testing.assert_array_equal(anchor, expected)
        for ratio in small:
            np.testing.assert_array_equal(small[ratio], big[ratio])
            self.assertEqual(len(big[ratio]), int(np.ceil(ratio*6)))
            self.assertEqual(len(np.unique(big[ratio])), len(big[ratio]))
        for a, b in zip((.5, 1, 2), (1, 2, 5)):
            np.testing.assert_array_equal(big[a], big[b][:len(big[a])])

    def test_insufficient_data_never_silently_oversampled(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient distinct'):
            curves.nested_positions(12, 12, 6, (1, 5), 42, 0, 0)

    def test_bad_ratios_and_rounding(self):
        for ratios in [[], [0, 1], [1, 1], [1, .5], [.5, 2], [1, np.nan], [1, np.inf]]:
            with self.assertRaises(ValueError):
                curves.validate_ratios(ratios)
        positions, _ = curves.nested_positions(20, 10, 5, (.5, 1, 2), 42, 0, 0)
        self.assertEqual(len(positions[.5]), 3)

    def test_resampling_matches_counts_and_flags_unsupported_tiers(self):
        source = ['tier_0', 'tier_0', 'tier_1']
        requested = ['tier_0']*2 + ['tier_1']*5
        chosen, meta = curves.matched_positions(source, requested, 9)
        self.assertEqual(pd.Series(np.asarray(source)[chosen]).value_counts().to_dict(),
                         pd.Series(requested).value_counts().to_dict())
        self.assertFalse(meta['support_restricted'])
        chosen, meta = curves.matched_positions(['tier_0'], ['tier_4']*7, 9)
        self.assertEqual(len(chosen), 7)
        self.assertTrue(meta['support_restricted'])
        self.assertEqual(meta['matching_policy'], 'uniform_available_tiers_no_overlap')
        self.assertEqual(json.loads(meta['unavailable_tiers_json']), ['tier_4'])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        output = patch('sys.stdout', new=io.StringIO())
        output.start()
        self.addCleanup(output.stop)
        self.real = questionnaire_fixture(12)
        self.real[schema.AGE_COL] = np.arange(12)+20
        self.synthetic = questionnaire_fixture(40)
        self.synthetic[schema.AGE_COL] = np.arange(40)+1000
        self.fits, self.tunings = [], []

    def select(self, x, y, classifier, seed, **kwargs):
        self.tunings.append((x.copy(), y.copy()))
        return 'fixture', {}, .5, 'fixture_inner_cv', 2

    def predict(self, estimator, x, y, test, *, diagnostics=None):
        self.fits.append((x.copy(), y.copy(), test.copy()))
        return np.repeat(y.iloc[0], len(test))

    @contextlib.contextmanager
    def fake_models(self):
        with patch.object(utility, '_select_estimator', side_effect=self.select), \
                patch.object(utility, 'fit_predict', side_effect=self.predict):
            yield

    def evaluate(self, **kwargs):
        with self.fake_models():
            return curves.evaluate_learning_curves(self.real, {'natural': self.synthetic},
                folds=2, repeats=1, domains=['depression'], **kwargs)

    def test_counts_protocols_no_test_leakage_and_no_target_predictors(self):
        result = self.evaluate()
        self.assertEqual(set(result.protocol), {'TRTR','TSTR','TAUG','TSTR_SYNTH_RESAMPLED',
                                               'TAUG_SYNTH_RESAMPLED','TRTR_RESAMPLED'})
        self.assertEqual(len(result[result.protocol == 'TRTR']), 4)
        for row in result.itertuples():
            target_n = int(np.ceil(row.ratio*row.n_real_train))
            expected = target_n if row.protocol.startswith('TSTR') else target_n+row.n_real_train
            self.assertEqual(row.n_fit, expected)
            self.assertEqual(np.asarray(json.loads(row.confusion_counts_json)).sum(), row.n_real_test)
        for x, y, test in self.fits:
            self.assertFalse(set(schema.item_columns('depression')) & set(x.columns))
            real_ages = set(x.loc[x[schema.AGE_COL] < 1000, schema.AGE_COL])
            self.assertFalse(real_ages & set(test[schema.AGE_COL]))
            self.assertTrue((test[schema.AGE_COL] < 1000).all())
        for x, _ in self.tunings:
            self.assertTrue(x[schema.AGE_COL].is_unique)
            self.assertTrue((x[schema.AGE_COL] < 1000).all() or (x[schema.AGE_COL] >= 1000).all())

    def test_one_x_matches_matched_utility(self):
        with self.fake_models():
            old = utility.evaluate_utility(self.real, {'natural': self.synthetic.iloc[:12]}, folds=2, repeats=1)
            new = curves.evaluate_learning_curves(self.real, {'natural': self.synthetic}, ratios=(1,),
                folds=2, repeats=1, domains=['depression'])
        keys = ['run','classifier','protocol','repeat','fold']
        old = old[(old.domain == 'depression') & old.protocol.isin(['TSTR','TAUG','TRTR'])]
        new = new[new.protocol.isin(['TSTR','TAUG','TRTR'])]
        for column in ['f1_macro','accuracy','tier_0_tp','tier_1_fp','tier_1_fn','split_fingerprint']:
            pd.testing.assert_series_equal(old.set_index(keys)[column].sort_index(),
                                           new.set_index(keys)[column].sort_index(), check_names=False)
        copied = new[new.protocol == 'TSTR'].iloc[0]
        original = old[old.protocol == 'TSTR'].iloc[0]
        self.assertEqual(copied.selection_sha256, curves.digest(json.loads(original.synthetic_train_positions_json)))

    def test_control_equals_generated_at_or_below_one_x(self):
        result = self.evaluate()
        keys = ['classifier','ratio','repeat','fold']
        for protocol in ['TSTR','TAUG']:
            original = result[(result.ratio <= 1) & (result.protocol == protocol)].set_index(keys)
            control = result[(result.ratio <= 1) & (result.protocol == protocol+'_SYNTH_RESAMPLED')].set_index(keys)
            for column in ['f1_macro','selection_sha256','confusion_counts_json']:
                pd.testing.assert_series_equal(original[column], control[column])
        resampled = result[(result.ratio > 1) & result.protocol.str.endswith('SYNTH_RESAMPLED')]
        self.assertTrue((resampled.n_unique_source_rows <= resampled.n_real_train).all())

    def test_summary_keeps_ratios_separate_and_pools_rare_counts(self):
        result = self.evaluate()
        summaries = curves.summarize(result)
        self.assertEqual(set(summaries.ratio), {0,.5,1,2,5})
        self.assertTrue((summaries.n_real_test == 12).all())
        self.assertTrue((summaries.tier_1_support == 6).all())
        self.assertTrue(summaries.tier_4_recall.isna().all())
        self.assertEqual(len(summaries[summaries.protocol == 'TRTR']), 2)

    def test_real_estimator_smoke_on_artificial_records(self):
        # Fit both actual estimators, using one candidate each to keep a smoke test bounded.
        candidates = utility.estimator_candidates
        with patch.object(utility, 'estimator_candidates', side_effect=lambda c,s: candidates(c,s)[:1]):
            result = curves.evaluate_learning_curves(self.real, {'natural': self.synthetic},
                ratios=(.5,1,2), folds=2, repeats=1, domains=['depression'])
        self.assertTrue(result.f1_macro.between(0,1).all())
        self.assertEqual(set(result.classifier), {'logreg','histgb'})

    def test_cli_defaults(self):
        args = parser().parse_args(['learning-curves','--real','real.csv','--run','pool','--out','out'])
        self.assertEqual(args.ratios, [.5,1,2,5])
        self.assertEqual(args.repeats, 3)
        self.assertFalse(args.resume)

    def test_upper_tier_recognition_counts_cross_predictions_and_false_positives(self):
        row={'run':'test','classifier':'logreg','protocol':'TSTR','ratio':2,'repeat':1,
             'domain':'panic','n_real_test':3,'support_restricted':False,
             'confusion_counts_json':json.dumps([[0,0,0,0,1],[0,0,0,0,0],[0,0,0,0,0],
                                                [0,0,0,0,1],[0,0,0,1,0]])}
        value=curves.summarize(pd.DataFrame([row])).iloc[0]
        self.assertEqual(value.upper_tier_recall,1)
        self.assertEqual(value.upper_tier_precision,2/3)
        self.assertEqual(value.upper_tier_false_positives,1)
        self.assertEqual(value.tier_4_recall,0)

    def manifest(self):
        return {'settings': {'model':'fixture','seed':42}, 'regime':'natural'}

    def test_checkpoint_resume_and_only_three_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            real_path=root/'real.csv'
            self.real.to_csv(real_path,index=False)
            run=root/'pool'
            run.mkdir()
            (run/'manifest.json').write_text('{}')
            self.synthetic.to_csv(run/'dataset.csv',index=False)
            out=root/'evaluation'
            actual_domain = curves.domain_rows
            def fail_second(*args, **kwargs):
                if args[2] == 'separation_anxiety':
                    raise RuntimeError('interrupted fixture')
                return actual_domain(*args, **kwargs)
            with self.fake_models(), patch.object(curves,'load_run',return_value=(self.manifest(),self.synthetic)), \
                    patch.object(curves,'domain_rows',side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError,'interrupted fixture'):
                    curves.evaluate_runs(real_path,[run],out,ratios=(1,2),folds=2,repeats=1)
            state=json.loads((out/'manifest.json').read_text())
            self.assertEqual(state['completed_domains'],['depression'])
            with (out/'utility_folds.csv').open('ab') as f:
                f.write(b'uncommitted crash tail')
            with self.fake_models(), patch.object(curves,'load_run',return_value=(self.manifest(),self.synthetic)):
                curves.evaluate_runs(real_path,[run],out,ratios=(1,2),folds=2,repeats=1,resume=True)
            self.assertEqual({p.name for p in out.iterdir()},{'manifest.json','metrics.csv','utility_folds.csv'})
            data=pd.read_csv(out/'utility_folds.csv')
            self.assertEqual(len(data[(data.domain=='depression') & (data.protocol=='TRTR')]),4)
            with patch.object(curves,'load_run',return_value=(self.manifest(),self.synthetic)), \
                    patch.object(curves,'domain_rows',side_effect=AssertionError('Must skip complete evaluation')):
                curves.evaluate_runs(real_path,[run],out,ratios=(1,2),folds=2,repeats=1,resume=True)
                with self.assertRaisesRegex(ValueError,'changed'):
                    curves.evaluate_runs(real_path,[run],out,ratios=(.5,1,2),folds=2,repeats=1,resume=True)
            metrics=pd.read_csv(out/'metrics.csv')
            with patch.object(curves,'load_run',return_value=(self.manifest(),self.synthetic)), \
                    patch.object(utility,'PROTOCOL_VERSION','incompatible_tuning_protocol'):
                with self.assertRaisesRegex(ValueError,'changed'):
                    curves.evaluate_runs(real_path,[run],out,ratios=(1,2),folds=2,repeats=1,resume=True)
            self.assertEqual(set(metrics.domain),set(schema.DISORDER_NAMES)|{'all_domains','eligible_domains'})
            self.assertTrue(metrics.loc[metrics.run!='real','generation_seed'].eq(42).all())
            self.assertTrue(metrics.loc[metrics.run=='real','generation_seed'].isna().all())


if __name__ == '__main__':
    unittest.main()
