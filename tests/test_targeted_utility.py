"""Targeted utility design checks using artificial records and local estimators."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cocast import schema, utility
from cocast.evaluation import evaluate_runs
from test_utility import questionnaire_fixture


ELIGIBLE = ['depression', 'specific_phobia', 'social_anxiety', 'panic', 'generalized_anxiety']


class TargetedUtilityTests(unittest.TestCase):
    def fake_select(self, x, y, classifier, seed, **kwargs):
        source = 'synthetic' if (x[schema.AGE_COL] >= 1000).all() else 'real'
        self.tuning_calls.append((source, x.copy(), y.copy()))
        return source, {'test_source': source}, 0.5, 'fixture_inner_cv', 2

    def fake_predict(self, estimator, x_train, y_train, x_test, *, diagnostics=None):
        self.fit_calls.append((estimator, x_train.copy(), y_train.copy(), x_test.copy()))
        return np.repeat(y_train.iloc[0], len(x_test))

    def setUp(self):
        self.tuning_calls = []
        self.fit_calls = []

    def evaluate_fast(self, real, synthetic, **kwargs):
        with patch('cocast.utility._select_estimator', side_effect=self.fake_select), \
                patch('cocast.utility.fit_predict', side_effect=self.fake_predict):
            return utility.evaluate_utility(real, synthetic, folds=2, repeats=1, **kwargs)

    def test_tstr_tuning_uses_only_synthetic_rows_and_its_own_estimator(self):
        real = questionnaire_fixture(12)
        synthetic = questionnaire_fixture(12)
        synthetic[schema.AGE_COL] = np.arange(12) + 1000
        results = self.evaluate_fast(real, {'natural': synthetic, 'targeted': synthetic.copy()})
        tstr = results[results.protocol == 'TSTR']
        self.assertTrue((tstr.tuning_source == 'synthetic_training_subset').all())
        self.assertEqual(set(tstr.selected_parameters_json.map(json.loads).map(lambda v: v['test_source'])), {'synthetic'})
        other = results[results.protocol != 'TSTR']
        self.assertEqual(set(other.selected_parameters_json.map(json.loads).map(lambda v: v['test_source'])), {'real'})
        for estimator, x_train, _, x_test in self.fit_calls:
            if estimator == 'synthetic':
                self.assertTrue((x_train[schema.AGE_COL] >= 1000).all())
            self.assertTrue((x_test[schema.AGE_COL].dropna() < 1000).all())
        for row in tstr.itertuples():
            indices = json.loads(row.synthetic_train_positions_json)
            candidates = [x for source, x, _ in self.tuning_calls if source == 'synthetic']
            self.assertTrue(any(x[schema.AGE_COL].tolist() == synthetic.iloc[indices][schema.AGE_COL].tolist()
                                for x in candidates))

    def test_replacement_copies_do_not_leak_across_synthetic_inner_cv(self):
        real = questionnaire_fixture(12)
        synthetic = questionnaire_fixture(2)
        synthetic[schema.AGE_COL] = [1000, 1001]
        results = self.evaluate_fast(real, {'small': synthetic})
        tstr = results[results.protocol == 'TSTR']
        self.assertTrue(tstr.synthetic_sampling_with_replacement.all())
        self.assertTrue((tstr.n_synth_train == 6).all())
        for source, x, _ in self.tuning_calls:
            if source == 'synthetic':
                self.assertLessEqual(len(x), 2)
                self.assertTrue(x[schema.AGE_COL].is_unique)
        self.assertTrue((tstr.n_tuning_records <= 2).all())

    def test_resampling_matches_counts_without_using_any_test_positions(self):
        real = questionnaire_fixture(12)
        results = self.evaluate_fast(real, {'targeted': real.copy()},
            matched_resampling_runs=['targeted'], targeted_domains={'targeted': ELIGIBLE})
        comparison = results[results.protocol == 'TRTR_RESAMPLED']
        self.assertEqual(set(comparison.domain), set(ELIGIBLE))
        self.assertEqual(set(results.loc[results.protocol == 'TSTR', 'domain']), set(schema.DISORDER_NAMES))
        self.assertFalse(comparison.resampling_support_restricted.any())
        for row in comparison.itertuples():
            selected = json.loads(row.resampled_real_positions_json)
            train = set(json.loads(row.real_train_positions_json))
            test = set(json.loads(row.real_test_positions_json))
            self.assertTrue(set(selected).issubset(train))
            self.assertFalse(set(selected) & test)
            self.assertEqual(len(selected), row.requested_resampled_rows)
            self.assertEqual(json.loads(row.requested_tier_counts_json), json.loads(row.resampled_tier_counts_json))
            achieved = schema.score_table(real.iloc[selected])[schema.tier_col(row.domain)].value_counts().to_dict()
            self.assertEqual(achieved, json.loads(row.resampled_tier_counts_json))

    def test_absent_training_tiers_are_not_fabricated_and_fallback_is_flagged(self):
        real = questionnaire_fixture(8, vary=False)
        synthetic = questionnaire_fixture(8, vary=False)
        synthetic[schema.item_columns('panic')] = 1
        result = self.evaluate_fast(real, {'targeted': synthetic},
            matched_resampling_runs=['targeted'], targeted_domains={'targeted': ['panic']})
        comparison = result[result.protocol == 'TRTR_RESAMPLED']
        self.assertTrue(comparison.resampling_support_restricted.all())
        self.assertTrue((comparison.resampling_match_policy == 'uniform_available_tiers_no_overlap').all())
        for row in comparison.itertuples():
            self.assertEqual(json.loads(row.unavailable_tiers_json), ['tier_1'])
            self.assertEqual(json.loads(row.resampled_tier_counts_json), {'tier_0': 4})
            self.assertEqual(row.n_resampled_train, row.requested_resampled_rows)
        self.assertTrue((comparison.tier_1_support == 0).all())
        self.assertTrue(comparison.tier_1_recall.isna().all())

    def test_resampling_requires_explicit_targeted_domain_provenance(self):
        real = questionnaire_fixture(8, vary=False)
        with self.assertRaisesRegex(ValueError, 'explicit targeted_domains'):
            self.evaluate_fast(real, {'targeted': real.copy()}, matched_resampling_runs=['targeted'])

    def test_repeat_summary_pools_counts_before_rare_tier_recall(self):
        rows = []
        # Tier 1 recall is 1/1 in fold 1 and 0/3 in fold 2. Pooled recall
        # must be 1/4, rather than the unweighted fold recall mean of 1/2.
        for fold, (true, pred) in enumerate([
            (['tier_0', 'tier_1'], ['tier_0', 'tier_1']),
            (['tier_0', 'tier_1', 'tier_1', 'tier_1'], ['tier_0'] * 4),
        ], 1):
            rows.append({'run': 'targeted', 'method': 'targeted', 'classifier': 'logreg',
                         'protocol': 'TSTR', 'repeat': 1, 'domain': 'depression',
                         'fold': fold, 'status': 'complete', 'n_real_test': len(true),
                         **utility.classification_metrics(true, np.array(pred))})
        result = utility.summarize_utility_repeats(pd.DataFrame(rows), ['depression'])
        domain = result[result.domain == 'depression'].set_index('metric').value
        self.assertEqual(domain['tier_1_recall'], 0.25)
        self.assertEqual(domain['tier_1_support'], 4)
        self.assertEqual(domain['recall_macro_observed'], 0.625)
        self.assertTrue(np.isnan(domain['tier_4_recall']))
        primary = result[result.domain == 'targeted_domains'].set_index('metric').value
        self.assertEqual(primary['recall_macro_observed'], 0.625)
        self.assertEqual(primary['n_supported_tiers'], 2)
        self.assertEqual(primary['n_declared_tiers'], 5)

    def test_evaluation_retains_generation_identity_and_only_three_output_files(self):
        real = questionnaire_fixture(8, vary=False)
        base_manifest = {'status': 'complete', 'n_patients': len(real), 'context': 'own',
                         'settings': {'model': 'fixture', 'seed': 44, 'backend': 'fixture', 'precision': 'bf16'}}
        manifests = {'natural': {**base_manifest, 'regime': 'natural'},
                     'targeted': {**base_manifest, 'regime': 'targeted',
                                  'profile_metadata': {'targeted_domains': ELIGIBLE}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_path, output = root / 'real.csv', root / 'review'
            real.to_csv(real_path, index=False)
            with patch('cocast.characterisation.characterise', return_value=[]), patch('cocast.evaluation.load_run', side_effect=lambda path: (manifests[Path(path).name], real.copy())), \
                    patch('cocast.utility._select_estimator', side_effect=self.fake_select), \
                    patch('cocast.utility.fit_predict', side_effect=self.fake_predict):
                evaluate_runs(real_path, [root / 'natural', root / 'targeted'], output,
                              utility=True, folds=2, repeats=1)
            self.assertEqual({p.name for p in output.iterdir()}, {'manifest.json', 'metrics.csv', 'utility_folds.csv'})
            folds = pd.read_csv(output / 'utility_folds.csv')
            self.assertTrue((folds.loc[folds.run != 'real', 'generation_seed'] == 44).all())
            self.assertTrue(folds.loc[folds.run == 'real', 'generation_seed'].isna().all())
            metrics = pd.read_csv(output / 'metrics.csv')
            self.assertFalse(((metrics.run == 'fixture_targeted_own_seed44') & metrics.family.isin(['fidelity', 'proximity'])).any())
            primary = metrics[(metrics.family == 'utility_repeat') & (metrics.domain == 'targeted_domains')
                              & (metrics.metric == 'recall_macro_observed') & (metrics.protocol == 'TSTR')]
            self.assertEqual(set(primary.run), {'fixture_natural_own_seed44', 'fixture_targeted_own_seed44'})
            metadata = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(metadata['utility']['targeted_domains'], {'fixture_targeted_own_seed44': ELIGIBLE})
            self.assertIn('synthetic training records only', metadata['utility']['tuning']['TSTR'])

    def test_old_targeted_manifest_without_domains_is_rejected_before_metrics(self):
        real = questionnaire_fixture(8, vary=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_path, output = root / 'real.csv', root / 'review'
            real.to_csv(real_path, index=False)
            with patch('cocast.evaluation.load_run', return_value=({'regime': 'targeted'}, real)), \
                    patch('cocast.evaluation.proximity') as proximity:
                with self.assertRaisesRegex(ValueError, 'targeted profile metadata must declare targeted_domains'):
                    evaluate_runs(real_path, [root / 'targeted'], output, utility=True)
                proximity.assert_not_called()
            self.assertFalse(output.exists())

    def test_utility_only_skips_fidelity_and_proximity(self):
        real = questionnaire_fixture(8, vary=False)
        manifest = {'status':'complete','n_patients':8,'regime':'natural',
                    'settings':{'model':'fixture','seed':42}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real.to_csv(root/'real.csv', index=False)
            with patch('cocast.evaluation.load_run',return_value=(manifest,real)), \
                    patch('cocast.evaluation.fidelity',side_effect=AssertionError('No fidelity call')), \
                    patch('cocast.evaluation.proximity',side_effect=AssertionError('No proximity call')), \
                    patch('cocast.utility._select_estimator',side_effect=self.fake_select), \
                    patch('cocast.utility.fit_predict',side_effect=self.fake_predict):
                evaluate_runs(root/'real.csv',[root/'fixture'],root/'evaluation',
                              utility_only=True,folds=2,repeats=1)
            metadata = json.loads((root/'evaluation/manifest.json').read_text())
            self.assertEqual(metadata['evaluation_scope'],'utility_only')
            self.assertEqual(metadata['utility']['protocol_version'],utility.PROTOCOL_VERSION)
            metrics = pd.read_csv(root/'evaluation/metrics.csv')
            self.assertEqual(set(metrics.family),{'utility_repeat'})


if __name__ == '__main__':
    unittest.main()
