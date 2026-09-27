"""Validate the data boundaries and independent choices of the revised protocol."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cocast import utility, learning_curves
from test_utility import questionnaire_fixture
from cocast import schema


class IndependentTuningTests(unittest.TestCase):
    def setUp(self):
        self.real_x = pd.DataFrame({'id': np.arange(12)})
        self.real_y = pd.Series(['tier_0', 'tier_1']*6)
        self.syn_x = pd.DataFrame({'id': np.arange(60)+1000})
        self.syn_y = pd.Series(['tier_0', 'tier_1']*30)

    def check_builder(self, x, y, classifier, seed, *, training_builder=None):
        self.assertIsNotNone(training_builder)
        splits, _, _ = utility.make_splits(y, 3, 1, seed)
        for train, validation in splits:
            fit_x, fit_y, audit = training_builder(train, validation)
            self.assertFalse(set(fit_x.id) & set(x.iloc[validation].id))
            self.assertEqual(len(fit_x), len(fit_y))
            self.observed.append((train, validation, fit_x, audit))
        return 'fixture', {}, 0, 'fixture', 3

    def test_augmentation_preserves_ratio_and_real_only_validation(self):
        for ratio in (.5, 1, 2, 5):
            self.observed = []
            with patch.object(utility, '_select_estimator', side_effect=self.check_builder):
                utility.select_augmented_estimator(self.real_x, self.real_y, 'logreg', 42,
                    ratio=ratio, x_synthetic=self.syn_x, y_synthetic=self.syn_y)
            for train, validation, x, audit in self.observed:
                self.assertEqual((x.id < 1000).sum(), len(train))
                self.assertEqual((x.id >= 1000).sum(), int(np.ceil(ratio*len(train))))
                self.assertTrue((self.real_x.iloc[validation].id < 1000).all())
                self.assertEqual(audit['construction'], 'synthetic_augmentation')

    def test_real_resampling_occurs_inside_inner_training(self):
        self.observed = []
        with patch.object(utility, '_select_estimator', side_effect=self.check_builder):
            utility.select_augmented_estimator(self.real_x, self.real_y, 'histgb', 42,
                ratio=5, requested_y=self.syn_y, resample_real=True)
        for train, _, x, audit in self.observed:
            self.assertEqual(len(x), 6*len(train))
            self.assertTrue(set(x.id) <= set(self.real_x.iloc[train].id))
            self.assertEqual(audit['construction'], 'real_resampling_after_inner_split')

    def test_synthetic_copies_never_cross_inner_boundary(self):
        ids = np.repeat(np.arange(12), 5)
        self.observed = []
        with patch.object(utility, '_select_estimator', side_effect=self.check_builder):
            utility.select_synthetic_estimator(self.syn_x.iloc[ids], self.syn_y.iloc[ids], ids,
                                              'logreg', 42)
        for train, _, x, audit in self.observed:
            self.assertEqual(len(x), 5*len(train))
            self.assertEqual(audit['construction'], 'synthetic_resampling_after_source_split')

    def test_missing_inner_training_tier_is_flagged(self):
        selected, missing = utility._inner_resample(['tier_0']*4, ['tier_0','tier_4'], 20, 42)
        self.assertEqual(missing, ['tier_4'])
        self.assertEqual(len(selected), 20)
        self.assertTrue(set(selected) <= set(range(4)))

    @staticmethod
    def condition_sensitive_predict(estimator, x, y, validation, *, diagnostics=None):
        # Deliberately give different optimal C values to the training conditions.
        ids = x[schema.AGE_COL]
        optimal = 10.0 if (ids >= 1000).any() else (1.0 if ids.duplicated().any() else .1)
        truth = np.where(validation[schema.AGE_COL].to_numpy()%2 == 0, 'tier_0', 'tier_1')
        if estimator.named_steps['model'].C != optimal:
            truth = np.where(truth == 'tier_0', 'tier_1', 'tier_0')
        return truth

    def test_standard_and_learning_curves_tune_augmentation_independently(self):
        real, synthetic = questionnaire_fixture(12), questionnaire_fixture(60)
        real[schema.AGE_COL] = np.arange(12)+20
        synthetic[schema.AGE_COL] = np.arange(60)+1000
        with patch.object(utility, 'CLASSIFIERS', ('logreg',)), \
                patch.object(utility, 'fit_predict', side_effect=self.condition_sensitive_predict):
            standard = utility.evaluate_utility(real, {'synth':synthetic.iloc[:12]}, folds=2,repeats=1,
                matched_resampling_runs=['synth'], targeted_domains={'synth':['depression']})
            curves = learning_curves.evaluate_learning_curves(real, {'synth':synthetic},
                folds=2,repeats=1,ratios=(1,5),domains=['depression'])
        for frame in (standard, curves):
            import json
            chosen = frame.selected_parameters_json.map(json.loads).map(lambda p:p['C'])
            self.assertTrue(chosen[frame.protocol == 'TRTR'].eq(.1).all())
            self.assertTrue(chosen[frame.protocol == 'TAUG'].eq(10).all())
            self.assertTrue(chosen[frame.protocol == 'TRTR_RESAMPLED'].eq(1).all())
            self.assertTrue(frame.utility_protocol_version.eq(utility.PROTOCOL_VERSION).all())


if __name__ == '__main__':
    unittest.main()
