"""Convergence failures, inner-fold stopping and complete-data refits."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cocast import utility


class StoppingTests(unittest.TestCase):
    def setUp(self):
        self.limits = threadpool_limits(limits=1)
        self.addCleanup(self.limits.restore_original_limits)
        rng = np.random.default_rng(82)
        self.x = pd.DataFrame(rng.normal(size=(120, 6)))
        self.y = pd.Series(np.where(self.x[0]+rng.normal(size=120) > 0, 'tier_1', 'tier_0'))

    def test_logistic_stops_before_cap_and_records_iterations(self):
        estimator = utility.estimator_candidates('logreg', 42)[0][1]
        audit = {}
        prediction = utility.fit_predict(estimator, self.x, self.y, self.x.iloc[:3], diagnostics=audit)
        self.assertEqual(len(prediction), 3)
        self.assertGreater(audit['fit_iterations'], 0)
        self.assertLess(audit['fit_iterations'], audit['fit_iteration_limit'])
        self.assertEqual(audit['fit_stopping_reason'], 'optimizer_converged')
        self.assertFalse(hasattr(estimator.named_steps['model'], 'coef_'))

    def test_nonconverged_final_fit_aborts_before_prediction(self):
        estimator = utility.estimator_candidates('logreg', 42)[0][1]
        estimator.set_params(model__max_iter=1)
        with self.assertRaisesRegex(utility.FitConvergenceError, 'No scores accepted'):
            utility.fit_predict(estimator, self.x, self.y, self.x.iloc[:3])

    def test_nonconverged_candidate_is_not_silently_skipped(self):
        estimator = utility.estimator_candidates('logreg', 42)[0][1]
        estimator.set_params(model__max_iter=1)
        with patch.object(utility, 'estimator_candidates', return_value=[({}, estimator)]):
            with self.assertRaises(utility.FitConvergenceError):
                utility._select_estimator(self.x, self.y, 'logreg', 42)

    def test_boosting_uses_explicit_disjoint_inner_validation_and_refits_all_rows(self):
        actual = utility.HistGradientBoostingClassifier.fit
        calls = []

        def spy(model, x, y, **kwargs):
            calls.append((x.index.tolist(), kwargs.get('X_val'), model.early_stopping))
            return actual(model, x, y, **kwargs)

        candidates = utility.estimator_candidates
        with patch.object(utility, 'estimator_candidates', side_effect=lambda c,s: candidates(c,s)[:1]), \
                patch.object(utility.HistGradientBoostingClassifier, 'fit', new=spy):
            estimator, params, _, _, folds = utility._select_estimator(self.x, self.y, 'histgb', 42)
            self.assertEqual(len(calls), folds)
            for training, validation, enabled in calls:
                self.assertTrue(enabled)
                self.assertIsNotNone(validation)
                self.assertFalse(set(training) & set(validation.index))
                self.assertEqual(set(training) | set(validation.index), set(self.x.index))
            audit = json.loads(utility.tuning_diagnostics(estimator)['tuning_stopping_json'])
            rounds = [row['best_round'] for row in audit['selected_folds']]
            self.assertEqual(params['max_iter'], int(np.ceil(np.median(rounds))))
            fit_audit = {}
            utility.fit_predict(estimator, self.x, self.y, self.x.iloc[:3], diagnostics=fit_audit)
            self.assertEqual(calls[-1][0], self.x.index.tolist())
            self.assertIsNone(calls[-1][1])
            self.assertFalse(calls[-1][2])
            self.assertEqual(fit_audit['fit_iterations'], params['max_iter'])
            self.assertEqual(fit_audit['fit_stopping_reason'], 'inner_cv_selected_rounds')

    def test_unseen_validation_tier_is_retained_for_predictions_and_flagged(self):
        estimator = utility.estimator_candidates('histgb', 42)[0][1]
        labels = self.y.iloc[90:].copy()
        labels.iloc[0] = 'tier_4'
        prediction, audit = utility._boost_validation_fit(
            estimator, self.x.iloc[:90], self.y.iloc[:90], self.x.iloc[90:], labels)
        self.assertEqual(len(prediction), len(labels))
        self.assertEqual(audit['n_unsupported_validation'], 1)
        self.assertEqual(audit['n_validation'], 30)
        self.assertAlmostEqual(audit['effective_tol'], utility.BOOST_TOL*30/29)
        self.assertEqual(audit['reason'], 'validation_loss_plateau')

    def test_boosting_cap_is_not_reported_as_a_plateau(self):
        estimator = utility.estimator_candidates('histgb', 42)[0][1]
        with patch.object(utility, 'BOOST_MAX_ITER', 1):
            with self.assertRaisesRegex(utility.FitConvergenceError, 'safety cap'):
                utility._boost_validation_fit(estimator, self.x.iloc[:90], self.y.iloc[:90],
                                              self.x.iloc[90:], self.y.iloc[90:])

    def test_no_supported_validation_class_is_explicitly_uninformative(self):
        estimator = utility.estimator_candidates('histgb', 42)[0][1]
        prediction, audit = utility._boost_validation_fit(
            estimator, self.x.iloc[:90], self.y.iloc[:90], self.x.iloc[90:],
            pd.Series(['tier_4']*30))
        self.assertEqual(len(prediction), 30)
        self.assertIsNone(audit['best_round'])
        self.assertEqual(audit['reason'], 'uninformative_validation')


if __name__ == '__main__':
    unittest.main()
