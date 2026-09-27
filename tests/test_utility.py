"""Classifier utility checks on artificial questionnaire tables only."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone as sklearn_clone

from cocast import schema, utility


def questionnaire_fixture(n: int = 12, *, vary: bool = True) -> pd.DataFrame:
    data = {
        schema.AGE_COL: np.arange(n, dtype=float) % 10 + 18,
        schema.SEX_COL: np.arange(n) % 2 + 1,
    }
    for disorder in schema.DISORDER_NAMES:
        for column in schema.item_columns(disorder):
            data[column] = np.arange(n) % 2 if vary else np.zeros(n, dtype=int)
    result = pd.DataFrame(data)
    result.loc[0, schema.AGE_COL] = np.nan
    return result


class UtilityMetricAndPipelineTests(unittest.TestCase):
    def test_target_items_never_enter_predictors(self):
        for disorder in schema.DISORDER_NAMES:
            predictors = utility.predictor_columns(disorder)
            self.assertFalse(set(predictors) & set(schema.item_columns(disorder)))
            self.assertEqual(len(predictors), 62 if disorder == "depression" else 61)
            self.assertIn(schema.AGE_COL, predictors)
            self.assertIn(schema.SEX_COL, predictors)

    def test_imputer_learns_only_classifier_training_values(self):
        training = pd.DataFrame({"age": [18, np.nan, 22, 24]})
        targets = pd.Series(["tier_0", "tier_1", "tier_0", "tier_1"])
        test = pd.DataFrame({"age": [1000, np.nan]})
        estimator = utility.estimator_candidates("logreg", seed=3)[0][1]
        clones = []

        def capture_clone(candidate):
            cloned = sklearn_clone(candidate)
            clones.append(cloned)
            return cloned

        with patch("cocast.utility.clone", side_effect=capture_clone):
            prediction = utility.fit_predict(estimator, training, targets, test)
        self.assertEqual(len(prediction), 2)
        self.assertEqual(clones[0].named_steps["imputer"].statistics_[0], 22)
        self.assertFalse(hasattr(estimator.named_steps["imputer"], "statistics_"))

    def test_histogram_boosting_refit_has_no_implicit_holdout(self):
        candidate = utility.estimator_candidates("histgb", 42)[0][1]
        self.assertEqual(list(candidate.named_steps), ["model"])
        self.assertFalse(candidate.named_steps["model"].early_stopping)

    def test_single_class_prediction_does_not_fit_classifier(self):
        estimator = utility.estimator_candidates("logreg", 2)[0][1]
        with patch("cocast.utility.clone", side_effect=AssertionError("No estimator fit should occur")):
            prediction = utility.fit_predict(
                estimator, pd.DataFrame({"age": [np.nan, 21]}),
                pd.Series(["tier_3", "tier_3"]), pd.DataFrame({"age": [22, 23]}),
            )
        self.assertEqual(prediction.tolist(), ["tier_3", "tier_3"])

    def test_missing_training_outcome_is_not_imputed(self):
        estimator = utility.estimator_candidates("logreg", 1)[0][1]
        with self.assertRaisesRegex(ValueError, "without imputation"):
            utility.fit_predict(
                estimator, pd.DataFrame({"age": [18, 20]}),
                pd.Series(["tier_0", np.nan]), pd.DataFrame({"age": [19]}),
            )

    def test_fold_strategy_handles_sparse_and_singleton_tiers(self):
        sparse = pd.Series(["tier_0"] * 8 + ["tier_1"] * 2)
        splits, strategy, actual = utility.make_splits(sparse, 5, 3, 42)
        self.assertEqual(actual, 2)
        self.assertEqual(len(splits), 6)
        self.assertEqual(strategy, "repeated_stratified")
        singleton = pd.Series(["tier_0"] * 8 + ["tier_4"])
        splits, strategy, actual = utility.make_splits(singleton, 5, 1, 42)
        self.assertEqual(actual, 5)
        self.assertEqual(strategy, "repeated_unstratified_singleton_tier")
        self.assertEqual(sorted(np.concatenate([test for _, test in splits]).tolist()), list(range(9)))

    def test_fixed_five_label_macro_and_absent_recall_are_explicit(self):
        labels = np.array(["tier_0", "tier_1", "tier_2", "tier_3"])
        metrics = utility.classification_metrics(labels, labels)
        self.assertAlmostEqual(metrics["f1_macro"], 0.8)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["tier_4_support"], 0)
        self.assertTrue(np.isnan(metrics["tier_4_recall"]))

    def test_per_tier_confusion_counts_are_reconstructable(self):
        metrics = utility.classification_metrics(
            np.array(["tier_0", "tier_0", "tier_1"]),
            np.array(["tier_0", "tier_1", "tier_1"]),
        )
        self.assertEqual((metrics["tier_0_tp"], metrics["tier_0_fp"], metrics["tier_0_fn"]), (1, 0, 1))
        self.assertEqual((metrics["tier_1_tp"], metrics["tier_1_fp"], metrics["tier_1_fn"]), (1, 1, 0))
        self.assertEqual(metrics["tier_0_recall"], 0.5)
        self.assertEqual(metrics["tier_1_recall"], 1)


class UtilityEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real = questionnaire_fixture()
        cls.results = utility.evaluate_utility(
            cls.real, {"aggregate_4b_s42": cls.real.copy(), "aggregate_4b_s43": cls.real.copy()},
            folds=2, repeats=1, seed=42,
        )

    def test_real_folds_shared_and_reference_only_emitted_once(self):
        results = self.results
        grouping = ["domain", "classifier", "repeat", "fold"]
        self.assertTrue((results.groupby(grouping)["split_fingerprint"].nunique() == 1).all())
        self.assertTrue((results.groupby(grouping)["real_test_positions_json"].nunique() == 1).all())
        reference = results[results["protocol"] == "TRTR"]
        self.assertEqual(len(reference), 7 * 2 * 2)
        self.assertEqual(set(reference["run"]), {"real"})
        self.assertFalse(reference.duplicated(grouping).any())
        self.assertTrue((results.loc[results["protocol"] == "TRTR", "tuning_source"]
                         == "real_outer_training_fold").all())
        self.assertTrue((results.loc[results["protocol"] == "TAUG", "tuning_source"]
                         == "real_inner_validation_augmented_training").all())
        self.assertTrue((results.loc[results["protocol"] == "TSTR", "tuning_source"]
                         == "synthetic_training_subset").all())

    def test_training_sizes_match_and_each_fit_has_a_stopping_audit(self):
        synthetic = self.results[self.results["protocol"].isin(["TSTR", "TAUG"])]
        self.assertTrue((synthetic["n_real_train"] == synthetic["n_synth_train"]).all())
        self.assertTrue(self.results.utility_protocol_version.eq(utility.PROTOCOL_VERSION).all())
        self.assertTrue(self.results.fit_stopping_reason.notna().all())
        self.assertTrue(self.results.tuning_stopping_json.notna().all())
        self.assertTrue((self.results["status"] == "complete").all())
        self.assertTrue(np.isfinite(self.results["f1_macro"]).all())

    def test_identical_synthetic_pools_receive_identical_samples_and_scores(self):
        first = self.results[self.results["run"] == "aggregate_4b_s42"].reset_index(drop=True)
        second = self.results[self.results["run"] == "aggregate_4b_s43"].reset_index(drop=True)
        pd.testing.assert_frame_equal(first.drop(columns=["run", "method"]), second.drop(columns=["run", "method"]))

    def test_real_train_test_positions_are_disjoint(self):
        for row in self.results[self.results["protocol"] == "TRTR"].itertuples():
            train = set(json.loads(row.real_train_positions_json))
            test = set(json.loads(row.real_test_positions_json))
            self.assertFalse(train & test)
            self.assertEqual(train | test, set(range(len(self.real))))

    def test_missing_real_target_is_excluded_only_for_that_domain(self):
        real = questionnaire_fixture(8, vary=False)
        synthetic = questionnaire_fixture(8, vary=False)
        real.loc[0, schema.item_columns("depression")[0]] = np.nan
        result = utility.evaluate_utility(real, {"test": synthetic}, folds=2, repeats=1)
        depression = result[result["domain"] == "depression"]
        self.assertTrue((depression["n_missing_real_target"] == 1).all())
        for row in depression.itertuples():
            self.assertNotIn(0, json.loads(row.real_test_positions_json))
            self.assertNotIn(0, json.loads(row.real_train_positions_json))
        panic = result[result["domain"] == "panic"]
        self.assertTrue((panic["n_missing_real_target"] == 0).all())
        self.assertIn(0, set(sum(panic["real_test_positions_json"].map(json.loads).tolist(), [])))

    def test_unscorable_synthetic_targets_report_failure_instead_of_fake_scores(self):
        real = questionnaire_fixture(8, vary=False)
        synthetic = real.copy()
        synthetic[schema.item_columns("depression")[0]] = np.nan
        result = utility.evaluate_utility(real, {"test": synthetic}, folds=2, repeats=1)
        affected = result[(result["domain"] == "depression") & (result["run"] == "test")]
        self.assertTrue((affected["status"] == "no_scorable_synthetic_target").all())
        self.assertTrue(affected["f1_macro"].isna().all())
        self.assertTrue((affected["n_missing_synthetic_target"] == 8).all())

    def test_small_synthetic_pool_discloses_replacement(self):
        real = questionnaire_fixture(8, vary=False)
        synthetic = questionnaire_fixture(2, vary=False)
        result = utility.evaluate_utility(real, {"small": synthetic}, folds=2, repeats=1)
        affected = result[result["run"] == "small"]
        self.assertTrue(affected["synthetic_sampling_with_replacement"].all())
        self.assertTrue((affected["n_synth_train"] == 4).all())


if __name__ == "__main__":
    unittest.main()
