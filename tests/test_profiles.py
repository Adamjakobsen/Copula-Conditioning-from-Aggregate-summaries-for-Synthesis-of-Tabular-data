"""Aggregate profile regressions without participant files or model calls."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from cocast import schema
from cocast.profiles import (
    attainable_tier_scores, compute_aggregates, generate_profiles,
    inverse_tier_cdf, load_aggregates, repair_correlation, validate_aggregates,
)


def summary_fixture() -> dict:
    correlation = np.full((7, 7), 0.45)
    np.fill_diagonal(correlation, 1)
    return {
        "n": 564,
        "disorders": list(schema.DISORDER_NAMES),
        "correlation_probit_2dp": correlation.tolist(),
        "tier_prevalences": {
            disorder: dict.fromkeys(schema.TIER_LABELS, 0.2)
            for disorder in schema.DISORDER_NAMES
        },
        "sex_ratio": {"female": 0.68, "male": 0.32},
        "age_histogram": {"18-18": 0.2, "19-19": 0.2, "20-20": 0.2, "21-22": 0.2, "23-51": 0.2},
        "age_missing_rate": 2 / 564,
    }


def real_fixture(n: int = 564) -> pd.DataFrame:
    random = np.random.default_rng(57)
    data = {
        schema.AGE_COL: random.integers(18, 30, size=n).astype(float),
        schema.SEX_COL: random.integers(1, 3, size=n),
    }
    for disorder, spec in schema.DISORDERS.items():
        for column in schema.item_columns(disorder):
            data[column] = random.integers(0, spec["item_max"] + 1, size=n)
    table = pd.DataFrame(data)
    table.loc[:1, schema.AGE_COL] = np.nan
    return table


class AggregateValidationTests(unittest.TestCase):
    def test_order_is_declared_and_not_silently_reinterpreted(self):
        summary = summary_fixture()
        summary["disorders"].reverse()
        with self.assertRaisesRegex(ValueError, "order"):
            validate_aggregates(summary)

    def test_rounded_prevalences_normalized_without_changing_input(self):
        summary = summary_fixture()
        summary["tier_prevalences"]["depression"] = dict(
            zip(schema.TIER_LABELS, [0.333, 0.333, 0.333, 0, 0])
        )
        before = deepcopy(summary)
        normalized = validate_aggregates(summary)
        self.assertEqual(summary, before)
        self.assertAlmostEqual(normalized["tier_prevalences"]["depression"]["tier_0"], 1 / 3)
        self.assertEqual(list(normalized["sex_ratio"]), ["male", "female"])

    def test_load_json_and_normalize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(summary_fixture()))
            loaded = load_aggregates(path)
        self.assertEqual(loaded["n"], 564)
        self.assertIn("correlation_repair", loaded)

    def test_small_rounding_violation_is_repaired_and_disclosed(self):
        correlation = np.full((7, 7), -0.167)
        np.fill_diagonal(correlation, 1)
        repaired, metadata = repair_correlation(correlation)
        self.assertTrue(metadata["applied"])
        self.assertGreater(np.linalg.eigvalsh(repaired).min(), 0)
        np.testing.assert_allclose(np.diag(repaired), 1)
        self.assertLess(metadata["maximum_absolute_change"], 0.001)

    def test_singular_valid_matrix_can_be_sampled(self):
        summary = summary_fixture()
        summary["correlation_probit_2dp"] = np.ones((7, 7)).tolist()
        self.assertEqual(len(generate_profiles(summary, 3, 42)), 3)

    def test_malformed_correlations_are_rejected(self):
        valid = np.eye(7)
        asymmetric = valid.copy()
        asymmetric[0, 1] = 0.3
        out_of_range = valid.copy()
        out_of_range[0, 1] = out_of_range[1, 0] = 1.1
        wrong_diagonal = valid.copy()
        wrong_diagonal[0, 0] = 0.9
        nonfinite = valid.copy()
        nonfinite[1, 1] = np.nan
        severe = np.full((7, 7), -0.4)
        np.fill_diagonal(severe, 1)
        for matrix in [np.eye(6), asymmetric, out_of_range, wrong_diagonal, nonfinite, severe]:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    repair_correlation(matrix)

    def test_invalid_prevalences_do_not_reach_sampling(self):
        for invalid in [0, -0.1, float("nan"), float("inf"), True, 1.1]:
            summary = summary_fixture()
            if invalid == 0:
                summary["tier_prevalences"]["depression"] = dict.fromkeys(schema.TIER_LABELS, 0)
            else:
                summary["tier_prevalences"]["depression"]["tier_0"] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    generate_profiles(summary, 3, 42)

    def test_missing_or_extra_tier_rejected(self):
        summary = summary_fixture()
        summary["tier_prevalences"]["panic"]["tier_5"] = 0
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_aggregates(summary)

    def test_overlapping_age_bins_rejected(self):
        summary = summary_fixture()
        summary["age_histogram"] = {"18-20": 0.5, "20-30": 0.5}
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_aggregates(summary)


class ProfileGenerationTests(unittest.TestCase):
    def test_every_tier_uses_attainable_endpoints(self):
        anxiety_endpoints = [(0, 0.4), (0.5, 1.4), (1.5, 2.4), (2.5, 3.4), (3.5, 4)]
        depression_endpoints = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 27)]
        for disorder in schema.DISORDER_NAMES:
            endpoints = depression_endpoints if disorder == "depression" else anxiety_endpoints
            for tier, (lower, upper) in enumerate(endpoints):
                with self.subTest(disorder=disorder, tier=tier):
                    support = attainable_tier_scores(disorder, tier)
                    self.assertEqual(support[0], lower)
                    self.assertEqual(support[-1], upper)
                    self.assertTrue(all(schema.tier_of(score, disorder) == tier for score in support))
                    prevalence = {label: float(index == tier) for index, label in enumerate(schema.TIER_LABELS)}
                    sampled = inverse_tier_cdf(np.array([0, 1e-15, 0.5, 1 - 1e-15, 1]), disorder, prevalence)
                    self.assertEqual(sampled[0], lower)
                    self.assertEqual(sampled[-1], upper)
                    self.assertTrue(set(sampled).issubset(set(support)))

    def test_zero_prevalence_tiers_never_sampled_at_boundaries(self):
        prevalence = dict(zip(schema.TIER_LABELS, [0, 0.333, 0, 0.333, 0]))
        uniforms = np.array([0, np.nextafter(0.5, 0), 0.5, np.nextafter(1.0, 0), 1])
        scores = inverse_tier_cdf(uniforms, "panic", prevalence)
        self.assertEqual([schema.tier_of(score, "panic") for score in scores], [1, 1, 3, 3, 3])

    def test_extension_preserves_all_existing_patient_fields(self):
        summary = summary_fixture()
        longest = generate_profiles(summary, 1001, 42)
        for size in [1, 7, 99, 564]:
            self.assertEqual(generate_profiles(summary, size, 42), longest[:size])

    def test_same_seed_reproduces_profiles_other_seed_changes_them(self):
        summary = summary_fixture()
        self.assertEqual(generate_profiles(summary, 12, 7), generate_profiles(summary, 12, 7))
        self.assertNotEqual(generate_profiles(summary, 12, 7), generate_profiles(summary, 12, 8))

    def test_profile_fields_and_target_consistency(self):
        profiles = generate_profiles(summary_fixture(), 300, 14)
        for patient in profiles:
            self.assertIn(patient["SEX"], (1, 2))
            self.assertEqual(len(patient), 24)
            for disorder in schema.DISORDER_NAMES:
                key = schema.disorder_key(disorder)
                score = patient[f"{key}_SCORE_TARGET"]
                tier = schema.tier_of(score, disorder)
                self.assertEqual(patient[f"{key}_TIER_CODE"], f"tier_{tier}")
                self.assertIn(score, attainable_tier_scores(disorder, tier))
                self.assertAlmostEqual(score * (1 if disorder == "depression" else 10), round(score * (1 if disorder == "depression" else 10)))
        self.assertEqual(len({profile["patient_id"] for profile in profiles}), len(profiles))

    def test_targeted_override_changes_only_requested_domain_fields(self):
        summary = summary_fixture()
        natural = generate_profiles(summary, 100, 3)
        override = {"depression": dict(zip(schema.TIER_LABELS, [0, 0, 0, 0, 1]))}
        targeted = generate_profiles(summary, 100, 3, prevalences=override)
        for original, changed in zip(natural, targeted):
            self.assertEqual(changed["DEPRESSION_TIER_CODE"], "tier_4")
            self.assertEqual(
                {key: value for key, value in original.items() if not key.startswith("DEPRESSION")},
                {key: value for key, value in changed.items() if not key.startswith("DEPRESSION")},
            )
        self.assertEqual(summary, summary_fixture())

    def test_missing_age_is_explicit_and_zero_rate_is_complete(self):
        summary = summary_fixture()
        summary["age_missing_rate"] = 1
        summary["age_histogram"] = {}
        self.assertTrue(all(patient["AGE"] is None for patient in generate_profiles(summary, 15, 2)))
        summary = summary_fixture()
        summary["age_missing_rate"] = 0
        self.assertTrue(all(patient["AGE"] is not None for patient in generate_profiles(summary, 15, 2)))

    def test_sampling_respects_marginals_and_independent_demographics(self):
        summary = summary_fixture()
        summary["age_missing_rate"] = 0.1
        generated = generate_profiles(summary, 5000, 51)
        self.assertAlmostEqual(np.mean([patient["SEX"] == 1 for patient in generated]), 0.32, delta=0.025)
        self.assertAlmostEqual(np.mean([patient["AGE"] is None for patient in generated]), 0.1, delta=0.025)
        tiers = [patient["PANIC_TIER_CODE"] for patient in generated]
        for tier in schema.TIER_LABELS:
            self.assertAlmostEqual(tiers.count(tier) / len(tiers), 0.2, delta=0.025)
        depression = [patient["DEPRESSION_SCORE_TARGET"] for patient in generated]
        panic = [patient["PANIC_SCORE_TARGET"] for patient in generated]
        sex = [patient["SEX"] for patient in generated]
        self.assertGreater(np.corrcoef(depression, panic)[0, 1], 0.3)
        self.assertLess(abs(np.corrcoef(depression, sex)[0, 1]), 0.05)


class ScoringAndExtractionTests(unittest.TestCase):
    def test_all_564_records_retained_despite_two_missing_ages(self):
        real = real_fixture()
        summary = compute_aggregates(real, source="Locally supplied study cohort")
        self.assertEqual(summary["n"], 564)
        self.assertEqual(summary["n_complete_scores"], 564)
        self.assertEqual(summary["n_observed_age"], 562)
        self.assertEqual(summary["n_missing_age"], 2)
        self.assertEqual(summary["age_missing_rate"], 2 / 564)
        self.assertIn("Computed from supplied cohort", summary["correlation_source"])
        self.assertEqual(summary["disorders"], schema.DISORDER_NAMES)
        self.assertTrue(np.all(np.asarray(summary["correlation_probit_2dp"]) == np.round(summary["correlation_probit_2dp"], 2)))
        for prevalences in summary["tier_prevalences"].values():
            self.assertTrue(all(probability == round(probability, 3) for probability in prevalences.values()))

    def test_missing_item_does_not_get_prorated_into_outcome(self):
        real = real_fixture(10)
        real.loc[0, schema.item_columns("depression")[0]] = np.nan
        scores = schema.score_table(real)
        self.assertTrue(pd.isna(scores.loc[0, schema.score_col("depression")]))
        self.assertTrue(pd.isna(scores.loc[0, schema.tier_col("depression")]))
        self.assertFalse(pd.isna(scores.loc[0, schema.score_col("panic")]))
        with self.assertRaisesRegex(ValueError, "missing responses"):
            compute_aggregates(real)

    def test_integer_and_range_validation_rejects_silent_coercion(self):
        for invalid in [2.9, -0.2, 4, np.inf, True]:
            real = real_fixture(10)
            column = schema.item_columns("depression")[0]
            real[column] = real[column].astype(object)
            real.loc[0, column] = invalid
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    schema.score_table(real)

    def test_invalid_sex_and_constant_scores_rejected(self):
        real = real_fixture(10)
        real.loc[0, schema.SEX_COL] = 3
        with self.assertRaisesRegex(ValueError, "sex codes"):
            compute_aggregates(real)
        real = real_fixture(10)
        real[schema.item_columns("depression")] = 0
        with self.assertRaisesRegex(ValueError, "must vary"):
            compute_aggregates(real)


if __name__ == "__main__":
    unittest.main()
