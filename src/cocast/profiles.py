"""Aggregate-only Gaussian-copula profiles with attainable score targets.

Generation accepts cohort summaries and never reads individual records.
``compute_aggregates`` is a separate data-custodian operation that extracts
those summaries from a locally supplied table.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import norm
import yaml

from . import schema

AGE_BIN_EDGES = (18, 19, 20, 21, 23, 52)
EIGENVALUE_FLOOR = 1e-8
# Rounding a seven-variable matrix to two decimals can perturb each of the
# six off-diagonal entries of a row by at most 0.005.
ROUNDING_EIGENVALUE_TOLERANCE = (len(schema.DISORDER_NAMES) - 1) * 0.005 + 1e-10


def _probabilities(values: dict, labels: list[str], name: str) -> dict[str, float]:
    if not isinstance(values, dict) or set(values) != set(labels):
        raise ValueError(f"{name} must contain exactly {labels}.")
    if any(isinstance(values[label], (bool, np.bool_)) for label in labels):
        raise ValueError(f"{name} must contain numeric probabilities, not Boolean values.")
    try:
        probabilities = np.asarray([values[label] for label in labels], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains nonnumeric probabilities.") from exc
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or probabilities.sum() <= 0
    ):
        raise ValueError(f"{name} requires finite probabilities in [0, 1] with a positive total.")
    probabilities /= probabilities.sum()
    return {label: float(probability) for label, probability in zip(labels, probabilities)}


def repair_correlation(matrix: object) -> tuple[np.ndarray, dict]:
    """Validate a rounded correlation matrix and repair small PSD violations.

    This is eigenvalue clipping followed by diagonal rescaling. It is not
    an optimization for the nearest correlation matrix. Large violations
    are rejected rather than silently changing the supplied dependence.
    """
    try:
        correlation = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Correlation matrix must be numeric.") from exc
    dimension = len(schema.DISORDER_NAMES)
    if correlation.shape != (dimension, dimension) or not np.isfinite(correlation).all():
        raise ValueError(f"Correlation matrix must be finite and have shape ({dimension}, {dimension}).")
    if not np.allclose(correlation, correlation.T, rtol=0, atol=1e-10):
        raise ValueError("Correlation matrix must be symmetric in the declared disorder order.")
    if not np.allclose(np.diag(correlation), 1, rtol=0, atol=1e-10):
        raise ValueError("Correlation matrix diagonal must equal one.")
    if np.any(np.abs(correlation) > 1 + 1e-10):
        raise ValueError("Correlation coefficients must lie in [-1, 1].")
    correlation = (correlation + correlation.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    minimum = float(eigenvalues.min())
    if minimum < -ROUNDING_EIGENVALUE_TOLERANCE:
        raise ValueError(
            "Correlation matrix has a negative eigenvalue too large to attribute "
            f"to two-decimal rounding ({minimum:.6g})."
        )
    repaired = correlation.copy()
    if minimum < EIGENVALUE_FLOOR:
        repaired = (eigenvectors * np.maximum(eigenvalues, EIGENVALUE_FLOOR)) @ eigenvectors.T
        scale = np.sqrt(np.diag(repaired))
        repaired /= np.outer(scale, scale)
        np.fill_diagonal(repaired, 1.0)
    metadata = {
        "method": "eigenvalue_clipping_then_diagonal_rescaling",
        "applied": bool(minimum < EIGENVALUE_FLOOR),
        "minimum_input_eigenvalue": minimum,
        "eigenvalue_floor": EIGENVALUE_FLOOR,
        "maximum_absolute_change": float(np.max(np.abs(repaired - correlation))),
    }
    return repaired, metadata


def parse_age_bin(label: str) -> tuple[int, int]:
    if not isinstance(label, str) or re.fullmatch(r"\d+-\d+", label) is None:
        raise ValueError(f"Age bin must have the form 'lower-upper': {label!r}")
    lower, upper = (int(part) for part in label.split("-"))
    if lower > upper:
        raise ValueError(f"Age bin endpoints are reversed: {label}")
    return lower, upper


def validate_aggregates(aggregates: dict) -> dict:
    """Return an independent validated copy with normalized sampling marginals."""
    if not isinstance(aggregates, dict):
        raise ValueError("Aggregate input must be a mapping.")
    result = deepcopy(aggregates)
    if result.get("disorders") != schema.DISORDER_NAMES:
        raise ValueError(f"Aggregate disorder order must be {schema.DISORDER_NAMES}.")
    correlation, repair = repair_correlation(result.get("correlation_probit_2dp"))
    result["correlation_sampling"] = correlation.tolist()
    result["correlation_repair"] = repair
    prevalence = result.get("tier_prevalences")
    if not isinstance(prevalence, dict) or set(prevalence) != set(schema.DISORDER_NAMES):
        raise ValueError("Tier prevalences must include exactly the seven declared disorders.")
    result["tier_prevalences"] = {
        disorder: _probabilities(prevalence[disorder], schema.TIER_LABELS, f"{disorder} prevalence")
        for disorder in schema.DISORDER_NAMES
    }
    result["sex_ratio"] = _probabilities(result.get("sex_ratio"), ["male", "female"], "Sex ratio")
    missing_rate = result.get("age_missing_rate", 0.0)
    if isinstance(missing_rate, (bool, np.bool_)):
        raise ValueError("Age missingness must be a probability.")
    try:
        missing_rate = float(missing_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("Age missingness must be a probability.") from exc
    if not np.isfinite(missing_rate) or not 0 <= missing_rate <= 1:
        raise ValueError("Age missingness must be a finite probability in [0, 1].")
    result["age_missing_rate"] = missing_rate
    histogram = result.get("age_histogram")
    if not isinstance(histogram, dict) or (not histogram and missing_rate < 1):
        raise ValueError("An age histogram is required when observed ages may be generated.")
    bins = sorted(histogram, key=parse_age_bin)
    endpoints = [parse_age_bin(label) for label in bins]
    if any(left[1] >= right[0] for left, right in zip(endpoints, endpoints[1:])):
        raise ValueError("Age histogram bins must not overlap.")
    result["age_histogram"] = _probabilities(histogram, bins, "Age histogram") if bins else {}
    if "n" in result and (
        isinstance(result["n"], (bool, np.bool_))
        or not isinstance(result["n"], (int, np.integer))
        or result["n"] < 1
    ):
        raise ValueError("Aggregate cohort size must be a positive integer.")
    return result


def load_aggregates(path: str | Path) -> dict:
    """Load JSON or YAML summary input without resolving any participant data."""
    with Path(path).open(encoding="utf-8") as handle:
        return validate_aggregates(yaml.safe_load(handle))


def attainable_tier_scores(disorder: str, tier: int) -> np.ndarray:
    """List scores realizable by valid integer questionnaire responses."""
    if (
        isinstance(tier, (bool, np.bool_))
        or not isinstance(tier, (int, np.integer))
        or tier not in range(5)
    ):
        raise ValueError("Tier must be an integer from 0 to 4.")
    spec = schema.DISORDERS[disorder]
    totals = np.arange(spec["n_items"] * spec["item_max"] + 1)
    scores = totals.astype(float) if spec["scoring"] == "sum" else totals / spec["n_items"]
    return scores[[schema.tier_of(score, disorder) == tier for score in scores]]


def inverse_tier_cdf(uniforms: np.ndarray, disorder: str, prevalences: dict) -> np.ndarray:
    """Choose a tier, then sample uniformly over its attainable scores."""
    values = np.asarray(uniforms, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Copula probabilities must lie in [0, 1].")
    probabilities = np.asarray(list(_probabilities(prevalences, schema.TIER_LABELS, "Tier prevalence").values()))
    active = np.flatnonzero(probabilities > 0)
    cumulative = np.r_[0.0, probabilities[active].cumsum()]
    cumulative[-1] = 1.0
    # A normal CDF can round to exactly one. Keep that draw in the final
    # positive-probability tier without assigning it to an empty tier.
    values = np.minimum(values, np.nextafter(1.0, 0.0))
    positions = np.searchsorted(cumulative[1:], values, side="right")
    tiers = active[positions]
    scores = np.empty(values.shape, dtype=float)
    for tier in range(5):
        selected = tiers == tier
        if not selected.any():
            continue
        support = attainable_tier_scores(disorder, tier)
        position = int(np.flatnonzero(active == tier)[0])
        fraction = (values[selected] - cumulative[position]) / probabilities[tier]
        index = np.minimum(np.floor(fraction * len(support)).astype(int), len(support) - 1)
        scores[selected] = support[index]
    return scores


def _category_indices(uniforms: np.ndarray, probabilities: list[float]) -> np.ndarray:
    weights = np.asarray(probabilities, dtype=float)
    active = np.flatnonzero(weights > 0)
    cumulative = np.cumsum(weights[active])
    cumulative[-1] = 1.0
    return active[np.searchsorted(cumulative, uniforms, side="right")]


def generate_profiles(
    aggregates: dict, n_patients: int, seed: int, prevalences: dict | None = None,
    *, dependence: str = "copula",
) -> list[dict]:
    """Generate flat profiles whose existing prefix survives larger requests.

    Optional prevalence overrides may change one or more domain marginals.
    The copula ablation independently samples each score from the same marginal CDF
    without applying the supplied cross-disorder correlations.
    Separate random streams make scores, sex, age and age missingness
    independent, and make each stream invariant to the requested row count.
    """
    if isinstance(n_patients, (bool, np.bool_)) or not isinstance(n_patients, (int, np.integer)) or n_patients < 1:
        raise ValueError("n_patients must be a positive integer.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if dependence not in ("copula", "independent"):
        raise ValueError("dependence must be 'copula' or 'independent'.")
    summary = validate_aggregates(aggregates)
    if prevalences is not None:
        if not isinstance(prevalences, dict) or not set(prevalences).issubset(schema.DISORDER_NAMES):
            raise ValueError("Prevalence overrides must use declared disorder names.")
        for disorder, distribution in prevalences.items():
            summary["tier_prevalences"][disorder] = _probabilities(
                distribution, schema.TIER_LABELS, f"{disorder} override",
            )
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(int(seed)).spawn(5)]
    score_rng, sex_rng, age_bin_rng, age_value_rng, missing_rng = streams
    independent = score_rng.standard_normal((n_patients, len(schema.DISORDER_NAMES)))
    if dependence == "copula":
        factor = np.linalg.cholesky(np.asarray(summary["correlation_sampling"]))
        # Explicit per-row contraction avoids batch-size-dependent BLAS paths.
        latent = np.einsum("ij,kj->ik", independent, factor, optimize=False)
    else:
        latent = independent
    uniforms = norm.cdf(latent)
    scores = {
        disorder: inverse_tier_cdf(uniforms[:, index], disorder, summary["tier_prevalences"][disorder])
        for index, disorder in enumerate(schema.DISORDER_NAMES)
    }
    sex_indices = _category_indices(sex_rng.random(n_patients), list(summary["sex_ratio"].values()))
    age_missing = missing_rng.random(n_patients) < summary["age_missing_rate"]
    ages: list[int | None] = [None] * n_patients
    histogram = summary["age_histogram"]
    if histogram:
        labels = list(histogram)
        age_bins = _category_indices(age_bin_rng.random(n_patients), list(histogram.values()))
        positions = age_value_rng.random(n_patients)
        for index in range(n_patients):
            lower, upper = parse_age_bin(labels[age_bins[index]])
            age = lower + int(np.floor(positions[index] * (upper - lower + 1)))
            ages[index] = None if age_missing[index] else age
    profiles = []
    for index in range(n_patients):
        patient = {
            "patient_id": f"syn_aggregate_{index + 1:05d}",
            "AGE": ages[index], "SEX": float(sex_indices[index] + 1),
        }
        for disorder in schema.DISORDER_NAMES:
            key = schema.disorder_key(disorder)
            score = float(scores[disorder][index])
            tier = schema.tier_of(score, disorder)
            family = "depression" if disorder == "depression" else "anxiety"
            patient[key] = schema.SEVERITY_LABELS[family][tier]
            patient[f"{key}_TIER_CODE"] = schema.TIER_LABELS[tier]
            patient[f"{key}_SCORE_TARGET"] = int(score) if family == "depression" else round(score, 1)
        profiles.append(patient)
    return profiles


def compute_aggregates(real_df: pd.DataFrame, source: str = "") -> dict:
    """Extract rounded summaries from complete questionnaires, retaining missing age.

    Correlations are calculated here from cohort records. They are not
    represented as correlations published by the source study. Age-bin
    proportions use observed ages only, with missingness reported separately.
    """
    schema.validate_real_table(real_df, source=source)
    if len(real_df) < 2:
        raise ValueError("At least two records are required to estimate correlations.")
    scored = schema.score_table(real_df)
    scores = scored[[schema.score_col(disorder) for disorder in schema.DISORDER_NAMES]]
    if (scores.nunique() < 2).any():
        raise ValueError("Every score domain must vary to estimate its probit correlation.")
    probabilities = scores.rank(method="average").to_numpy() / (len(scores) + 1)
    correlation = np.corrcoef(norm.ppf(probabilities).T)
    tiers = {}
    for disorder in schema.DISORDER_NAMES:
        counts = scored[schema.tier_col(disorder)].value_counts(normalize=True)
        tiers[disorder] = {
            tier: round(float(counts.get(tier, 0)), 3) for tier in schema.TIER_LABELS
        }
    sex = pd.to_numeric(real_df[schema.SEX_COL]).value_counts(normalize=True)
    observed_age = pd.to_numeric(real_df[schema.AGE_COL]).dropna()
    if not observed_age.empty and (
        observed_age.min() < AGE_BIN_EDGES[0] or observed_age.max() >= AGE_BIN_EDGES[-1]
    ):
        raise ValueError("Observed ages fall outside the declared study age bins (18 to 51 years).")
    histogram = {}
    if not observed_age.empty:
        counts, _ = np.histogram(observed_age, bins=AGE_BIN_EDGES)
        histogram = {
            f"{lower}-{upper - 1}": round(float(count / len(observed_age)), 3)
            for lower, upper, count in zip(AGE_BIN_EDGES, AGE_BIN_EDGES[1:], counts)
        }
    result = {
        "description": "Cohort-computed aggregate inputs for questionnaire profile generation.",
        "source": source,
        "correlation_source": "Computed from supplied cohort records using average-rank probit scores.",
        "n": int(len(real_df)),
        "n_complete_scores": int(len(scores)),
        "n_observed_age": int(len(observed_age)),
        "n_missing_age": int(real_df[schema.AGE_COL].isna().sum()),
        "disorders": list(schema.DISORDER_NAMES),
        "correlation_probit_2dp": np.round(correlation, 2).tolist(),
        "tier_prevalences": tiers,
        "sex_ratio": {
            label: round(float(sex.get(code, 0)), 3) for code, label in schema.SEX_LABELS.items()
        },
        "age_histogram": histogram,
        "age_missing_rate": float(real_df[schema.AGE_COL].isna().mean()),
        "rounding": {"correlation_decimals": 2, "marginal_decimals": 3},
    }
    # Validate without replacing the rounded release values by normalized
    # sampling probabilities. Loading/generation performs that normalization.
    validate_aggregates(result)
    return result
