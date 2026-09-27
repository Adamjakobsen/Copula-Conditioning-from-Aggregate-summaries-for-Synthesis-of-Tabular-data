"""Declared questionnaire schema and scoring without outcome imputation.

The baseline cohort contains 69 questionnaire items across seven domains,
recorded sex and age. Missing age does not make a questionnaire incomplete.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AGE_COL = "W1_age_r"
SEX_COL = "W1_sex_r"
ID_COL = "ID"
SEX_LABELS = {1: "male", 2: "female"}

DISORDERS = {
    "depression": {
        "n_items": 9, "item_min": 0, "item_max": 3,
        "scoring": "sum", "tier_cutoffs": [4, 9, 14, 19],
    },
    **{
        disorder: {
            "n_items": 10, "item_min": 0, "item_max": 4,
            "scoring": "mean", "tier_cutoffs": [0.5, 1.5, 2.5, 3.5],
        }
        for disorder in (
            "separation_anxiety", "specific_phobia", "social_anxiety",
            "panic", "agoraphobia", "generalized_anxiety",
        )
    },
}
DISORDER_NAMES = list(DISORDERS)
TIER_LABELS = [f"tier_{index}" for index in range(5)]
SEVERITY_LABELS = {
    "depression": ["none", "mild", "moderate", "moderately severe", "severe"],
    "anxiety": ["none", "mild", "moderate", "severe", "extreme"],
}


def item_columns(disorder: str) -> list[str]:
    """Return the declared item order for one domain."""
    return [
        f"W1_{disorder}_it{index}"
        for index in range(1, DISORDERS[disorder]["n_items"] + 1)
    ]


ALL_ITEM_COLUMNS = [column for d in DISORDER_NAMES for column in item_columns(d)]
REAL_COLUMNS = [AGE_COL, SEX_COL, *ALL_ITEM_COLUMNS]


def tier_col(disorder: str) -> str:
    return f"{disorder}_tier_code"


def score_col(disorder: str) -> str:
    return f"{disorder}_score"


def disorder_key(disorder: str) -> str:
    """Return the domain key used in flat prompt profiles."""
    if disorder not in DISORDERS:
        raise ValueError(f"Unknown disorder: {disorder}")
    return disorder.upper().replace("_", " ")


def tier_of(score: float, disorder: str) -> int:
    """Assign an attainable sum or mean score to its operational tier."""
    spec = DISORDERS[disorder]
    value = float(score)
    maximum = spec["item_max"] * (spec["n_items"] if spec["scoring"] == "sum" else 1)
    if not np.isfinite(value) or value < 0 or value > maximum:
        raise ValueError(f"Invalid {disorder} score: {score!r}")
    side = "left" if spec["scoring"] == "sum" else "right"
    return int(np.searchsorted(spec["tier_cutoffs"], value, side=side))


def _numbers(series: pd.Series, column: str) -> np.ndarray:
    if series.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(f"{column} contains Boolean values instead of numeric responses.")
    try:
        numeric = pd.to_numeric(series, errors="raise")
        return numeric.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} contains nonnumeric values.") from exc


def _validate_items(df: pd.DataFrame, *, allow_missing: bool) -> None:
    missing = [column for column in ALL_ITEM_COLUMNS if column not in df]
    if missing:
        raise ValueError(f"Missing questionnaire columns: {missing}")
    for disorder, spec in DISORDERS.items():
        for column in item_columns(disorder):
            values = _numbers(df[column], column)
            observed = values[~np.isnan(values)]
            if not allow_missing and np.isnan(values).any():
                raise ValueError(f"{column} has missing responses.")
            if (
                not np.isfinite(observed).all()
                or np.any(observed != np.floor(observed))
                or np.any(observed < spec["item_min"])
                or np.any(observed > spec["item_max"])
            ):
                raise ValueError(
                    f"{column} must contain integer responses from "
                    f"{spec['item_min']} to {spec['item_max']}."
                )


def validate_real_table(df: pd.DataFrame, source: str = "") -> None:
    """Validate all study variables, retaining records whose age is missing."""
    if not df.columns.is_unique:
        raise ValueError(f"Real table {source or '<dataframe>'} has duplicate columns.")
    missing = [column for column in REAL_COLUMNS if column not in df]
    extra = [column for column in df if column not in REAL_COLUMNS]
    if missing or extra or df.empty:
        raise ValueError(
            f"Real table {source or '<dataframe>'} does not match the schema. "
            f"Missing: {missing}. Extra: {extra}. Rows: {len(df)}."
        )
    _validate_items(df, allow_missing=False)
    sex = _numbers(df[SEX_COL], SEX_COL)
    if not np.isin(sex, list(SEX_LABELS)).all():
        raise ValueError(f"{SEX_COL} must contain recorded sex codes 1 or 2 without missingness.")
    age = _numbers(df[AGE_COL], AGE_COL)
    observed_age = age[~np.isnan(age)]
    if (
        not np.isfinite(observed_age).all()
        or np.any(observed_age < 0)
        or np.any(observed_age != np.floor(observed_age))
    ):
        raise ValueError(f"{AGE_COL} must contain nonnegative integer ages or missing values.")


def validate_synthetic_items(
    df: pd.DataFrame, source: str = "", require_id: bool = True,
    *, allow_missing: bool = True,
) -> None:
    """Validate responses without truncating fractional or out-of-range values."""
    if not df.columns.is_unique:
        raise ValueError(f"Synthetic table {source or '<dataframe>'} has duplicate columns.")
    if require_id:
        if ID_COL not in df or df[ID_COL].isna().any() or df[ID_COL].duplicated().any():
            raise ValueError(f"Synthetic table {source or '<dataframe>'} requires unique nonmissing IDs.")
    _validate_items(df, allow_missing=allow_missing)


def score_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return seven scores and tier codes with the input index preserved.

    A domain with any missing response receives a missing score and tier.
    Observed responses must still be integers within their permitted range.
    No item proration or outcome imputation is performed.
    """
    if not df.columns.is_unique:
        raise ValueError("Questionnaire table has duplicate columns.")
    _validate_items(df, allow_missing=True)
    result = pd.DataFrame(index=df.index)
    for disorder, spec in DISORDERS.items():
        items = df[item_columns(disorder)].apply(pd.to_numeric)
        score = items.sum(axis=1, min_count=spec["n_items"])
        if spec["scoring"] == "mean":
            score = score / spec["n_items"]
        result[score_col(disorder)] = score
        result[tier_col(disorder)] = score.map(
            lambda value: np.nan if pd.isna(value) else TIER_LABELS[tier_of(value, disorder)]
        )
    return result
