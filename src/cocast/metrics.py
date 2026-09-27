"""Distributional fidelity and record resemblance on one categorical representation."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, jensenshannon

from . import schema


def categorical_tables(real, synthetic, age_bins=5):
    """Quantile edges use observed real ages. Missingness is a separate category."""
    tables = [frame[schema.REAL_COLUMNS].copy() for frame in (real, synthetic)]
    ages = pd.to_numeric(tables[0][schema.AGE_COL]).dropna().to_numpy()
    edges = np.unique(np.quantile(ages, np.linspace(0, 1, age_bins + 1))) if len(ages) else np.array([])
    if len(edges) >= 2:
        edges[0], edges[-1] = -np.inf, np.inf
        for table in tables:
            table[schema.AGE_COL] = pd.cut(table[schema.AGE_COL], edges, labels=False, include_lowest=True)
    else:
        for table in tables:
            table[schema.AGE_COL] = np.where(table[schema.AGE_COL].isna(), np.nan, 0)
    arrays = []
    for col in schema.REAL_COLUMNS:
        combined = pd.concat([tables[0][col], tables[1][col]], ignore_index=True)
        codes, _ = pd.factorize(combined, use_na_sentinel=False)
        arrays.append(codes)
    matrix = np.column_stack(arrays)
    return matrix[:len(real)], matrix[len(real):]


def cramers_v(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    r, c, n = int(a.max()) + 1, int(b.max()) + 1, len(a)
    if n < 2 or r < 2 or c < 2:
        return 0.0
    counts = np.bincount(a * c + b, minlength=r * c).reshape(r, c)
    expected = np.outer(counts.sum(1), counts.sum(0)) / n
    phi = (((counts - expected) ** 2 / expected).sum() / n)
    corrected = max(0.0, phi - (r - 1) * (c - 1) / (n - 1))
    denominator = min(r - 1 - (r - 1) ** 2 / (n - 1), c - 1 - (c - 1) ** 2 / (n - 1))
    return float(np.sqrt(corrected / denominator)) if denominator > 0 else 0.0


def fidelity(real, synthetic):
    """Return JSD, association MAE and sqrt-Hamming energy, without per-column files."""
    x, y = categorical_tables(real, synthetic)
    if min(len(x), len(y)) < 2:
        raise ValueError('Energy U-statistic requires at least two rows per dataset.')
    divergences = []
    for j in range(x.shape[1]):
        size = max(x[:, j].max(), y[:, j].max()) + 1
        divergences.append(jensenshannon(np.bincount(x[:, j], minlength=size),
                                        np.bincount(y[:, j], minlength=size), base=2) ** 2)
    domains = {column: domain for domain in schema.DISORDER_NAMES for column in schema.item_columns(domain)}
    differences, between, within = [], [], []
    for j, a in enumerate(schema.REAL_COLUMNS):
        for k in range(j + 1, len(schema.REAL_COLUMNS)):
            b = schema.REAL_COLUMNS[k]
            delta = abs(cramers_v(x[:, j], x[:, k]) - cramers_v(y[:, j], y[:, k]))
            differences.append(delta)
            if a in domains and b in domains:
                (within if domains[a] == domains[b] else between).append(delta)
    # Square-root EACH Hamming distance, before averaging. Plain Hamming loses dependence.
    estimate = (2 * np.sqrt(cdist(x, y, 'hamming')).mean()
                - np.sqrt(pdist(x, 'hamming')).mean() - np.sqrt(pdist(y, 'hamming')).mean())
    return {'jsd': float(np.mean(divergences)), 'mae_v': float(np.mean(differences)),
            'energy': float(np.sqrt(max(0, estimate))),
            'between_disorder_mae_v': float(np.mean(between)),
            'within_questionnaire_mae_v': float(np.mean(within))}


def proximity(real, synthetic=None):
    """Normalized Hamming DCR and NNDR. Exact percentage is 0..100, not a fraction."""
    loo = synthetic is None
    x, y = categorical_tables(real, real if loo else synthetic)
    if len(x) < 2 + int(loo):
        raise ValueError('NNDR requires two eligible neighbours.')
    distances = cdist(y, x, 'hamming')
    if loo:
        np.fill_diagonal(distances, np.inf)
    neighbours = np.partition(distances, 1, axis=1)[:, :2]
    neighbours.sort(axis=1)
    d1, d2 = neighbours.T
    ratio = np.divide(d1, d2, out=np.ones_like(d1), where=d2 > 0)
    return {'dcr_mean': float(d1.mean()), 'dcr_p05': float(np.quantile(d1, .05)),
            'exact_percent': float(100 * (d1 == 0).mean()), 'nndr_p05': float(np.quantile(ratio, .05))}
