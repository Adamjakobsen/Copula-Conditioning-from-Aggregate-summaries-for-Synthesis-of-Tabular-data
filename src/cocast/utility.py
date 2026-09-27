"""Same-cohort classifier utility with auditable, shared real-data folds.

Synthetic generators may use summaries or training records from the full
reference cohort. These classifier folds therefore do not establish utility
on participants excluded from generator development. TSTR hyperparameters
are tuned using only synthetic training records. Augmentation and resampling
are tuned independently against real inner-validation records, with each
training condition rebuilt inside the inner split. Outer test folds are shared.

The scoring convention is explicit: macro-F1 averages
all five declared tiers, assigning zero to undefined tier F1 scores.
Per-tier recall is missing when a test fold contains no such tier.
"""

from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import schema

CLASSIFIERS = ("logreg", "histgb")
ALL_TIER_LABELS = list(schema.TIER_LABELS)
PROTOCOL_VERSION = 'independent_inner_cv_stopping_v2'
LOGREG_MAX_ITER = 5000
LOGREG_TOL = 1e-4
BOOST_MAX_ITER = 1000
BOOST_PATIENCE = 20
BOOST_TOL = 1e-7


class FitConvergenceError(RuntimeError):
    """A safety cap or failed optimizer must not produce a completed evaluation."""


def stopping_policy():
    return {
        'protocol_version': PROTOCOL_VERSION,
        'tuning': 'Independent per training condition, classifier, outcome, generation seed, ratio and outer fold. Shared candidate grids and macro-F1 selection.',
        'augmentation_validation': 'Real inner-validation only, ceil(ratio * inner real training rows).',
        'resampling': 'Split unique sources first and reconstruct resampling inside inner training.',
        'logreg': {'max_iter': LOGREG_MAX_ITER, 'tol': LOGREG_TOL,
                   'rule': 'L-BFGS convergence. Convergence warnings abort evaluation.'},
        'histgb': {'max_iter': BOOST_MAX_ITER, 'n_iter_no_change': BOOST_PATIENCE,
                   'tol': BOOST_TOL, 'scoring': 'loss',
                   'validation': 'Existing inner CV validation fold only. No extra holdout.',
                   'refit': 'Ceiling of median best round across informative inner folds.',
                   'unsupported_validation_classes': 'Exclude from stopping loss only, retain in macro-F1. Scale tolerance by total/retained validation count.',
                   'uninformative_folds': 'Constant training labels or no supported validation labels have no round estimate. If all folds are uninformative, use one round and flag.',
                   'cap_policy': 'Abort if a fit reaches the safety cap.'},
    }


def tuning_diagnostics(estimator):
    return {'utility_protocol_version': PROTOCOL_VERSION,
            'tuning_stopping_json': json.dumps(getattr(estimator, '_stopping_audit', {}), sort_keys=True)}


def predictor_columns(disorder: str) -> list[str]:
    """Exclude every item used to construct the target domain's outcome."""
    target_items = set(schema.item_columns(disorder))
    return [column for column in schema.REAL_COLUMNS if column not in target_items]


def estimator_candidates(classifier: str, seed: int) -> list[tuple[dict, Pipeline]]:
    """Declare the two classifier grids in stable tie-breaking order."""
    candidates = []
    if classifier == "logreg":
        for strength in (0.1, 1.0, 10.0):
            for weighting in (None, "balanced"):
                parameters = {"C": strength, "class_weight": weighting}
                candidate = Pipeline([
                    # A training-only empty age column remains present with a
                    # constant value, rather than changing feature dimension.
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(
                        **parameters, solver="lbfgs", max_iter=LOGREG_MAX_ITER,
                        tol=LOGREG_TOL, random_state=seed,
                    )),
                ])
                candidates.append((parameters, candidate))
    elif classifier == "histgb":
        for depth in (None, 3):
            for weighting in (None, "balanced"):
                parameters = {"max_depth": depth, "class_weight": weighting}
                candidate = Pipeline([("model", HistGradientBoostingClassifier(
                    **parameters, learning_rate=0.1, max_iter=BOOST_MAX_ITER,
                    max_leaf_nodes=31, min_samples_leaf=20,
                    l2_regularization=0.0, early_stopping=False, random_state=seed,
                ))])
                candidates.append((parameters, candidate))
    else:
        raise ValueError(f"Unknown classifier {classifier!r}. Expected one of {CLASSIFIERS}.")
    return candidates


def make_splits(
    targets: pd.Series, folds: int, repeats: int, seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, int]:
    """Reduce stratified folds for sparse tiers, using unstratified folds for singletons."""
    if targets.isna().any():
        raise ValueError("Missing outcomes must be excluded before constructing classifier folds.")
    if len(targets) < 2:
        raise ValueError("At least two scorable real outcomes are required for cross-validation.")
    counts = targets.value_counts()
    smallest = int(counts.min())
    if smallest >= 2:
        actual = min(folds, smallest)
        splitter = RepeatedStratifiedKFold(n_splits=actual, n_repeats=repeats, random_state=seed)
        strategy = "repeated_stratified"
    else:
        actual = min(folds, len(targets))
        splitter = RepeatedKFold(n_splits=actual, n_repeats=repeats, random_state=seed)
        strategy = "repeated_unstratified_singleton_tier"
    indices = list(splitter.split(np.zeros((len(targets), 1)), targets))
    return indices, strategy, actual


def fit_predict(
    estimator: Pipeline, x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, *, diagnostics: dict | None = None,
) -> np.ndarray:
    """Fit an independent estimator, with a constant predictor for one-class training."""
    if y_train.empty or y_train.isna().any():
        raise ValueError("Classifier training requires observed outcomes without imputation.")
    labels = y_train.astype(str)
    unique = labels.unique()
    if len(unique) == 1:
        if diagnostics is not None:
            diagnostics.update(fit_iterations=0, fit_stopping_reason='constant_training_label')
        return np.repeat(unique[0], len(x_test))
    fitted = clone(estimator)
    # A failed numerical fit cannot enter either tuning scores or final results.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            fitted.fit(x_train, labels)
    except ConvergenceWarning as exc:
        raise FitConvergenceError(
            f'Classifier failed to converge on {len(labels)} training rows. '
            f'No scores accepted. Settings: {fitted.named_steps["model"].get_params()}. {exc}'
        ) from exc
    model = fitted.named_steps['model']
    iterations = int(np.max(np.atleast_1d(model.n_iter_)))
    if isinstance(model, LogisticRegression):
        if not (np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()):
            raise FitConvergenceError('Logistic regression produced non-finite parameters.')
        reason = 'optimizer_converged'
    else:
        reason = 'inner_cv_selected_rounds'
    if diagnostics is not None:
        diagnostics.update(fit_iterations=iterations, fit_iteration_limit=int(model.max_iter),
                           fit_stopping_reason=reason)
    return fitted.predict(x_test)


def _boost_validation_fit(estimator, x_train, y_train, x_validation, y_validation):
    """Stop on inner-fold loss, then score the best round on all validation rows.

    Classes absent from inner training cannot inform stopping, but still count
    in macro-F1. Scaling tol keeps the full validation-row denominator. No outer
    test inputs or labels are accepted here.
    """
    train_labels, validation_labels = y_train.astype(str), y_validation.astype(str)
    supported = validation_labels.isin(train_labels.unique()).to_numpy()
    audit = {'n_validation': len(validation_labels),
             'n_unsupported_validation': int((~supported).sum())}
    if train_labels.nunique() == 1 or not supported.any():
        prediction = np.repeat(train_labels.iloc[0], len(validation_labels))
        # With no overlap every prediction is wrong, regardless of the round.
        audit.update(reason='uninformative_validation', best_round=None, fitted_rounds=0)
        return prediction, audit
    fitted = clone(estimator.named_steps['model']).set_params(
        early_stopping=True, validation_fraction=None, scoring='loss',
        max_iter=BOOST_MAX_ITER, n_iter_no_change=BOOST_PATIENCE,
        tol=BOOST_TOL*len(validation_labels)/int(supported.sum()),
    )
    fitted.fit(x_train, train_labels,
               X_val=x_validation.iloc[np.flatnonzero(supported)],
               y_val=validation_labels.iloc[np.flatnonzero(supported)])
    scores = np.asarray(fitted.validation_score_, dtype=float)
    if not np.isfinite(scores).all():
        raise FitConvergenceError('Boosting produced non-finite validation losses.')
    if fitted.n_iter_ >= BOOST_MAX_ITER:
        raise FitConvergenceError(f'Boosting reached the {BOOST_MAX_ITER}-round safety cap. '
                                  'No scores accepted. Inspect stopping before rerunning.')
    # Round zero is intercept-only and is not a valid max_iter for refitting.
    best_round = int(np.argmax(scores[1:])) + 1
    for iteration, probabilities in enumerate(fitted.staged_predict_proba(x_validation), start=1):
        if iteration == best_round:
            prediction = fitted.classes_[np.argmax(probabilities, axis=1)]
            break
    audit.update(reason='validation_loss_plateau', best_round=best_round,
                 fitted_rounds=int(fitted.n_iter_), effective_tol=float(fitted.tol))
    return prediction, audit


def _select_estimator(
    x_train: pd.DataFrame, y_train: pd.Series, classifier: str, seed: int,
    *, training_builder=None,
) -> tuple[Pipeline, dict, float, str, int]:
    candidates = estimator_candidates(classifier, seed)
    if (y_train.nunique() == 1 and training_builder is None) or len(y_train) < 2:
        parameters, estimator = candidates[0]
        return estimator, parameters, np.nan, "constant_training_target", 0
    splits, strategy, actual = make_splits(y_train, folds=3, repeats=1, seed=seed)
    # Build once per split, before candidate selection. Every candidate sees the
    # same training records, while only the original validation source is scored.
    inner = []
    for training, validation in splits:
        if training_builder is None:
            x_fit, y_fit, construction = x_train.iloc[training], y_train.iloc[training], {}
        else:
            x_fit, y_fit, construction = training_builder(training, validation)
        inner.append((x_fit, y_fit, x_train.iloc[validation], y_train.iloc[validation], construction))
    best = None
    audit = []
    for parameters, estimator in candidates:
        scores = []
        rounds = []
        for x_fit, y_fit, x_valid, y_valid, construction in inner:
            if classifier == 'histgb':
                prediction, fit_audit = _boost_validation_fit(
                    estimator, x_fit, y_fit, x_valid, y_valid)
                if fit_audit['best_round'] is not None:
                    rounds.append(fit_audit['best_round'])
            else:
                fit_audit = {}
                prediction = fit_predict(
                    estimator, x_fit, y_fit, x_valid, diagnostics=fit_audit)
            audit.append({'parameters': parameters, 'fold': len(scores)+1,
                          'n_fitting_rows': len(y_fit), 'n_validation_rows': len(y_valid),
                          **construction, **fit_audit})
            scores.append(f1_score(
                y_valid.astype(str), prediction,
                labels=ALL_TIER_LABELS, average="macro", zero_division=0,
            ))
        mean = float(np.mean(scores))
        if best is None or mean > best[2]:
            selected_parameters = dict(parameters)
            if classifier == 'histgb':
                selected_parameters['max_iter'] = max(1, int(np.ceil(np.median(rounds)))) if rounds else 1
                estimator = clone(estimator).set_params(model__max_iter=selected_parameters['max_iter'])
            best = (estimator, selected_parameters, mean, strategy, actual)
    selected = {k: v for k, v in best[1].items() if k != 'max_iter'}
    best[0]._stopping_audit = {
        'candidate_fits': len(audit),
        'max_iterations': max(a.get('fit_iterations', a.get('fitted_rounds', 0)) for a in audit),
        'selected_folds': [{k: v for k, v in a.items() if k != 'parameters'}
                           for a in audit if a['parameters'] == selected],
    }
    return best


def _requested_labels(labels, n):
    """Deterministic largest-remainder counts at the requested inner-fold size."""
    counts = pd.Series(np.asarray(labels, dtype=str)).value_counts()
    if counts.empty or n < 1:
        raise ValueError('Tuning requires a nonempty requested tier mix and positive size.')
    levels = sorted(counts.index)
    expected = counts.loc[levels].to_numpy(dtype=float) / counts.sum() * n
    allocated = np.floor(expected).astype(int)
    order = np.argsort(-(expected-allocated), kind='stable')
    allocated[order[:n-int(allocated.sum())]] += 1
    return np.repeat(levels, allocated)


def _inner_resample(available, requested, n, seed):
    """Resample only inner-training sources, explicitly flagging missing tiers."""
    available = pd.Series(np.asarray(available, dtype=str))
    desired = pd.Series(_requested_labels(requested, n))
    missing = sorted(set(desired)-set(available))
    if missing:
        retained = desired[desired.isin(set(available))]
        desired = pd.Series(_requested_labels(retained if len(retained) else sorted(set(available)), n))
    indices = matched_resample_positions(available, desired, seed)
    return indices, missing


def select_synthetic_estimator(x, y, source_ids, classifier, seed):
    """Split unique source IDs before recreating copies inside training folds.

    Validation contains one record per held-out source. For a copied condition,
    the fitting size scales with the inner training-source fraction. Its tier
    mix is the achieved outer synthetic training mix, restricted when needed.
    """
    ids = np.asarray(source_ids)
    if len(ids) != len(y) or len(x) != len(y):
        raise ValueError('Synthetic source IDs must align with the training rows.')
    _, first = np.unique(ids, return_index=True)
    first.sort()
    unique_x, unique_y = x.iloc[first].reset_index(drop=True), y.iloc[first].reset_index(drop=True)
    if len(first) == len(y):
        return _select_estimator(unique_x, unique_y, classifier, seed)

    def build(training, validation):
        n = int(np.ceil(len(y)*len(training)/len(first)))
        chosen, missing = _inner_resample(unique_y.iloc[training], y, n,
                                          np.random.SeedSequence([seed, len(training), 531]))
        positions = training[chosen]
        return unique_x.iloc[positions], unique_y.iloc[positions], {
            'construction': 'synthetic_resampling_after_source_split',
            'n_unique_training_sources': len(np.unique(positions)),
            'unavailable_tiers': missing,
        }
    return _select_estimator(unique_x, unique_y, classifier, seed, training_builder=build)


def select_augmented_estimator(x_real, y_real, classifier, seed, *, ratio,
                               x_synthetic=None, y_synthetic=None, source_ids=None,
                               requested_y=None, resample_real=False, resample_synthetic=False):
    """Tune a specific augmentation on real inner-validation records only.

    The added size is ceil(ratio * inner real training size). Real resampling
    happens after splitting, never from the outer fold's pre-resampled rows.
    Synthetic rows come only from the selected outer synthetic training set.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError('Augmentation tuning requires a positive finite ratio.')
    if requested_y is None:
        requested_y = y_synthetic
    if requested_y is None or not len(requested_y):
        raise ValueError('Supply the achieved synthetic tier mix for augmentation.')
    if not resample_real:
        if x_synthetic is None or y_synthetic is None or len(x_synthetic) != len(y_synthetic):
            raise ValueError('Supply aligned synthetic predictors and targets.')
        ids = np.arange(len(y_synthetic)) if source_ids is None else np.asarray(source_ids)
        if len(ids) != len(y_synthetic):
            raise ValueError('Synthetic source IDs must align with the training rows.')
        _, first = np.unique(ids, return_index=True)
        first.sort()
        if resample_synthetic or len(first) != len(ids):
            resample_synthetic = True
            x_synthetic = x_synthetic.iloc[first].reset_index(drop=True)
            y_synthetic = y_synthetic.iloc[first].reset_index(drop=True)
        # Fixed order makes smaller requested sizes nested and candidates paired.
        order = np.random.default_rng(np.random.SeedSequence([seed, 532])).permutation(len(y_synthetic))

    def build(training, validation):
        real_x, real_y = x_real.iloc[training], y_real.iloc[training]
        n = int(np.ceil(ratio*len(training)))
        missing = []
        if resample_real:
            chosen, missing = _inner_resample(real_y, requested_y, n,
                                              np.random.SeedSequence([seed, len(training), 533]))
            added_x, added_y = real_x.iloc[chosen], real_y.iloc[chosen]
            kind = 'real_resampling_after_inner_split'
        elif resample_synthetic:
            chosen, missing = _inner_resample(y_synthetic, requested_y, n,
                                              np.random.SeedSequence([seed, len(training), 534]))
            added_x, added_y = x_synthetic.iloc[chosen], y_synthetic.iloc[chosen]
            kind = 'synthetic_resampled_augmentation'
        else:
            if n > len(order):
                raise ValueError('Insufficient distinct synthetic rows for inner augmentation.')
            chosen = order[:n]
            added_x, added_y = x_synthetic.iloc[chosen], y_synthetic.iloc[chosen]
            kind = 'synthetic_augmentation'
        return (pd.concat([real_x, added_x], ignore_index=True),
                pd.concat([real_y, added_y], ignore_index=True),
                {'construction': kind, 'n_real_fitting': len(training), 'n_added': n,
                 'n_unique_added_sources': len(np.unique(chosen)), 'unavailable_tiers': missing})

    return _select_estimator(x_real, y_real, classifier, seed, training_builder=build)


def classification_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict:
    """Return aggregate scores and counts sufficient to reconstruct tier metrics."""
    true = np.asarray(y_true, dtype=str)
    prediction = np.asarray(y_pred, dtype=str)
    if len(true) == 0 or len(true) != len(prediction):
        raise ValueError("Metrics require nonempty, equally sized observed and predicted outcomes.")
    if not np.isin(true, ALL_TIER_LABELS).all() or not np.isin(prediction, ALL_TIER_LABELS).all():
        raise ValueError("Classification outcomes must use the five declared tier labels.")
    matrix = confusion_matrix(true, prediction, labels=ALL_TIER_LABELS)
    metrics = {
        "f1_macro": float(f1_score(true, prediction, labels=ALL_TIER_LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(true, prediction)),
    }
    for index, tier in enumerate(ALL_TIER_LABELS):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        support = tp + fn
        metrics.update({
            f"{tier}_tp": tp, f"{tier}_fp": fp, f"{tier}_fn": fn,
            f"{tier}_support": support,
            f"{tier}_precision": float(tp / (tp + fp)) if tp + fp else np.nan,
            f"{tier}_recall": float(tp / support) if support else np.nan,
        })
    return metrics


def matched_resample_positions(training_tiers, requested_tiers, seed):
    """Sample only supplied real training positions, matching requested severity counts exactly."""
    training = np.asarray(training_tiers, dtype=str)
    requested = np.asarray(requested_tiers, dtype=str)
    if not np.isin(requested, ALL_TIER_LABELS).all():
        raise ValueError('Targeted synthetic outcomes must be observed.')
    random = np.random.default_rng(seed)
    selected = []
    for tier in ALL_TIER_LABELS:
        count = int((requested == tier).sum())
        eligible = np.flatnonzero(training == tier)
        if count and not len(eligible):
            raise ValueError(f'No real training records available for requested {tier}.')
        if count:
            selected.extend(random.choice(eligible, size=count, replace=True).tolist())
    random.shuffle(selected)
    return np.asarray(selected, dtype=int)


def _numeric_table(table: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [column for column in schema.REAL_COLUMNS if column not in table]
    if missing or table.empty:
        raise ValueError(f"{name} must contain age, sex and all questionnaire columns. Missing: {missing}.")
    schema.validate_synthetic_items(table, source=name, require_id=False, allow_missing=True)
    result = table[schema.REAL_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError(f"{name} contains infinite predictor values.")
    age = result[schema.AGE_COL].dropna()
    if (age < 0).any() or (age != np.floor(age)).any():
        raise ValueError(f"{name} contains invalid ages.")
    if not result[schema.SEX_COL].dropna().isin(schema.SEX_LABELS).all():
        raise ValueError(f"{name} contains invalid recorded sex codes.")
    return result.reset_index(drop=True)


def _evaluate(
    real_df: pd.DataFrame, synth_tables: dict[str, pd.DataFrame],
    folds: int, repeats: int, seed: int, matched_resampling_runs=(), targeted_domains=None,
) -> pd.DataFrame:
    real = _numeric_table(real_df, "real")
    real_scores = schema.score_table(real)
    synthetic = {name: _numeric_table(table, name) for name, table in synth_tables.items()}
    synthetic_scores = {name: schema.score_table(table) for name, table in synthetic.items()}
    rows = []
    for domain_index, disorder in enumerate(schema.DISORDER_NAMES):
        columns = predictor_columns(disorder)
        labels = real_scores[schema.tier_col(disorder)]
        observed = labels.notna().to_numpy()
        positions = np.flatnonzero(observed)
        x_real = real.loc[observed, columns].reset_index(drop=True)
        y_real = labels[observed].astype(str).reset_index(drop=True)
        splits, strategy, actual_folds = make_splits(y_real, folds, repeats, seed)
        pools = {}
        for name, table in synthetic.items():
            outcomes = synthetic_scores[name][schema.tier_col(disorder)]
            keep = outcomes.notna().to_numpy()
            pools[name] = (
                table.loc[keep, columns].reset_index(drop=True),
                outcomes[keep].astype(str).reset_index(drop=True), int((~keep).sum()),
                np.flatnonzero(keep),
            )
        for split_index, (train, test) in enumerate(splits):
            repeat = split_index // actual_folds + 1
            fold = split_index % actual_folds + 1
            x_train = x_real.iloc[train].reset_index(drop=True)
            y_train = y_real.iloc[train].reset_index(drop=True)
            x_test = x_real.iloc[test].reset_index(drop=True)
            y_test = y_real.iloc[test].reset_index(drop=True)
            identity = {
                "domain": disorder, "train": positions[train].tolist(), "test": positions[test].tolist(),
            }
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            sample_seed = np.random.SeedSequence([seed, domain_index, split_index])
            subsets = {}
            for name, (x_pool, y_pool, n_missing, pool_positions) in pools.items():
                if y_pool.empty:
                    subsets[name] = None
                    continue
                # Identical-size pools use the same sampled row positions.
                # Each selected subset is reused for both classifiers and protocols.
                random = np.random.default_rng(sample_seed)
                replacement = len(y_pool) < len(train)
                selected = random.choice(len(y_pool), size=len(train), replace=replacement)
                subsets[name] = (
                    x_pool.iloc[selected].reset_index(drop=True),
                    y_pool.iloc[selected].reset_index(drop=True), replacement,
                    pool_positions[selected],
                )
            base = {
                "domain": disorder, "repeat": repeat, "fold": fold,
                "outer_strategy": strategy, "outer_folds": actual_folds,
                "outer_repeats": repeats, "evaluation_seed": seed,
                "n_real_train": len(train), "n_real_test": len(test),
                "n_missing_real_target": int((~observed).sum()),
                "n_predictors": len(columns),
                "split_fingerprint": fingerprint,
                "real_test_positions_json": json.dumps(positions[test].tolist()),
                "real_train_positions_json": json.dumps(positions[train].tolist()),
                "target_label_policy": "all_five_declared_tiers_zero_undefined_f1",
                "fit_input_scope": "same_cohort",
            }
            for classifier in CLASSIFIERS:
                estimator, parameters, inner_score, inner_strategy, inner_folds = _select_estimator(
                    x_train, y_train, classifier, seed,
                )
                setting = {
                    **base, "classifier": classifier, **tuning_diagnostics(estimator),
                    "tuning_source": "real_outer_training_fold",
                    "n_tuning_records": len(y_train),
                    "selected_parameters_json": json.dumps(parameters, sort_keys=True),
                    "inner_f1_macro": inner_score, "inner_strategy": inner_strategy,
                    "inner_folds": inner_folds,
                }
                fit_audit = {}
                prediction = fit_predict(estimator, x_train, y_train, x_test, diagnostics=fit_audit)
                rows.append({
                    **setting, **fit_audit, "run": "real", "method": "real", "protocol": "TRTR",
                    "status": "complete", "n_synth_train": 0,
                    "n_missing_synthetic_target": 0, "synthetic_sampling_with_replacement": False,
                    **classification_metrics(y_test, prediction),
                })
                for name, subset in subsets.items():
                    n_missing = pools[name][2]
                    if subset is None:
                        for protocol in ("TSTR", "TAUG"):
                            failed_setting = setting
                            if protocol == "TSTR":
                                failed_setting = {**setting,
                                    "tuning_source": "synthetic_training_subset",
                                    "n_tuning_records": 0,
                                    "selected_parameters_json": None,
                                    "inner_f1_macro": np.nan,
                                    "inner_strategy": "no_scorable_synthetic_target",
                                    "inner_folds": 0}
                            rows.append({
                                **failed_setting, "run": name, "method": name, "protocol": protocol,
                                "status": "no_scorable_synthetic_target", "n_synth_train": 0,
                                "n_missing_synthetic_target": n_missing,
                                "synthetic_sampling_with_replacement": False,
                                "f1_macro": np.nan, "accuracy": np.nan,
                            })
                        continue
                    x_synth, y_synth, replacement, synthetic_positions = subset
                    # Split unique source records before reconstructing copies
                    # inside inner training. Validation never contains a copy
                    # of a source used by the inner classifier fit.
                    _, unique_indices = np.unique(synthetic_positions, return_index=True)
                    unique_indices.sort()
                    synth_estimator, synth_parameters, synth_score, synth_strategy, synth_folds = select_synthetic_estimator(
                        x_synth, y_synth, synthetic_positions, classifier, seed,
                    )
                    synth_setting = {
                        **base, "classifier": classifier, **tuning_diagnostics(synth_estimator),
                        "tuning_source": "synthetic_training_subset",
                        "n_tuning_records": len(unique_indices),
                        "selected_parameters_json": json.dumps(synth_parameters, sort_keys=True),
                        "inner_f1_macro": synth_score, "inner_strategy": synth_strategy,
                        "inner_folds": synth_folds,
                    }
                    aug_estimator, aug_parameters, aug_score, aug_strategy, aug_folds = select_augmented_estimator(
                        x_train, y_train, classifier, seed, ratio=1,
                        x_synthetic=x_synth, y_synthetic=y_synth, source_ids=synthetic_positions,
                    )
                    aug_setting = {**base, 'classifier': classifier, **tuning_diagnostics(aug_estimator),
                                   'tuning_source': 'real_inner_validation_augmented_training',
                                   'n_tuning_records': len(y_train),
                                   'selected_parameters_json': json.dumps(aug_parameters, sort_keys=True),
                                   'inner_f1_macro': aug_score, 'inner_strategy': aug_strategy,
                                   'inner_folds': aug_folds}
                    training_sets = {
                        "TSTR": (x_synth, y_synth),
                        "TAUG": (
                            pd.concat([x_train, x_synth], ignore_index=True),
                            pd.concat([y_train, y_synth], ignore_index=True),
                        ),
                    }
                    for protocol, (x_fit, y_fit) in training_sets.items():
                        protocol_estimator = synth_estimator if protocol == "TSTR" else aug_estimator
                        protocol_setting = synth_setting if protocol == "TSTR" else aug_setting
                        fit_audit = {}
                        prediction = fit_predict(protocol_estimator, x_fit, y_fit, x_test, diagnostics=fit_audit)
                        rows.append({
                            **protocol_setting, **fit_audit, "run": name, "method": name, "protocol": protocol,
                            "status": "complete", "n_synth_train": len(y_synth),
                            "n_missing_synthetic_target": n_missing,
                            "synthetic_sampling_with_replacement": replacement,
                            "synthetic_train_positions_json": json.dumps(synthetic_positions.tolist()),
                            **classification_metrics(y_test, prediction),
                        })
                    if name in matched_resampling_runs and disorder in targeted_domains[name]:
                        comparison = {**setting, 'run': name, 'method': 'real_resampled',
                                      'protocol': 'TRTR_RESAMPLED', 'matched_to_run': name,
                                      'resampling_domain': disorder, 'n_synth_train': 0,
                                      'requested_resampled_rows': len(y_synth),
                                      'requested_tier_counts_json': json.dumps(
                                          y_synth.value_counts().to_dict(), sort_keys=True)}
                        requested = y_synth
                        available = set(y_train.astype(str))
                        missing_tiers = sorted(set(y_synth.astype(str)) - available)
                        match_policy = 'exact_achieved_synthetic_tier_counts'
                        if missing_tiers:
                            # Real-only resampling cannot invent unobserved classes.
                            # Keep the same added N and condition the requested mix
                            # on the tiers available in this training fold.
                            labels = sorted(available)
                            weights = np.array([(y_synth == t).sum() for t in labels], dtype=float)
                            match_policy = 'renormalized_available_tier_counts'
                            if weights.sum() == 0:
                                weights[:] = 1
                                match_policy = 'uniform_available_tiers_no_overlap'
                            expected = weights / weights.sum() * len(y_synth)
                            counts = np.floor(expected).astype(int)
                            remainder = len(y_synth) - counts.sum()
                            order = np.argsort(-(expected - counts), kind='stable')
                            counts[order[:remainder]] += 1
                            requested = pd.Series(np.repeat(labels, counts))
                        added = matched_resample_positions(
                            y_train, requested,
                            np.random.SeedSequence([seed, domain_index, split_index, 917]),
                        )
                        comparison.update(
                            resampling_support_restricted=bool(missing_tiers),
                            resampling_match_policy=match_policy,
                            unavailable_tiers_json=json.dumps(missing_tiers),
                            resampled_tier_counts_json=json.dumps(
                                requested.value_counts().to_dict(), sort_keys=True),
                        )
                        resampled_estimator, resampled_parameters, resampled_score, resampled_strategy, resampled_folds = select_augmented_estimator(
                            x_train, y_train, classifier, seed, ratio=1,
                            requested_y=y_synth, resample_real=True,
                        )
                        comparison.update(**tuning_diagnostics(resampled_estimator),
                            tuning_source='real_inner_validation_resampled_training',
                            selected_parameters_json=json.dumps(resampled_parameters, sort_keys=True),
                            inner_f1_macro=resampled_score, inner_strategy=resampled_strategy,
                            inner_folds=resampled_folds)
                        fit_audit = {}
                        prediction = fit_predict(
                            resampled_estimator, pd.concat([x_train, x_train.iloc[added]], ignore_index=True),
                            pd.concat([y_train, y_train.iloc[added]], ignore_index=True), x_test, diagnostics=fit_audit,
                        )
                        rows.append({**comparison, **fit_audit, 'status': 'complete', 'n_resampled_train': len(added),
                                     'resampled_real_positions_json': json.dumps(positions[train][added].tolist()),
                                     **classification_metrics(y_test, prediction)})
    return pd.DataFrame(rows)


def evaluate_utility(
    real_df: pd.DataFrame, synth_tables: dict[str, pd.DataFrame],
    folds: int = 5, repeats: int = 3, seed: int = 42,
    matched_resampling_runs=None, targeted_domains=None,
) -> pd.DataFrame:
    """Return raw paired fold metrics without file writes or inferential statistics.

    Every synthetic classifier-training subset has the real training fold's
    size. TSTR uses that subset alone. TAUG adds the same subset to the real
    training fold, giving a 1:1 augmentation ratio. Sampling uses replacement
    only when the scorable synthetic pool is smaller, and records that fact.
    TRTR is emitted once for each real fold and classifier, not once per run.
    TSTR selects hyperparameters by inner CV on unique selected synthetic
    records. No real training labels or predictors enter that tuning. TRTR uses real-only inner fitting. TAUG and real-data resampling are tuned
    independently with their own inner training construction and real validation.
    Resampling comparisons are restricted to the explicitly targeted domains.
    """
    for name, value, minimum in (("folds", folds, 2), ("repeats", repeats, 1), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    if not isinstance(synth_tables, dict) or any(not isinstance(name, str) or not name for name in synth_tables):
        raise ValueError("synth_tables must map nonempty run names to response tables.")
    if "real" in synth_tables:
        raise ValueError("The run name 'real' is reserved for the TRTR reference.")
    matched_resampling_runs = set(matched_resampling_runs or ())
    if not matched_resampling_runs.issubset(synth_tables):
        raise ValueError('Resampling comparators require explicitly supplied synthetic runs.')
    targeted_domains = targeted_domains or {}
    if not isinstance(targeted_domains, dict) or not set(targeted_domains).issubset(synth_tables):
        raise ValueError('targeted_domains must map supplied run names to declared disorder names.')
    for name, domains in targeted_domains.items():
        if (not isinstance(domains, (list, tuple)) or not domains
                or len(set(domains)) != len(domains)
                or not set(domains).issubset(schema.DISORDER_NAMES)):
            raise ValueError(f'{name}: targeted_domains must list unique, declared disorders.')
    if not matched_resampling_runs.issubset(targeted_domains):
        raise ValueError('Resampling comparators require explicit targeted_domains for each targeted run.')
    # Bound nested numerical libraries, allowing callers to parallelize
    # independent evaluations without multiplying native thread pools.
    with threadpool_limits(limits=1):
        return _evaluate(real_df, synth_tables, int(folds), int(repeats), int(seed),
                         matched_resampling_runs, targeted_domains)


def summarize_utility_repeats(fold_data: pd.DataFrame, primary_domains=()) -> pd.DataFrame:
    """Pool held-out confusion counts before calculating recall per repeat.

    This avoids giving a rare-tier case a different weight because its fold
    has a different size. Undefined recall stays missing. Per-domain macro-F1
    and precision use all five declared tiers, with undefined terms set to
    zero. The targeted recall summary gives each eligible domain and each
    observed tier equal weight and discloses the supported tier count.
    These repeat summaries are not generation-seed uncertainty estimates.
    """
    if fold_data.empty:
        return pd.DataFrame()
    keys = ["run", "method", "classifier", "protocol", "repeat"]
    summaries = []
    for identity, group in fold_data.groupby(keys + ["domain"], dropna=False, sort=False):
        base = dict(zip(keys + ["domain"], identity))
        if (group["status"] != "complete").any():
            summaries.append({**base, "status": "incomplete_fold_results"})
            continue
        tier_recalls, tier_precisions, tier_f1 = [], [], []
        total_correct = 0
        values = {}
        for tier in ALL_TIER_LABELS:
            tp, fp, fn = (int(group[f"{tier}_{suffix}"].sum()) for suffix in ("tp", "fp", "fn"))
            support = tp + fn
            recall = tp / support if support else np.nan
            precision = tp / (tp + fp) if tp + fp else np.nan
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            values.update({f"{tier}_tp": tp, f"{tier}_fp": fp, f"{tier}_fn": fn,
                           f"{tier}_recall": recall, f"{tier}_precision": precision,
                           f"{tier}_support": support})
            tier_recalls.append(recall)
            tier_precisions.append(precision if np.isfinite(precision) else 0.0)
            tier_f1.append(f1)
            total_correct += tp
        values.update(
            recall_macro_observed=float(np.nanmean(tier_recalls)),
            precision_macro=float(np.mean(tier_precisions)),
            f1_macro=float(np.mean(tier_f1)),
            accuracy=total_correct / int(group["n_real_test"].sum()),
            n_supported_tiers=int(np.isfinite(tier_recalls).sum()),
            n_declared_tiers=len(ALL_TIER_LABELS),
        )
        summaries.append({**base, "status": "complete", **values})
    domains = pd.DataFrame(summaries)
    scopes = {"all_domains": tuple(schema.DISORDER_NAMES)}
    if primary_domains:
        scopes["targeted_domains"] = tuple(primary_domains)
    for identity, group in domains.groupby(keys, dropna=False, sort=False):
        for scope, included in scopes.items():
            selected = group[group["domain"].isin(included)]
            if set(selected["domain"]) != set(included):
                # The real-resampling reference intentionally exists only for
                # targeted outcomes and therefore has no seven-domain average.
                continue
            base = {**dict(zip(keys, identity)), "domain": scope}
            if (selected["status"] != "complete").any():
                summaries.append({**base, "status": "incomplete_fold_results"})
                continue
            summaries.append({**base, "status": "complete",
                **{metric: float(selected[metric].mean()) for metric in
                   ("recall_macro_observed", "precision_macro", "f1_macro", "accuracy")},
                "n_supported_tiers": int(selected["n_supported_tiers"].sum()),
                "n_declared_tiers": len(included) * len(ALL_TIER_LABELS)})
            if scope == 'targeted_domains':
                pooled = {**base, 'domain': 'targeted_tiers_pooled', 'status': 'complete'}
                for tier in ALL_TIER_LABELS:
                    tp, fp, fn = (int(selected[f'{tier}_{suffix}'].sum()) for suffix in ('tp', 'fp', 'fn'))
                    pooled.update({f'{tier}_tp': tp, f'{tier}_fp': fp, f'{tier}_fn': fn,
                                   f'{tier}_support': tp + fn,
                                   f'{tier}_recall': tp / (tp + fn) if tp + fn else np.nan,
                                   f'{tier}_precision': tp / (tp + fp) if tp + fp else np.nan})
                summaries.append(pooled)
    result = pd.DataFrame(summaries)
    result["aggregation"] = "pooled_real_test_confusion_counts_within_repeat_then_equal_domain_mean"
    identifiers = keys + ["domain", "status", "aggregation"]
    return result.melt(id_vars=identifiers, var_name="metric", value_name="value")
