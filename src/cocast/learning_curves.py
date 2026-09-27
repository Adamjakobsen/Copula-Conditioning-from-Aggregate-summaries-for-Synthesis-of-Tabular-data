"""Utility learning curves from completed pools, with no generation side effects.

The first reference_n rows anchor the matched-size subset. Larger subsets add distinct source rows.
Real-fold oversampling and reuse of the 1x synthetic subset are explicit controls.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from . import REVISION, schema, utility
from .evaluation import load_run
from .io import atomic_text, read_real, sha256, write_json

VERSION = 'utility_learning_curves_v2'
RATIOS = (0.5, 1.0, 2.0, 5.0)
ELIGIBLE = ('depression', 'specific_phobia', 'social_anxiety', 'panic', 'generalized_anxiety')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_ratios(ratios):
    values = tuple(float(r) for r in ratios)
    if (not values or any(not math.isfinite(r) or r <= 0 for r in values)
            or tuple(sorted(set(values))) != values or 1.0 not in values):
        raise ValueError('Ratios must be positive, finite, unique and increasing, and include 1.')
    return values


def nested_positions(pool_n, reference_n, train_n, ratios, seed, domain_index, split_index):
    """Anchor 1x to the matched-size sampler, independently of the expanded pool size."""
    ratios = validate_ratios(ratios)
    maximum = math.ceil(max(ratios) * train_n)
    if not 0 < train_n <= reference_n <= pool_n or maximum > pool_n:
        raise ValueError(f'Insufficient distinct synthetic rows: need {maximum}, have {pool_n}. '
                         'Learning curves never silently replace missing generated rows.')
    random = np.random.default_rng(np.random.SeedSequence([seed, domain_index, split_index]))
    anchor = random.choice(reference_n, size=train_n, replace=False)
    remaining = np.setdiff1d(np.arange(reference_n), anchor, assume_unique=True)
    extension = np.random.default_rng(np.random.SeedSequence([seed, domain_index, split_index, 728]))
    # New profiles already arrive in random sampling order. Append them by ID
    # so an existing 2x subset cannot change when the pool later grows to 5x.
    order = np.r_[anchor, extension.permutation(remaining), np.arange(reference_n, pool_n)]
    return {ratio: order[:math.ceil(ratio * train_n)] for ratio in ratios}, anchor


def matched_positions(available, requested, seed):
    """Match achieved tier counts using only available training-source records.

    Absent tiers are removed and their mass is redistributed proportionally.
    The same largest-remainder convention as the matched-size utility evaluator
    is retained. Every fallback is returned explicitly for reporting.
    """
    available = pd.Series(np.asarray(available, dtype=str))
    requested = pd.Series(np.asarray(requested, dtype=str))
    if available.empty or requested.empty:
        raise ValueError('Oversampling requires nonempty source and requested labels.')
    support = set(available)
    missing = sorted(set(requested) - support)
    match = requested
    policy = 'exact_achieved_synthetic_tier_counts'
    if missing:
        labels = sorted(support)
        weights = np.array([(requested == t).sum() for t in labels], dtype=float)
        policy = 'renormalized_available_tier_counts'
        if weights.sum() == 0:
            weights[:] = 1
            policy = 'uniform_available_tiers_no_overlap'
        expected = weights / weights.sum() * len(requested)
        counts = np.floor(expected).astype(int)
        order = np.argsort(-(expected - counts), kind='stable')
        counts[order[:len(requested) - int(counts.sum())]] += 1
        match = pd.Series(np.repeat(labels, counts))
    chosen = utility.matched_resample_positions(available, match, seed)
    return chosen, {
        'support_restricted': bool(missing), 'unavailable_tiers_json': json.dumps(missing),
        'matching_policy': policy,
        'requested_tier_counts_json': json.dumps(requested.value_counts().to_dict(), sort_keys=True),
        'training_tier_counts_json': json.dumps(available.iloc[chosen].value_counts().to_dict(), sort_keys=True),
    }


def validate_inputs(real, tables, reference_n, ratios, folds, repeats, seed, domains):
    ratios = validate_ratios(ratios)
    for value, minimum, label in [(reference_n, 2, 'reference_n'), (folds, 2, 'folds'),
                                   (repeats, 1, 'repeats'), (seed, 0, 'seed')]:
        if type(value) is not int or value < minimum:
            raise ValueError(f'{label} must be an integer >= {minimum}.')
    if not tables or 'real' in tables or not domains or not set(domains) <= set(schema.DISORDER_NAMES):
        raise ValueError('Supply named synthetic pools and valid outcome domains.')
    if len(set(domains)) != len(domains):
        raise ValueError('Outcome domains must be unique.')
    schema.validate_real_table(real)
    scores = schema.score_table(real)
    maximum = 0
    for domain in domains:
        splits, _, _ = utility.make_splits(scores[schema.tier_col(domain)].astype(str), folds, repeats, seed)
        train_n = max(len(train) for train, _ in splits)
        if train_n > reference_n:
            raise ValueError('The reference pool is too small for the anchored 1x subset.')
        maximum = max(maximum, math.ceil(max(ratios) * train_n))
    for name, table in tables.items():
        schema.validate_real_table(table)
        if len(table) < max(maximum, reference_n):
            raise ValueError(f'{name}: need at least {max(maximum, reference_n)} complete rows, '
                             f'found {len(table)}. Generate the missing rows first.')
    return ratios


def domain_rows(real, tables, domain, *, reference_n, ratios, folds, repeats, seed):
    """One outcome checkpoint. All arms share real folds and preprocessing rules."""
    domain_index = schema.DISORDER_NAMES.index(domain)
    columns = utility.predictor_columns(domain)
    x_real = real[columns].reset_index(drop=True)
    y_real = schema.score_table(real)[schema.tier_col(domain)].astype(str)
    pools = {name: (table[columns].reset_index(drop=True),
                   schema.score_table(table)[schema.tier_col(domain)].astype(str))
             for name, table in tables.items()}
    splits, strategy, actual = utility.make_splits(y_real, folds, repeats, seed)
    rows = []
    for split_index, (train, test) in enumerate(splits):
        x_train, y_train = x_real.iloc[train], y_real.iloc[train]
        x_test, y_test = x_real.iloc[test], y_real.iloc[test]
        identity = {'domain': domain, 'train': train.tolist(), 'test': test.tolist()}
        base = dict(domain=domain, repeat=split_index // actual + 1, fold=split_index % actual + 1,
                    outer_strategy=strategy, outer_folds=actual, evaluation_seed=seed,
                    n_real_train=len(train), n_real_test=len(test), n_predictors=len(columns),
                    split_fingerprint=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                    reference_pool_n=reference_n)
        selections = {name: nested_positions(len(x), reference_n, len(train), ratios,
                                             seed, domain_index, split_index)
                      for name, (x, _) in pools.items()}
        for classifier in utility.CLASSIFIERS:
            estimator, params, _, _, _ = utility._select_estimator(x_train, y_train, classifier, seed)
            real_tuning = dict(**utility.tuning_diagnostics(estimator), tuning_source='real_outer_training_fold', n_tuning_records=len(train),
                               selected_parameters_json=json.dumps(params, sort_keys=True))

            def measure(run, ratio, protocol, model, x, y, positions, *, source, tuning, match=None):
                fit_audit = {}
                predicted = utility.fit_predict(model, x, y, x_test, diagnostics=fit_audit)
                matrix = confusion_matrix(y_test, predicted, labels=utility.ALL_TIER_LABELS)
                matched = dict(support_restricted=False, unavailable_tiers_json='[]', matching_policy='none',
                               requested_tier_counts_json='{}', training_tier_counts_json='{}')
                matched.update(match or {})
                return {**base, **fit_audit, 'run': run, 'classifier': classifier, 'ratio': ratio, 'protocol': protocol,
                        'status': 'complete', 'training_source': source, 'n_fit': len(y),
                        'n_added': len(positions) if protocol != 'TRTR' else 0,
                        'n_unique_source_rows': len(np.unique(positions)),
                        'selection_sha256': digest(np.asarray(positions).tolist()),
                        'confusion_counts_json': json.dumps(matrix.tolist()), **tuning, **matched,
                        **utility.classification_metrics(y_test, predicted)}

            rows.append(measure('real', 0.0, 'TRTR', estimator, x_train, y_train, train,
                                source='real', tuning=real_tuning))
            for name, (x_pool, y_pool) in pools.items():
                selected_by_ratio, anchor = selections[name]
                tuning_cache = {}
                for ratio in ratios:
                    selected = selected_by_ratio[ratio]
                    x_synth, y_synth = x_pool.iloc[selected], y_pool.iloc[selected]
                    if ratio <= 1:
                        copied = selected
                        copy_match = dict(support_restricted=False, unavailable_tiers_json='[]',
                                          matching_policy='identical_subset_at_or_below_1x')
                    else:
                        chosen, copy_match = matched_positions(y_pool.iloc[anchor], y_synth,
                            np.random.SeedSequence([seed, domain_index, split_index, 918]))
                        copied = anchor[chosen]
                    for control, positions, match in [('generated', selected, None),
                                                      ('synthetic_resampled', copied, copy_match)]:
                        # Cache only identical training conditions. Multiplicities
                        # matter for tuning resampling controls at larger ratios.
                        key = tuple(positions)
                        if key not in tuning_cache:
                            tuned, parameters, _, _, _ = utility.select_synthetic_estimator(
                                x_pool.iloc[positions], y_pool.iloc[positions], positions, classifier, seed)
                            tuning_cache[key] = (tuned, dict(**utility.tuning_diagnostics(tuned),
                                tuning_source='synthetic_training_subset_unique',
                                n_tuning_records=len(np.unique(positions)),
                                selected_parameters_json=json.dumps(parameters, sort_keys=True)))
                        tuned, tuning = tuning_cache[key]
                        x, y = x_pool.iloc[positions], y_pool.iloc[positions]
                        counts = dict(requested_tier_counts_json=json.dumps(y_synth.value_counts().to_dict(), sort_keys=True),
                                      training_tier_counts_json=json.dumps(y.value_counts().to_dict(), sort_keys=True))
                        counts.update(match or {})
                        suffix = '' if control == 'generated' else '_SYNTH_RESAMPLED'
                        rows.append(measure(name, ratio, 'TSTR'+suffix, tuned, x, y, positions,
                                            source=control, tuning=tuning, match=counts))
                        aug_key = ('augmentation', ratio, tuple(positions))
                        if aug_key not in tuning_cache:
                            aug, aug_params, _, _, _ = utility.select_augmented_estimator(
                                x_train, y_train, classifier, seed, ratio=ratio,
                                x_synthetic=x, y_synthetic=y, source_ids=positions,
                                requested_y=y_synth, resample_synthetic=control=='synthetic_resampled' and ratio>1)
                            tuning_cache[aug_key] = (aug, dict(**utility.tuning_diagnostics(aug),
                                tuning_source='real_inner_validation_augmented_training',
                                n_tuning_records=len(y_train), selected_parameters_json=json.dumps(aug_params, sort_keys=True)))
                        aug, aug_tuning = tuning_cache[aug_key]
                        rows.append(measure(name, ratio, 'TAUG'+suffix, aug,
                            pd.concat([x_train, x], ignore_index=True), pd.concat([y_train, y], ignore_index=True),
                            positions, source=control, tuning=aug_tuning, match=counts))
                    added, match = matched_positions(y_train, y_synth,
                        np.random.SeedSequence([seed, domain_index, split_index, 917]))
                    resampled, resampled_params, _, _, _ = utility.select_augmented_estimator(
                        x_train, y_train, classifier, seed, ratio=ratio,
                        requested_y=y_synth, resample_real=True)
                    resampled_tuning = dict(**utility.tuning_diagnostics(resampled),
                        tuning_source='real_inner_validation_resampled_training',
                        n_tuning_records=len(y_train), selected_parameters_json=json.dumps(resampled_params, sort_keys=True))
                    rows.append(measure(name, ratio, 'TRTR_RESAMPLED', resampled,
                        pd.concat([x_train, x_train.iloc[added]], ignore_index=True),
                        pd.concat([y_train, y_train.iloc[added]], ignore_index=True), train[added],
                        source='real_resampled', tuning=resampled_tuning, match=match))
        print(f'{domain}: fold {split_index + 1}/{len(splits)} complete', flush=True)
    return pd.DataFrame(rows)


def evaluate_learning_curves(real, tables, *, reference_n=None, ratios=RATIOS, folds=5, repeats=3, seed=42,
                             domains=tuple(schema.DISORDER_NAMES)):
    """In-memory API used by tests and small studies. No files or providers."""
    reference_n = len(real) if reference_n is None else reference_n
    ratios = validate_inputs(real, tables, reference_n, ratios, folds, repeats, seed, domains)
    with threadpool_limits(limits=1):
        return pd.concat([domain_rows(real, tables, d, reference_n=reference_n, ratios=ratios,
                                    folds=folds, repeats=repeats, seed=seed) for d in domains], ignore_index=True)


def summarize(folds):
    """Pool confusion matrices within repeats, never across ratios or generation seeds."""
    keys = ['run', 'model', 'generation_seed', 'regime', 'classifier', 'protocol', 'ratio', 'repeat']
    data = folds.copy()
    for key in ('model', 'generation_seed', 'regime'):
        if key not in data:
            data[key] = None
    rows = []
    for identity, group in data.groupby(keys + ['domain'], dropna=False, sort=False):
        matrix = np.sum([np.asarray(json.loads(v), dtype=int) for v in group.confusion_counts_json], axis=0)
        if matrix.sum() != group.n_real_test.sum():
            raise ValueError('Confusion counts do not cover the recorded real test rows.')
        tp, support, predicted = matrix.diagonal(), matrix.sum(axis=1), matrix.sum(axis=0)
        precision = np.divide(tp, predicted, out=np.zeros(5, dtype=float), where=predicted > 0)
        recall = np.divide(tp, support, out=np.full(5, np.nan), where=support > 0)
        f1 = np.divide(2*tp, support+predicted, out=np.zeros(5, dtype=float), where=(support+predicted) > 0)
        high_tp = matrix[3:, 3:].sum()
        values = dict(f1_macro=float(f1.mean()), recall_macro_observed=float(np.nanmean(recall)),
                      precision_macro=float(precision.mean()), accuracy=float(tp.sum()/matrix.sum()),
                      upper_tier_recall=float(high_tp/support[3:].sum()) if support[3:].sum() else np.nan,
                      upper_tier_precision=float(high_tp/predicted[3:].sum()) if predicted[3:].sum() else np.nan,
                      upper_tier_false_positives=int(matrix[:3, 3:].sum()), upper_tier_support=int(support[3:].sum()),
                      n_supported_tiers=int((support > 0).sum()), n_real_test=int(matrix.sum()),
                      support_restricted=bool(group.support_restricted.any()))
        for i, tier in enumerate(schema.TIER_LABELS):
            values.update({f'{tier}_tp': int(tp[i]), f'{tier}_fp': int(predicted[i]-tp[i]),
                           f'{tier}_fn': int(support[i]-tp[i]), f'{tier}_support': int(support[i]),
                           f'{tier}_recall': recall[i], f'{tier}_precision': precision[i] if predicted[i] else np.nan})
        rows.append({**dict(zip(keys+['domain'], identity)), **values})
    domains = pd.DataFrame(rows)
    averages = ['f1_macro', 'recall_macro_observed', 'precision_macro', 'accuracy',
                'upper_tier_recall', 'upper_tier_precision']
    for identity, group in domains.groupby(keys, dropna=False, sort=False):
        for scope, included in [('all_domains', schema.DISORDER_NAMES), ('eligible_domains', ELIGIBLE)]:
            selected = group[group.domain.isin(included)]
            if set(selected.domain) != set(included):
                continue
            rows.append({**dict(zip(keys, identity)), 'domain': scope,
                         **{m: selected[m].mean() for m in averages},
                         'n_supported_tiers': int(selected.n_supported_tiers.sum()),
                         'support_restricted': bool(selected.support_restricted.any())})
    return pd.DataFrame(rows)


def evaluate_runs(real_path, runs, outdir, *, ratios=RATIOS, reference_n=None, folds=5, repeats=3, seed=42, resume=False):
    """Three files, resumable at outcome boundaries, with frozen input fingerprints."""
    real = read_real(real_path)
    reference_n = len(real) if reference_n is None else reference_n
    tables, provenance, identities = {}, {}, {}
    if len({str(Path(p).resolve()) for p in runs}) != len(runs):
        raise ValueError('A run was supplied more than once.')
    for path in runs:
        path = Path(path)
        manifest, table = load_run(path)
        settings = manifest.get('settings', {})
        generation_seed = manifest.get('seed', settings.get('seed'))
        name = f'{path.name}_seed{generation_seed}'
        if name in tables:
            raise ValueError('Run names and seeds must identify distinct inputs.')
        domains = manifest.get('profile_metadata', {}).get('targeted_domains')
        if manifest['regime'] == 'targeted' and set(domains or []) != set(ELIGIBLE):
            raise ValueError('This learning-curve comparison uses the five declared eligible domains.')
        tables[name] = table
        identities[name] = dict(model=manifest.get('method', settings.get('model')),
                                generation_seed=generation_seed, regime=manifest['regime'])
        provenance[name] = dict(manifest_sha256=sha256(path/'manifest.json'), dataset_sha256=sha256(path/'dataset.csv'),
                                n_patients=len(table), **identities[name])
    ratios = validate_inputs(real, tables, reference_n, ratios, folds, repeats, seed, schema.DISORDER_NAMES)
    fingerprint = dict(version=VERSION, stopping_policy=utility.stopping_policy(), revision=REVISION, real_sha256=sha256(real_path), runs=provenance,
        ratios=list(ratios), reference_n=reference_n, folds=folds, repeats=repeats, evaluation_seed=seed,
        sources={n: sha256(Path(__file__).with_name(n+'.py')) for n in
                 ('learning_curves', 'utility', 'schema', 'evaluation', 'io')},
        packages={n: importlib.metadata.version(n) for n in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl')})
    out = Path(outdir)
    manifest_path, raw_path = out/'manifest.json', out/'utility_folds.csv'
    if out.exists() and any(out.iterdir()) and (not resume or not manifest_path.exists()):
        raise FileExistsError('Use a new output directory, or --resume for a compatible checkpoint.')
    out.mkdir(parents=True, exist_ok=True)
    with raw_path.open('a+b') as handle, threadpool_limits(limits=1):
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another evaluator is writing this output.') from exc
        state = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if state:
            if not resume or state['fingerprint'] != fingerprint:
                raise ValueError('Evaluation inputs, settings, source code or package versions changed.')
            handle.seek(0)
            committed = handle.read(state['committed_bytes'])
            if len(committed) != state['committed_bytes'] or hashlib.sha256(committed).hexdigest() != state['raw_sha256']:
                raise ValueError('Committed evaluation rows changed.')
            if state['status'] == 'complete':
                if sha256(raw_path) != state['raw_sha256'] or sha256(out/'metrics.csv') != state['metrics_sha256']:
                    raise ValueError('Completed evaluation files changed.')
                print(f'Learning curves already complete: {out}', flush=True)
                return state
            # Only an uncommitted final outcome block may be discarded after interruption.
            handle.truncate(state['committed_bytes'])
            handle.seek(0, os.SEEK_END)
        else:
            if os.fstat(handle.fileno()).st_size:
                raise ValueError('A nonempty fold file has no checkpoint manifest.')
            state = dict(fingerprint=fingerprint, status='running', completed_domains=[], committed_bytes=0,
                         raw_sha256=hashlib.sha256(b'').hexdigest(),
                         interpretation='Internal same-cohort utility. Classifier test folds contributed to generation inputs.',
                         sampling='Fixed 1x anchor, nested distinct-row extensions, ceil(ratio * n_real_train).',
                         controls='Real training-fold resampling and resampling the fixed 1x synthetic subset, matched to achieved outcome-tier counts where supported.',
                         uncertainty='Pool held-out confusion counts within each CV repeat. Generation seeds and repeats remain separate. Real-only is emitted once per evaluation.',
                         files=['manifest.json', 'utility_folds.csv', 'metrics.csv'])
            write_json(manifest_path, state)
        for domain in schema.DISORDER_NAMES:
            if domain in state['completed_domains']:
                continue
            chunk = domain_rows(real, tables, domain, reference_n=reference_n, ratios=ratios,
                                folds=folds, repeats=repeats, seed=seed)
            for key in ('model', 'generation_seed', 'regime'):
                chunk[key] = chunk.run.map({n: d[key] for n, d in identities.items()})
            payload = chunk.to_csv(index=False, header=state['committed_bytes'] == 0).encode()
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            state['completed_domains'].append(domain)
            state.update(committed_bytes=handle.tell(), raw_sha256=sha256(raw_path))
            write_json(manifest_path, state)
        data = pd.read_csv(raw_path)
        atomic_text(out/'metrics.csv', summarize(data).to_csv(index=False))
        state.update(status='complete', metrics_sha256=sha256(out/'metrics.csv'))
        write_json(manifest_path, state)
    print(f'Learning curves complete: {out} (3 files)', flush=True)
    return state
