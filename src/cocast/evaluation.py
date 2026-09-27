"""Evaluate explicit, completed runs without discovering or mixing old results."""
import importlib.metadata
import json
from pathlib import Path

import pandas as pd

from . import REVISION, schema
from .io import atomic_text, read_real, sha256, write_json
from .metrics import fidelity, proximity


def load_run(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('revision') != REVISION:
        raise ValueError(f'{path}: wrong or missing experiment revision.')
    if manifest.get('status') != 'complete':
        raise ValueError(f'{path}: run is incomplete. Resume generation first.')
    dataset = path / 'dataset.csv'
    if sha256(dataset) != manifest.get('dataset_sha256'):
        raise ValueError(f'{path}: dataset checksum does not match its manifest.')
    if manifest.get('kind') != 'baseline':
        if manifest.get('context') not in {'own', 'full'}:
            raise ValueError(f'{path}: unsupported prompt context for the study.')
        if manifest.get('profile_metadata', {}).get('dependence', 'copula') not in {'copula', 'independent'}:
            raise ValueError(f'{path}: unsupported severity dependence model.')
        if sha256(path / 'responses.jsonl') != manifest.get('journal_sha256'):
            raise ValueError(f'{path}: response journal checksum mismatch.')
    table = pd.read_csv(dataset)
    if schema.ID_COL in table:
        if table[schema.ID_COL].isna().any() or table[schema.ID_COL].duplicated().any():
            raise ValueError(f'{path}: missing or duplicate synthetic IDs.')
        table = table.drop(columns=schema.ID_COL)
    schema.validate_real_table(table)
    if len(table) != manifest.get('n_patients'):
        raise ValueError(f'{path}: row count differs from manifest.')
    if manifest.get('kind') == 'baseline':
        regime = manifest.get('regime')
    else:
        regime = manifest.get('profile_metadata', {}).get('regime')
    if regime not in {'natural', 'targeted'}:
        raise ValueError(f'{path}: natural/targeted provenance is required for evaluation.')
    manifest['regime'] = regime
    return manifest, table[schema.REAL_COLUMNS]


def evaluate_runs(real_path, runs, outdir, *, utility=False, utility_only=False, folds=5, repeats=3, seed=42):
    """Store manifest and long-format metrics, plus one fold table if requested.

    Generation-seed and CV-repeat aggregation are intentionally separate. Raw
    rows preserve both identifiers and never produce automatic significance claims.
    """
    utility = utility or utility_only
    out = Path(outdir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Evaluation directory is not empty. Choose a new output directory.')
    if len({str(Path(p).resolve()) for p in runs}) != len(runs):
        raise ValueError('A run was supplied more than once.')
    real = read_real(real_path)
    tables, manifests, rows, locations = {}, {}, [], {}
    for path in runs:
        manifest, table = load_run(path)
        settings = manifest.get('settings', {})
        identity = manifest.get('method', settings.get('model', Path(path).name)).replace('/', '_')
        name = f"{identity}_{manifest['regime']}_{manifest.get('context', 'baseline')}_seed{manifest.get('seed', settings.get('seed'))}"
        if manifest.get('profile_metadata', {}).get('dependence') == 'independent':
            name += '_independent'
        if name in tables:
            raise ValueError('Duplicate model, condition and seed in one evaluation.')
        locations[name] = Path(path)
        tables[name], manifests[name] = table, manifest
    if not utility and any(m['regime'] == 'targeted' for m in manifests.values()):
        raise ValueError('Targeted runs require --utility. Fidelity/proximity are not targeted endpoints.')
    targeted_domains = {}
    for name, manifest in manifests.items():
        if manifest['regime'] != 'targeted':
            continue
        domains = manifest.get('profile_metadata', {}).get('targeted_domains')
        if (not isinstance(domains, list) or not domains
                or len(set(domains)) != len(domains)
                or not set(domains).issubset(schema.DISORDER_NAMES)):
            raise ValueError(
                f'{name}: targeted profile metadata must declare targeted_domains. '
                'Regenerate the profile and prompt bundle with the current preparation code. '
                'Do not infer targeted outcomes from observed responses or alter a completed run manifest.'
            )
        targeted_domains[name] = domains
    domain_sets = {tuple(sorted(domains)) for domains in targeted_domains.values()}
    if len(domain_sets) > 1:
        raise ValueError('Compare targeted runs with the same targeted domains in one evaluation.')
    if not utility_only:
        for metric, value in proximity(real).items():
            rows.append({'run': 'real', 'family': 'proximity', 'metric': metric, 'value': value})
    run_fields = {}
    for name, table in tables.items():
        info = manifests[name]
        settings = info.get('settings', {})
        fields = {'run': name, 'model': info.get('method', settings.get('model')),
                  'generation_seed': info.get('seed', settings.get('seed')), 'regime': info['regime'],
                  'backend': info.get('device', settings.get('backend')),
                  'precision': settings.get('precision'), 'context': info.get('context')}
        fields['dependence'] = ('baseline' if info.get('kind') == 'baseline' else
                                info.get('profile_metadata', {}).get('dependence', 'copula'))
        run_fields[name] = fields
        print(f'Evaluating {name}: {len(table)} synthetic rows, {info["regime"]}')
        # Targeted data have a deliberately changed distribution. Their endpoint is utility.
        if info['regime'] == 'natural' and not utility_only:
            for family, values in [('fidelity', fidelity(real, table)), ('proximity', proximity(real, table))]:
                rows.extend({**fields, 'family': family, 'metric': metric, 'value': value}
                            for metric, value in values.items())
        if not utility_only:
            from .characterisation import characterise
            rows.extend({**fields, **row} for row in characterise(table, locations[name], info))
    fold_data = None
    if utility:
        from . import utility as utility_module
        from .utility import evaluate_utility, summarize_utility_repeats
        targeted = [name for name, manifest in manifests.items() if manifest['regime'] == 'targeted']
        fold_data = evaluate_utility(real, tables, folds=folds, repeats=repeats, seed=seed,
                                    matched_resampling_runs=targeted, targeted_domains=targeted_domains)
        # Preserve dataset identity on every fold, including the real-resampling
        # comparator whose matching distribution came from a synthetic run.
        for column in ('model', 'generation_seed', 'regime', 'backend', 'precision', 'context', 'dependence'):
            fold_data[column] = fold_data['run'].map({name: fields[column] for name, fields in run_fields.items()})
        primary_domains = next(iter(domain_sets), ())
        repeat_metrics = summarize_utility_repeats(fold_data, primary_domains)
        for row in repeat_metrics.to_dict('records'):
            fields = run_fields.get(row['run'], {'run': 'real'})
            rows.append({**fields, **row, 'family': 'utility_repeat'})
    out.mkdir(parents=True, exist_ok=True)
    atomic_text(out / 'metrics.csv', pd.DataFrame(rows).to_csv(index=False))
    if fold_data is not None:
        atomic_text(out / 'utility_folds.csv', fold_data.to_csv(index=False))
    metadata = {'revision': REVISION, 'evaluation_scope': 'utility_only' if utility_only else 'all_requested', 'status': 'complete', 'n_real': len(real),
                'n_missing_age': int(real[schema.AGE_COL].isna().sum()),
                'real_sha256': sha256(real_path), 'runs': manifests,
                'utility': {'enabled': utility, 'folds_requested': folds, 'repeats': repeats, 'seed': seed,
                            'protocol_version': utility_module.PROTOCOL_VERSION if utility else None,
                            'stopping_policy': utility_module.stopping_policy() if utility else None,
                            'tuning_metric': 'Five-tier macro-F1, with zero for undefined class F1.',
                            'targeted_domains': targeted_domains,
                            'matched_resampling': 'Targeted runs: match achieved outcome-tier counts within real training folds. Absent tiers are excluded and the restricted mix is flagged.',
                            'targeted_primary': 'Targeted versus natural TSTR: macro-recall across the declared targeted domains and five tiers, evaluated on unchanged real test folds. Pool held-out confusion counts within each repeat before computing recall. Undefined recall remains missing and supported tier counts are reported.',
                            'targeted_secondary': 'Per-tier recall and precision, five-tier macro-F1, seven-domain summaries, augmentation and training-fold real-data resampling.',
                            'interpretation': 'Same-cohort utility, not unseen-patient validation.',
                            'labels': [0, 1, 2, 3, 4],
                            'tuning': {'TSTR': 'Inner CV on unique selected synthetic training records only.',
                                       'TRTR': 'Inner CV on each real outer training fold.',
                                       'TAUG': 'Independent tuning of each augmented training condition against real inner-validation records.',
                                       'TRTR_RESAMPLED': 'Independent tuning with resampling performed after each real inner split.'},
                            'summary_policy': 'metrics.csv utility_repeat rows pool test-fold confusion counts per repeat before computing metrics. utility_folds.csv retains raw fold metrics. Repeat variability and generation-seed variability must be summarized separately.'},
                'uncertainty': 'Raw generation seeds and CV repeats retained separately. No confidence intervals inferred.',
                'files': {name: sha256(out / name) for name in ['metrics.csv'] + (['utility_folds.csv'] if utility else [])},
                'source_sha256': {name: sha256(Path(__file__).with_name(name))
                                  for name in ['evaluation.py', 'metrics.py', 'schema.py'] + (['utility.py'] if utility else [])},
                'versions': {name: importlib.metadata.version(name)
                             for name in ['numpy', 'pandas', 'scipy'] + (['scikit-learn', 'threadpoolctl'] if utility else [])}}
    write_json(out / 'manifest.json', metadata)
    print(f'Evaluation complete -> {out} ({len(metadata["files"]) + 1} files)')
