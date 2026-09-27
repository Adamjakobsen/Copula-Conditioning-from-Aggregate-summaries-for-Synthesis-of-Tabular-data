"""Numerical paper summaries, with generation seeds and CV repeats kept distinct."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import sha256, write_json, atomic_text


def read_evaluation(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError(f'Incomplete evaluation: {path}.')
    files = manifest.get('files')
    if isinstance(files, dict):
        for name, digest in files.items():
            if sha256(path / name) != digest:
                raise ValueError(f'Evaluation checksum mismatch: {path / name}.')
        data = pd.read_csv(path / 'metrics.csv')
        data['experiment'] = 'matched'
        data['ratio'] = np.where(data['family'].eq('utility_repeat'), 1., np.nan)
    else:
        if (sha256(path / 'metrics.csv') != manifest['metrics_sha256']
                or sha256(path / 'utility_folds.csv') != manifest['raw_sha256']):
            raise ValueError(f'Learning-curve evaluation checksum mismatch: {path}.')
        wide = pd.read_csv(path / 'metrics.csv')
        identifiers = ['run', 'model', 'generation_seed', 'regime', 'classifier', 'protocol', 'ratio', 'repeat', 'domain']
        values = [c for c in wide if c not in identifiers and c not in ('support_restricted',)]
        data = wide.melt(id_vars=identifiers, value_vars=values, var_name='metric', value_name='value')
        data['family'] = 'utility_repeat'
        data['context'] = 'own'
        data['experiment'] = 'volume'
    for field in ('model', 'generation_seed', 'regime', 'context', 'classifier', 'protocol', 'repeat', 'domain'):
        if field not in data:
            data[field] = np.nan
    real = data['run'].eq('real')
    data.loc[real, ['model', 'regime', 'context']] = ['real', 'real', 'real']
    data.loc[real, 'ratio'] = 0.
    data['context'] = data['context'].fillna('baseline')
    data['domain'] = data['domain'].fillna('all_variables')
    data['condition'] = data['regime'] + ':' + data['context']
    if 'dependence' in data:
        independent = data['dependence'].eq('independent')
        data.loc[independent, 'condition'] += ':independent'
    data.loc[data['domain'].eq('targeted_domains'), 'domain'] = 'eligible_domains'
    return data, manifest


def summarize(evaluations, output):
    if len({str(Path(p).resolve()) for p in evaluations}) != len(evaluations):
        raise ValueError('Supply each evaluation only once.')
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Summary output directory is not empty.')
    parts, inputs, signatures = [], {}, []
    for path in evaluations:
        frame, manifest = read_evaluation(path)
        fingerprint = manifest.get('fingerprint', {})
        policy = manifest.get('utility', {})
        signatures.append((manifest.get('real_sha256', fingerprint.get('real_sha256')),
                           policy.get('protocol_version', fingerprint.get('stopping_policy', {}).get('protocol_version'))))
        parts.append(frame)
        inputs[str(Path(path).resolve())] = sha256(Path(path) / 'manifest.json')
    if len({s[0] for s in signatures}) > 1 or len({s[1] for s in signatures if s[1]}) > 1:
        raise ValueError('Do not combine different reference cohorts or utility protocols.')
    data = pd.concat(parts, ignore_index=True)
    data['value'] = pd.to_numeric(data['value'], errors='raise')
    keys = ['experiment', 'family', 'model', 'condition', 'classifier', 'protocol', 'ratio', 'domain', 'metric']
    identity = keys + ['generation_seed', 'repeat']
    real = data.model.eq('real')
    for _, group in data[real].groupby(identity, dropna=False):
        finite = group.value.dropna()
        if len(finite) and not np.allclose(finite, finite.iloc[0], atol=1e-12, rtol=0):
            raise ValueError('Real-only references disagree. Do not combine different evaluation settings.')
    data = pd.concat([data[~real], data[real].drop_duplicates(identity)], ignore_index=True)
    if data[~data.model.eq('real')].duplicated(identity).any():
        raise ValueError('Duplicate condition/seed/repeat results. Do not mix overlapping evaluations.')
    rows = []
    for key, group in data.groupby(keys, dropna=False, sort=False):
        utility = key[1] == 'utility_repeat'
        if utility:
            seed_sets = group.groupby('repeat').generation_seed.apply(lambda x: tuple(sorted(x.dropna().unique())))
            if seed_sets.nunique() > 1:
                raise ValueError('Generation seeds are incomplete across CV repeats.')
            samples = group.groupby('repeat').value.mean()
            seed_means = group.groupby('generation_seed').value.mean()
            uncertainty = 'cv_repeat_after_seed_mean'
        else:
            samples = group.value
            seed_means = samples
            uncertainty = 'generation_seed' if key[2] != 'real' else 'none'
        rows.append({**dict(zip(keys, key)), 'mean': samples.mean(),
                     'sd': samples.std(ddof=1) if uncertainty != 'none' else np.nan,
                     'n_repeats': len(samples) if utility else np.nan,
                     'n_generation_seeds': group.generation_seed.nunique(),
                     'generation_seed_sd': seed_means.std(ddof=1), 'uncertainty': uncertainty})
    pair_keys = ['experiment', 'model', 'classifier', 'protocol', 'ratio', 'domain', 'metric', 'generation_seed', 'repeat']
    utilities = data[data.family.eq('utility_repeat') & data.experiment.eq('matched')]
    natural = utilities[utilities.condition.eq('natural:own')]
    balanced = utilities[utilities.condition.eq('targeted:own')]
    paired = balanced.merge(natural, on=pair_keys, suffixes=('_balanced', '_natural'), validate='one_to_one')
    paired['value'] = paired.value_balanced - paired.value_natural
    differences = []
    for key, group in paired.groupby(pair_keys[:-2], dropna=False, sort=False):
        samples = group.groupby('repeat').value.mean()
        differences.append({**dict(zip(pair_keys[:-2], key)), 'mean': samples.mean(), 'sd': samples.std(ddof=1),
                            'n_generation_seeds': group.generation_seed.nunique(), 'n_repeats': len(samples),
                            'difference': 'balanced_minus_natural', 'uncertainty': 'cv_repeat_after_paired_seed_mean'})
    out.mkdir(parents=True, exist_ok=True)
    atomic_text(out / 'summary.csv', pd.DataFrame(rows).to_csv(index=False))
    atomic_text(out / 'targeting_changes.csv', pd.DataFrame(differences, columns=pair_keys[:-2]+[
        'mean','sd','n_generation_seeds','n_repeats','difference','uncertainty']).to_csv(index=False))
    write_json(out / 'manifest.json', dict(status='complete', inputs=inputs,
        files={name:sha256(out/name) for name in ('summary.csv', 'targeting_changes.csv')},
        interpretation='Descriptive means and sample SDs. Recall differences are fractions, multiply by 100 for percentage points. No significance or equivalence inference.'))
    print(f'Numerical summaries -> {out}', flush=True)
