"""Reproduce selected study conditions from explicit configuration and inputs."""
from contextlib import nullcontext
import fcntl
from importlib import resources
import json
from pathlib import Path
import shutil

import yaml

from . import REVISION
from .generation import generate, resolve_settings, _check_resume
from .io import sha256, write_json, read_real
from .profiles import generate_profiles, load_aggregates, compute_aggregates
from .prompts import load_bundle, prepare_bundle
from .serving import serve, check_server

CONDITIONS = ('natural', 'full', 'balanced',
              'expanded_natural', 'expanded_balanced', 'independent')


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Study configuration must be a mapping.')
    required = {'aggregates', 'target_prevalences', 'models', 'seeds', 'n', 'expanded_n',
                'conditions', 'baselines', 'evaluation'}
    if set(config) != required:
        raise ValueError(f'Study configuration keys must be {sorted(required)}.')
    if config['n'] < 2 or config['expanded_n'] < config['n']:
        raise ValueError('Use at least two reference records and an expanded size no smaller than n.')
    for key in ('aggregates', 'target_prevalences'):
        value = config[key]
        config[key] = (Path(str(resources.files('cocast').joinpath('resources/aggregate_example.yaml')))
                       if value == '@aggregate_example' else path.parent / value)
    for model, file in config['models'].items():
        config['models'][model] = resolve_settings(yaml.safe_load((path.parent / file).read_text()))
    if set(config['conditions']) != set(CONDITIONS):
        raise ValueError(f'Declare the six paper conditions: {CONDITIONS}.')
    expected = {'natural': ('own', 'natural'), 'full': ('full', 'natural'),
                'balanced': ('own', 'targeted'),
                'expanded_natural': ('own', 'natural'), 'expanded_balanced': ('own', 'targeted'),
                'independent': ('own', 'natural')}
    for name, condition in config['conditions'].items():
        if (condition['context'], condition['regime']) != expected[name]:
            raise ValueError(f'Incorrect context or prevalence regime for {name}.')
        if not set(condition['models']).issubset(config['models']):
            raise ValueError(f'Unknown model in {name}.')
    return config


def selections(config, models, conditions, seeds):
    explicit_models = models is not None and conditions is not None and 'independent' in conditions
    models = list(config['models']) if models is None else models
    conditions = list(config['conditions']) if conditions is None else conditions
    seeds = config['seeds'] if seeds is None else seeds
    for name, values, allowed in [('models', models, config['models']),
                                 ('conditions', conditions, config['conditions'])]:
        if not values or len(set(values)) != len(values) or not set(values).issubset(allowed):
            raise ValueError(f'Invalid or duplicate {name}.')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Supply distinct nonnegative integer seeds.')
    jobs = []
    for model in models:
        for seed in seeds:
            for name in conditions:
                condition = config['conditions'][name]
                # Explicit selectors permit additional model sizes for the copula ablation.
                if model not in condition['models'] and (name != 'independent' or not explicit_models):
                    continue
                jobs.append((model, seed, name))
    if not jobs:
        raise ValueError('No jobs match the selection.')
    return jobs, seeds


def paths(out, model, seed, condition):
    return out / f'seed{seed}' / f'{condition}_{model}'


def prepare_job(config, out, model, seed, condition):
    spec = config['conditions'][condition]
    n = config['expanded_n'] if condition.startswith('expanded_') else config['n']
    mix = spec['regime']
    dependence = 'independent' if condition == 'independent' else 'copula'
    input_name = 'independent_natural' if dependence == 'independent' else mix
    inputs = out / f'seed{seed}' / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    summary = load_aggregates(config['aggregates'])
    prevalence = yaml.safe_load(config['target_prevalences'].read_text()) if mix == 'targeted' else None
    data = {'metadata': dict(revision=REVISION, seed=seed, n_patients=n, regime=mix,
                            aggregate_sha256=sha256(config['aggregates']),
                            prevalences_sha256=sha256(config['target_prevalences']) if prevalence else None),
            'profiles': generate_profiles(summary, n, seed, prevalences=prevalence, dependence=dependence)}
    if dependence == 'independent':
        data['metadata']['dependence'] = dependence
        data['metadata']['sampling_method'] = 'independent_normal_cdf_then_discrete_tier_inverse_cdf'
    if prevalence:
        data['metadata']['targeted_domains'] = list(prevalence)
    profile = inputs / f'{input_name}_{n}.json'
    if profile.exists():
        if json.loads(profile.read_text()) != data:
            raise ValueError(f'Existing profiles differ: {profile}. Use a separate output directory.')
    else:
        write_json(profile, data)
    bundle = inputs / f'{input_name}_{spec["context"]}_{n}'
    prepare_bundle(profile, bundle, summary_path=config['aggregates'], context=spec['context'])
    return bundle


def settings_for(config, model, seed, port):
    return resolve_settings({**config['models'][model], 'seed': seed,
                             'base_url': f'http://127.0.0.1:{port}/v1'})


def completed(run, bundle, settings):
    if not (run / 'manifest.json').exists():
        return False
    manifest = json.loads((run / 'manifest.json').read_text())
    info, records = load_bundle(bundle)
    _check_resume(manifest, info, records, settings)
    if manifest.get('status') != 'complete' or manifest['n_requests'] != len(records):
        return False
    from .evaluation import load_run
    load_run(run)
    return True


def generate_jobs(config, out, jobs, port, existing_server, tensor_parallel_size, extra_attempts=0):
    # Expanded runs explicitly depend on a completed matched-size prefix.
    expanded = list(jobs)
    for model, seed, name in jobs:
        if name.startswith('expanded_'):
            dependency = (model, seed, name.removeprefix('expanded_'))
            if dependency not in expanded:
                expanded.append(dependency)
    expanded.sort(key=lambda j: (j[0], j[1], j[2].startswith('expanded_'), j[2]))
    if existing_server and len({j[0] for j in expanded}) != 1:
        raise ValueError('--existing-server requires a single selected model.')
    for model in dict.fromkeys(j[0] for j in expanded):
        pending = []
        for _, seed, name in (j for j in expanded if j[0] == model):
            bundle = prepare_job(config, out, model, seed, name)
            settings = settings_for(config, model, seed, port)
            run = paths(out, model, seed, name)
            if completed(run, bundle, settings):
                print(f'Validated complete: {run}', flush=True)
            else:
                pending.append((seed, name, bundle, settings, run))
        if not pending:
            continue
        server = nullcontext() if existing_server else serve(
            pending[0][3], [j[2] for j in pending], out / f'vllm_{model}.log',
            tensor_parallel_size=tensor_parallel_size)
        if existing_server:
            check_server(pending[0][3])
        with server:
            for seed, name, bundle, settings, run in pending:
                if name.startswith('expanded_') and not run.exists():
                    source = paths(out, model, seed, name.removeprefix('expanded_'))
                    from .evaluation import load_run
                    load_run(source)
                    run.mkdir(parents=True)
                    for filename in ('manifest.json', 'responses.jsonl', 'dataset.csv'):
                        shutil.copy2(source / filename, run / filename)
                resuming = (run / 'manifest.json').exists()
                result = generate(bundle, run, settings, resume=resuming, extra_attempts=extra_attempts if resuming else 0)
                if result['status'] != 'complete':
                    raise RuntimeError(f'{run}: incomplete responses. Inspect responses.jsonl before resuming.')


def check_reference(config, real):
    table = read_real(real)
    derived = compute_aggregates(table)
    declared = yaml.safe_load(config['aggregates'].read_text())
    fields = ('n', 'tier_prevalences', 'correlation_probit_2dp', 'sex_ratio', 'age_histogram', 'age_missing_rate')
    if any(derived.get(k) != declared.get(k) for k in fields) or len(table) != config['n']:
        raise ValueError('Reference records do not reproduce the configured cohort summaries and size.')
    return table


def baseline_jobs(config, out, real, seeds, methods):
    from .baselines import run_baseline
    from .evaluation import load_run
    check_reference(config, real)
    for seed in seeds:
        for method in methods:
            run = out / f'seed{seed}' / method
            if run.exists():
                manifest, _ = load_run(run)
                expected = dict(method=method, seed=seed, epochs=config['baselines']['epochs'],
                                device=config['baselines']['device'], n_patients=config['n'], real_sha256=sha256(real))
                if any(manifest.get(k) != v for k, v in expected.items()):
                    raise ValueError(f'Baseline inputs or settings differ: {run}.')
                print(f'Validated complete: {run}', flush=True)
                continue
            run_baseline(real, run, method=method, n=config['n'], seed=seed,
                         epochs=config['baselines']['epochs'], device=config['baselines']['device'])


def evaluate_jobs(config, out, real, jobs, seeds, methods, fidelity_only=False):
    from .evaluation import evaluate_runs
    from .learning_curves import evaluate_runs as evaluate_curves
    check_reference(config, real)
    options = {k: config['evaluation'][k] for k in ('folds', 'repeats', 'seed')}
    for seed in seeds:
        selected = [(m, c) for m, s, c in jobs if s == seed]
        main = [paths(out, m, seed, c) for m, c in selected if not c.startswith('expanded_')]
        main += [out / f'seed{seed}' / method for method in methods]
        if main:
            evaluate_runs(real, main, out / 'evaluation' / f'seed{seed}',
                          utility=not fidelity_only, **options)
        volume = [paths(out, m, seed, c) for m, c in selected if c.startswith('expanded_')]
        if volume and not fidelity_only:
            evaluate_curves(real, volume, out / 'evaluation' / f'volume_seed{seed}',
                            reference_n=config['n'], ratios=config['evaluation']['ratios'], resume=True, **options)


def run(args):
    config = load_config(args.config)
    jobs, seeds = selections(config, args.models, args.conditions, args.seeds)
    methods = config['baselines']['methods'] if args.baselines is None else args.baselines
    if args.no_baselines:
        methods = []
    if args.stage in ('evaluate', 'baselines') and not args.real:
        raise ValueError('--real is required for baseline training and evaluation.')
    if args.extra_attempts < 0:
        raise ValueError('Additional attempt allowance must be nonnegative.')
    if args.extra_attempts and args.stage != 'generate':
        raise ValueError('--extra-attempts is only applicable to generation.')
    if not 1024 <= args.port <= 65535 or args.tensor_parallel_size < 1:
        raise ValueError('Use a port from 1024 to 65535 and a positive tensor parallel size.')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.study.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another study process is using this output directory.') from exc
        if args.stage == 'prepare':
            for model, seed, condition in jobs:
                bundle = prepare_job(config, out, model, seed, condition)
                print(f'Prepared {model}, seed {seed}, {condition}: {bundle}', flush=True)
        elif args.stage == 'generate':
            generate_jobs(config, out, jobs, args.port, args.existing_server, args.tensor_parallel_size, args.extra_attempts)
        elif args.stage == 'baselines':
            baseline_jobs(config, out, args.real, seeds, methods)
        elif args.stage == 'evaluate':
            evaluate_jobs(config, out, args.real, jobs, seeds, methods, args.fidelity_only)
