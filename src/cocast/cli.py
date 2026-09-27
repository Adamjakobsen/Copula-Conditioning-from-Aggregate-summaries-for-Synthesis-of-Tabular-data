"""Explicit stages with compact outputs. No command runs another stage implicitly."""
import argparse
import json
from pathlib import Path

import yaml

from . import REVISION
from .io import atomic_text, read_real, sha256, write_json


def parser():
    p = argparse.ArgumentParser(description='Generate psychiatric questionnaire responses from aggregate summaries.')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('summarize', help='Custodian stage: derive summaries from a baseline CSV or SPSS file.')
    a.add_argument('--real', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--export-real', help='Optional baseline CSV for local evaluation, never needed by generation.')
    a = sub.add_parser('profiles', help='Sample profiles using only the summary file.')
    a.add_argument('--aggregates', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--n', type=int, required=True)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--prevalences', help='Optional YAML with per-domain tier prevalences for targeted generation.')
    a.add_argument('--dependence', choices=['copula', 'independent'], default='copula')
    a = sub.add_parser('prepare', help='Assemble reusable prompts from existing profiles without inference.')
    a.add_argument('--profiles', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--summary')
    a.add_argument('--context', choices=['full', 'own'], default='own')
    a = sub.add_parser('check', help='Verify a bundle and optionally measure its complete token lengths.')
    a.add_argument('--bundle', required=True)
    a.add_argument('--tokenizer', help='Downloaded tokenizer path or cached model ID. Never downloads.')
    a.add_argument('--max-tokens', type=int, default=2048)
    a.add_argument('--context-window', type=int, default=8192)
    a = sub.add_parser('generate', help='Call an already running inference server.')
    a.add_argument('--bundle', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--config', required=True)
    a.add_argument('--resume', action='store_true')
    a.add_argument('--extra-attempts', type=int, default=0, help='Explicit additional attempts for unanswered requests, requires --resume.')
    a.add_argument('--dry-run', action='store_true', help='Validate inputs and print workload without inference or output files.')
    a = sub.add_parser('baseline', help='Fit and sample one CTGAN or TVAE baseline, explicitly.')
    a.add_argument('--real', required=True)
    a.add_argument('--method', choices=['ctgan', 'tvae'], required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--n', type=int)
    a.add_argument('--epochs', type=int, default=500)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    a = sub.add_parser('evaluate', help='Evaluate explicit completed runs. No directory discovery or figures.')
    a.add_argument('--real', required=True)
    a.add_argument('--run', action='append', required=True, help='Run directory, repeat to compare runs.')
    a.add_argument('--out', required=True)
    a.add_argument('--utility', action='store_true', help='Also run the two declared downstream classifiers.')
    a.add_argument('--utility-only', action='store_true', help='Rerun utility without fidelity, proximity or generation.')
    a.add_argument('--folds', type=int, default=5)
    a.add_argument('--repeats', type=int, default=3)
    a.add_argument('--seed', type=int, default=42)
    a = sub.add_parser('learning-curves', help='Utility-only size curves from explicit completed pools. No generation.')
    a.add_argument('--real', required=True)
    a.add_argument('--run', action='append', required=True, help='Completed maximum-size pool. Repeat for mixes/seeds.')
    a.add_argument('--out', required=True)
    a.add_argument('--ratios', type=float, nargs='+', default=[.5, 1, 2, 5])
    a.add_argument('--reference-n', type=int, help='Reference prefix pool size. Defaults to the real cohort size.')
    a.add_argument('--folds', type=int, default=5)
    a.add_argument('--repeats', type=int, default=3)
    a.add_argument('--seed', type=int, default=42, help='Evaluation seed, distinct from generation seeds.')
    a.add_argument('--resume', action='store_true')
    a = sub.add_parser('study', help='Run selected conditions from the paper configuration.')
    a.add_argument('stage', choices=['prepare', 'generate', 'baselines', 'evaluate'])
    a.add_argument('--config', default='configs/paper.yaml')
    a.add_argument('--out', required=True)
    a.add_argument('--models', nargs='+', choices=['4b', '9b', '27b'])
    a.add_argument('--conditions', nargs='+', choices=['natural', 'full', 'balanced', 'expanded_natural', 'expanded_balanced', 'independent'])
    a.add_argument('--seeds', nargs='+', type=int)
    a.add_argument('--real')
    a.add_argument('--baselines', nargs='+', choices=['ctgan', 'tvae'])
    a.add_argument('--no-baselines', action='store_true')
    a.add_argument('--fidelity-only', action='store_true')
    a.add_argument('--extra-attempts', type=int, default=0, help='Explicit additional attempt allowance when resuming incomplete generation.')
    a.add_argument('--existing-server', action='store_true')
    a.add_argument('--port', type=int, default=8000)
    a.add_argument('--tensor-parallel-size', type=int, default=1)
    a = sub.add_parser('results', help='Aggregate verified evaluation outputs into numerical summaries.')
    a.add_argument('--evaluation', action='append', required=True)
    a.add_argument('--out', required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        execute(args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as exc:
        p.exit(2, f'Error: {exc}\n')


def execute(args):
    if args.command == 'study':
        from .study import run
        run(args)
    elif args.command == 'results':
        from .results import summarize
        summarize(args.evaluation, args.out)
    elif args.command == 'summarize':
        from .profiles import compute_aggregates
        frame = read_real(args.real)
        summary = compute_aggregates(frame, source='Computed from the supplied baseline cohort, not a published correlation table.')
        summary['source_sha256'] = sha256(args.real)
        atomic_text(args.output, yaml.safe_dump(summary, sort_keys=False))
        if args.export_real:
            atomic_text(args.export_real, frame.to_csv(index=False))
        print(f'Summarized {len(frame)} participants -> {args.output}')
    elif args.command == 'profiles':
        from .profiles import generate_profiles, load_aggregates
        agg = load_aggregates(args.aggregates)
        prevalence = yaml.safe_load(Path(args.prevalences).read_text()) if args.prevalences else None
        if args.prevalences and (not isinstance(prevalence, dict) or not prevalence):
            raise ValueError('A targeted prevalence file must contain a nonempty mapping.')
        if prevalence and 'tier_prevalences' in prevalence:
            prevalence = prevalence['tier_prevalences']
            if not isinstance(prevalence, dict) or not prevalence:
                raise ValueError('tier_prevalences must contain a nonempty mapping.')
        profiles = generate_profiles(agg, args.n, args.seed, prevalences=prevalence, dependence=args.dependence)
        data = {'metadata': {'revision': REVISION, 'seed': args.seed, 'n_patients': args.n,
                'regime': 'targeted' if args.prevalences else 'natural',
                'aggregate_sha256': sha256(args.aggregates),
                'prevalences_sha256': sha256(args.prevalences) if args.prevalences else None},
                'profiles': profiles}
        if args.dependence == 'independent':
            data['metadata']['dependence'] = args.dependence
            data['metadata']['sampling_method'] = 'independent_normal_cdf_then_discrete_tier_inverse_cdf'
        if args.prevalences:
            data['metadata']['targeted_domains'] = list(prevalence)
        path = Path(args.output)
        if path.exists() and json.loads(path.read_text()) != data:
            previous = json.loads(path.read_text())
            old = previous.get('profiles', [])
            old_meta = {k: v for k, v in previous.get('metadata', {}).items() if k != 'n_patients'}
            new_meta = {k: v for k, v in data['metadata'].items() if k != 'n_patients'}
            if old_meta != new_meta or not old or profiles[:len(old)] != old:
                raise ValueError('Existing profiles differ. Use a new output path.')
        write_json(path, data)
        print(f'Prepared {args.n} profiles -> {args.output}')
    elif args.command == 'prepare':
        from .prompts import prepare_bundle
        m = prepare_bundle(args.profiles, args.out, summary_path=args.summary, context=args.context)
        print(f"Prepared {m['n_requests']} requests -> {args.out}")
    elif args.command == 'check':
        from .prompts import load_bundle
        if args.max_tokens < 1 or args.context_window <= args.max_tokens:
            raise ValueError('Use positive token limits with room for the input prompt.')
        m, records = load_bundle(args.bundle)
        print(f"Bundle valid: {m['n_patients']} patients, {len(records)} requests, {m['context']} context")
        if args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
            lengths = [len(tokenizer.apply_chat_template(r['messages'], tokenize=True,
                       add_generation_prompt=True, enable_thinking=False, return_dict=False)) for r in records]
            peak = max(lengths)
            if peak + args.max_tokens > args.context_window:
                raise ValueError(f'{peak} input + {args.max_tokens} output tokens exceed {args.context_window}. Criteria were not truncated.')
            print(f'Tokens per complete request: {min(lengths)} to {peak}. Output reserve: {args.max_tokens}. Context: {args.context_window}.')
    elif args.command == 'generate':
        from .prompts import load_bundle
        from .generation import resolve_settings
        settings = yaml.safe_load(Path(args.config).read_text())
        if not isinstance(settings, dict):
            raise ValueError('Generation configuration must be a mapping.')
        settings = resolve_settings(settings)
        if args.dry_run:
            m, _ = load_bundle(args.bundle)
            print(f"Ready to request {m['n_requests']} completions from {settings['model']}. No generation performed.")
        else:
            from .generation import generate
            print(f"Generating with {settings['model']} ({settings['backend']}, {settings['precision']}) -> {args.out}")
            result = generate(args.bundle, args.out, settings, resume=args.resume, extra_attempts=args.extra_attempts)
            if result['status'] != 'complete':
                raise RuntimeError(f"Run {result['status']}: {result['completed_requests']}/{result['n_requests']} valid requests. See responses.jsonl.")
            print(f"Complete: {result['n_patients']} patients, {result['n_attempts']} attempts, 3 output files.")
    elif args.command == 'baseline':
        from .baselines import run_baseline
        run_baseline(args.real, args.out, method=args.method, n=args.n, epochs=args.epochs,
                     seed=args.seed, device=args.device)
    elif args.command == 'learning-curves':
        from .learning_curves import evaluate_runs
        evaluate_runs(args.real, args.run, args.out, ratios=args.ratios, reference_n=args.reference_n,
                      folds=args.folds, repeats=args.repeats, seed=args.seed, resume=args.resume)
    elif args.command == 'evaluate':
        from .evaluation import evaluate_runs
        evaluate_runs(args.real, args.run, args.out, utility=args.utility, utility_only=args.utility_only, folds=args.folds,
                      repeats=args.repeats, seed=args.seed)
