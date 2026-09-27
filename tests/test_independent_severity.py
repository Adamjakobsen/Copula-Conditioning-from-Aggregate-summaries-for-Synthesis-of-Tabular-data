"""Independent severity sampling, prompt compatibility and run isolation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cocast import cli, schema, study
from cocast.profiles import generate_profiles, attainable_tier_scores
from cocast.prompts import load_bundle
from test_profiles import summary_fixture
from test_cli_evaluation import FakeProvider
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]


class IndependentSeverityTests(unittest.TestCase):
    def test_independence_preserves_marginal_score_probabilities(self):
        summary = summary_fixture()
        for domain in schema.DISORDER_NAMES:
            summary['tier_prevalences'][domain] = dict(zip(schema.TIER_LABELS, [.5, .2, .15, .1, .05]))
        rows = generate_profiles(summary, 20000, 42, dependence='independent')
        matrix = np.array([[r[f'{schema.disorder_key(d)}_SCORE_TARGET']
                            for d in schema.DISORDER_NAMES] for r in rows])
        correlations = np.corrcoef(matrix.T)
        self.assertLess(np.abs(correlations[np.triu_indices(7, 1)]).max(), .035)
        for column, domain in enumerate(schema.DISORDER_NAMES):
            for tier, probability in enumerate([.5, .2, .15, .1, .05]):
                support = attainable_tier_scores(domain, tier)
                selected = np.isin(matrix[:, column], support)
                self.assertAlmostEqual(selected.mean(), probability, delta=.012)
                for score in support:
                    self.assertAlmostEqual(np.mean(matrix[:, column] == score),
                                           probability / len(support), delta=.01)

    def test_correlations_unused_demographics_and_prefix_preserved(self):
        summary = summary_fixture()
        before = deepcopy(summary)
        ablation = generate_profiles(summary, 600, 43, dependence='independent')
        natural = generate_profiles(summary, 600, 43)
        self.assertEqual(summary, before)
        for a, n in zip(ablation, natural):
            self.assertEqual([a[k] for k in ('patient_id', 'AGE', 'SEX')],
                             [n[k] for k in ('patient_id', 'AGE', 'SEX')])
        self.assertEqual(generate_profiles(summary, 564, 43, dependence='independent'), ablation[:564])
        summary['correlation_probit_2dp'] = np.eye(7).tolist()
        self.assertEqual(generate_profiles(summary, 600, 43, dependence='independent'), ablation)
        self.assertEqual(generate_profiles(summary, 600, 43), ablation)
        self.assertNotEqual(generate_profiles(summary, 600, 44, dependence='independent'), ablation)

    def test_zero_probability_tiers_and_invalid_mode(self):
        summary = summary_fixture()
        summary['tier_prevalences']['panic'] = dict(zip(schema.TIER_LABELS, [0, .5, 0, .5, 0]))
        rows = generate_profiles(summary, 564, 44, dependence='independent')
        self.assertEqual({r['PANIC_TIER_CODE'] for r in rows}, {'tier_1', 'tier_3'})
        with self.assertRaisesRegex(ValueError, 'dependence'):
            generate_profiles(summary, 2, 42, dependence='unknown')

    def test_preparation_is_offline_separate_and_prompt_content_unchanged(self):
        config = study.load_config(ROOT / 'configs/paper.yaml')
        config.update(n=3, expanded_n=4)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch('cocast.study.generate', side_effect=AssertionError('inference')), \
                 patch('cocast.study.serve', side_effect=AssertionError('server')):
                natural = study.prepare_job(config, out, '27b', 42, 'natural')
                independent = study.prepare_job(config, out, '27b', 42, 'independent')
                self.assertEqual(independent, study.prepare_job(config, out, '9b', 42, 'independent'))
            self.assertNotEqual(natural, independent)
            natural_records, independent_records = load_bundle(natural)[1], load_bundle(independent)[1]
            self.assertEqual(len(independent_records), 21)
            for a, n in zip(independent_records, natural_records):
                self.assertIsNotNone(a['target'])
                self.assertEqual(a['messages'][0], n['messages'][0])
                self.assertEqual(a['messages'][-1]['content'].split('=== DSM-5 CONTEXT ===')[1],
                                 n['messages'][-1]['content'].split('=== DSM-5 CONTEXT ===')[1])
            profile = out / 'seed42/inputs/independent_natural_3.json'
            data = json.loads(profile.read_text())
            self.assertEqual(data['metadata']['dependence'], 'independent')
            data['metadata']['dependence'] = 'copula'
            profile.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'profiles differ'):
                study.prepare_job(config, out, '27b', 42, 'independent')

    def test_default_paper_includes_copula_ablation_and_cli_available(self):
        paper = study.load_config(ROOT / 'configs/paper.yaml')
        jobs, _ = study.selections(paper, None, None, None)
        self.assertEqual([j for j in jobs if j[2] == 'independent'],
                         [('27b', seed, 'independent') for seed in (42, 43, 44)])
        self.assertEqual(len(jobs), 36)
        args = cli.parser().parse_args(['profiles', '--aggregates', 'a.yaml', '--output', 'p.json',
                                       '--n', '564', '--dependence', 'independent'])
        self.assertEqual(args.dependence, 'independent')

    def test_generation_resumes_and_rejects_natural_bundle(self):
        config = study.load_config(ROOT / 'configs/paper.yaml')
        config.update(n=2, expanded_n=4)
        config['models']['27b']['workers'] = 1
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            provider = FakeProvider()
            with patch('cocast.study.serve', return_value=nullcontext()), \
                 patch('cocast.generation.check_context', return_value={}), \
                 patch('cocast.generation.get_provider', return_value=provider):
                study.generate_jobs(config, out, [('27b', 42, 'independent')], 8019, False, 1)
            self.assertEqual(provider.calls, 14)
            with patch('cocast.study.serve', side_effect=AssertionError('already complete')):
                study.generate_jobs(config, out, [('27b', 42, 'independent')], 8019, False, 1)
            natural = study.prepare_job(config, out, '27b', 42, 'natural')
            with self.assertRaises(ValueError):
                study.completed(study.paths(out, '27b', 42, 'independent'), natural,
                                study.settings_for(config, '27b', 42, 8019))
            with patch('cocast.study.serve', return_value=nullcontext()), \
                 patch('cocast.generation.check_context', return_value={}), \
                 patch('cocast.generation.get_provider', return_value=provider):
                study.generate_jobs(config, out, [('27b', 42, 'natural')], 8019, False, 1)
            from cocast.evaluation import evaluate_runs
            from cocast.results import summarize
            from test_profiles import real_fixture
            import pandas as pd
            real_fixture(8).to_csv(out / 'real.csv', index=False)
            evaluate_runs(out / 'real.csv', [study.paths(out, '27b', 42, name)
                                           for name in ('natural', 'independent')], out / 'evaluation')
            summarize([out / 'evaluation'], out / 'summary')
            summary = pd.read_csv(out / 'summary/summary.csv')
            self.assertTrue({'natural:own', 'natural:own:independent'}.issubset(set(summary.condition)))


if __name__ == '__main__':
    unittest.main()
