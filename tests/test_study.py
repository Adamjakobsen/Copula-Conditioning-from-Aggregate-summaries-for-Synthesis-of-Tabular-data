"""Offline study preparation and generation orchestration contracts."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cocast import cli, study
from cocast.prompts import load_bundle
from cocast.io import write_json
from test_cli_evaluation import FakeProvider

CONFIG = Path(__file__).resolve().parents[1] / 'configs/paper.yaml'


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = study.load_config(CONFIG)
        self.config.update(n=2, expanded_n=4)

    def test_preparation_is_offline_and_shared_between_models(self):
        with patch('cocast.study.generate', side_effect=AssertionError('generation')), \
             patch('cocast.study.serve', side_effect=AssertionError('server')):
            one = study.prepare_job(self.config, self.root, '4b', 42, 'natural')
            two = study.prepare_job(self.config, self.root, '27b', 42, 'natural')
        self.assertEqual(one, two)
        self.assertEqual(len(load_bundle(one)[1]), 14)

    def test_ablation_and_extension_preserve_patient_inputs(self):
        own = study.prepare_job(self.config, self.root, '27b', 42, 'natural')
        full = study.prepare_job(self.config, self.root, '27b', 42, 'full')
        independent = study.prepare_job(self.config, self.root, '27b', 42, 'independent')
        expanded = study.prepare_job(self.config, self.root, '27b', 42, 'expanded_natural')
        records = load_bundle(own)[1]
        self.assertEqual(load_bundle(expanded)[1][:14], records)
        self.assertEqual([r['profile'] for r in records], [r['profile'] for r in load_bundle(full)[1]])
        for r, reference in zip(load_bundle(independent)[1], records):
            self.assertIsNotNone(r['target'])
            self.assertEqual(r['profile'], reference['profile'])
            self.assertIn('=== DSM-5 CONTEXT ===', r['messages'][-1]['content'])

    def test_default_and_explicit_ablation_models(self):
        jobs, _ = study.selections(self.config, None, ['independent'], [42])
        self.assertEqual(jobs, [('27b', 42, 'independent')])
        jobs, _ = study.selections(self.config, ['4b', '9b', '27b'], ['independent'], [42])
        self.assertEqual(len(jobs), 3)

    def test_changed_profiles_rejected(self):
        study.prepare_job(self.config, self.root, '27b', 42, 'natural')
        file = self.root / 'seed42/inputs/natural_2.json'
        write_json(file, {})
        with self.assertRaisesRegex(ValueError, 'profiles differ'):
            study.prepare_job(self.config, self.root, '27b', 42, 'natural')

    def test_generation_from_empty_directory_and_expansion(self):
        config = deepcopy(self.config)
        config['models']['27b']['workers'] = 1
        provider = FakeProvider()
        with patch('cocast.study.serve', return_value=nullcontext()) as server, \
             patch('cocast.generation.check_context', return_value={}), \
             patch('cocast.generation.get_provider', return_value=provider):
            study.generate_jobs(config, self.root, [('27b', 42, 'expanded_natural')], 8000, False, 1)
        self.assertEqual(provider.calls, 28)
        self.assertEqual(server.call_count, 1)
        source = self.root / 'seed42/natural_27b/responses.jsonl'
        extended = self.root / 'seed42/expanded_natural_27b/responses.jsonl'
        self.assertTrue(extended.read_bytes().startswith(source.read_bytes()))
        with patch('cocast.study.serve', side_effect=AssertionError('completed server')):
            study.generate_jobs(config, self.root, [('27b', 42, 'expanded_natural')], 8000, False, 1)
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            study.generate_jobs(config, self.root, [('27b', 42, 'natural')], 8001, False, 1)

    def test_server_multi_model_rejected(self):
        with self.assertRaisesRegex(ValueError, 'single selected model'):
            study.generate_jobs(self.config, self.root, [('4b',42,'natural'),('27b',42,'natural')],8000,True,1)

    def test_cli_stages_do_not_require_participant_records_for_preparation(self):
        args = cli.parser().parse_args(['study','prepare','--out',str(self.root),'--models','27b','--seeds','42'])
        with patch('cocast.study.load_config', return_value=self.config), \
             patch('cocast.study.generate_jobs', side_effect=AssertionError('inference')), \
             patch('cocast.study.baseline_jobs', side_effect=AssertionError('baseline')):
            study.run(args)
