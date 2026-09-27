"""Offline integration checks using artificial tables and seven fake responses."""

import contextlib
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from cocast import REVISION, cli, generation, schema
from cocast.evaluation import evaluate_runs, load_run
from cocast.io import sha256
from cocast.profiles import compute_aggregates
from cocast.prompts import prepare_bundle


SETTINGS = {
    "model": "offline-fixture", "model_revision": "a" * 40, "precision": "bf16",
    "base_url": "http://127.0.0.1:9999/v1", "backend": "vllm", "seed": 42,
    "max_retries": 0, "workers": 1,
}


class FakeProvider:
    """Return a valid zero-score questionnaire without contacting a server."""
    def __init__(self):
        self.calls = 0

    def complete(self, messages, *, seed):
        self.calls += 1
        items = messages[1]["content"].split("=== QUESTIONNAIRE ITEMS ===")[1]
        keys = re.findall(r"^(W1_[a-z_]+_it\d+):", items, flags=re.MULTILINE)
        if len(keys) not in (9, 10):
            raise AssertionError("Unexpected questionnaire prompt in test fixture.")
        return {"content": json.dumps({"scores": {key: 0 for key in keys}}),
                "response_model": SETTINGS["model"], "finish_reason": "stop"}


class CliEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.real = self.root / "real.csv"
        self.summary = self.root / "summary.yaml"
        self.config = self.root / "config.yaml"
        self.out = self.root / "evaluation"
        values = {schema.AGE_COL: [18, 19, 20, 21, 22, 23, None, None],
                  schema.SEX_COL: [1, 2] * 4}
        for disorder, spec in schema.DISORDERS.items():
            for key in schema.item_columns(disorder):
                values[key] = [row % (spec["item_max"] + 1) for row in range(8)]
        self.real_table = pd.DataFrame(values)[schema.REAL_COLUMNS]
        self.real_table.to_csv(self.real, index=False)
        self.summary.write_text(yaml.safe_dump(compute_aggregates(self.real_table), sort_keys=False))
        self.config.write_text(yaml.safe_dump(SETTINGS))

    def execute(self, arguments):
        """Exercise parser plus dispatch while letting ValueErrors reach tests."""
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.execute(cli.parser().parse_args(arguments))

    def profile_input(self, name="profiles", regime="natural"):
        profile = {"patient_id": "fixture_1", "AGE": None, "SEX": 2}
        for disorder in schema.DISORDER_NAMES:
            key = schema.disorder_key(disorder)
            profile[key + "_TIER_CODE"] = "tier_0"
            profile[key + "_SCORE_TARGET"] = 0
        metadata = {"revision": REVISION, "seed": 42, "n_patients": 1, "regime": regime,
                    "aggregate_sha256": sha256(self.summary),
                    "prevalences_sha256": "b" * 64 if regime == "targeted" else None}
        path = self.root / (name + ".json")
        path.write_text(json.dumps({"metadata": metadata, "profiles": [profile]}))
        return path

    def generation_run(self, name="llm_run", regime="natural"):
        profile = self.profile_input(name + "_profiles", regime)
        bundle = self.root / (name + "_bundle")
        run = self.root / name
        prepare_bundle(profile, bundle, summary_path=self.summary)
        provider = FakeProvider()
        with patch("cocast.generation.get_provider", return_value=provider), \
                patch("cocast.generation.check_context", return_value={"max_input_tokens": 10}):
            manifest = generation.generate(bundle, run, SETTINGS, progress_every=0)
        self.assertEqual(provider.calls, 7)
        self.assertEqual(manifest["status"], "complete")
        return run

    def baseline_run(self, name="baseline"):
        """Artificial completed baseline fixture with at least two rows for ED."""
        run = self.root / name
        run.mkdir()
        table = self.real_table.copy()
        table.insert(0, schema.ID_COL, [f"fake_{i}" for i in range(len(table))])
        table.to_csv(run / "dataset.csv", index=False)
        manifest = {"revision": REVISION, "status": "complete", "kind": "baseline",
                    "method": "ctgan", "regime": "natural", "seed": 42, "device": "cpu",
                    "n_patients": len(table), "dataset_sha256": sha256(run / "dataset.csv")}
        (run / "manifest.json").write_text(json.dumps(manifest))
        return run

    def test_empty_target_prevalence_input_fails_before_profiles_are_created(self):
        path = self.root / "target.yaml"
        output = self.root / "target_profiles.json"
        for content in ("", "{}", "tier_prevalences: {}\n", "tier_prevalences: null\n", "[]"):
            with self.subTest(content=content):
                path.write_text(content)
                with patch("cocast.profiles.generate_profiles") as generator:
                    with self.assertRaisesRegex(ValueError, "nonempty mapping"):
                        self.execute(["profiles", "--aggregates", str(self.summary), "--output", str(output),
                                      "--n", "1", "--prevalences", str(path)])
                    generator.assert_not_called()
                self.assertFalse(output.exists())

    def test_cli_prepare_and_dry_run_never_call_providers_or_create_results(self):
        profiles = self.profile_input()
        bundle, run = self.root / "bundle", self.root / "run"
        with patch("cocast.generation.get_provider", side_effect=AssertionError("provider")), \
                patch("cocast.generation.generate", side_effect=AssertionError("generation")), \
                patch("cocast.profiles.generate_profiles", side_effect=AssertionError("profiles")), \
                patch("cocast.providers.urlopen", side_effect=AssertionError("network")):
            self.execute(["prepare", "--profiles", str(profiles), "--out", str(bundle),
                          "--summary", str(self.summary)])
            snapshot = {path.name: path.read_bytes() for path in bundle.iterdir()}
            self.execute(["generate", "--bundle", str(bundle), "--out", str(run),
                          "--config", str(self.config), "--dry-run"])
        self.assertEqual(set(snapshot), {"manifest.json", "prompts.jsonl"})
        self.assertEqual(snapshot, {path.name: path.read_bytes() for path in bundle.iterdir()})
        self.assertFalse(run.exists())

    def test_dry_run_rejects_unpinned_configuration(self):
        profiles, bundle = self.profile_input(), self.root / "bundle"
        prepare_bundle(profiles, bundle)
        self.config.write_text(yaml.safe_dump({**SETTINGS, "model_revision": "UNSET"}))
        with self.assertRaisesRegex(ValueError, "model_revision|UNSET"):
            self.execute(["generate", "--bundle", str(bundle), "--out", str(self.root / "run"),
                          "--config", str(self.config), "--dry-run"])
        self.assertFalse((self.root / "run").exists())

    def test_generated_fixture_loads_across_real_prompt_and_generator_contracts(self):
        run = self.generation_run()
        manifest, table = load_run(run)
        self.assertEqual(manifest["regime"], "natural")
        self.assertEqual(list(table), schema.REAL_COLUMNS)
        self.assertEqual(len(table), 1)
        self.assertTrue(table[schema.ALL_ITEM_COLUMNS].eq(0).all().all())
        self.assertTrue(table[schema.AGE_COL].isna().all())

    def test_evaluation_rejects_mixed_revisions_before_writing(self):
        valid = self.baseline_run("valid")
        old = self.baseline_run("incompatible")
        path = old / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["revision"] = "incompatible_protocol"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "revision"):
            evaluate_runs(self.real, [valid, old], self.out)
        self.assertFalse(self.out.exists())

    def test_evaluation_rejects_unsupported_context_and_dependence(self):
        run = self.generation_run()
        path = run / "manifest.json"
        original = json.loads(path.read_text())
        manifest = json.loads(path.read_text())
        manifest['context'] = 'unsupported'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'unsupported prompt context'):
            evaluate_runs(self.real, [run], self.out)
        original['profile_metadata']['dependence'] = 'unsupported'
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, 'unsupported severity dependence'):
            evaluate_runs(self.real, [run], self.out)
        self.assertFalse(self.out.exists())

    def test_evaluation_rejects_incomplete_run_before_writing(self):
        run = self.generation_run()
        path = run / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["status"] = "interrupted"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluate_runs(self.real, [run], self.out)
        self.assertFalse(self.out.exists())

    def test_evaluation_rejects_modified_dataset_or_journal(self):
        original = self.generation_run()
        for filename, message in (("dataset.csv", "dataset checksum"), ("responses.jsonl", "journal checksum")):
            with self.subTest(filename=filename):
                run = self.root / filename.replace(".", "_")
                shutil.copytree(original, run)
                path = run / filename
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_runs(self.real, [run], self.out)
                self.assertFalse(self.out.exists())

    def test_targeted_run_requires_utility_before_output_files(self):
        run = self.generation_run(regime="targeted")
        with self.assertRaisesRegex(ValueError, "require --utility"):
            self.execute(["evaluate", "--real", str(self.real), "--run", str(run), "--out", str(self.out)])
        self.assertFalse(self.out.exists())

    def test_natural_evaluation_writes_only_manifest_and_metrics(self):
        run = self.baseline_run()
        self.execute(["evaluate", "--real", str(self.real), "--run", str(run), "--out", str(self.out)])
        self.assertEqual({path.name for path in self.out.iterdir()}, {"manifest.json", "metrics.csv"})
        manifest = json.loads((self.out / "manifest.json").read_text())
        metrics = pd.read_csv(self.out / "metrics.csv")
        self.assertEqual(manifest["n_real"], 8)
        self.assertEqual(manifest["n_missing_age"], 2)
        self.assertEqual(manifest["files"]["metrics.csv"], sha256(self.out / "metrics.csv"))
        self.assertEqual(set(metrics["family"]), {"fidelity", "proximity", "characterisation"})
        self.assertEqual(set(metrics["run"]), {"real", "ctgan_natural_baseline_seed42"})


if __name__ == "__main__":
    unittest.main()
