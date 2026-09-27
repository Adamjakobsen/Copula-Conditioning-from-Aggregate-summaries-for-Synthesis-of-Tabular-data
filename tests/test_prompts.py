"""Offline tests for prompt content, provenance and portable bundle validation."""

import copy
import hashlib
from importlib import resources
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cocast.prompts import DISORDERS, REVISION, load_bundle, prepare_bundle, record_hash


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profiles = self.root / "profiles.json"
        self.bundle = self.root / "bundle"
        self.graph = self.root / "graph.json"
        self.questionnaire = self.root / "questionnaire.json"
        package = resources.files("cocast").joinpath("resources")
        self.graph.write_bytes(package.joinpath("knowledge_graph.json").read_bytes())
        self.questionnaire.write_bytes(package.joinpath("questionnaire.json").read_bytes())
        self.kwargs = {"knowledge_graph_path": self.graph, "questionnaire_path": self.questionnaire}
        self.rows = [self.profile(1)]
        self.write_profiles()

    @staticmethod
    def profile(number):
        profile = {"patient_id": f"patient_{number}", "AGE": 21, "SEX": 2}
        for disorder in DISORDERS:
            key = disorder.replace("_", " ").upper()
            profile[key + "_TIER_CODE"] = "tier_0"
            profile[key + "_SCORE_TARGET"] = 0.0
        return profile

    def write_profiles(self, metadata=None):
        value = self.rows if metadata is None else {"metadata": metadata, "profiles": self.rows}
        self.profiles.write_text(json.dumps(value), encoding="utf-8")

    def prepare(self, **kwargs):
        return prepare_bundle(self.profiles, self.bundle, **(self.kwargs | kwargs))

    def rewrite_bundle(self, records):
        """Refresh checksums to test structural checks beyond raw file hashing."""
        manifest = json.loads((self.bundle / "manifest.json").read_text())
        for record in records:
            record["prompt_sha256"] = record_hash(record)
        content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode()
        (self.bundle / "prompts.jsonl").write_bytes(content)
        manifest["bundle_sha256"] = hashlib.sha256(content).hexdigest()
        manifest["request_hashes"] = {r["request_id"]: r["prompt_sha256"] for r in records}
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))

    def test_complete_criteria_notes_pages_and_unchanged_questionnaire(self):
        self.prepare()
        manifest, records = load_bundle(self.bundle)
        self.assertEqual(manifest["revision"], REVISION)
        self.assertEqual(manifest["n_requests"], 7)
        self.assertEqual([r["disorder"] for r in records], list(DISORDERS))
        graph = json.loads(self.graph.read_text())
        questionnaire = json.loads(self.questionnaire.read_text())
        for record in records:
            key = record["disorder"].replace("_", " ").upper()
            entry = graph[key]
            text = record["messages"][1]["content"]
            self.assertIn(entry["clinical_definition"], text)
            self.assertIn(entry["source"]["edition"], text)
            for field in ("criteria_printed_pages", "criteria_pdf_pages", "description_printed_pages", "description_pdf_pages"):
                self.assertIn(", ".join(map(str, entry["source"][field])), text)
            positions = []
            for identifier, criterion in entry["dsm5_criteria"].items():
                positions.append(text.index(f"{identifier}. {criterion}"))
            self.assertEqual(positions, sorted(positions))
            for note in entry.get("criteria_notes", []):
                self.assertIn(f"{note['label']}: {note['text']}", text)
            for item in questionnaire[key]["items"]:
                self.assertIn(f"{item['key']}: {item['label']}", text)
            self.assertEqual(record["target"]["tier_code"], "tier_0")
            self.assertIn("exactly one top-level key, scores", text)
        self.assertEqual({p.name for p in self.bundle.iterdir()}, {"manifest.json", "prompts.jsonl"})

    def test_full_profile_and_own_target_ablation(self):
        self.prepare(context="full")
        _, full = load_bundle(self.bundle)
        other = self.root / "own"
        prepare_bundle(self.profiles, other, context="own", **self.kwargs)
        _, own = load_bundle(other)
        for complete, isolated in zip(full, own):
            full_text, own_text = complete["messages"][1]["content"], isolated["messages"][1]["content"]
            for disorder in DISORDERS:
                line = disorder.replace("_", " ").upper() + ": target "
                self.assertIn(line, full_text)
                self.assertEqual(line in own_text, disorder == isolated["disorder"])
            self.assertEqual(complete["target"], isolated["target"])
            self.assertEqual(full_text.split("=== DSM-5 CONTEXT ===")[1],
                             own_text.split("=== DSM-5 CONTEXT ===")[1])

    def test_source_change_rejected_without_overwriting_existing_bundle(self):
        old = self.prepare()
        self.graph.write_bytes(self.graph.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.prepare()
        self.assertEqual(load_bundle(self.bundle)[0], old)

    def test_graph_completeness_checked_before_preparation(self):
        graph = json.loads(self.graph.read_text())
        del graph["SOCIAL ANXIETY"]["dsm5_criteria"]["J"]
        self.graph.write_text(json.dumps(graph))
        with self.assertRaisesRegex(ValueError, "criteria"):
            self.prepare()
        self.assertFalse(self.bundle.exists())

    def test_missing_numbered_symptom_rejected(self):
        graph = json.loads(self.graph.read_text())
        graph["PANIC"]["dsm5_criteria"]["A"] = graph["PANIC"]["dsm5_criteria"]["A"].replace("13. Fear of dying.", "")
        self.graph.write_text(json.dumps(graph))
        with self.assertRaisesRegex(ValueError, "numbered"):
            self.prepare()

    def test_missing_criterion_note_rejected(self):
        graph = json.loads(self.graph.read_text())
        graph["DEPRESSION"]["criteria_notes"] = []
        self.graph.write_text(json.dumps(graph))
        with self.assertRaisesRegex(ValueError, "notes"):
            self.prepare()

    def test_exact_prefix_extension_and_idempotent_reuse(self):
        old = self.prepare()
        self.assertEqual(self.prepare(), old)
        _, old_records = load_bundle(self.bundle)
        self.rows.append(self.profile(2))
        self.write_profiles()
        new = self.prepare()
        _, records = load_bundle(self.bundle)
        self.assertEqual(new["n_patients"], 2)
        self.assertEqual(records[:7], old_records)
        self.assertEqual({k: new["request_hashes"][k] for k in old["request_hashes"]}, old["request_hashes"])

    def test_changed_prior_profile_and_shrink_rejected(self):
        self.rows.append(self.profile(2))
        self.write_profiles()
        old = self.prepare()
        self.rows[0]["AGE"] = 22
        self.write_profiles()
        with self.assertRaisesRegex(ValueError, "prior patient"):
            self.prepare()
        self.rows = [self.profile(1)]
        self.write_profiles()
        with self.assertRaisesRegex(ValueError, "shrink"):
            self.prepare()
        self.assertEqual(load_bundle(self.bundle)[0], old)

    def test_revision_mismatch_and_context_change_rejected(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "context"):
            self.prepare(context="own")
        path = self.bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["revision"] = "dsm5_criteria_v1"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "revision"):
            load_bundle(self.bundle)

    def test_portable_load_does_not_need_original_sources(self):
        self.prepare()
        moved = self.root / "another_machine" / "bundle"
        shutil.copytree(self.bundle, moved)
        self.graph.unlink()
        self.questionnaire.unlink()
        self.profiles.unlink()
        manifest, records = load_bundle(moved)
        self.assertEqual(len(records), 7)
        self.assertTrue(all("/" not in key for key in manifest["source_hashes"]))

    def test_tampered_bundle_rejected(self):
        self.prepare()
        path = self.bundle / "prompts.jsonl"
        path.write_bytes(path.read_bytes().replace(b"clinical", b"altered", 1))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_bundle(self.bundle)

    def test_reordering_and_duplicate_patient_rejected_even_with_new_hashes(self):
        self.rows.append(self.profile(2))
        self.write_profiles()
        self.prepare()
        _, records = load_bundle(self.bundle)
        reordered = copy.deepcopy(records)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        self.rewrite_bundle(reordered)
        with self.assertRaisesRegex(ValueError, "reordered"):
            load_bundle(self.bundle)
        self.rewrite_bundle(records[:7] + records[:7])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_bundle(self.bundle)

    def test_unattainable_or_wrong_tier_score_rejected(self):
        for score in (0.375, 0.5, float("nan"), True):
            with self.subTest(score=score):
                self.rows[0]["SEPARATION ANXIETY_SCORE_TARGET"] = score
                self.write_profiles()
                with self.assertRaisesRegex(ValueError, "unattainable"):
                    self.prepare()
        self.assertFalse(self.bundle.exists())

    def test_anxiety_attainable_bounds_and_missing_age(self):
        self.rows[0]["AGE"] = None
        self.rows[0]["PANIC_TIER_CODE"] = "tier_1"
        self.rows[0]["PANIC_SCORE_TARGET"] = 1.4
        self.write_profiles()
        self.prepare()
        _, records = load_bundle(self.bundle)
        panic = records[4]
        self.assertEqual((panic["target"]["allowed_min"], panic["target"]["allowed_max"]), (0.5, 1.4))
        self.assertIn("Age: not reported", panic["messages"][1]["content"])

    def test_preparation_cannot_start_subprocess_or_network(self):
        with patch.object(socket, "create_connection", side_effect=AssertionError("network")), \
                patch.object(subprocess, "Popen", side_effect=AssertionError("subprocess")):
            self.prepare()
        self.assertFalse((self.bundle / "responses.csv").exists())

    def test_metadata_envelope_summary_fingerprint_and_extension(self):
        summary = self.root / "summary.yaml"
        summary.write_text("cohort_size: 564\n")
        metadata = {"revision": REVISION, "seed": 42, "n_patients": 1, "regime": "natural",
                    "aggregate_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                    "prevalences_sha256": None}
        self.write_profiles(metadata)
        self.prepare(summary_path=summary)
        self.assertEqual(load_bundle(self.bundle)[0]["profile_metadata"], metadata)
        self.rows.append(self.profile(2))
        metadata["n_patients"] = 2
        self.write_profiles(metadata)
        self.prepare(summary_path=summary)
        metadata["regime"] = "targeted"
        metadata["prevalences_sha256"] = "a" * 64
        self.write_profiles(metadata)
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            self.prepare(summary_path=summary)

    def test_wrong_summary_or_metadata_count_rejected(self):
        summary = self.root / "summary.yaml"
        summary.write_text("cohort_size: 564\n")
        metadata = {"revision": REVISION, "seed": 42, "n_patients": 1, "regime": "natural",
                    "aggregate_sha256": "b" * 64, "prevalences_sha256": "a" * 64}
        self.write_profiles(metadata)
        with self.assertRaisesRegex(ValueError, "aggregate fingerprint"):
            self.prepare(summary_path=summary)
        metadata["n_patients"] = 2
        self.write_profiles(metadata)
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
