"""Offline tests for strict responses and durable, compatible generation resumes."""

import copy
import csv
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cocast import generation, schema
from cocast.prompts import REVISION, record_hash
from cocast.providers import OpenAICompatibleProvider, ProviderError


SETTINGS = {"model": "Qwen/test", "model_revision": "a" * 40, "precision": "bf16",
            "base_url": "http://127.0.0.1:8087/v1", "backend": "vllm", "seed": 42,
            "max_retries": 1, "workers": 1}


def bundle(n=1):
    records = []
    for patient in range(1, n + 1):
        for disorder in schema.DISORDERS:
            record = {"request_id": f"{patient}:{disorder}", "patient_id": patient, "disorder": disorder,
                      "profile": {"AGE": 38, "SEX": 1},
                      "target": {"score_target": 0, "tier_code": "tier_0", "allowed_min": 0,
                                 "allowed_max": 4 if disorder == "depression" else 0.4,
                                 "measurement": "sum" if disorder == "depression" else "average"},
                      "messages": [{"role": "user", "content": disorder}]}
            record["prompt_sha256"] = record_hash(record)
            records.append(record)
    manifest = {"revision": REVISION, "n_patients": n, "n_requests": len(records), "context": "full",
                "source_hashes": {"profiles": str(n), "questionnaire": "q", "knowledge_graph": "kg", "renderer": "r"},
                "request_hashes": {record["request_id"]: record["prompt_sha256"] for record in records},
                "bundle_sha256": str(n), "profile_metadata": {"regime": "natural", "seed": 42, "n_patients": n}}
    return manifest, records


def good_response(disorder, value=0):
    return json.dumps({"scores": {column: value for column in schema.item_columns(disorder)}})


class FakeProvider:
    def __init__(self, *, invalid_first=False, invalid_disorder=None):
        self.calls = []
        self.invalid_first = invalid_first
        self.invalid_disorder = invalid_disorder

    def complete(self, messages, *, seed):
        disorder = messages[-1]["content"]
        self.calls.append((disorder, seed))
        invalid = (self.invalid_first and len(self.calls) == 1) or disorder == self.invalid_disorder
        return {"content": "not JSON" if invalid else good_response(disorder),
                "usage": {"prompt_tokens": 5, "completion_tokens": 20}, "response_model": "Qwen/test",
                "finish_reason": "stop"}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run_dir = Path(self.directory.name) / "run"
        self.fixture = bundle()
        self.loader = patch("cocast.generation.load_bundle", return_value=self.fixture).start()
        self.context = patch("cocast.generation.check_context", return_value={"max_input_tokens": 5}).start()
        self.addCleanup(patch.stopall)

    def generate(self, provider=None, **kwargs):
        with patch("cocast.generation.get_provider", return_value=provider or FakeProvider()) as factory:
            result = generation.generate("unused", self.run_dir, kwargs.pop("settings", SETTINGS), progress_every=0, **kwargs)
        return result, factory

    def test_only_three_files_and_all_pairs_required(self):
        result, _ = self.generate()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_requests"], 7)
        self.assertEqual({p.name for p in self.run_dir.iterdir()}, {"manifest.json", "responses.jsonl", "dataset.csv"})
        self.assertEqual(result["dataset_sha256"], hashlib.sha256((self.run_dir / "dataset.csv").read_bytes()).hexdigest())
        self.assertEqual(result["journal_sha256"], hashlib.sha256((self.run_dir / "responses.jsonl").read_bytes()).hexdigest())
        with (self.run_dir / "dataset.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertTrue(all(rows[0][column] == "0" for column in schema.ALL_ITEM_COLUMNS))

    def test_bounded_additional_attempts_preserve_successful_responses(self):
        failed, _ = self.generate(FakeProvider(invalid_disorder='depression'))
        self.assertEqual(failed['status'], 'failed')
        provider = FakeProvider()
        complete, _ = self.generate(provider, resume=True, extra_attempts=1)
        self.assertEqual(complete['status'], 'complete')
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(complete['n_attempts'], failed['n_attempts'] + 1)
        self.assertEqual(len(complete['attempt_extensions']), 1)
        with patch('cocast.generation.get_provider', side_effect=AssertionError('completed')):
            generation.generate('unused', self.run_dir, SETTINGS, resume=True)

    def test_csv_rebuild_needs_no_provider_or_tokenizer(self):
        self.generate()
        expected = (self.run_dir / "dataset.csv").read_bytes()
        (self.run_dir / "dataset.csv").unlink()
        self.context.reset_mock()
        result, factory = self.generate(resume=True)
        self.assertEqual(result["status"], "complete")
        factory.assert_not_called()
        self.context.assert_not_called()
        self.assertEqual((self.run_dir / "dataset.csv").read_bytes(), expected)

    def test_crash_after_durable_responses_rebuilds_without_provider(self):
        with patch("cocast.generation._export", side_effect=OSError("simulated crash before CSV")):
            with self.assertRaisesRegex(OSError, "simulated crash"):
                self.generate()
        self.assertFalse((self.run_dir / "dataset.csv").exists())
        self.assertEqual(len((self.run_dir / "responses.jsonl").read_text().splitlines()), 7)
        result, factory = self.generate(resume=True)
        self.assertEqual(result["status"], "complete")
        factory.assert_not_called()

    def test_torn_last_append_is_removed_and_recorded(self):
        self.generate()
        manifest_path = self.run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest))
        tail = b'{"request_id":"2:depre'
        with (self.run_dir / "responses.jsonl").open("ab") as handle:
            handle.write(tail)
        result, factory = self.generate(resume=True)
        self.assertEqual(result["recovered_tail_bytes"], len(tail))
        self.assertEqual(len((self.run_dir / "responses.jsonl").read_text().splitlines()), 7)
        factory.assert_not_called()

    def test_complete_corrupt_journal_row_is_not_discarded(self):
        self.generate()
        with (self.run_dir / "responses.jsonl").open("ab") as handle:
            handle.write(b"broken\n")
        with patch("cocast.generation.get_provider") as factory:
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                generation.generate("unused", self.run_dir, SETTINGS, resume=True)
        factory.assert_not_called()

    def test_valid_but_altered_completed_journal_is_rejected(self):
        self.generate()
        journal = self.run_dir / "responses.jsonl"
        entries = [json.loads(line) for line in journal.read_text().splitlines()]
        entries[0]["raw_completion"] += " "
        journal.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            self.generate(resume=True)

    def test_strict_invalid_fraction_strings_bools_and_extra_keys(self):
        disorder = "depression"
        target = self.fixture[1][0]["target"]
        for value in (2.1, "2", True, -0.2, 4, None, float("nan"), float("inf")):
            with self.subTest(value=value):
                _, validation = generation.validate_response(good_response(disorder, value), disorder, target)
                self.assertFalse(validation["valid"])
        original = json.loads(good_response(disorder))
        for mutation in (lambda x: x.update(extra=1), lambda x: x["scores"].update(extra=1),
                         lambda x: x["scores"].pop(schema.item_columns(disorder)[0])):
            changed = copy.deepcopy(original)
            mutation(changed)
            _, validation = generation.validate_response(json.dumps(changed), disorder, target)
            self.assertFalse(validation["valid"])
        duplicate = good_response(disorder).replace('"scores":', '"scores":{}, "scores":', 1)
        self.assertFalse(generation.validate_response(duplicate, disorder, target)[1]["valid"])

    def test_integral_float_is_safe_and_target_deviation_is_logged(self):
        parsed, validation = generation.validate_response(good_response("depression", 2.0), "depression", self.fixture[1][0]["target"])
        self.assertTrue(validation["valid"])
        self.assertTrue(all(type(value) is int for value in parsed["scores"].values()))
        self.assertEqual(validation["achieved_score"], 18)
        self.assertEqual(validation["absolute_score_deviation"], 18)
        self.assertFalse(validation["tier_match"])

    def test_retries_are_journaled_with_distinct_deterministic_seeds(self):
        provider = FakeProvider(invalid_first=True)
        result, _ = self.generate(provider)
        self.assertEqual(result["status"], "complete")
        entries = [json.loads(line) for line in (self.run_dir / "responses.jsonl").read_text().splitlines()]
        self.assertEqual(len(entries), 8)
        self.assertFalse(entries[0]["validation"]["valid"])
        self.assertEqual(entries[0]["raw_completion"], "not JSON")
        self.assertEqual(entries[1]["attempt"], 2)
        self.assertNotEqual(entries[0]["seed"], entries[1]["seed"])

    def test_exhausted_invalid_domain_cannot_mark_run_complete(self):
        result, _ = self.generate(FakeProvider(invalid_disorder="depression"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["completed_requests"], 6)
        with (self.run_dir / "dataset.csv").open() as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row[schema.item_columns("depression")[0]], "")
        result, factory = self.generate(resume=True)
        factory.assert_not_called()
        self.assertEqual(result["status"], "failed")

    def test_scaling_extension_preserves_existing_prompts(self):
        self.generate()
        self.loader.return_value = bundle(2)
        provider = FakeProvider()
        result, _ = self.generate(provider, resume=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["n_patients"], 2)
        self.assertEqual(len(provider.calls), 7)
        self.assertEqual(result["completed_requests"], 14)

    def test_changed_prefix_revision_sources_and_metadata_rejected_before_provider(self):
        self.generate()
        for kind in ("prompt", "revision", "context", "source", "metadata", "order"):
            with self.subTest(kind=kind):
                manifest, records = copy.deepcopy(bundle(2))
                if kind == "prompt":
                    manifest["request_hashes"][records[0]["request_id"]] = "changed"
                elif kind in {"revision", "context"}:
                    manifest[kind] = "changed"
                elif kind == "source":
                    manifest["source_hashes"]["questionnaire"] = "changed"
                elif kind == "metadata":
                    manifest["profile_metadata"]["regime"] = "targeted"
                else:
                    records[:2] = reversed(records[:2])
                self.loader.return_value = manifest, records
                with patch("cocast.generation.get_provider") as factory:
                    with self.assertRaises(ValueError):
                        generation.generate("unused", self.run_dir, SETTINGS, resume=True)
                factory.assert_not_called()

    def test_changed_settings_rejected_before_provider(self):
        self.generate()
        for field, value in (("temperature", 0.5), ("seed", 43), ("precision", "4bit"), ("model_revision", "b" * 40)):
            with self.subTest(field=field), patch("cocast.generation.get_provider") as factory:
                with self.assertRaisesRegex(ValueError, "settings differ"):
                    generation.generate("unused", self.run_dir, {**SETTINGS, field: value}, resume=True)
                factory.assert_not_called()

    def test_context_failure_prevents_provider_initialization(self):
        self.context.side_effect = ValueError("context overflow")
        with patch("cocast.generation.get_provider") as factory:
            with self.assertRaisesRegex(ValueError, "context overflow"):
                generation.generate("unused", self.run_dir, SETTINGS)
        factory.assert_not_called()

    def test_concurrent_attempts_produce_a_valid_journal(self):
        self.loader.return_value = bundle(3)
        result, _ = self.generate(settings={**SETTINGS, "workers": 3})
        self.assertEqual(result["completed_requests"], 21)
        _, factory = self.generate(settings={**SETTINGS, "workers": 3}, resume=True)
        factory.assert_not_called()

    def test_permanent_provider_error_stops_new_requests(self):
        provider = Mock()
        provider.complete.side_effect = ProviderError("Unknown model", retryable=False, status=404)
        result, _ = self.generate(provider)
        self.assertEqual(provider.complete.call_count, 1)
        self.assertEqual(result["status"], "failed")


class ProviderTests(unittest.TestCase):
    def test_context_counts_tokens_rather_than_encoding_dictionary_keys(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = {"input_ids": list(range(7000)), "attention_mask": [1] * 7000}
        auto_tokenizer = Mock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        module = types.SimpleNamespace(AutoTokenizer=auto_tokenizer)
        with patch.dict("sys.modules", {"transformers": module}):
            with self.assertRaisesRegex(ValueError, "7000 input tokens"):
                generation.check_context(bundle()[1], generation.resolve_settings(SETTINGS))
        self.assertTrue(auto_tokenizer.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["return_dict"])

    def test_context_accepts_complete_one_dimensional_token_lists(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = list(range(1500))
        auto_tokenizer = Mock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        with patch.dict("sys.modules", {"transformers": types.SimpleNamespace(AutoTokenizer=auto_tokenizer)}):
            checked = generation.check_context(bundle()[1], generation.resolve_settings(SETTINGS))
        self.assertEqual(checked["max_input_tokens"], 1500)
        self.assertEqual(checked["n_requests_checked"], 7)

    def test_vllm_backends_share_request_flags_and_metadata(self):
        body = json.dumps({"id": "test-id", "model": "test-model", "choices": [{"message": {"content": "{}"},
                           "finish_reason": "stop"}], "usage": {"prompt_tokens": 3}}).encode()
        for backend in ("vllm", "vllm"):
            with self.subTest(backend=backend):
                response = Mock()
                response.__enter__ = Mock(return_value=response)
                response.__exit__ = Mock(return_value=False)
                response.read.return_value = body
                response.headers = {"Server": "test-server"}
                settings = generation.resolve_settings({**SETTINGS, "backend": backend})
                with patch("cocast.providers.urlopen", return_value=response) as http:
                    result = OpenAICompatibleProvider(settings).complete([{"role": "user", "content": "test"}], seed=13)
                request = http.call_args.args[0]
                payload = json.loads(request.data)
                self.assertEqual(request.full_url, "http://127.0.0.1:8087/v1/chat/completions")
                self.assertEqual(payload["response_format"], {"type": "json_object"})
                self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                self.assertEqual(payload["seed"], 13)
                self.assertEqual(payload["top_p"], 1.0)
                self.assertEqual(payload["top_k"], -1)
                self.assertEqual(payload["min_p"], 0.0)
                self.assertEqual(payload["model"], SETTINGS["model"])
                self.assertEqual(settings["model"], SETTINGS["model"])
                self.assertNotIn("truncate_prompt_tokens", payload)
                self.assertEqual(result["response_model"], "test-model")
                self.assertEqual(result["server_header"], "test-server")
                self.assertEqual(http.call_count, 1)

    def test_sampling_controls_are_validated(self):
        settings = generation.resolve_settings(SETTINGS)
        self.assertEqual(settings["top_k"], -1)
        self.assertNotIn("api_model", settings)
        self.assertEqual(generation.resolve_settings({**SETTINGS, "top_k": 20})["top_k"], 20)
        for override in ({"top_p": 0}, {"top_p": 1.1}, {"min_p": -0.1}, {"min_p": 1.1},
                         {"top_k": 0}, {"top_k": -2}, {"top_k": 1.5}, {"top_k": True},
                         {"api_model": "unused_alias"}, {"backend": "unsupported"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                generation.resolve_settings({**SETTINGS, **override})

    def test_unset_revision_and_thinking_are_rejected(self):
        for override in ({"model_revision": "UNSET"}, {"model_revision": "main"}, {"model_revision": "latest"},
                         {"chat_template_kwargs": {"enable_thinking": True}},
                         {"chat_template_kwargs": {"enable_thinking": 0}}, {"response_format_json": "true"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                generation.resolve_settings({**SETTINGS, **override})

    def test_json_mode_can_be_disabled_explicitly_for_server_compatibility(self):
        body = json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}).encode()
        for backend in ("vllm", "vllm"):
            with self.subTest(backend=backend):
                response = Mock()
                response.__enter__ = Mock(return_value=response)
                response.__exit__ = Mock(return_value=False)
                response.read.return_value = body
                response.headers = {}
                settings = generation.resolve_settings({**SETTINGS, "backend": backend, "response_format_json": False})
                with patch("cocast.providers.urlopen", return_value=response) as http:
                    OpenAICompatibleProvider(settings).complete([{"role": "user", "content": "test"}], seed=3)
                self.assertNotIn("response_format", json.loads(http.call_args.args[0].data))

    def test_runtime_records_client_package_versions_without_claiming_server_environment(self):
        with patch("cocast.generation.version", side_effect=lambda package: f"fixture-{package}"):
            runtime = generation._runtime()
        self.assertEqual(runtime["scope"], "client_process")
        self.assertNotIn("vllm-metal", runtime["local_packages"])
        self.assertNotIn("mlx", runtime["local_packages"])


if __name__ == "__main__":
    unittest.main()
