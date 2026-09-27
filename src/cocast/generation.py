"""Validated questionnaire generation with a durable, append-only attempt journal.

The journal is authoritative. The CSV is a rebuildable view, including explicit
empty cells for unfinished domains. Only the manifest marks a complete run.
"""

import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import schema
from .prompts import load_bundle
from .providers import ProviderError, get_provider


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic(path, writer):
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_settings(settings):
    """Resolve defaults once so every scientifically relevant setting is saved."""
    resolved = {
        "temperature": 1.0, "max_tokens": 2048, "context_window": 8192,
        "top_p": 1.0, "top_k": -1, "min_p": 0.0,
        "workers": 1, "max_retries": 2, "seed": 42, "timeout": 600.0,
        "chat_template_kwargs": {"enable_thinking": False},
        **settings,
    }
    allowed = {"model", "model_revision", "precision", "backend", "base_url",
               "temperature", "max_tokens", "context_window", "workers", "max_retries",
               "seed", "timeout", "response_format_json", "chat_template_kwargs", "tokenizer_path",
               "top_p", "top_k", "min_p"}
    if set(resolved) - allowed:
        raise ValueError(f"Unknown generation settings: {sorted(set(resolved) - allowed)}")
    for key in ("model", "model_revision", "precision", "backend", "base_url"):
        if not isinstance(resolved.get(key), str) or not resolved[key].strip():
            raise ValueError(f"A nonempty {key} must be recorded explicitly.")
    if resolved["backend"] not in {"vllm"}:
        raise ValueError("backend must be vllm.")
    if re.fullmatch(r"[0-9a-fA-F]{40}", resolved["model_revision"]) is None:
        raise ValueError("model_revision must be an immutable 40-character Hugging Face commit hash.")
    resolved["model_revision"] = resolved["model_revision"].lower()
    template = resolved["chat_template_kwargs"]
    if not isinstance(template, dict) or set(template) != {"enable_thinking"} or template["enable_thinking"] is not False:
        raise ValueError("This protocol requires chat_template_kwargs={enable_thinking: false}.")
    if "tokenizer_path" in resolved and not isinstance(resolved["tokenizer_path"], str):
        raise ValueError("tokenizer_path must be a local path or cached model identifier.")
    resolved.setdefault("response_format_json", True)
    if not isinstance(resolved["response_format_json"], bool):
        raise ValueError("response_format_json must be a boolean.")
    for key in ("max_tokens", "context_window", "workers", "max_retries", "seed"):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key in {"max_retries", "seed"} else 1):
            raise ValueError(f"{key} must be a {'nonnegative' if key in {'max_retries', 'seed'} else 'positive'} integer.")
    if (isinstance(resolved["top_k"], bool) or not isinstance(resolved["top_k"], int)
            or (resolved["top_k"] != -1 and resolved["top_k"] < 1)):
        raise ValueError("top_k must be -1 (disabled) or a positive integer.")
    for key in ("temperature", "timeout", "top_p", "min_p"):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
    if resolved["temperature"] < 0 or resolved["timeout"] <= 0:
        raise ValueError("temperature must be nonnegative and timeout positive.")
    if not 0 < resolved["top_p"] <= 1 or not 0 <= resolved["min_p"] <= 1:
        raise ValueError("top_p must be in (0, 1] and min_p in [0, 1].")
    if resolved["max_tokens"] >= resolved["context_window"]:
        raise ValueError("context_window must leave space for the prompt in addition to max_tokens.")
    return resolved


def check_context(records, settings):
    """Count every complete chat prompt with the cached model tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the transformers tokenizer dependency before generation.") from exc
    tokenizer_source = settings.get("tokenizer_path", settings["model"])
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, revision=settings["model_revision"], local_files_only=True,
            trust_remote_code=False,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError("The matching tokenizer must already be downloaded. Set tokenizer_path if needed.") from exc
    counts = []
    for record in records:
        encoded = tokenizer.apply_chat_template(
            record["messages"], tokenize=True, add_generation_prompt=True, return_dict=False,
            **settings["chat_template_kwargs"],
        )
        # Transformers versions differ in their default chat-template return type.
        tokens = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if tokens and isinstance(tokens[0], (list, tuple)):
            if len(tokens) != 1:
                raise ValueError("Expected one tokenized conversation per request.")
            tokens = tokens[0]
        counts.append(len(tokens))
    maximum = max(counts, default=0)
    if maximum + settings["max_tokens"] > settings["context_window"]:
        raise ValueError(f"A complete prompt needs {maximum} input tokens plus {settings['max_tokens']} output tokens, "
                         f"exceeding context_window={settings['context_window']}. Criteria will not be truncated.")
    return {"tokenizer_source": tokenizer_source, "tokenizer_revision": settings["model_revision"],
            "max_input_tokens": maximum, "max_tokens": settings["max_tokens"],
            "context_window": settings["context_window"], "n_requests_checked": len(records)}


def _runtime():
    packages = {}
    for package in ("cocast", "transformers", "vllm", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    return {"scope": "client_process", "python": platform.python_version(),
            "platform": platform.platform(), "local_packages": packages}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite numbers are not valid responses.")
    return value


def validate_response(raw, disorder, target):
    """Reject malformed scores without coercing strings, fractions, or extra keys."""
    parsed = None
    errors = []
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object, parse_float=_finite_float,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON number: {value}")))
        if not isinstance(parsed, dict) or set(parsed) != {"scores"}:
            raise ValueError("The JSON object must contain exactly one key, scores.")
        scores = parsed["scores"]
        columns = schema.item_columns(disorder)
        if not isinstance(scores, dict) or set(scores) != set(columns):
            raise ValueError("scores must contain exactly the required questionnaire item keys.")
        spec = schema.DISORDERS[disorder]
        for column in columns:
            value = scores[column]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value):
                raise ValueError(f"{column} must be an integral JSON number.")
            if not spec["item_min"] <= value <= spec["item_max"]:
                raise ValueError(f"{column} is outside the allowed item range.")
        parsed = {"scores": {column: int(scores[column]) for column in columns}}
    except (ValueError, TypeError, OverflowError) as exc:
        errors.append(str(exc))
    validation = {"valid": not errors, "errors": errors}
    if not errors:
        total = sum(parsed["scores"].values())
        score = total if spec["scoring"] == "sum" else total / len(columns)
        tier = int(schema.tier_of(score, disorder))
        validation.update(achieved_score=score, achieved_tier=tier)
        if target is not None:
            deviation = score - float(target["score_target"])
            validation.update(score_deviation=deviation, absolute_score_deviation=abs(deviation),
                              tier_match=schema.TIER_LABELS[tier] == target["tier_code"])
    return parsed, validation


def _seed(seed, request_id, attempt):
    digest = hashlib.sha256(f"{seed}:{request_id}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _attempt(provider, record, attempt, settings):
    started = time.perf_counter()
    seed = _seed(settings["seed"], record["request_id"], attempt)
    entry = {"request_id": record["request_id"], "prompt_sha256": record["prompt_sha256"],
             "attempt": attempt, "seed": seed, "raw_completion": None, "parsed_response": None}
    try:
        completion = dict(provider.complete(record["messages"], seed=seed))
        entry["raw_completion"] = completion.pop("content")
        entry["provider"] = completion
        entry["parsed_response"], entry["validation"] = validate_response(
            entry["raw_completion"], record["disorder"], record["target"])
        entry["retryable"] = True
    except ProviderError as exc:
        entry.update(validation={"valid": False, "errors": [str(exc)]},
                     retryable=exc.retryable, provider={"http_status": exc.status})
    entry.update(completed_at=_now(), elapsed_s=round(time.perf_counter() - started, 6))
    return entry


def _check_bundle(bundle, records):
    if len(records) != bundle["n_requests"]:
        raise ValueError("Bundle request count does not match records.")
    ids = [record["request_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Bundle contains duplicate request IDs.")
    patients = {}
    for record in records:
        patient = str(record["patient_id"])
        if record["request_id"] != f"{patient}:{record['disorder']}":
            raise ValueError("Request ID must be patient_id:disorder.")
        if record["disorder"] not in schema.DISORDERS:
            raise ValueError("Unknown questionnaire disorder.")
        domains, profile = patients.setdefault(patient, (set(), record["profile"]))
        if profile != record["profile"]:
            raise ValueError("Patient demographics differ between disorder requests.")
        domains.add(record["disorder"])
    if len(patients) != bundle["n_patients"] or any(domains != set(schema.DISORDERS) for domains, _ in patients.values()):
        raise ValueError("A bundle must contain every disorder exactly once for every patient.")


def _check_resume(previous, bundle, records, settings):
    if previous.get("settings_sha256") != _hash(previous["settings"]):
        raise ValueError("Stored settings fingerprint is inconsistent.")
    if previous["settings"] != settings:
        raise ValueError("Generation settings differ from this run. Use a separate run directory.")
    for key in ("revision", "context"):
        if previous[key] != bundle[key]:
            raise ValueError(f"Bundle {key} differs from this run.")
    old_hashes, new_hashes = previous["request_hashes"], bundle["request_hashes"]
    order = [record["request_id"] for record in records]
    if (order[:len(previous["request_order"])] != previous["request_order"]
            or any(new_hashes.get(key) != value for key, value in old_hashes.items())):
        raise ValueError("Existing patient prompts changed or were removed. Scaling requires an identical prefix.")
    if bundle["n_requests"] < previous["n_requests"] or bundle["n_patients"] < previous["n_patients"]:
        raise ValueError("A run cannot be resumed with a smaller bundle.")
    for key in ("source_hashes", "profile_metadata"):
        old, new = dict(previous.get(key, {})), dict(bundle.get(key, {}))
        if key == "source_hashes":
            old.pop("profiles", None)
            new.pop("profiles", None)
        else:
            old.pop("n_patients", None)
            new.pop("n_patients", None)
        if old != new:
            raise ValueError(f"Bundle {key} differs from this run.")


def _read_journal(handle, records, settings):
    """Discard only a torn final append. Never ignore a corrupted complete row."""
    index = {record["request_id"]: record for record in records}
    successes, attempts, entries = {}, {}, []
    handle.seek(0)
    content = handle.read()
    offset = 0
    recovered = 0
    for line in content.splitlines(keepends=True):
        try:
            entry = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if offset + len(line) == len(content) and not line.endswith(b"\n"):
                recovered = len(line)
                handle.seek(offset)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                break
            raise ValueError("The response journal contains a corrupted complete row.")
        request_id = entry["request_id"]
        if request_id not in index or entry["prompt_sha256"] != index[request_id]["prompt_sha256"]:
            raise ValueError("Journal request does not match the selected bundle.")
        if entry["attempt"] != attempts.get(request_id, 0) + 1 or request_id in successes:
            raise ValueError("Journal attempt ordering is inconsistent.")
        if entry["seed"] != _seed(settings["seed"], request_id, entry["attempt"]):
            raise ValueError("Journal sampling seed does not match this run.")
        attempts[request_id] = entry["attempt"]
        if entry["raw_completion"] is not None:
            parsed, validation = validate_response(entry["raw_completion"], index[request_id]["disorder"], index[request_id]["target"])
            if parsed != entry["parsed_response"] or validation != entry["validation"]:
                raise ValueError("Journal parsed scores or validation disagree with its raw response.")
        elif entry["validation"]["valid"]:
            raise ValueError("Journal marks a missing raw response as valid.")
        if entry["validation"]["valid"]:
            successes[request_id] = entry
        entries.append(entry)
        offset += len(line)
    handle.seek(0, os.SEEK_END)
    if not recovered and content and not content.endswith(b"\n"):
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return successes, attempts, entries, recovered


def _export(path, records, successes):
    rows = {}
    for record in records:
        row = rows.setdefault(str(record["patient_id"]), {
            schema.ID_COL: record["patient_id"], schema.AGE_COL: record["profile"]["AGE"],
            schema.SEX_COL: record["profile"]["SEX"],
        })
        if record["request_id"] in successes:
            row.update(successes[record["request_id"]]["parsed_response"]["scores"])
    columns = [schema.ID_COL, schema.AGE_COL, schema.SEX_COL] + list(schema.ALL_ITEM_COLUMNS)
    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows.values())
    _atomic(path, write)


def generate(bundle_dir, run_dir, settings, *, resume=False, progress_every=25, extra_attempts=0):
    """Generate or resume a run. Completed journal entries never call a provider."""
    if type(extra_attempts) is not int or extra_attempts < 0 or (extra_attempts and not resume):
        raise ValueError('Additional attempts require an explicit resume and a nonnegative integer allowance.')
    settings = resolve_settings(settings)
    bundle, records = load_bundle(bundle_dir)
    _check_bundle(bundle, records)
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and not resume:
        raise FileExistsError("Run already exists. Pass resume=True to validate and resume it.")
    if previous:
        _check_resume(previous, bundle, records, settings)
    elif (run_dir / "responses.jsonl").exists() and (run_dir / "responses.jsonl").stat().st_size:
        raise ValueError("A nonempty journal has no manifest. Refusing to overwrite it.")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {**(previous or {}), **{key: bundle[key] for key in (
        "revision", "n_patients", "n_requests", "context", "source_hashes", "request_hashes", "bundle_sha256")},
        "request_order": [record["request_id"] for record in records],
        "profile_metadata": bundle.get("profile_metadata", {}), "settings": settings,
        "settings_sha256": _hash(settings), "created_at": (previous or {}).get("created_at", _now()),
        "status": "running", "runtime": (previous or {}).get("runtime", _runtime())}
    started = time.perf_counter()
    with (run_dir / "responses.jsonl").open("a+b") as journal:
        try:
            fcntl.flock(journal.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is already writing this run.") from exc
        # A process could have completed between the first manifest read and lock acquisition.
        current = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if current != previous:
            raise RuntimeError("This run changed while acquiring its journal lock. Retry with resume=True.")
        if previous and previous["status"] == "complete" and _file_hash(run_dir / "responses.jsonl") != previous.get("journal_sha256"):
            raise ValueError("The completed response journal fingerprint changed. Refusing to reuse altered results.")
        successes, attempts, entries, recovered = _read_journal(journal, records, settings)
        if len(successes) < len(records):
            manifest.pop("completed_at", None)
        manifest["recovered_tail_bytes"] = (previous or {}).get("recovered_tail_bytes", 0) + recovered
        manifest["updated_at"] = _now()
        _atomic(manifest_path, lambda handle: handle.write(_json(manifest) + "\n"))
        latest = {entry["request_id"]: entry for entry in entries}
        limits = manifest.setdefault('additional_attempt_limits', {})
        if any(k not in {r['request_id'] for r in records} or type(v) is not int or v < 1 for k, v in limits.items()):
            raise ValueError('Invalid additional-attempt limits in manifest.')
        if extra_attempts:
            changed = {}
            for record in records:
                key = record['request_id']
                if key not in successes and latest.get(key, {}).get('retryable', True):
                    limits[key] = max(limits.get(key, settings['max_retries'] + 1), attempts.get(key, 0)) + extra_attempts
                    changed[key] = limits[key]
            if changed:
                manifest.setdefault('attempt_extensions', []).append({'at': _now(), 'additional_attempts': extra_attempts, 'limits': changed})
                _atomic(manifest_path, lambda handle: handle.write(_json(manifest) + "\n"))
        remaining = [record for record in records if record["request_id"] not in successes
                     and attempts.get(record["request_id"], 0) < limits.get(record['request_id'], settings['max_retries'] + 1)
                     and latest.get(record["request_id"], {}).get("retryable", True)]
        interrupted = False
        try:
            if remaining:
                history = manifest.setdefault("runtime_history", [manifest["runtime"]])
                if _runtime() not in history:
                    history.append(_runtime())
                manifest["context_check"] = check_context(records, settings)
                _atomic(manifest_path, lambda handle: handle.write(_json(manifest) + "\n"))
                provider = get_provider(settings)
                queue = iter(remaining)
                with ThreadPoolExecutor(max_workers=settings["workers"]) as executor:
                    futures = {}
                    def submit(record):
                        number = attempts.get(record["request_id"], 0) + 1
                        futures[executor.submit(_attempt, provider, record, number, settings)] = record
                    for _ in range(min(settings["workers"], len(remaining))):
                        submit(next(queue))
                    while futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            record = futures.pop(future)
                            entry = future.result()
                            journal.write((_json(entry) + "\n").encode())
                            journal.flush()
                            os.fsync(journal.fileno())
                            entries.append(entry)
                            attempts[record["request_id"]] = entry["attempt"]
                            if entry["validation"]["valid"]:
                                successes[record["request_id"]] = entry
                            elif entry["retryable"] and entry["attempt"] < limits.get(record["request_id"], settings["max_retries"] + 1):
                                submit(record)
                                continue
                            elif not entry["retryable"]:
                                # Authentication/model/configuration errors should not hit every patient.
                                queue = iter(())
                            if progress_every and (len(successes) % progress_every == 0 or len(successes) == len(records)):
                                print(f"Valid responses: {len(successes)}/{len(records)}. Attempts: {len(entries)}.", flush=True)
                            next_record = next(queue, None)
                            if next_record is not None:
                                submit(next_record)
        except BaseException:
            interrupted = True
            raise
        finally:
            _export(run_dir / "dataset.csv", records, successes)
            manifest.update(
                status="interrupted" if interrupted else ("complete" if len(successes) == len(records) else "failed"),
                completed_requests=len(successes), missing_requests=len(records) - len(successes),
                n_attempts=len(entries), last_invocation_seconds=round(time.perf_counter() - started, 6),
                journal_request_seconds=round(sum(entry["elapsed_s"] for entry in entries), 6), updated_at=_now(),
                dataset_sha256=_file_hash(run_dir / "dataset.csv"),
                journal_sha256=_file_hash(run_dir / "responses.jsonl"),
                server_observations={key: sorted({str(entry.get("provider", {}).get(key)) for entry in entries
                                                if entry.get("provider", {}).get(key) is not None})
                                     for key in ("response_model", "system_fingerprint", "server_header")},
            )
            if manifest["status"] == "complete":
                manifest["completed_at"] = manifest.get("completed_at", _now())
            else:
                manifest.pop("completed_at", None)
            _atomic(manifest_path, lambda handle: handle.write(_json(manifest) + "\n"))
    return manifest
