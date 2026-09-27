"""Assemble auditable questionnaire prompts.

"""

from __future__ import annotations

import hashlib
from importlib import resources
import json
import math
import os
from pathlib import Path
import re
import tempfile


REVISION = "aggregate_dsm5_v2"
DISORDERS = (
    "depression", "separation_anxiety", "specific_phobia", "social_anxiety",
    "panic", "agoraphobia", "generalized_anxiety",
)
SYSTEM_MESSAGE = (
    "You are a clinical questionnaire completion assistant. "
    "Respond ONLY with a valid JSON object. "
    "Do not include any text outside the JSON."
)
_CRITERIA = dict(zip(DISORDERS, ("ABCDE", "ABCD", "ABCDEFG", "ABCDEFGHIJ", "ABCD", "ABCDEFGHI", "ABCDEF")))
_CHILDREN = {
    "depression": {"A": 9}, "separation_anxiety": {"A": 8},
    "panic": {"A": 13, "B": 2}, "agoraphobia": {"A": 5},
    "generalized_anxiety": {"C": 6},
}
_NOTES = {"depression": ["Footnote 1"], "specific_phobia": ["Specify if"],
          "social_anxiety": ["Specify if"]}
_DEP_RANGES = ((0, 4), (5, 9), (10, 14), (15, 19), (20, 27))
_ANX_RANGES = ((0.0, 0.4), (0.5, 1.4), (1.5, 2.4), (2.5, 3.4), (3.5, 4.0))
_LABELS = ("none", "mild", "moderate", "severe", "extreme")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def record_hash(record):
    """Hash all request content, including targets and demographics."""
    return _hash(_json_bytes({k: v for k, v in record.items() if k != "prompt_sha256"}))


def _read_json(data, label):
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {label}.") from exc


def _resource_bytes(path, name):
    return (Path(path).read_bytes() if path is not None else
            resources.files("cocast").joinpath("resources", name).read_bytes())


def _key(disorder):
    return disorder.replace("_", " ").upper()


def _validate_sources(questionnaire, graph):
    """Reject missing criteria, numbered symptoms, source pages, or items."""
    if not isinstance(questionnaire, dict) or not isinstance(graph, dict):
        raise ValueError("Questionnaire and knowledge graph must be JSON objects.")
    blocks = {}
    for disorder in DISORDERS:
        key = _key(disorder)
        entry = graph.get(key, {})
        criteria = entry.get("dsm5_criteria", {})
        if list(criteria) != list(_CRITERIA[disorder]) or not all(
                isinstance(t, str) and t.strip() for t in criteria.values()):
            raise ValueError(f"Incomplete or reordered DSM-5 criteria for {key}.")
        for criterion, count in _CHILDREN.get(disorder, {}).items():
            numbers = re.findall(r"(?:^|\s)(\d+)\.\s", criteria[criterion])
            if numbers != [str(i) for i in range(1, count + 1)]:
                raise ValueError(f"Incomplete numbered DSM-5 criteria for {key} {criterion}.")
        source = entry.get("source", {})
        required = ("title", "author", "edition", "year", "criteria_printed_pages",
                    "criteria_pdf_pages", "description_printed_pages", "description_pdf_pages")
        if not entry.get("clinical_definition") or not entry.get("full_name") or not all(
                source.get(field) for field in required):
            raise ValueError(f"Missing DSM-5 description or source references for {key}.")
        if not isinstance(entry.get("criteria_notes", []), list) or any(
                not note.get("label") or not note.get("text") for note in entry.get("criteria_notes", [])):
            raise ValueError(f"Invalid DSM-5 criteria notes for {key}.")
        if [note["label"] for note in entry.get("criteria_notes", [])] != _NOTES.get(disorder, []):
            raise ValueError(f"Missing or reordered DSM-5 criteria notes for {key}.")
        count = 9 if disorder == "depression" else 10
        expected = [f"W1_{disorder}_it{i}" for i in range(1, count + 1)]
        candidates = [block for block in questionnaire.values()
                      if [item.get("key") for item in block.get("items", [])] == expected]
        if len(candidates) != 1 or not all(
                isinstance(item.get("label"), str) and item["label"].strip()
                for item in candidates[0]["items"]):
            raise ValueError(f"Missing, duplicated or reordered questionnaire items for {key}.")
        blocks[disorder] = candidates[0]
    if sum(len(block.get("items", [])) for block in questionnaire.values()) != 69:
        raise ValueError("The questionnaire must contain exactly 69 items.")
    return blocks


def _target(profile, disorder):
    key = _key(disorder)
    try:
        tier_code = profile[f"{key}_TIER_CODE"]
        raw_score = profile[f"{key}_SCORE_TARGET"]
        if tier_code not in [f"tier_{i}" for i in range(5)] or isinstance(raw_score, bool):
            raise ValueError
        score = float(raw_score)
        tier = int(tier_code[-1])
        low, high = (_DEP_RANGES if disorder == "depression" else _ANX_RANGES)[tier]
        scale = 1 if disorder == "depression" else 10
        if (not math.isfinite(score) or not low <= score <= high or
                not math.isclose(score * scale, round(score * scale), abs_tol=1e-8)):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Missing or unattainable score/tier target for {key}.") from exc
    return {"score_target": score, "tier_code": tier_code, "allowed_min": low,
            "allowed_max": high, "measurement": "sum" if disorder == "depression" else "average"}


def _demographics(profile):
    if "AGE" not in profile or "SEX" not in profile:
        raise ValueError("Every profile must provide AGE and SEX (AGE may be null).")
    age, sex = profile["AGE"], profile["SEX"]
    if age is not None and (isinstance(age, bool) or not isinstance(age, (int, float))
                            or not math.isfinite(age) or age < 0):
        raise ValueError("AGE must be a finite nonnegative number or null.")
    if isinstance(sex, bool) or sex not in (1, 2):
        raise ValueError("SEX must use the source coding 1 or 2.")
    return {"AGE": age, "SEX": sex}


def _score_text(disorder, target):
    score = format(target["score_target"], ".1f") if disorder != "depression" else str(int(target["score_target"]))
    labels = ("none", "mild", "moderate", "moderately severe", "severe") if disorder == "depression" else _LABELS
    return (f"{_key(disorder)}: target {target['measurement']} {score}, "
            f"{target['tier_code']} ({labels[int(target['tier_code'][-1])]}).")


def _render(patient_id, demographics, disorder, targets, block, entry, context):
    target = targets[disorder]
    count = len(block["items"])
    maximum = 3 if disorder == "depression" else 4
    age = "not reported" if demographics["AGE"] is None else f"{demographics['AGE']:g} years"
    sex = {1: "male", 2: "female"}[demographics["SEX"]]
    lines = [
        "Complete the following structured psychological questionnaire as the described patient.",
        'Return only a JSON object with this schema: {"scores": {"ITEM_KEY": INTEGER, ...}}.',
        "The object must have exactly one top-level key, scores. Do not add other keys or commentary.",
        f"Include exactly the {count} item keys listed below, using integers from 0 to {maximum}.",
        "", "=== PATIENT CONTEXT ===", f"Patient ID: {patient_id}",
        f"Age: {age}. Recorded sex: {sex}.", "",
    ]
    lines.append("=== ASSIGNED SYMPTOM PROFILE ===" if context == "full" else "=== ASSIGNED TARGET ===")
    shown = DISORDERS if context == "full" else (disorder,)
    lines.extend(_score_text(d, targets[d]) for d in shown)
    if context == "full":
        lines.append("Use the other assigned domain scores as context for a coherent symptom profile.")
    lines.extend(["", f"Answer only the {_key(disorder)} questionnaire in this request."])
    lines.extend([
        f"Make the {target['measurement']} of its item responses equal the assigned target "
        f"{target['score_target']:g}. Its severity-tier range is "
        f"{target['allowed_min']:g} to {target['allowed_max']:g}, inclusive.",
        "Keep the item pattern clinically coherent while matching this score.",
    ])
    lines.extend(["", "=== DSM-5 CONTEXT ===", entry["full_name"]])
    source = entry["source"]
    pages = lambda field: ", ".join(map(str, source[field]))
    lines.extend([
        f"Source: {source['author']}, {source['title']}, {source['edition']}, {source['year']}.",
        f"Description: printed pp. {pages('description_printed_pages')} "
        f"(PDF pp. {pages('description_pdf_pages')}).",
        entry["clinical_definition"],
        f"Diagnostic criteria: printed pp. {pages('criteria_printed_pages')} "
        f"(PDF pp. {pages('criteria_pdf_pages')}).",
    ])
    lines.extend(f"{identifier}. {text}" for identifier, text in entry["dsm5_criteria"].items())
    lines.extend(f"{note['label']}: {note['text']}" for note in entry.get("criteria_notes", []))
    lines.extend([
        "", "Use this DSM-5 context to answer the severity-measure questions according to the assigned score.",
        "", "=== QUESTIONNAIRE ITEMS ===",
    ])
    # Preserve every questionnaire item key and label verbatim.
    lines.extend(f"{item['key']}: {item['label']}" for item in block["items"])
    record = {"request_id": f"{patient_id}:{disorder}", "patient_id": patient_id,
              "disorder": disorder, "profile": demographics, "target": target,
              "messages": [{"role": "system", "content": SYSTEM_MESSAGE},
                           {"role": "user", "content": "\n".join(lines)}]}
    record["prompt_sha256"] = record_hash(record)
    return record


def _atomic_write(path, content):
    """Publish complete individual files, leaving no temporary file on success."""
    fd, temporary = tempfile.mkstemp(prefix=".prepare-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_bundle(profiles_path, output_dir, *, summary_path=None, context="full",
                   questionnaire_path=None, knowledge_graph_path=None):
    """Prepare seven requests per existing patient, without any model access.

    Existing bundles can only grow by appending complete patients with unchanged
    prior prompts and unchanged questionnaire, graph, renderer and summary.
    Conditioned profiles must contain attainable questionnaire scores.
    """
    if context not in ("full", "own"):
        raise ValueError("context must be full or own.")
    inputs = {"profiles": Path(profiles_path).read_bytes(),
              "questionnaire": _resource_bytes(questionnaire_path, "questionnaire.json"),
              "knowledge_graph": _resource_bytes(knowledge_graph_path, "knowledge_graph.json"),
              "renderer": Path(__file__).read_bytes()}
    if summary_path is not None:
        inputs["summary"] = Path(summary_path).read_bytes()
    profiles = _read_json(inputs["profiles"], "profiles")
    metadata = None
    if isinstance(profiles, dict):
        metadata = profiles.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("A profiles envelope must include scenario metadata.")
        profiles = profiles.get("profiles")
    questionnaire = _read_json(inputs["questionnaire"], "questionnaire")
    graph = _read_json(inputs["knowledge_graph"], "knowledge graph")
    blocks = _validate_sources(questionnaire, graph)
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Profiles must be a nonempty JSON list.")
    if metadata is not None:
        _validate_metadata(metadata, len(profiles))
        if "summary" in inputs and metadata["aggregate_sha256"] != _hash(inputs["summary"]):
            raise ValueError("Profile aggregate fingerprint differs from the supplied summary.")
    records, patient_ids = [], set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("Each profile must be a JSON object.")
        patient_id = profile.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id.strip() or ":" in patient_id or patient_id in patient_ids:
            raise ValueError("Every profile needs a unique nonempty patient_id without ':'.")
        patient_ids.add(patient_id)
        demographics = _demographics(profile)
        targets = {d: _target(profile, d) for d in DISORDERS}
        records.extend(_render(patient_id, demographics, d, targets, blocks[d], graph[_key(d)], context)
                       for d in DISORDERS)
    content = b"".join(_json_bytes(record) + b"\n" for record in records)
    manifest = {"revision": REVISION, "context": context, "n_patients": len(profiles),
                "n_requests": len(records), "source_hashes": {k: _hash(v) for k, v in inputs.items()},
                "request_hashes": {r["request_id"]: r["prompt_sha256"] for r in records},
                "bundle_sha256": _hash(content)}
    if metadata is not None:
        manifest["profile_metadata"] = metadata
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        previous, old_records = load_bundle(output)
        if previous["context"] != context:
            raise ValueError("Cannot change the context of an existing bundle.")
        stable_sources = lambda m: {k: v for k, v in m["source_hashes"].items() if k != "profiles"}
        if stable_sources(previous) != stable_sources(manifest):
            raise ValueError("Prompt source changed. Prepare a separate bundle.")
        stable_metadata = lambda m: {k: v for k, v in m.get("profile_metadata", {}).items() if k != "n_patients"}
        if stable_metadata(previous) != stable_metadata(manifest):
            raise ValueError("Profile scenario metadata changed. Prepare a separate bundle.")
        if len(records) < len(old_records) or records[:len(old_records)] != old_records:
            raise ValueError("Cannot shrink a bundle or change prior patient prompts.")
        if len(records) == len(old_records):
            if previous["source_hashes"] != manifest["source_hashes"]:
                raise ValueError("Profiles source changed without a patient-prefix extension.")
            return previous
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write(output / "prompts.jsonl", content)
    _atomic_write(output / "manifest.json", _json_bytes(manifest) + b"\n")
    return manifest


def _validate_metadata(metadata, n_patients):
    if (not isinstance(metadata, dict) or metadata.get("revision") != REVISION
            or type(metadata.get("n_patients")) is not int
            or metadata.get("n_patients") != n_patients
            or metadata.get("regime") not in ("natural", "targeted")
            or type(metadata.get("seed")) is not int):
        raise ValueError("Invalid profile scenario metadata.")
    for field in ("aggregate_sha256", "prevalences_sha256"):
        if field == "prevalences_sha256" and metadata.get("regime") == "natural" and metadata.get(field) is None:
            continue
        if not isinstance(metadata.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", metadata[field]):
            raise ValueError("Invalid profile scenario fingerprint.")


def load_bundle(output_dir):
    """Load and validate saved requests without requiring their original inputs."""
    output = Path(output_dir)
    try:
        manifest = _read_json((output / "manifest.json").read_bytes(), "manifest")
        content = (output / "prompts.jsonl").read_bytes()
    except OSError as exc:
        raise ValueError("Incomplete prompt bundle: manifest.json and prompts.jsonl are required.") from exc
    if not isinstance(manifest, dict) or manifest.get("revision") != REVISION:
        raise ValueError("Incompatible prompt revision.")
    if manifest.get("context") not in ("full", "own"):
        raise ValueError("Invalid prompt context.")
    if _hash(content) != manifest.get("bundle_sha256"):
        raise ValueError("Prompt bundle checksum mismatch.")
    sources = manifest.get("source_hashes", {})
    if not isinstance(sources, dict) or not {"profiles", "questionnaire", "knowledge_graph", "renderer"} <= sources.keys() or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in sources.values()):
        raise ValueError("Missing or invalid source fingerprints.")
    records = [_read_json(line, "prompt record") for line in content.splitlines() if line.strip()]
    if (not records or not all(isinstance(record, dict) for record in records)
            or type(manifest.get("n_patients")) is not int or type(manifest.get("n_requests")) is not int
            or len(records) % 7 or len(records) != manifest.get("n_requests")
            or len(records) // 7 != manifest.get("n_patients")):
        raise ValueError("Prompt bundle patient/request count mismatch.")
    if "profile_metadata" in manifest:
        _validate_metadata(manifest["profile_metadata"], len(records) // 7)
        if ("summary" in sources and
                manifest["profile_metadata"]["aggregate_sha256"] != sources["summary"]):
            raise ValueError("Profile aggregate fingerprint differs from the summary fingerprint.")
    seen, hashes = set(), {}
    for start in range(0, len(records), 7):
        block = records[start:start + 7]
        patient_id = block[0].get("patient_id")
        if not isinstance(patient_id, str) or not patient_id.strip() or ":" in patient_id or patient_id in seen:
            raise ValueError("Invalid or duplicate patient ID in bundle.")
        seen.add(patient_id)
        demographics = _demographics(block[0].get("profile", {}))
        for disorder, record in zip(DISORDERS, block):
            if (record.get("patient_id") != patient_id or record.get("disorder") != disorder or
                    record.get("request_id") != f"{patient_id}:{disorder}" or record.get("profile") != demographics):
                raise ValueError("Incomplete, inconsistent or reordered patient questionnaire block.")
            if record.get("prompt_sha256") != record_hash(record):
                raise ValueError("Prompt record checksum mismatch.")
            messages = record.get("messages", [])
            if (not isinstance(messages, list) or len(messages) != 2 or not isinstance(messages[1], dict)
                    or messages[0] != {"role": "system", "content": SYSTEM_MESSAGE}
                    or messages[1].get("role") != "user" or not isinstance(messages[1].get("content"), str)
                    or not messages[1]["content"].strip()):
                raise ValueError("Missing or malformed assembled prompt messages.")
            marker = "=== ASSIGNED SYMPTOM PROFILE ===" if manifest["context"] == "full" else "=== ASSIGNED TARGET ==="
            if marker not in messages[1]["content"]:
                raise ValueError("Prompt content does not match its context designation.")
            target = record.get("target", {})
            if not isinstance(target, dict):
                raise ValueError("Invalid target object in bundle.")
            validated = _target({_key(disorder) + "_TIER_CODE": target.get("tier_code"),
                                 _key(disorder) + "_SCORE_TARGET": target.get("score_target")}, disorder)
            if target != validated:
                raise ValueError("Invalid target bounds or measurement in bundle.")
            hashes[record["request_id"]] = record["prompt_sha256"]
    if hashes != manifest.get("request_hashes"):
        raise ValueError("Request fingerprint map mismatch.")
    return manifest, records
