"""Synthetic integrity tests; no existing experimental outputs are modified."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_rsi_release_ledger as builder


def put(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sealed_inputs(tmp_path):
    root = tmp_path
    base = root / "outputs/rsi_cycle_001"
    capability, nested, replay = [base / name for name in
                                  ("capability_eval_v1", "nested_analysis_v1", "checkpoint_replay_v1")]
    source_paths = {name: put(root / "synthetic-inputs" / f"{name}.json", {"source": name})
                    for name in ("capability", "nested", "replay")}
    capability_protocol = put(capability / "evaluation_protocol.json", {
        "source_input_files_sha256": {str(source_paths["capability"].relative_to(root)): sha(source_paths["capability"])}
    })
    metrics = put(capability / "case_metrics.json", {"main": {"23": []}})
    nested_summary = put(nested / "nested_summary.json", {"synthetic_result": True})
    replay_summary = put(replay / "summary.json", {"synthetic_result": True})
    seals = {
        "capability": put(capability / "result_seal.json", {
            "format": "phyevo-rsi-capability-seal-v1",
            "files": {p.name: sha(p) for p in (capability_protocol, metrics)}}),
        "nested": put(nested / "result_seal.json", {
            "format": "phyevo-rsi-nested-seal-v1",
            "input_files_sha256": {str(source_paths["nested"].relative_to(root)): sha(source_paths["nested"])},
            "output_files_sha256": {nested_summary.name: sha(nested_summary)}}),
        "replay": put(replay / "seal.json", {
            "format": "phyevo-rsi-replay-seal-v1",
            "source_input_files_sha256": {str(source_paths["replay"].relative_to(root)): sha(source_paths["replay"])},
            "artifacts_sha256": {replay_summary.name: sha(replay_summary)}}),
    }
    return {"root": root, "directories": (capability, nested, replay),
            "sources": source_paths, "seals": seals, "metrics": metrics,
            "results": {"capability": metrics, "nested": nested_summary, "replay": replay_summary}}


def preflight(fixture):
    return builder.verify_input_seals(fixture["root"], *fixture["directories"])


def rewrite_synthetic(path, value):
    # Only task-created synthetic fixtures are intentionally mutated in tests.
    path.write_text(json.dumps(value), encoding="utf-8")


def test_all_three_input_and_output_mappings_are_verified(sealed_inputs):
    result = preflight(sealed_inputs)
    assert result["mapping_file_counts"] == {
        "capability_outputs": 2, "capability_inputs": 1,
        "nested_outputs": 1, "nested_inputs": 1,
        "replay_outputs": 1, "replay_inputs": 1}
    assert result["unique_verified_files"] == 7
    assert result["checked_before_output_creation"]
    assert len(result["seal_files_sha256"]) == 3


@pytest.mark.parametrize("name", ["capability", "nested", "replay"])
def test_changed_result_fails_before_output_directory_exists(sealed_inputs, monkeypatch, name):
    rewrite_synthetic(sealed_inputs["results"][name], {"result_was_changed": True})
    monkeypatch.setattr(builder, "ROOT", sealed_inputs["root"])
    output = sealed_inputs["root"] / "new-release"
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.build(output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["capability", "nested", "replay"])
def test_changed_source_fails_even_when_outputs_are_unchanged(sealed_inputs, monkeypatch, name):
    rewrite_synthetic(sealed_inputs["sources"][name], {"input_was_changed": True})
    monkeypatch.setattr(builder, "ROOT", sealed_inputs["root"])
    output = sealed_inputs["root"] / "new-release"
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.build(output)
    assert not output.exists()


@pytest.mark.parametrize("name,key", [("capability", "files"), ("nested", "output_files_sha256"),
                                      ("replay", "artifacts_sha256")])
def test_missing_recorded_file_is_rejected_without_deleting_fixture(sealed_inputs, name, key):
    path = sealed_inputs["seals"][name]
    seal = json.loads(path.read_text())
    seal[key]["never-created.json"] = "a" * 64
    rewrite_synthetic(path, seal)
    with pytest.raises(ValueError, match="Missing sealed input/output"):
        preflight(sealed_inputs)


def test_missing_seal_is_not_treated_as_completed(tmp_path):
    directories = [tmp_path / name for name in ("capability", "nested", "replay")]
    for directory in directories:
        directory.mkdir()
    with pytest.raises(ValueError, match="seal missing"):
        builder.verify_input_seals(tmp_path, *directories)


@pytest.mark.parametrize("entry", [("../escaped.json", "a" * 64),
                                   ("/absolute/path.json", "a" * 64),
                                   ("file.json", "not-a-sha256")])
def test_malformed_or_escaping_entries_are_rejected(sealed_inputs, entry):
    path = sealed_inputs["seals"]["nested"]
    seal = json.loads(path.read_text())
    seal["input_files_sha256"] = {entry[0]: entry[1]}
    rewrite_synthetic(path, seal)
    with pytest.raises(ValueError, match="Malformed checksum entry"):
        preflight(sealed_inputs)


@pytest.mark.parametrize("replacement", [{}, [], None])
def test_empty_or_missing_input_mapping_fails_closed(sealed_inputs, replacement):
    path = sealed_inputs["seals"]["replay"]
    seal = json.loads(path.read_text())
    seal["source_input_files_sha256"] = replacement
    rewrite_synthetic(path, seal)
    with pytest.raises(ValueError, match="Missing or empty checksum mapping"):
        preflight(sealed_inputs)


def test_required_consumed_result_cannot_be_omitted_from_mapping(sealed_inputs):
    path = sealed_inputs["seals"]["capability"]
    seal = json.loads(path.read_text())
    seal["files"].pop("case_metrics.json")
    rewrite_synthetic(path, seal)
    with pytest.raises(ValueError, match="Required consumed artefact absent"):
        preflight(sealed_inputs)


def test_wrong_seal_schema_is_rejected(sealed_inputs):
    path = sealed_inputs["seals"]["nested"]
    seal = json.loads(path.read_text())
    seal["format"] = "some-other-result"
    rewrite_synthetic(path, seal)
    with pytest.raises(ValueError, match="Wrong experiment seal format"):
        preflight(sealed_inputs)
