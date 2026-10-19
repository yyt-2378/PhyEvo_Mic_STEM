#!/usr/bin/env python3
"""Record real development capability proposals and honest production refusals.

Each card describes one fixed seed's actual held-out result, not an ensemble
whose averaged metric is misleadingly attached to a single checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re

import numpy as np

from tfree_detection.expert_supervision import contained_path, sha256_file, write_new_json
from tfree_detection.rsi_policy import (ArtifactRef, CapabilityCard, CapabilityLedger, CapabilityScope,
                                      FeatureRange, PairedMetric, content_hash, evidence_contract,
                                      evaluate_capability, verify_artifacts)

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def verify_input_seals(project_root, capability, nested, replay):
    """Verify every recorded input/output byte before creating a new ledger.

    The three producers use different seal schemas. Capability source hashes
    live in its sealed evaluation protocol; nested/replay source hashes live in
    their seals. Merely finding these JSON files is not an integrity check.
    These are checksum receipts, not protection against hostile rewriting of
    an entire receipt chain. Existing release ledgers remain untouched.
    """
    root = Path(project_root).resolve(strict=True)
    capability, nested, replay = [contained_path(root, p) for p in (capability, nested, replay)]
    verified = {}
    mapping_counts = {}
    seal_hashes = {}

    def verify_mapping(mapping, base, label, required=()):
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Missing or empty checksum mapping: {label}")
        if not set(required).issubset(mapping):
            raise ValueError(f"Required consumed artefact absent from checksum mapping: {label}")
        for name, expected in mapping.items():
            if (not isinstance(name, str) or not name or Path(name).is_absolute()
                    or ".." in Path(name).parts or not isinstance(expected, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected)):
                raise ValueError(f"Malformed checksum entry: {label}: {name}")
            target = contained_path(base, name)
            if not target.is_file():
                raise ValueError(f"Missing sealed input/output: {label}: {name}")
            actual = sha256_file(target)
            if actual != expected:
                raise ValueError(f"Sealed input/output checksum mismatch: {label}: {name}")
            relative = str(target.relative_to(root))
            if relative in verified and verified[relative] != actual:
                raise ValueError(f"Input changed between checksum mappings: {relative}")
            verified[relative] = actual
        mapping_counts[label] = len(mapping)

    def load_seal(directory, filename, format_name):
        path = contained_path(root, directory / filename)
        if not path.is_file():
            raise ValueError(f"Required experiment seal missing: {path.relative_to(root)}")
        value = read(path)
        if not isinstance(value, dict) or value.get("format") != format_name:
            raise ValueError(f"Wrong experiment seal format: {path.relative_to(root)}")
        seal_hashes[str(path.relative_to(root))] = sha256_file(path)
        return value

    capability_seal = load_seal(capability, "result_seal.json", "phyevo-rsi-capability-seal-v1")
    verify_mapping(capability_seal.get("files"), capability, "capability_outputs",
                   ("evaluation_protocol.json", "case_metrics.json"))
    capability_protocol = read(capability / "evaluation_protocol.json")
    verify_mapping(capability_protocol.get("source_input_files_sha256"), root, "capability_inputs")
    nested_seal = load_seal(nested, "result_seal.json", "phyevo-rsi-nested-seal-v1")
    verify_mapping(nested_seal.get("output_files_sha256"), nested, "nested_outputs",
                   ("nested_summary.json",))
    verify_mapping(nested_seal.get("input_files_sha256"), root, "nested_inputs")
    replay_seal = load_seal(replay, "seal.json", "phyevo-rsi-replay-seal-v1")
    verify_mapping(replay_seal.get("artifacts_sha256"), replay, "replay_outputs", ("summary.json",))
    verify_mapping(replay_seal.get("source_input_files_sha256"), root, "replay_inputs")
    return {"format": "phyevo-rsi-release-input-preflight-v1",
            "mapping_file_counts": mapping_counts, "unique_verified_files": len(verified),
            "seal_files_sha256": seal_hashes, "verified_files_sha256": verified,
            "checked_before_output_creation": True}


def build(output):
    output = contained_path(ROOT, output)
    if output.exists():
        raise FileExistsError("Refusing to replace a release ledger")
    base = ROOT / "outputs/rsi_cycle_001"
    capability = base / "capability_eval_v1"
    nested = base / "nested_analysis_v1"
    replay = base / "checkpoint_replay_v1"
    preflight = verify_input_seals(ROOT, capability, nested, replay)
    rows = read(capability / "case_metrics.json")["main"]["23"]
    outer_path = ROOT / "outputs/expert_cycle_001/student_gpu_s23_v1/protocol.json"
    if str(outer_path.relative_to(ROOT)) not in preflight["verified_files_sha256"]:
        raise ValueError("Outer training protocol was not covered by the input seals")
    outer = read(outer_path)
    for fold in outer["folds"]:
        for path in (nested / "routers" / f"{fold['fold_id']}.json",
                     outer_path.parent / fold["fold_id"] / "teacher_plus_expert/step-001200/checkpoint.json"):
            if str(path.relative_to(ROOT)) not in preflight["verified_files_sha256"]:
                raise ValueError(f"Consumed router/checkpoint was not covered by input seals: {path.relative_to(ROOT)}")
    protocol = read(capability / "evaluation_protocol.json")
    budget_hash = content_hash(protocol["protocols"]["main"])
    output.mkdir(parents=True)
    cards_dir = output / "cards"
    cards_dir.mkdir()
    write_new_json(output / "input_preflight.json", preflight)
    ledger = CapabilityLedger(ROOT, output / "events")
    proposals = []

    def ref(role, path):
        return ArtifactRef(role, str(path.relative_to(ROOT)), sha256_file(path))

    for fold in outer["folds"]:
        group = fold["held_out_group"]
        subset = [r for r in rows if r["group"] == group and r["main_comparison_eligible"]]
        geometry = subset[0]["geometry"]
        task = "atomic_point_defect" if geometry == "point" else "phase_boundary"
        metric_name = "marked_point_recall" if geometry == "point" else "annotated_band_coverage"
        checkpoint = (ROOT / "outputs/expert_cycle_001/student_gpu_s23_v1" / fold["fold_id"]
                      / "teacher_plus_expert/step-001200/checkpoint.json")
        train_groups = tuple(read(checkpoint)["train_groups"])
        nested_router = read(nested / "routers" / f"{fold['fold_id']}.json")
        specifications = [("student_expert", None, "fixed"), ("hybrid_fixed_budget", None, "fixed")]
        for empty in (False, True):
            key = geometry + ("::teacher_empty" if empty else "::teacher_nonempty")
            chosen = nested_router["scopes"][key]["selected_branch"]
            if chosen != "physics_budgeted" and any(r["teacher_empty"] == empty for r in subset):
                specifications.append((chosen, empty, "nested_group_cv"))
        for index, (method, empty, selection) in enumerate(specifications):
            selected_rows = subset if empty is None else [r for r in subset if r["teacher_empty"] == empty]
            reference = float(np.mean([r["methods"]["physics_budgeted"]["positive_coverage"] for r in selected_rows]))
            candidate = float(np.mean([r["methods"][method]["positive_coverage"] for r in selected_rows]))
            version = f"{fold['fold_id']}-s23-{index}-{method}"
            ranges = () if empty is None else (FeatureRange("teacher_empty", int(empty), int(empty)),)
            scope = CapabilityScope(geometry + ("-all" if empty is None else f"-empty-{int(empty)}"),
                                    (task,), geometry, ranges)
            branch = "student" if method == "student_expert" else "hybrid"
            if selection == "nested_group_cv":
                router_path = nested / "routers" / f"{fold['fold_id']}.json"
                key = geometry + ("::teacher_empty" if empty else "::teacher_nonempty")
                selection_groups = tuple(sorted(nested_router["scopes"][key]["inner_group_metrics"]))
            else:
                router_path = cards_dir / f"{version}-fixed-policy.json"
                write_new_json(router_path, {"format": "phyevo-fixed-development-policy-v1",
                    "selected_branch": method, "evaluation_protocol_sha256": sha256_file(capability / "evaluation_protocol.json"),
                    "scope": scope.scope_id, "selection_uses_labels": False})
                selection_groups = ()
            artifacts = (ref("checkpoint", checkpoint), ref("protocol", capability / "evaluation_protocol.json"),
                         ref("router", router_path), ref("replay_audit", replay / "summary.json"))
            metric = PairedMetric(metric_name, reference, candidate, True, budget_hash, budget_hash)
            card = CapabilityCard(version, scope, branch, "development", artifacts, (metric,),
                (group,), train_groups, frozen_before_evaluation=True,
                selection_method=selection, selection_group_ids=selection_groups,
                protected_scope_ids=("point", "interface"),
                caveats=("seed23_fixed_reference_not_best_seed", "one_held_out_acquisition_not_population",
                         "positive_unknown_labels_no_complete_false_positive_risk", "no_complete_old_capability_certification"))
            evidence_path = cards_dir / f"{version}-evidence.json"
            evidence = {**evidence_contract(card), "actual_case_numbers": [r["case_number"] for r in selected_rows],
                        "metric_aggregation": "macro_case_coverage_fixed_seed23",
                        "case_metrics_path": str((capability / "case_metrics.json").relative_to(ROOT)),
                        "case_metrics_sha256": sha256_file(capability / "case_metrics.json"),
                        "method": method, "independent_source_group_count": 1}
            write_new_json(evidence_path, evidence)
            card = replace(card, artifacts=card.artifacts + (ref("evidence", evidence_path),))
            card_path = cards_dir / f"{version}.json"
            write_new_json(card_path, card.to_dict())
            if not verify_artifacts(card, ROOT):
                raise ValueError("Capability artifact binding failed")
            event = ledger.record(card)
            proposals.append({"card": str(card_path.relative_to(ROOT)), "group": group,
                              "method": method, "selection": selection, "teacher_empty_scope": empty,
                              "reference": reference, "candidate": candidate,
                              "decision": evaluate_capability(card).to_dict(), "activated": event["activated"]})
    if ledger.active_cards():
        raise AssertionError("Current development evidence must not replace production physics")
    # This is an explicit retain-physics event, not a pretended reversal of a
    # production update that never happened. Real routing toggles are tested
    # separately by the runtime, and synthetic tests cover approved rollback.
    before = {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}
    ledger.rollback("all-research-proposals", reason="Retain physics default; no production proposal was activated")
    for name, digest in before.items():
        if sha256_file(output / name) != digest:
            raise AssertionError("Appending retain-physics event changed old evidence")
    summary = {"format": "phyevo-rsi-release-result-v1", "proposals": proposals,
               "production_activated_count": 0, "default_detector": "unchanged_training_free_physics",
               "old_events_unchanged_on_append": True, "events": len(ledger.events()),
               "rollback_semantics": "explicit_baseline_retention_not_reversal_of_an_activated_release",
               "retrospective_checkpoint_replay": str((replay / "summary.json").relative_to(ROOT)),
               "input_preflight": "input_preflight.json",
               "development_cohort_non_regression_is_not_external_safety": True}
    write_new_json(output / "release_summary.json", summary)
    files = {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}
    write_new_json(output / "seal.json", {"format": "phyevo-rsi-release-seal-v1", "files_sha256": files,
                                         "deleted_files": [], "original_data_modified": False})
    print(json.dumps({"output": str(output), "proposals": len(proposals), "production_activated": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/rsi_cycle_001/release_ledger_v1"))
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
