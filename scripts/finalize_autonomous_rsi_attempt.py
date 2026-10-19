#!/usr/bin/env python3
"""Seal the first autonomous attempt, including its negative transfer result.

This seals an EXPERIMENT, not achievement of sustained positive RSI. No current
production/default model is switched. All failed trials remain in the archive.
"""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

import numpy as np

from tfree_detection.expert_supervision import contained_path, sha256_file, write_new_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/autonomous_rsi_20261015_v1"
FILES = [
    "scripts/freeze_autonomous_rsi_baseline.py", "scripts/run_autonomous_rsi.py",
    "scripts/verify_autonomous_rsi.py", "scripts/report_autonomous_rsi.py",
    "scripts/run_autonomous_detection.py", "scripts/run_autonomous_rsi_cycle.py",
    "scripts/finalize_autonomous_rsi_attempt.py", "src/tfree_detection/autonomous_program.py",
    "src/tfree_detection/autonomous_rsi.py", "src/tfree_detection/autonomous_runtime.py",
    "tests/test_autonomous_program.py", "tests/test_autonomous_rsi.py", "tests/test_autonomous_runner.py",
    "tests/test_autonomous_verification.py", "docs/learning/index.md",
    "docs/learning/entries/2026-10-15-autonomous-program-synthesis-rsi.md",
    "benchmarks/AUTONOMOUS_RSI_EVIDENCE_v1.yaml"]


def read(p):
    return json.loads(p.read_text())


def main():
    protocol = read(OUT / "protocol.json")
    for name, digest in protocol["source_hashes"].items():
        if sha256_file(contained_path(ROOT, name)) != digest:
            raise ValueError("Frozen autonomous algorithm modified")
    baseline = read(OUT / "baseline/manifest.json")
    for key in ("protected_files", "source_files"):
        for name, digest in baseline[key].items():
            if sha256_file(contained_path(ROOT, name)) != digest:
                raise ValueError("Protected baseline changed")
    for stage in ("cache", "search", "audit"):
        if read(OUT / f"workers-{stage}/execution.json")["failed"]:
            raise ValueError("Unfinished experiment stage")
    if read(OUT / "workers-verification-v2/execution.json")["failed"]:
        raise ValueError("Unverified search")
    verifications = [read(OUT / "verification" / f'search-{s["index"]:02d}.json') for s in protocol["searches"]]
    tests = ET.parse(OUT / "focused_tests.xml").getroot()[0].attrib
    if int(tests["failures"]) or int(tests["errors"]):
        raise ValueError("Related tests failed")
    outer_rows = 0
    for search in protocol["searches"]:
        directory = OUT / "audit" / f'search-{search["index"]:02d}'
        result = read(directory / "result.json")
        for row in result["rows"]:
            prediction = directory / "predictions" / row["prediction"]
            if sha256_file(prediction) != row["sha256"]:
                raise ValueError("Audit mask changed")
            with np.load(prediction, allow_pickle=False) as z:
                mask = z["mask"]
            with np.load(ROOT / "outputs/expert_cycle_001/learning_dataset_v1/planes" / f'case-{row["case_number"]:03d}.npz', allow_pickle=False) as z:
                human = z["human"].astype(bool)
            coverage = int((mask & human).sum()) / max(1, int(human.sum()))
            if coverage != row["coverage"] or int(mask.sum()) != row["selected_pixels"] or int(mask.sum()) > row["budget"]:
                raise ValueError("Outer metric or budget mismatch")
            outer_rows += 1
    raw_checks = []
    for name, n, index in [("zro2", 103, 0), ("ori", 105, 3), ("orii", 135, 6)]:
        raw = OUT / f"raw_runtime_{name}"
        receipt = read(raw / "receipt.json")
        if sha256_file(raw / "prediction.npz") != receipt["arrays_sha256"]:
            raise ValueError("Raw runtime artifact changed")
        with np.load(raw / "prediction.npz", allow_pickle=False) as z:
            mask = z["mask"]; student = z["student_score"]; old_mask = z["baseline_mask"]
        with np.load(OUT / "audit" / f"search-{index:02d}" / "predictions" / f"case-{n:03d}-seed-23.npz", allow_pickle=False) as z:
            offline = z["mask"]
        with np.load(ROOT / "outputs/interface_improvement_20261015_v1/release_v2/predictions" / f"case-{n:03d}-seed-23.npz", allow_pickle=False) as z:
            old = z["mask"]; old_student = z["student_score"]
        raw_checks.append({"case_number": n, "raw_runtime": str(raw.relative_to(ROOT)),
            "synthesized_mask_exact": bool(np.array_equal(mask, offline)),
            "synthesized_mask_differing_pixels": int((mask != offline).sum()),
            "base_mask_exact": bool(np.array_equal(old_mask, old)),
            "max_student_score_difference": float(np.max(np.abs(student - old_student))),
            "area_budget_respected": int(mask.sum()) <= receipt["budget"]})
    write_new_json(OUT / "raw_runtime_verification.json", {"records": raw_checks,
        "all_candidate_masks_exact": all(r["synthesized_mask_exact"] for r in raw_checks),
        "scores_bitwise_exact_claimed": False})
    paths = {name: sha256_file(contained_path(ROOT, name)) for name in FILES}
    write_new_json(OUT / "change_manifest.json", {"files": paths,
        "preexisting_dirty_file_touched": "docs/learning/index.md: appended one new row only",
        "original_framework_source_modified": False, "deletions": [], "git_commit_or_push": False})
    with ZipFile(OUT / "delivery_source.zip", "x", compression=ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(ROOT / name, name)
    result = {"attempt": "autonomous-program-synthesis-v1", "tests_passed": int(tests["tests"]),
        "independent_inner_metric_rows": sum(v["metric_rows_independently_recomputed"] for v in verifications),
        "independent_outer_mask_metric_rows": outer_rows, "source_excluded_searches": len(verifications),
        "frozen_controller_sources_unchanged": True, "baseline_protected_files_unchanged": len(baseline["protected_files"]),
        "baseline_source_files_unchanged": len(baseline["source_files"]),
        "all_raw_runtime_budgets_respected": all(r["area_budget_respected"] for r in raw_checks),
        "raw_runtime_replay": raw_checks,
        "rejected_final_update": not read(OUT / "release_v1/results.json")["final_audit_gates"]["feedback"]["accepted"],
        "sustained_positive_transfer_rsi_achieved": False,
        "original_best_version_still_current": True,
        "initial_verifier_failure": "Tuple/list JSON representation only; frozen algorithm and metrics unchanged; logs retained",
        "one_command_pipeline": "scripts/run_autonomous_rsi_cycle.py; stage order unit-tested, constituent stages actually executed",
        "no_dataset_annotation_or_successful_checkpoint_deleted_or_overwritten": True}
    write_new_json(OUT / "attempt_verification.json", result)
    write_new_json(OUT / "attempt_seal.json", {"files": {str(p.relative_to(OUT)): sha256_file(p)
        for p in OUT.rglob("*") if p.is_file()}, "controller_source_hashes": protocol["source_hashes"],
        "meaning": "Sealed experiment with negative transfer result; not proof of sustained positive RSI"})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
