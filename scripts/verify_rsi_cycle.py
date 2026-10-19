#!/usr/bin/env python3
"""Read actual RSI artifacts, recompute routing, and create a new audit receipt.

No training, database writes, deletion, promotion, or service changes. Output is
exclusive-create. Hashes establish artifact consistency, not scientific truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from tfree_detection.rsi_policy import CapabilityCard, CapabilityLedger, evaluate_capability, verify_artifacts
from tfree_detection.rsi_router import fit_router, select_branch
from tfree_detection.rsi_runtime import load_binding

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("outputs/rsi_cycle_001")


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Audit:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.files = {}
        self.stats = {}

    def path(self, path):
        path = Path(path)
        actual = (path if path.is_absolute() else self.root / path).resolve()
        require(actual != self.root and actual.is_relative_to(self.root), "Path outside project")
        return actual

    def check(self, path, expected=None, *, force=False):
        path = self.path(path)
        name = str(path.relative_to(self.root))
        stat = path.stat()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if not force and self.stats.get(name) == signature:
            digest = self.files[name]
        else:
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
            digest = checksum.hexdigest()
        require(expected is None or digest == expected, f"Hash mismatch: {path}")
        require(name not in self.files or self.files[name] == digest, f"Input changed during audit: {path}")
        self.files[name] = digest
        self.stats[name] = signature
        return digest

    def read(self, path):
        # Same-size rewrites can share a metadata signature on some filesystems.
        # Bind the actual JSON bytes, not just the cached stat tuple.
        digest = self.check(path, force=True)
        payload = self.path(path).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == digest,
                f"Input changed during audit: {path}")
        return json.loads(payload)

    def hashes(self, mapping, base=Path()):
        for path, digest in mapping.items():
            self.check(base / path, digest)


def disjoint_training(entries, metadata, excluded_groups, expected_cases):
    require(metadata["train_case_numbers"] == expected_cases, "Training case inventory differs")
    groups = {entries[n]["group"] for n in expected_cases}
    require(not groups.intersection(excluded_groups), "Held-out acquisition leaked into learner")


def junit_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    require(bool(suites), "No JUnit test suites")
    result = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    require(result["tests"] > 0 and result["failures"] == result["errors"] == 0, "Tests not green")
    return result


def verify_experiments(audit, runtime_receipt):
    frozen = Path("outputs/expert_cycle_001/frozen_round")
    audit.hashes(audit.read(frozen / "seal.json")["files"], frozen)
    dataset = Path("outputs/expert_cycle_001/learning_dataset_v1")
    audit.check(dataset / "manifest.json", audit.read(dataset / "seal.json")["manifest_sha256"])
    manifest = audit.read(dataset / "manifest.json")
    entries = {e["case_number"]: e for e in manifest["entries"]}
    require(len(entries) == 154, "Unexpected expert cohort")
    for entry in entries.values():
        audit.check(dataset / entry["plane"], entry["plane_sha256"])
        audit.check(entry["source_path"], entry["source_file_sha256"])

    capability = BASE / "capability_eval_v1"
    audit.hashes(audit.read(capability / "result_seal.json")["files"], capability)
    audit.hashes(audit.read(capability / "evaluation_protocol.json")["source_input_files_sha256"])
    case_metrics = audit.read(capability / "case_metrics.json")["main"]
    require(set(case_metrics) == {"23", "41", "67"}, "Missing training seed")
    for seed, rows in case_metrics.items():
        require(len({r["case_number"] for r in rows}) == len(rows) == 152, "Wrong outer cohort")
        require(sum(r["main_comparison_eligible"] for r in rows) == 151, "Wrong corrected cohort")
        require(len({r["group"] for r in rows}) == 5, "Wrong acquisition count")
        for row in rows:
            for name, metric in row["methods"].items():
                require(metric["precision"] is None and metric["f1"] is None,
                        "Partial-positive labels cannot provide complete precision/F1")
                if name != "physics_native":
                    count = metric.get("candidate_count", metric.get("selected_pixels"))
                    if count is not None:
                        require(count <= metric["budget"], "Candidate ceiling exceeded")

    nested = BASE / "nested_gpu_v1"
    protocol = audit.read(nested / "protocol.json")
    audit.check(nested / "protocol.json", audit.read(nested / "completed.json")["protocol_sha256"])
    audit.hashes(protocol["source_hashes"])
    nested_models, scores = 0, 0
    for seed in protocol["seeds"]:
        for fold in protocol["folds"]:
            run = nested / f"seed-{seed}" / fold["fold_id"]
            report = audit.read(run / "evaluation_manifest.json")
            marker = audit.read(run / "worker_completed.json")
            audit.check(run / "evaluation_manifest.json", marker["evaluation_sha256"])
            for step in (400, 800, 1200):
                checkpoint = run / f"step-{step:06d}"
                metadata = audit.read(checkpoint / "checkpoint.json")
                audit.check(checkpoint / "state.npz", metadata["state_sha256"])
                require(metadata["protocol_sha256"] == audit.files[str(nested / "protocol.json")], "Checkpoint protocol mismatch")
                disjoint_training(entries, metadata, fold["excluded_groups"], fold["train_case_numbers"])
            audit.check(Path(report["checkpoint"]) / "checkpoint.json", report["checkpoint_sha256"])
            require({r["case_number"] for r in report["cases"]} == set(fold["validation_case_numbers"]), "Nested score cohort mismatch")
            for row in report["cases"]:
                require(entries[row["case_number"]]["group"] in fold["excluded_groups"], "Invalid nested holdout")
                audit.check(row["score_path"], row["score_sha256"])
                scores += 1
            nested_models += 1
    require((nested_models, scores) == (30, 1824), "Incomplete nested experiment")

    analysis = BASE / "nested_analysis_v1"
    seal = audit.read(analysis / "result_seal.json")
    audit.hashes(seal["input_files_sha256"])
    audit.hashes(seal["output_files_sha256"], analysis)
    inner = audit.read(analysis / "inner_metrics.json")
    outer = audit.read(analysis / "outer_metrics.json")
    summary = audit.read(analysis / "nested_summary.json")
    for group, rows in inner.items():
        require(fit_router(rows, outer_group=group) == summary["routers"][group], "Refit router differs")
    for seed, rows in outer.items():
        original = {row["case_number"]: row for row in case_metrics[seed]}
        require(len(rows) == len(original), "Routed cohort differs")
        for row in rows:
            branch = select_branch(summary["routers"][row["group"]], geometry=row["geometry"], teacher_empty=row["teacher_empty"])
            require(branch == row["selected_branch"], "Stored routing differs")
            require(row["methods"]["nested_routed"] == original[row["case_number"]]["methods"][branch], "Routed metric mismatch")
    for path in summary["runtime_bindings"]:
        audit.read(path)
        load_binding(audit.root, path)
    require(len(summary["runtime_bindings"]) == 15, "Incomplete runtime bindings")

    replay = BASE / "checkpoint_replay_v1"
    seal = audit.read(replay / "seal.json")
    audit.hashes(seal["source_input_files_sha256"])
    audit.hashes(seal["artifacts_sha256"], replay)
    trajectory = audit.read(replay / "summary.json")
    require(trajectory["new_gpu_predictions"] == 912 and trajectory["reused_historical_predictions"] == 456, "Replay incomplete")
    require(not trajectory["actual_checkpoint_selection_or_promotion"], "Descriptive replay changed deployed step")

    release = BASE / "release_ledger_v1"
    audit.hashes(audit.read(release / "seal.json")["files_sha256"], release)
    release_summary = audit.read(release / "release_summary.json")
    ledger = CapabilityLedger(audit.root, release / "events")
    require(len(ledger.events()) == 13 and not ledger.active_cards(), "Unexpected production activation")
    rows = case_metrics["23"]
    for proposal in release_summary["proposals"]:
        card = CapabilityCard.from_dict(audit.read(proposal["card"]))
        require(verify_artifacts(card, audit.root), "Card bytes or evidence contract differ")
        require(not evaluate_capability(card).production_approved and not proposal["activated"], "Development card promoted")
        selected = [r for r in rows if r["main_comparison_eligible"] and r["group"] == proposal["group"]
                    and (proposal["teacher_empty_scope"] is None or r["teacher_empty"] == proposal["teacher_empty_scope"])]
        for key, method in (("reference", "physics_budgeted"), ("candidate", proposal["method"])):
            value = float(np.mean([r["methods"][method]["positive_coverage"] for r in selected]))
            require(abs(value - proposal[key]) < 1e-12, "Card metric not tied to fixed seed23 cases")
    require(len(release_summary["proposals"]) == 12, "Incomplete proposal set")

    runtime = audit.read(runtime_receipt)
    audit.hashes(runtime["source_hashes"])
    runtime_checks = []
    for case in runtime["cases"]:
        number = case["case_number"]
        audit.check(case["summary"], case["summary_sha256"])
        result = audit.read(case["summary"])
        for ref in result["candidate_artifacts"].values():
            audit.check(ref["path"], ref["sha256"])
        for ref in result["lineage_artifacts"].values():
            audit.check(ref["path"], ref["sha256"])
        checkpoint = Path(result["checkpoint"])
        metadata = audit.read(checkpoint / "checkpoint.json")
        audit.check(checkpoint / "state.npz", metadata["state_sha256"])
        require(number not in metadata["train_case_numbers"] and entries[number]["group"] not in metadata["train_groups"], "Runtime learner trained on evaluated acquisition")
        with np.load(audit.path(dataset / entries[number]["plane"]), allow_pickle=False) as plane:
            physical_path = Path(case["summary"]).parent / "physics_original/mask.npy"
            audit.check(physical_path)
            require(np.array_equal(np.load(audit.path(physical_path), allow_pickle=False), plane["teacher"]), "Physics mask changed")
        score_path = checkpoint.parent / "held_out_scores" / f"case-{number:03d}.npz"
        audit.check(score_path)
        with np.load(audit.path(score_path), allow_pickle=False) as old:
            with np.load(audit.path(result["candidate_artifacts"]["student_score"]["path"]), allow_pickle=False) as new:
                require(np.array_equal(old["score"], new["score"]), "Runtime score not bitwise equal to outer prediction")
        runtime_checks.append({"case_number": number, "physics_equal": True, "student_score_equal": True,
                               "primary_branch": result["primary_branch"]})
    require({r["case_number"] for r in runtime_checks} == {63, 103, 135}, "Runtime probes incomplete")

    old = audit.read("paper/reproducibility/release_manifest_v15.json")
    preserved = {"paper/revision_agent/manuscript_v21_candidate.md", "paper/PhyEvo_Mic_manuscript_v40_submission_layout.docx"}
    for row in old["files"]:
        if row["path"] in preserved:
            audit.check(row["path"], row["sha256"])
            preserved.remove(row["path"])
    require(not preserved, "Historical manuscript inventory missing")
    return {"expert_cases": 154, "corrected_evaluation_cases": 151, "independent_evaluation_sources": 5,
            "new_nested_training_models": nested_models, "nested_score_records": scores,
            "new_checkpoint_replay_predictions": 912, "refitted_routers": len(inner),
            "runtime_checks": runtime_checks, "development_proposals": 12, "production_promotions": 0,
            "historical_v21_manuscript_and_word_preserved": True,
            "population_non_regression_or_complete_precision_claimed": False}


def verify_manuscript(audit, directory):
    receipt = audit.read(directory / "build_receipt.json")
    require(receipt["status"] == "rendered_final_candidate" and not receipt["unresolved_markers"], "Manuscript incomplete")
    for ref in [receipt["source"], receipt["figure_manifest"], *receipt["figures"], *receipt["outputs"].values()]:
        audit.check(ref["path"], ref["sha256"])
    audit.check("paper/build_manuscript_rsi_v22.py", receipt["builder_sha256"])
    figure_dir = Path(receipt["figure_manifest"]["path"]).parent
    figure_seal = audit.read(figure_dir / "seal.json")
    audit.hashes(figure_seal["artifacts_sha256"], figure_dir)
    audit.hashes(figure_seal["input_files_sha256"])
    audit.check("paper/build_rsi_figures_v22.py", figure_seal["builder_sha256"])
    require(len(receipt["figures"]) == 5, "Five figures required")
    with zipfile.ZipFile(audit.path(receipt["outputs"]["docx"]["path"])) as document:
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        for name in ("word/document.xml", "word/styles.xml"):
            tree = ET.fromstring(document.read(name))
            for value in tree.findall(".//w:sz", ns):
                require(int(value.attrib[f"{{{ns['w']}}}val"]) % 2 == 0, "Noninteger DOCX font size")
        require(b'Arial' in document.read("word/styles.xml"), "Arial style missing")
    return {"source": receipt["source"], "outputs": receipt["outputs"], "figure_count": 5,
            "typography": receipt["typography"], "warnings": receipt["warnings"]}


def verify_runtime_probes(audit, version):
    """Recheck three completed CLI runs, including raw-image normalization."""
    from tfree_detection.expert_supervision import normalize_image
    from tfree_detection.preprocessing import load_grayscale
    from tfree_detection.rsi_runtime import pixel_sha256

    dataset = Path("outputs/expert_cycle_001/learning_dataset_v1")
    entries = {e["case_number"]: e for e in audit.read(dataset / "manifest.json")["entries"]}
    records_path = Path("outputs/expert_cycle_001/frozen_round/cases.jsonl")
    audit.check(records_path)
    records = {r["case_number"]: r for r in map(json.loads, audit.path(records_path).read_text().splitlines())}
    rows = []
    for number, fold in ((63, "group-00"), (103, "group-02"), (135, "group-04")):
        directory = BASE / f"runtime_case{number:03d}_research_v{version}"
        path = directory / "rsi_detection.json"
        result = audit.read(path)
        require(result["status"] == "research_routed" and not result["production_approved"], "Wrong probe mode")
        frozen = load_binding(audit.root, BASE / "nested_analysis_v1/routers" / f"{fold}-seed-23-binding.json")
        require(result["checkpoint_sha256"] == frozen.artifacts["checkpoint"].sha256, "Wrong outer checkpoint")
        entry, sample = entries[number], records[number]["sample"]
        audit.check(entry["source_path"], entry["source_file_sha256"])
        raw = load_grayscale(audit.path(entry["source_path"]), frame_index=sample.get("frame_index"), dataset_path=sample.get("dataset_path"))
        require(pixel_sha256(raw) == result["raw_pixel_sha256"], "Runtime source pixels differ")
        with np.load(audit.path(dataset / entry["plane"]), allow_pickle=False) as plane:
            require(np.array_equal(normalize_image(raw), plane["image"]), "Runtime normalization differs")
            require(np.array_equal(np.load(audit.path(directory / "physics_original/mask.npy"), allow_pickle=False), plane["teacher"]), "Physics changed")
        for name, ref in result["candidate_artifacts"].items():
            audit.check(ref["path"], ref["sha256"])
            if ref["path"].endswith(".npz"):
                with np.load(audit.path(ref["path"]), allow_pickle=False) as current:
                    with np.load(audit.path(BASE / f"runtime_case{number:03d}_research_v2" / Path(ref["path"]).name), allow_pickle=False) as previous:
                        require(set(current.files) == set(previous.files) and all(np.array_equal(current[k], previous[k]) for k in current.files), "Ledger fix changed research candidates")
        score_path = frozen.checkpoint.parent / "held_out_scores" / f"case-{number:03d}.npz"
        with np.load(score_path, allow_pickle=False) as reference:
            with np.load(audit.path(result["candidate_artifacts"]["student_score"]["path"]), allow_pickle=False) as current:
                require(np.array_equal(current["score"], reference["score"]), "Runtime differs from held-out score")
        require(number not in frozen.checkpoint_metadata["train_case_numbers"] and entry["group"] not in frozen.checkpoint_metadata["train_groups"], "Probe is not held out")
        rows.append({"case_number": number, "summary": str(path), "summary_sha256": audit.check(path),
                     "primary_branch": result["primary_branch"], "raw_normalization_equal": True,
                     "physics_mask_equal": True, "student_outer_score_bitwise_equal": True,
                     "all_candidate_arrays_equal_to_v2": True})
    return {"format": "phyevo-rsi-runtime-verification-v1", "physical_gpu_index": 2, "seed": 23,
            "scope": "Three completed raw-image research CLI probes after activation-ledger fix; not a statistical external test",
            "production_approved": False, "cases": rows,
            "source_hashes": {path: audit.check(path) for path in ["src/tfree_detection/rsi_runtime.py", "scripts/run_rsi_detection.py", "tests/test_rsi_runtime.py"]},
            "previous_source_archive": {"path": str(BASE / "runtime_source_before_ledger_fix.zip"),
                                        "sha256": audit.check(BASE / "runtime_source_before_ledger_fix.zip")},
            "deleted_files": [], "previous_results_overwritten": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New receipt JSON inside project")
    parser.add_argument("--runtime-receipt", type=Path, default=BASE / "runtime_verification_v3.json")
    parser.add_argument("--manuscript-dir", type=Path)
    parser.add_argument("--junit", type=Path, action="append", default=[])
    parser.add_argument("--runtime-probe-version", type=int,
                        help="Only verify these already-completed probe directories and create a new runtime receipt")
    args = parser.parse_args()
    audit = Audit(ROOT)
    output = audit.path(args.output)
    require(not output.exists() and output.parent.is_dir(), "Output must be new with an existing project parent")
    if args.runtime_probe_version is not None:
        require(args.runtime_probe_version >= 3, "Do not replace historical runtime receipts")
        result = verify_runtime_probes(audit, args.runtime_probe_version)
        for path, digest in list(audit.files.items()):
            audit.check(path, digest, force=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    summary = verify_experiments(audit, args.runtime_receipt)
    if args.manuscript_dir:
        summary["manuscript"] = verify_manuscript(audit, args.manuscript_dir)
    tests = []
    for path in args.junit:
        audit.check(path)
        tests.append({"path": str(path), **junit_counts(audit.path(path))})
    for path, digest in list(audit.files.items()):
        audit.check(path, digest, force=True)
    result = {"format": "phyevo-rsi-independent-verification-v1", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "summary": summary, "tests": tests, "verified_file_count": len(audit.files),
              "input_files_sha256": audit.files, "deleted_files": [], "overwritten_files": [],
              "scope": "Artifact, split, route, runtime equality and listed tests; not an external statistical safety certificate"}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt": str(output), "verified_files": len(audit.files), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
