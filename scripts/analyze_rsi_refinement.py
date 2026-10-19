#!/usr/bin/env python3
"""Freeze, evaluate and select refinement policies without outer-label tuning.

Evaluate writes per-run immutable caches and can run while training continues.
Summarize requires the complete registered experiment, fits inner selectors
first, and only then reads outer outcomes. No artifact is overwritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from scripts.run_rsi_refinement import read_protocol, verify_run
from tfree_detection.expert_supervision import contained_path, load_learning_dataset, sha256_file, write_new_json
from tfree_detection.rsi_evaluation import EvaluationProtocol, evaluate_case
from tfree_detection.rsi_feedback import (
    BRANCHES, embedding_reference_context, feedback_predictions,
    fit_feedback_selector, selected_configuration,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("scripts/analyze_rsi_refinement.py", "src/tfree_detection/rsi_feedback.py",
           "src/tfree_detection/rsi_evaluation.py", "src/tfree_detection/expert_supervision.py")


def read(path):
    return json.loads(path.read_text())


def sources():
    return {name: sha256_file(ROOT/name) for name in SOURCES}


def prepare(training, output):
    training, output = contained_path(ROOT, training), contained_path(ROOT, output)
    protocol, dataset = read_protocol(training)
    if output.exists():
        raise FileExistsError("Choose a new analysis version")
    output.mkdir(parents=True)
    (output/"run_metrics").mkdir()
    write_new_json(output/"analysis_protocol.json", {
        "format": "phyevo-refinement-analysis-v1", "training": str(training.relative_to(ROOT)),
        "training_protocol_sha256": sha256_file(training/"protocol.json"), "source_hashes": sources(),
        "evaluation": asdict(EvaluationProtocol()),
        "selection": {"minimum_groups": 2, "minimum_gain": .01, "maximum_group_loss": .01},
        "embedding_threshold": "training-bank leave-one-reference-out nearest cosine distance q95, minimum 1e-6; fewer than 2 references veto",
        "stage_order": "evaluate per-run; fit and persist all inner selectors; read outer metrics for selected-policy evaluation",
        "evidence_kind": "reused_development_cohort_not_new_locked_external_test",
        "branches": list(BRANCHES), "production_approved": False,
    })
    return output


def load_analysis(output):
    output = contained_path(ROOT, output)
    protocol = read(output/"analysis_protocol.json")
    if protocol["source_hashes"] != sources():
        raise ValueError("Analysis producer changed; choose a new version")
    training = contained_path(ROOT, protocol["training"])
    if sha256_file(training/"protocol.json") != protocol["training_protocol_sha256"]:
        raise ValueError("Training protocol changed")
    train_protocol, dataset = read_protocol(training)
    return output, protocol, training, train_protocol, dataset


def jobs(protocol):
    return [(policy, seed, fold) for policy in protocol["policies"] for seed in protocol["seeds"] for fold in protocol["folds"]]


def metric_path(output, policy, seed, fold):
    return output/"run_metrics"/f"{policy}--seed-{seed}--{fold['fold_id']}.json"


@lru_cache(maxsize=1)
def loaded_data(dataset_string):
    manifest, arrays = load_learning_dataset(ROOT, Path(dataset_string))
    frozen = contained_path(ROOT, manifest["frozen_bundle"])
    records = {row["case_number"]: row for row in map(json.loads, (frozen/"cases.jsonl").read_text().splitlines())}
    return {e["case_number"]: e for e in manifest["entries"]}, arrays, records


def evaluate_run(output_string, policy, seed, fold):
    output, analysis, training, protocol, dataset = load_analysis(Path(output_string))
    run = training/policy/f"seed-{seed}"/fold["fold_id"]
    report = verify_run(run, analysis["training_protocol_sha256"])
    entries, arrays, frozen = loaded_data(str(dataset))
    train_groups = sorted({entries[n]["group"] for n in fold["train_case_numbers"]})
    geometry_support = sorted({entries[n]["geometry"] for n in fold["train_case_numbers"]})
    if set(train_groups) & set(fold["excluded_groups"]):
        raise ValueError("Training groups leaked")
    ep = EvaluationProtocol(**analysis["evaluation"])
    rows, inputs = [], {}
    for checkpoint in report["checkpoints"]:
        location = contained_path(ROOT, checkpoint["checkpoint"])
        bank_meta = read(location/"training_embedding_bank.json")
        if not {r["case_number"] for r in bank_meta["cases"]} <= set(fold["train_case_numbers"]):
            raise ValueError("Embedding bank contains non-training cases")
        with np.load(location/"training_embedding_bank.npz", allow_pickle=False) as archive:
            bank = archive["embeddings"]
        if sorted(r["case_number"] for r in checkpoint["cases"]) != sorted(fold["validation_case_numbers"]):
            raise ValueError("Prediction inventory mismatch")
        for case in checkpoint["cases"]:
            number = case["case_number"]
            entry, plane = entries[number], arrays[number]
            with np.load(contained_path(ROOT, case["score_path"]), allow_pickle=False) as archive:
                score, embedding = archive["score"], archive["embedding"]
            predictions = feedback_predictions(plane["image"], plane["teacher"], frozen[number]["prediction"]["candidates"],
                                               score, entry["geometry"], protocol=ep)
            row = evaluate_case(entry, plane, predictions, protocol=ep)
            context = embedding_reference_context(embedding, bank, bank_meta["cases"], entry["geometry"])
            row.update({"policy": policy, "seed": seed, "step": checkpoint["step"], "fold_id": fold["fold_id"],
                        "stage": fold["stage"], "model_train_groups": train_groups,
                        "model_geometry_support": geometry_support, "checkpoint": checkpoint["checkpoint"],
                        "embedding_context": context, "score_diagnostics": case["diagnostics"],
                        "score_path": case["score_path"]})
            if fold["stage"] == "inner":
                row["outer_group"] = next(g for g in fold["excluded_groups"] if g != entry["group"])
            rows.append(row)
        inputs.update(checkpoint["files"])
    inputs[str((run/"evaluation_manifest.json").relative_to(ROOT))] = sha256_file(run/"evaluation_manifest.json")
    target = metric_path(output, policy, seed, fold)
    write_new_json(target, {"analysis_protocol_sha256": sha256_file(output/"analysis_protocol.json"),
                           "fold": fold, "rows": rows, "sampler_audits": [{"step": c["step"], "audit": c["sampler_audit"]} for c in report["checkpoints"]]})
    write_new_json(target.with_suffix(".seal.json"), {"sha256": sha256_file(target), "inputs": inputs})
    return {"policy": policy, "seed": seed, "fold": fold["fold_id"], "rows": len(rows), "pid": os.getpid()}


def verified_metrics(path, *, verify_inputs=False):
    seal = read(path.with_suffix(".seal.json"))
    if seal["sha256"] != sha256_file(path):
        raise ValueError("Run metrics changed")
    if verify_inputs:
        for name, digest in seal["inputs"].items():
            if sha256_file(contained_path(ROOT, name)) != digest:
                raise ValueError(f"Run metric input changed: {name}")
    return read(path)


def evaluate_available(output, max_workers):
    output, analysis, training, protocol, _ = load_analysis(output)
    if not 1 <= max_workers <= 16:
        raise ValueError("Use 1 to 16 bounded evaluation workers")
    pending, existing, waiting = [], 0, 0
    for policy, seed, fold in jobs(protocol):
        target = metric_path(output, policy, seed, fold)
        if target.exists() or target.with_suffix(".seal.json").exists():
            verified_metrics(target)
            existing += 1
        elif (training/policy/f"seed-{seed}"/fold["fold_id"]/"worker_completed.json").exists():
            pending.append((str(output), policy, seed, fold))
        else:
            waiting += 1
    print(json.dumps({"ready": len(pending), "already_evaluated": existing, "waiting_for_training": waiting}), flush=True)
    # These subprocesses do not use GPUs or start persistent services.
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(evaluate_run, *args) for args in pending]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


def macro_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        if not row["main_comparison_eligible"]:
            continue
        for method, metric in row["methods"].items():
            key = row["group"], row["geometry"], row["policy"], row["step"], method, row["seed"]
            grouped[key].append(metric["positive_coverage"])
    combined = defaultdict(list)
    for (*key, seed), values in grouped.items():
        combined[tuple(key)].append({"seed": seed, "macro_case_coverage": float(np.mean(values)), "cases": len(values)})
    result = []
    for (group, geometry, policy, step, method), values in sorted(combined.items()):
        array = [v["macro_case_coverage"] for v in values]
        result.append({"group": group, "geometry": geometry, "policy": policy, "step": step, "method": method,
                       "mean": float(np.mean(array)), "seed_std": float(np.std(array)), "by_seed": values,
                       "seed_std_is_not_independent_acquisition_uncertainty": True})
    return result


def save_once(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("Previously saved selection/summary changed")
    else:
        write_new_json(path, value)


def summarize(output):
    output, analysis, training, protocol, _ = load_analysis(output)
    inventory = jobs(protocol)
    for policy, seed, fold in inventory:
        target = metric_path(output, policy, seed, fold)
        if not target.is_file() or not target.with_suffix(".seal.json").is_file():
            raise ValueError(f"Complete registered experiment required; missing {target.name}")
    inner = defaultdict(list)
    for policy, seed, fold in inventory:
        if fold["stage"] == "inner":
            for row in verified_metrics(metric_path(output, policy, seed, fold), verify_inputs=True)["rows"]:
                inner[row["outer_group"]].append(row)
    selectors = {}
    selector_dir = output/"selectors"
    selector_dir.mkdir(exist_ok=True)
    for fold in protocol["folds"]:
        if fold["stage"] != "outer":
            continue
        group = fold["excluded_groups"][0]
        selector = fit_feedback_selector(inner[group], outer_group=group, **analysis["selection"])
        save_once(selector_dir/(fold["fold_id"]+".json"), selector)
        selectors[group] = selector
    # No outer outcomes are read until all selectors above have been persisted.
    outer, directed = [], []
    for policy, seed, fold in inventory:
        if fold["stage"] in ("outer", "directed"):
            dest = outer if fold["stage"] == "outer" else directed
            dest.extend(verified_metrics(metric_path(output, policy, seed, fold), verify_inputs=True)["rows"])
    lookup = {(r["group"],r["case_number"],r["seed"],r["policy"],r["step"]): r for r in outer}
    routed = []
    for row in outer:
        if row["policy"] != "control" or row["step"] != 400:
            continue
        choice = selected_configuration(selectors[row["group"]], geometry=row["geometry"], teacher_empty=row["teacher_empty"])
        target, branch = row, "physics_budgeted"
        if choice:
            target = lookup[row["group"],row["case_number"],row["seed"],choice["policy"],choice["step"]]
            if not choice["embedding_veto"] or not target["embedding_context"]["novelty_veto"]:
                branch = choice["branch"]
        routed.append({**target, "policy": "nested_selected", "step": 0, "selected_configuration": choice,
                       "selected_branch": branch, "methods": {
                           "physics_budgeted": target["methods"]["physics_budgeted"],
                           "nested_feedback": target["methods"][branch]}})
    save_once(output/"selected_outer_cases.json", routed)
    save_once(output/"summary.json", {"registered_models": len(inventory), "outer_comparisons": macro_summary(outer),
               "directed_same_material_comparisons": macro_summary(directed), "nested_selected_comparison": macro_summary(routed),
               "selector_paths": [str(p.relative_to(ROOT)) for p in sorted(selector_dir.glob("*.json"))],
               "production_approved": False, "external_validation": False,
               "main_metric": "known-positive point recall or annotated-band coverage at fixed budget, not accuracy/F1",
               "limitations": ["same development cohort already seen", "incomplete negatives", "few independent acquisitions",
                               "embedding reference frames not independent calibration data", "line/region not validated",
                               "single-source point transfer cannot pass two-source route gate"]})
    produced = [p for p in output.rglob("*") if p.is_file() and p.name != "result_seal.json"]
    save_once(output/"result_seal.json", {"files": {str(p.relative_to(output)): sha256_file(p) for p in sorted(produced)},
                                         "analysis_protocol_sha256": sha256_file(output/"analysis_protocol.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("summarize")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.training, args.output)
    elif args.command == "evaluate":
        evaluate_available(args.output, args.workers)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
