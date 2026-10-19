#!/usr/bin/env python3
"""Fit inner-only observable routers, then score their untouched outer predictions.

All data in this workflow remain developmental; source-group exclusion does not
undo prior analyst exposure to this cohort. No outer labels choose a branch.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_rsi_capabilities import across_seeds
from tfree_detection.expert_supervision import contained_path, load_learning_dataset, sha256_file, write_new_json
from tfree_detection.rsi_evaluation import EvaluationProtocol, aggregate_capabilities, evaluate_case, predict_methods
from tfree_detection.rsi_router import RouterProtocol, fit_router, select_branch

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(path.read_text())


def analyze(output, nested, capability):
    output, nested, capability = [contained_path(ROOT, p) for p in (output, nested, capability)]
    if output.exists():
        raise FileExistsError("Refusing to overwrite nested analysis")
    protocol = load_json(nested / "protocol.json")
    if load_json(nested / "completed.json")["protocol_sha256"] != sha256_file(nested / "protocol.json"):
        raise ValueError("Nested training not completed under this protocol")
    if load_json(capability / "result_seal.json")["files"]["case_metrics.json"] != sha256_file(capability / "case_metrics.json"):
        raise ValueError("Outer comparison scores changed")
    dataset = contained_path(ROOT, protocol["learning_dataset"])
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    entries = {e["case_number"]: e for e in manifest["entries"]}
    frozen = contained_path(ROOT, manifest["frozen_bundle"])
    records = {r["case_number"]: r for r in map(json.loads, (frozen / "cases.jsonl").read_text().splitlines())}
    eval_protocol = EvaluationProtocol(**protocol["evaluation"])
    files = {str(path.relative_to(ROOT)): sha256_file(path) for path in
             [nested / "protocol.json", dataset / "manifest.json", frozen / "cases.jsonl",
              capability / "case_metrics.json", capability / "evaluation_protocol.json",
              ROOT / "scripts/analyze_rsi_nested.py", ROOT / "src/tfree_detection/rsi_router.py",
              ROOT / "src/tfree_detection/rsi_evaluation.py"]}
    output.mkdir(parents=True)
    routers_dir = contained_path(ROOT, output / "routers")
    routers_dir.mkdir()
    write_new_json(output / "analysis_protocol.json", {
        "format": "phyevo-rsi-nested-analysis-v1", "input_hashes_initial": files,
        "evaluation": asdict(eval_protocol), "router_selection": asdict(RouterProtocol()),
        "outer_labels_never_fit_routers": True, "new_locked_test_claimed": False,
        "stage_order": "inner_metrics -> frozen_routers -> outer_metrics",
    })
    inner = defaultdict(list)
    count = 0
    for seed in protocol["seeds"]:
        for fold in protocol["folds"]:
            run = nested / f"seed-{seed}" / fold["fold_id"]
            report_path = run / "evaluation_manifest.json"
            report = load_json(report_path)
            marker = load_json(run / "worker_completed.json")
            if marker["evaluation_sha256"] != sha256_file(report_path):
                raise ValueError("Nested evaluation manifest changed")
            checkpoint = contained_path(ROOT, report["checkpoint"])
            metadata = load_json(checkpoint / "checkpoint.json")
            if (sha256_file(checkpoint / "checkpoint.json") != report["checkpoint_sha256"]
                    or sha256_file(checkpoint / "state.npz") != metadata["state_sha256"]
                    or metadata["train_case_numbers"] != fold["train_case_numbers"]
                    or metadata["protocol_sha256"] != sha256_file(nested / "protocol.json")):
                raise ValueError("Nested training/checkpoint inventory mismatch")
            train_groups = {entries[n]["group"] for n in metadata["train_case_numbers"]}
            if train_groups & set(fold["excluded_groups"]):
                raise ValueError("Nested leakage")
            for path in [report_path, checkpoint / "checkpoint.json", checkpoint / "state.npz"]:
                files[str(path.relative_to(ROOT))] = sha256_file(path)
            for record in report["cases"]:
                number = record["case_number"]
                entry, plane = entries[number], arrays[number]
                if entry["group"] not in fold["excluded_groups"]:
                    raise ValueError("Inner sample not in excluded pair")
                outer_group = next(g for g in fold["excluded_groups"] if g != entry["group"])
                path = contained_path(ROOT, record["score_path"])
                if sha256_file(path) != record["score_sha256"]:
                    raise ValueError("Nested score changed")
                files[str(path.relative_to(ROOT))] = record["score_sha256"]
                with np.load(path, allow_pickle=False) as archive:
                    score = archive["score"]
                predictions = predict_methods(plane["image"], plane["teacher"],
                                              records[number]["prediction"]["candidates"],
                                              score, score, entry["geometry"], protocol=eval_protocol)
                # Only one actual inner model exists. Do not mislabel its copied
                # input as a separately trained teacher-only ablation.
                del predictions["student_teacher_only"]
                row = evaluate_case(entry, plane, predictions, protocol=eval_protocol)
                row.update({"seed": seed, "outer_group": outer_group,
                            "model_train_groups": sorted(train_groups),
                            "model_geometry_support": report["geometry_support"],
                            "nested_model": str(checkpoint.relative_to(ROOT)),
                            "score_path": record["score_path"]})
                inner[outer_group].append(row)
                count += 1
            print(f"verified nested seed {seed} {fold['fold_id']}; {count} score records", flush=True)
    routers = {}
    for fold in manifest["folds"]:
        group = fold["held_out_group"]
        router = fit_router(inner[group], outer_group=group)
        path = routers_dir / f"{fold['fold_id']}.json"
        write_new_json(path, router)
        routers[group] = router
    write_new_json(output / "inner_metrics.json", dict(inner))
    # Actual outer metrics enter only after every choice has been sealed above.
    outer = load_json(capability / "case_metrics.json")["main"]
    routed, summaries = {}, {}
    for seed, rows in outer.items():
        routed[seed] = []
        for row in rows:
            router = routers[row["group"]]
            branch = select_branch(router, geometry=row["geometry"], teacher_empty=row["teacher_empty"])
            value = copy.deepcopy(row)
            value["selected_branch"] = branch
            value["methods"]["nested_routed"] = copy.deepcopy(row["methods"][branch])
            routed[seed].append(value)
        summaries[int(seed)] = aggregate_capabilities(routed[seed])
    aggregate = across_seeds(summaries)
    retention = []
    for row in aggregate:
        if row["dimension"] != "source_group":
            continue
        methods = row["methods"]
        baseline = methods["physics_budgeted"]["macro_case_positive_coverage_mean"]
        retention.append({"group": row["value"], "geometry": row["geometry"], "physics_macro": baseline,
                          "candidate_deltas_macro": {name: v["macro_case_positive_coverage_mean"] - baseline
                                                     for name, v in methods.items() if name != "physics_native"}})
    bindings = []
    for fold in manifest["folds"]:
        for seed in protocol["seeds"]:
            run = ROOT / "outputs/expert_cycle_001" / f"student_gpu_s{seed}_v1"
            checkpoint = run / fold["fold_id"] / "teacher_plus_expert/step-001200/checkpoint.json"
            refs = {"router": routers_dir / f"{fold['fold_id']}.json",
                    "evaluation_protocol": capability / "evaluation_protocol.json",
                    "checkpoint": checkpoint, "training_protocol": run / "protocol.json",
                    "nested_protocol": nested / "protocol.json"}
            binding = {"format": "phyevo-rsi-runtime-binding-v1", "artifacts": [
                {"role": role, "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for role, path in refs.items()], "evidence_kind": "nested_development",
                "production_approved": False, "all_label_checkpoint_allowed": False}
            path = routers_dir / f"{fold['fold_id']}-seed-{seed}-binding.json"
            write_new_json(path, binding)
            bindings.append(str(path.relative_to(ROOT)))
    summary = {"format": "phyevo-rsi-nested-result-v1", "nested_models": 30,
               "inner_score_records_verified": count, "outer_unique_cases": len(next(iter(routed.values()))),
               "outer_corrected_unique_cases": sum(r["main_comparison_eligible"] for r in next(iter(routed.values()))),
               "routers": routers, "across_seeds": aggregate, "retention_by_group": retention,
               "runtime_bindings": bindings, "production_promoted": False,
               "limitations": ["development_cohort_previously_exposed", "no_exhaustive_negative_labels",
                               "seed_repeats_not_independent_acquisitions", "few_source_groups",
                               "inner_model_train_size_differs_from_outer", "no_unbounded_RSI_claim"]}
    write_new_json(output / "outer_metrics.json", routed)
    write_new_json(output / "nested_summary.json", summary)
    for name, expected in files.items():
        if sha256_file(contained_path(ROOT, name)) != expected:
            raise ValueError(f"Evidence changed during analysis: {name}")
    produced = sorted(path for path in output.rglob("*") if path.is_file())
    write_new_json(output / "result_seal.json", {
        "format": "phyevo-rsi-nested-seal-v1", "input_files_sha256": files,
        "output_files_sha256": {str(p.relative_to(output)): sha256_file(p) for p in produced},
        "all_input_hashes_rechecked": True, "both_inner_outer_excluded_from_every_inner_learner": True,
        "production_default_changed": False, "deleted_files": [],
    })
    print(json.dumps({"output": str(output), "models": 30, "inner_scores": count}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/rsi_cycle_001/nested_analysis_v1"))
    parser.add_argument("--nested", type=Path, default=Path("outputs/rsi_cycle_001/nested_gpu_v1"))
    parser.add_argument("--capability", type=Path, default=Path("outputs/rsi_cycle_001/capability_eval_v1"))
    args = parser.parse_args()
    analyze(args.output, args.nested, args.capability)


if __name__ == "__main__":
    main()
