#!/usr/bin/env python3
"""Audit frozen out-of-fold scores and build a new controlled-RSI capability atlas.

No training, existing artifact writes, model selection, or production promotion.
The outer folds are exploratory assisted-Silver evidence, not a new locked test.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from tfree_detection.expert_supervision import (
    contained_path, load_learning_dataset, sha256_file, verify_frozen_round, write_new_json,
)
from tfree_detection.rsi_evaluation import (
    EvaluationProtocol, Prediction, aggregate_capabilities, evaluate_case,
    predict_methods, student_mask, student_points,
)


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("teacher_only", "teacher_plus_expert")
PROTOCOLS = {
    "low_budget": EvaluationProtocol(point_budget_per_512=20, area_budget_fraction=0.05),
    "main": EvaluationProtocol(),
    "high_budget": EvaluationProtocol(point_budget_per_512=100, area_budget_fraction=0.20),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audited_inputs(root: Path, dataset: Path, runs: list[Path]):
    """Verify score lineage, disjoint acquisition groups, and every input digest."""
    manifest, arrays = load_learning_dataset(root, dataset)
    records = verify_frozen_round(root, contained_path(root, manifest["frozen_bundle"]))
    records = {record["case_number"]: record for record in records}
    files: dict[str, str] = {}

    def remember(path, expected=None):
        path = contained_path(root, path)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Input checksum mismatch: {path.relative_to(root)}")
        files[str(path.relative_to(root))] = digest
        return digest

    remember(dataset / "manifest.json")
    remember(dataset / "seal.json")
    frozen = contained_path(root, manifest["frozen_bundle"])
    for filename in ("cases.jsonl", "seal.json", "summary.json", "review_store_snapshot.sqlite3"):
        remember(frozen / filename)
    for entry in manifest["entries"]:
        remember(contained_path(dataset, entry["plane"]), entry["plane_sha256"])
    entries = {entry["case_number"]: entry for entry in manifest["entries"]}
    inventory = {}
    for directory in runs:
        protocol_path = directory / "protocol.json"
        protocol = read_json(protocol_path)
        protocol_hash = remember(protocol_path)
        if protocol["learning_manifest_sha256"] != files[str((dataset / "manifest.json").relative_to(root))]:
            raise ValueError("Run uses a different learning dataset")
        if protocol["folds"] != manifest["folds"] or tuple(protocol["variants"]) != VARIANTS:
            raise ValueError("Run fold/variant inventory mismatch")
        seed = int(protocol["seed"])
        if seed in inventory:
            raise ValueError("Duplicate training seed")
        scores = {}
        for fold in protocol["folds"]:
            for variant in VARIANTS:
                run = directory / fold["fold_id"] / variant
                marker = read_json(run / "worker_completed.json")
                remember(run / "worker_completed.json")
                checkpoint = contained_path(root, marker["checkpoint"])
                remember(checkpoint / "checkpoint.json", marker["checkpoint_sha256"])
                metadata = read_json(checkpoint / "checkpoint.json")
                remember(checkpoint / "state.npz", metadata["state_sha256"])
                if marker["protocol_sha256"] != protocol_hash or metadata["protocol_sha256"] != protocol_hash:
                    raise ValueError("Checkpoint protocol mismatch")
                if metadata["train_case_numbers"] != fold["train_case_numbers"]:
                    raise ValueError("Checkpoint train inventory mismatch")
                if metadata["held_out_group"] != fold["held_out_group"] or fold["held_out_group"] in metadata["train_groups"]:
                    raise ValueError("Acquisition group leakage")
                if metadata["variant"] != variant or int(metadata["seed"]) != seed or metadata["step"] != protocol["steps"]:
                    raise ValueError("Checkpoint variant/seed/completion mismatch")
                training_groups = {entries[number]["group"] for number in metadata["train_case_numbers"]}
                if training_groups != set(metadata["train_groups"]):
                    raise ValueError("Checkpoint train groups do not match cases")
                report_path = run / "evaluation.json"
                remember(report_path)
                report = read_json(report_path)
                if report["checkpoint_sha256"] != marker["checkpoint_sha256"]:
                    raise ValueError("OOF prediction checkpoint mismatch")
                if sorted(row["case_number"] for row in report["cases"]) != sorted(fold["validation_case_numbers"]):
                    raise ValueError("OOF validation inventory mismatch")
                for row in report["cases"]:
                    number = int(row["case_number"])
                    if entries[number]["group"] != fold["held_out_group"] or number in metadata["train_case_numbers"]:
                        raise ValueError("Case or acquisition leakage")
                    path = contained_path(run, row["score_path"])
                    remember(path, row["score_sha256"])
                    key = (variant, number)
                    if key in scores:
                        raise ValueError("Case has duplicate out-of-fold prediction")
                    scores[key] = {"path": path, "old_metrics": row,
                                   "fold_id": fold["fold_id"], "checkpoint": str(checkpoint.relative_to(root))}
        inventory[seed] = scores
    return manifest, arrays, records, inventory, files


def spatial_controls(image, geometry: str, case_number: int, protocol: EvaluationProtocol, draw: int):
    """Three reproducible independent spatial draws; no annotation-informed tuning."""
    shape = image.shape
    rng = np.random.default_rng(137_000 + 1000 * draw + int(case_number))
    noise = gaussian_filter(rng.random(shape), 1)
    metadata = {"selection_uses_expert_labels": False, "calibrated": False,
                "spatial_null_not_model": True, "draw": draw, "budget_matched": True}
    if geometry == "point":
        budget = protocol.point_budget(shape)
        yy, xx = np.indices(shape)
        distance = np.minimum.reduce([xx, yy, shape[1] - 1 - xx, shape[0] - 1 - yy])
        scores = {"control_uniform": noise,
                  "control_border24": np.where(distance < 24, noise, 0),
                  "control_border32": np.where(distance < 32, noise, 0)}
        return {name: Prediction(geometry, points_xy=student_points(score, budget, protocol.point_min_distance_px),
                                 budget=budget, metadata={**metadata, "selection": name})
                for name, score in scores.items()}
    budget = protocol.area_budget(shape)
    # IID pixel ranking, not smoothed topography, gives the uniform-area control.
    return {"control_uniform": Prediction(geometry, mask=student_mask(rng.random(shape), budget),
                                           budget=budget, metadata={**metadata, "selection": "uniform_pixel_ranking"})}


def across_seeds(seed_summaries: dict[int, dict]) -> list[dict]:
    groups = {}
    for seed, summary in seed_summaries.items():
        for row in summary["strata"]:
            key = (row["dimension"], row["value"], row["geometry"])
            groups.setdefault(key, {})[seed] = row
    result = []
    for key, rows in sorted(groups.items()):
        first = next(iter(rows.values()))
        methods = {}
        for method in first["methods"]:
            values = {}
            for metric in ("micro_positive_coverage", "macro_case_positive_coverage"):
                numbers = [row["methods"][method][metric] for row in rows.values()]
                values[metric + "_mean"] = float(np.mean(numbers))
                values[metric + "_seed_std"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else 0.0
                values[metric + "_by_seed"] = {str(seed): row["methods"][method][metric] for seed, row in rows.items()}
            methods[method] = values
        result.append({"dimension": key[0], "value": key[1], "geometry": key[2],
                       "unique_case_count": first["case_count"],
                       "independent_source_group_count": first["independent_source_group_count"],
                       "seed_count": len(rows), "methods": methods,
                       "seed_std_is_not_population_confidence_interval": True})
    return result


def complementarity(row, predictions, plane):
    """Descriptive matched-positive overlap, never used to select predictions."""
    if not row["main_comparison_eligible"]:
        return None
    physics, student = "physics_budgeted", "student_expert"
    if row["geometry"] == "point":
        teacher_set = set(row["methods"][physics]["matched_expert_indices"])
        student_set = set(row["methods"][student]["matched_expert_indices"])
        overlap, teacher_only, student_only = len(teacher_set & student_set), len(teacher_set - student_set), len(student_set - teacher_set)
    else:
        positive = plane["human"].astype(bool)
        teacher = predictions[physics].mask & positive
        student = predictions[student].mask & positive
        overlap = int((teacher & student).sum())
        teacher_only, student_only = int((teacher & ~student).sum()), int((student & ~teacher).sum())
    return {"both_matched_positive_count": overlap, "physics_only_positive_count": teacher_only,
            "student_only_positive_count": student_only,
            "not_a_deployable_oracle_router_or_equal_budget_union": True}


def run(root: Path, output: Path):
    root = root.resolve()
    output = contained_path(root, output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite capability evaluation")
    base = contained_path(root, "outputs/expert_cycle_001")
    dataset = base / "learning_dataset_v1"
    runs = [base / f"student_gpu_s{seed}_v1" for seed in (23, 41, 67)]
    manifest, arrays, records, inventory, files = audited_inputs(root, dataset, runs)
    for relative in ("src/tfree_detection/rsi_evaluation.py", "scripts/evaluate_rsi_capabilities.py"):
        files[relative] = sha256_file(contained_path(root, relative))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    protocol = {
        "format": "phyevo-controlled-rsi-capability-protocol-v1", "frozen_at_utc": utc_now(),
        "evaluation_kind": "retrospective_assisted_silver_acquisition_group_cross_validation",
        "confirmatory_preregistration_claimed": False,
        "protocols": {name: asdict(value) for name, value in PROTOCOLS.items()},
        "training_seeds": sorted(inventory), "source_input_files_sha256": files,
        "original_physics": "frozen_native_centroids_and_masks_not_teacher_only_student",
        "physics_area_budgeting": "support_only_coordinate_hash_thinning_no_dense_score_available",
        "student_and_hybrid_have_identical_budget_ceiling": True,
        "hybrid": "fixed_half_physics_quota_then_student_fill_no_label_based_routing",
        "spatial_controls": {"draws": [0, 1, 2], "point": ["uniform", "24px_border", "32px_border"],
                             "area": ["uniform_pixel_ranking"], "main_budget_only": True},
        "accepted_teacher_label_excluded_from_main_comparison": True,
        "unknown_pixels_are_not_negative": True, "production_release_eligible": False,
        "default_detector_changed": False, "previous_artifact_writes": False,
        "population_significance_or_chemical_identity_claimed": False,
    }
    write_new_json(output / "evaluation_protocol.json", protocol)
    entries = {entry["case_number"]: entry for entry in manifest["entries"]}
    results = {name: {} for name in PROTOCOLS}
    summaries = {name: {} for name in PROTOCOLS}
    old_equal = 0
    for seed, scores in inventory.items():
        per_protocol = {name: [] for name in PROTOCOLS}
        numbers = sorted(number for variant, number in scores if variant == "teacher_plus_expert")
        for index, number in enumerate(numbers):
            entry, plane = entries[number], arrays[number]
            score_arrays = {}
            for variant in VARIANTS:
                with np.load(scores[(variant, number)]["path"], allow_pickle=False) as archive:
                    score_arrays[variant] = archive["score"]
            for name, budget in PROTOCOLS.items():
                predictions = predict_methods(plane["image"], plane["teacher"], records[number]["prediction"]["candidates"],
                                              score_arrays["teacher_only"], score_arrays["teacher_plus_expert"],
                                              entry["geometry"], protocol=budget)
                row = evaluate_case(entry, plane, predictions, protocol=budget)
                row["seed"] = seed
                row["fold_id"] = scores[("teacher_plus_expert", number)]["fold_id"]
                row["complementarity"] = complementarity(row, predictions, plane)
                if name == "main" and row["main_comparison_eligible"]:
                    for method, variant in (("student_teacher_only", "teacher_only"), ("student_expert", "teacher_plus_expert")):
                        previous = scores[(variant, number)]["old_metrics"]
                        expected = (previous["point_recall_at_budget"]["50"]["recall_of_marked_points"]
                                    if entry["geometry"] == "point" else previous["human_positive_coverage_at_10pct_area"])
                        if row["methods"][method]["positive_coverage"] != expected:
                            raise ValueError("New student evaluation differs from frozen historical evaluation")
                        old_equal += 1
                per_protocol[name].append(row)
            if (index + 1) % 25 == 0:
                print(f"seed {seed}: audited {index + 1}/{len(numbers)} cases at 3 budgets", flush=True)
        for name, rows in per_protocol.items():
            results[name][str(seed)] = rows
            summaries[name][seed] = aggregate_capabilities(rows)
        print(f"seed {seed}: complete", flush=True)
    controls = {}
    primary_numbers = sorted({row["case_number"] for row in results["main"][str(min(inventory))]})
    for draw in (0, 1, 2):
        rows = []
        for number in primary_numbers:
            entry, plane = entries[number], arrays[number]
            predictions = spatial_controls(plane["image"], entry["geometry"], number, PROTOCOLS["main"], draw)
            rows.append(evaluate_case(entry, plane, predictions, protocol=PROTOCOLS["main"]))
        controls[str(draw)] = {"cases": rows, "summary": aggregate_capabilities(rows)}
    capabilities = {
        "format": "phyevo-controlled-rsi-capabilities-v1",
        "evaluation_protocol_sha256": sha256_file(output / "evaluation_protocol.json"),
        "by_seed": {name: {str(seed): value for seed, value in values.items()} for name, values in summaries.items()},
        "across_seeds": {name: across_seeds(values) for name, values in summaries.items()},
        "spatial_controls_across_draws": across_seeds({int(draw): result["summary"] for draw, result in controls.items()}),
        "spatial_control_dispersion_is_draw_std_not_training_seed_std": True,
        "unique_oof_case_count": len(primary_numbers),
        "unique_independent_corrected_case_count": sum(entries[n]["action"] == "correct_geometry" for n in primary_numbers),
        "unique_held_out_source_groups": sorted({entries[n]["group"] for n in primary_numbers}),
        "seed_case_repetitions_are_not_independent_samples": True,
        "legacy_student_case_metrics_exact_matches": old_equal,
        "unvalidated_geometries": manifest["unvalidated_geometries"],
        "production_release_eligible": False,
        "limitations": ["positive_unknown_labels_not_exhaustive_negatives", "assisted_annotations_not_blind_Gold",
                        "single_acquisition_frames_correlated", "seed_std_not_population_confidence_interval",
                        "posthoc_capability_strata_not_causal_or_confirmatory", "teacher_empty_not_no_defect",
                        "native_teacher_cost_differs_from_equal_budget", "area_teacher_no_dense_ranking",
                        "interface_roi_not_centerline", "all_expert_labels_checkpoint_excluded_from_testing"],
    }
    write_new_json(output / "case_metrics.json", results)
    write_new_json(output / "spatial_controls.json", controls)
    write_new_json(output / "capabilities.json", capabilities)
    fields = ["protocol", "dimension", "value", "geometry", "method", "unique_case_count", "source_groups", "seed_count",
              "macro_case_coverage_mean", "macro_case_coverage_seed_std", "micro_positive_coverage_mean", "micro_positive_coverage_seed_std"]
    with (output / "capability_table.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, strata in capabilities["across_seeds"].items():
            for row in strata:
                for method, values in row["methods"].items():
                    writer.writerow({"protocol": name, "dimension": row["dimension"], "value": row["value"],
                                     "geometry": row["geometry"], "method": method, "unique_case_count": row["unique_case_count"],
                                     "source_groups": row["independent_source_group_count"], "seed_count": row["seed_count"],
                                     "macro_case_coverage_mean": values["macro_case_positive_coverage_mean"],
                                     "macro_case_coverage_seed_std": values["macro_case_positive_coverage_seed_std"],
                                     "micro_positive_coverage_mean": values["micro_positive_coverage_mean"],
                                     "micro_positive_coverage_seed_std": values["micro_positive_coverage_seed_std"]})
    for relative, expected in files.items():
        if sha256_file(contained_path(root, relative)) != expected:
            raise ValueError(f"Input changed during evaluation: {relative}")
    produced = ("evaluation_protocol.json", "case_metrics.json", "spatial_controls.json", "capabilities.json", "capability_table.csv")
    write_new_json(output / "result_seal.json", {
        "completed_at_utc": utc_now(), "format": "phyevo-rsi-capability-seal-v1",
        "files": {name: sha256_file(output / name) for name in produced},
        "all_input_files_rechecked_unchanged": True, "verified_input_file_count": len(files),
        "verified_oof_score_file_count": sum(len(scores) for scores in inventory.values()),
        "deleted_files": [], "production_default_changed": False,
    })
    return capabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/rsi_cycle_001/capability_eval_v1"))
    args = parser.parse_args()
    result = run(args.project_root, args.output)
    for row in result["across_seeds"]["main"]:
        if row["dimension"] == "source_group":
            print(json.dumps({"group": row["value"], "n": row["unique_case_count"],
                              "macro_mean": {name: value["macro_case_positive_coverage_mean"] for name, value in row["methods"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
