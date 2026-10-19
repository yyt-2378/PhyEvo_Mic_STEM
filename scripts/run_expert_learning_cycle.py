#!/usr/bin/env python3
"""Prepare and train an immutable expert-learning cycle, with resumable group CV.

Uses only project-local inputs/outputs. Never modifies the review service,
annotation ledger, source registry, source images or default physics detector.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from tfree_detection.expert_supervision import (
    build_learning_dataset, contained_path, load_learning_dataset,
    sha256_file, write_new_json,
)
from tfree_detection.expert_student import (
    evaluate_case, expert_adaptation_loss, make_expert_student, predict_scores,
    restore_checkpoint, sample_batch, save_checkpoint, torch,
)
from tfree_detection.cycle import CycleStage, SelfImprovementCycle


ROOT = Path(__file__).resolve().parents[1]


def code_hashes():
    names = ["scripts/run_expert_learning_cycle.py", "src/tfree_detection/expert_supervision.py",
             "src/tfree_detection/expert_student.py", "src/tfree_detection/student.py",
             "src/tfree_detection/cycle.py"]
    return {name: sha256_file(ROOT / name) for name in names}


def train_model(entries, arrays, run, protocol, protocol_hash, fold, variant, *, resume):
    numbers = fold["train_case_numbers"]
    selected = [entry for entry in entries if entry["case_number"] in numbers]
    if not selected or set(numbers) != {entry["case_number"] for entry in selected}:
        raise ValueError("Training case inventory mismatch")
    train_groups = {entry["group"] for entry in selected}
    if fold["held_out_group"] in train_groups:
        raise ValueError("Held-out acquisition leaked into training")
    device, seed = protocol["device"], protocol["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_expert_student(protocol["base_channels"]).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=protocol["learning_rate"])
    start = 0
    if run.exists() and not resume:
        raise FileExistsError("Existing run requires explicit --resume")
    run.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(path for path in run.glob("step-*") if (path / "checkpoint.json").is_file())
    if checkpoints:
        model, optimizer, rng, metadata = restore_checkpoint(checkpoints[-1], device=device, resume=True)
        for key, expected in (("protocol_sha256", protocol_hash), ("variant", variant),
                              ("train_case_numbers", numbers), ("held_out_group", fold["held_out_group"])):
            if metadata[key] != expected:
                raise ValueError(f"Resume mismatch: {key}")
        start = int(metadata["step"])
    clock_start = time.monotonic()
    metadata = {"base_channels": protocol["base_channels"], "learning_rate": protocol["learning_rate"],
                "seed": seed, "variant": variant, "protocol_sha256": protocol_hash,
                "learning_manifest_sha256": protocol["learning_manifest_sha256"],
                "train_case_numbers": numbers, "train_groups": sorted(train_groups),
                "held_out_group": fold["held_out_group"],
                "geometry_support": sorted({e["geometry"] for e in selected}),
                "unvalidated_geometries": protocol.get("unvalidated_geometries", ["line", "region"]),
                "device": device, "torch_version": torch.__version__}
    model.train()
    final_checkpoint = checkpoints[-1] if checkpoints else None
    for step in range(start + 1, protocol["steps"] + 1):
        batch = sample_batch(selected, arrays, rng, batch_size=protocol["batch_size"],
                             patch_size=protocol["patch_size"], device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["image"])
        loss, parts = expert_adaptation_loss(output, batch, use_expert=variant == "teacher_plus_expert")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite expert learning loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == protocol["steps"]:
            event = {"step": step, "elapsed_seconds_this_invocation": round(time.monotonic() - clock_start, 3),
                     **{key: float(value.detach().cpu()) for key, value in parts.items()}}
            with (run / "training_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            print(f"{fold['fold_id']} {variant} step {step}/{protocol['steps']} loss={float(loss):.4f}", flush=True)
        if step % protocol["checkpoint_every"] == 0 or step == protocol["steps"]:
            final_checkpoint = run / f"step-{step:06d}"
            save_checkpoint(final_checkpoint, model, optimizer, rng, {**metadata, "step": step})
    if final_checkpoint is None:
        raise ValueError("No trained checkpoint available")
    return model, final_checkpoint


def evaluate_fold(entries, arrays, model, run, fold, protocol, checkpoint):
    report_path = run / "evaluation.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["checkpoint_sha256"] != sha256_file(checkpoint / "checkpoint.json"):
            raise ValueError("Evaluation refers to a different checkpoint")
        for record in report["cases"]:
            path = contained_path(run, record["score_path"])
            if sha256_file(path) != record["score_sha256"]:
                raise ValueError("Saved evaluation score checksum mismatch")
        return report
    scores_dir = run / "held_out_scores"
    scores_dir.mkdir(exist_ok=True)
    cases = []
    for entry in entries:
        if entry["case_number"] not in fold["validation_case_numbers"]:
            continue
        if entry["group"] != fold["held_out_group"]:
            raise ValueError("Evaluation fold crosses acquisition groups")
        plane = arrays[entry["case_number"]]
        score = predict_scores(model, plane["image"], entry["geometry_index"], device=protocol["device"],
                               normalized=True, tile_size=protocol["inference_tile_size"], overlap=64)
        path = scores_dir / f"case-{entry['case_number']:03d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved["score"], score):
                    raise ValueError("Existing score differs on deterministic resume")
        else:
            with path.open("xb") as stream:
                np.savez_compressed(stream, score=score)
        result = evaluate_case(entry, plane, score)
        result.update({"score_path": str(path.relative_to(run)), "score_sha256": sha256_file(path)})
        cases.append(result)
    report = {"fold_id": fold["fold_id"], "held_out_group": fold["held_out_group"],
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
              "evaluation_kind": "group_held_out_assisted_silver", "cases": cases}
    write_new_json(report_path, report)
    return report


def summarize(reports):
    grouped = defaultdict(list)
    for variant, report in reports:
        for case in report["cases"]:
            grouped[(variant, case["geometry"], case["group"])].append(case)
    result = []
    for (variant, geometry, group), cases in sorted(grouped.items()):
        item = {"variant": variant, "geometry": geometry, "group": group, "cases": len(cases)}
        for metric in ("human_positive_coverage_at_10pct_area", "teacher_human_positive_coverage", "fraction_above_0_5"):
            values = [c[metric] for c in cases if metric in c]
            item[metric] = None if not values else float(np.mean(values))
        points = [c for c in cases if "point_recall_at_budget" in c]
        if points:
            for density in (20, 50, 100):
                item[f"marked_point_recall_at_{density}"] = float(np.mean([
                    c["point_recall_at_budget"][str(density)]["recall_of_marked_points"] for c in points]))
                for control in ("uniform_spatial_null", "border_only_null"):
                    item[f"{control}_recall_at_{density}"] = float(np.mean([
                        c["null_controls"][control][str(density)]["matches"] / c["annotated_points"] for c in points]))
            item["annotated_point_count"] = sum(c["annotated_points"] for c in points)
        result.append(item)
    return result


def final_training_fold(entries):
    return {"fold_id": "all_expert_labels", "held_out_group": None,
            "train_case_numbers": [e["case_number"] for e in entries if e["human_known_pixels"] > 0],
            "validation_case_numbers": []}


def run_worker(args):
    output = contained_path(ROOT, args.output)
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["code_hashes"] != code_hashes():
        raise ValueError("Worker code differs from frozen protocol")
    dataset = contained_path(ROOT, protocol["learning_dataset"])
    if sha256_file(dataset / "manifest.json") != protocol["learning_manifest_sha256"]:
        raise ValueError("Worker data differs from frozen protocol")
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    folds = protocol["folds"] + [final_training_fold(manifest["entries"])]
    matching = [f for f in folds if f["fold_id"] == args.fold_id]
    if len(matching) != 1 or args.variant not in protocol["variants"]:
        raise ValueError("Unknown worker fold/variant")
    fold = matching[0]
    if fold["fold_id"] == "all_expert_labels" and args.variant != "teacher_plus_expert":
        raise ValueError("Final model uses expert supervision")
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    run = output / fold["fold_id"]
    if fold["fold_id"] != "all_expert_labels":
        run = run / args.variant
    model, checkpoint = train_model(manifest["entries"], arrays, run, protocol,
                                    sha256_file(protocol_path), fold, args.variant, resume=args.resume)
    if fold["validation_case_numbers"]:
        evaluate_fold(manifest["entries"], arrays, model, run, fold, protocol, checkpoint)
    marker = run / "worker_completed.json"
    if not marker.exists():
        write_new_json(marker, {"protocol_sha256": sha256_file(protocol_path),
                               "checkpoint": str(checkpoint.relative_to(ROOT)),
                               "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
                               "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})


def launch_gpu_workers(output, protocol, gpu_indices, *, resume):
    """One process per idle GPU. Record PIDs; never terminate unrelated jobs."""
    if len(set(gpu_indices)) != len(gpu_indices) or not gpu_indices:
        raise ValueError("GPU indices must be nonempty and unique")
    inventory = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    available = {}
    for line in inventory.splitlines():
        index, used, utilization = [int(value.strip()) for value in line.split(",")]
        available[index] = (used, utilization)
    for gpu in gpu_indices:
        if gpu not in available or available[gpu][0] > 1024 or available[gpu][1] > 5:
            raise ValueError(f"GPU {gpu} is not idle; refusing to interfere with existing work")
    jobs = [(fold["fold_id"], variant) for fold in protocol["folds"] for variant in protocol["variants"]]
    jobs.append(("all_expert_labels", "teacher_plus_expert"))
    workers = output / "workers"
    workers.mkdir(exist_ok=True)
    active, failures = {}, []
    journal = workers / "process_events.jsonl"

    def event(payload):
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(), **payload}) + "\n")

    while jobs or active:
        for gpu in gpu_indices:
            if gpu in active or not jobs:
                continue
            fold_id, variant = jobs.pop(0)
            log = (workers / f"{fold_id}--{variant}.log").open("a", encoding="utf-8")
            temporary = tempfile.mkdtemp(prefix=f"expert-gpu{gpu}-", dir=ROOT / ".codex-tmp")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": temporary, "OMP_NUM_THREADS": str(protocol["threads"])}
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--output", str(output),
                       "--fold-id", fold_id, "--variant", variant]
            if resume:
                command.append("--resume")
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            active[gpu] = (process, log, fold_id, variant)
            event({"event": "started", "pid": process.pid, "gpu": gpu, "fold": fold_id, "variant": variant})
            print(f"GPU {gpu}: started {fold_id} {variant}, PID {process.pid}", flush=True)
        for gpu, (process, log, fold_id, variant) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            event({"event": "finished", "pid": process.pid, "gpu": gpu, "fold": fold_id,
                   "variant": variant, "returncode": code})
            print(f"GPU {gpu}: {fold_id} {variant} exited {code}", flush=True)
            if code:
                failures.append((fold_id, variant, code))
            del active[gpu]
        if active:
            time.sleep(2)
    if failures:
        raise RuntimeError(f"Workers failed; preserved logs/checkpoints for explicit resume: {failures}")


def close_exploratory_cycle(output, dataset, manifest, checkpoint):
    """A completed experiment may reject production promotion; do not fake Gold."""
    frozen = contained_path(ROOT, manifest["frozen_bundle"])
    records = [json.loads(line) for line in (frozen / "cases.jsonl").read_text().splitlines() if line]
    files = {
        "locked_student_test_manifest.json": {"assets": [], "scope": "expert_cycle_student",
                                             "note": "No independent locked student test is claimed"},
        "review_time_report.json": {
            "cases": len(records), "summed_recorded_seconds": sum(r["revision"]["annotation_duration_seconds"] for r in records),
            "scope": "recorded review timer, includes possible idle time; not measured labor savings"},
        "held_out_groups_manifest.json": {"folds": manifest["folds"], "kind": "assisted_silver_group_cv"},
        "release_decision.json": {"production_accepted": False, "shadow_inference_available": True,
                                   "reason": "uncalibrated_silver_labels_and_no_locked_student_test",
                                   "default_physics_unchanged": True,
                                   "checkpoint": str(checkpoint.relative_to(ROOT)),
                                   "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json")},
    }
    for name, payload in files.items():
        path = output / name
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise ValueError(f"Cycle closure artifact changed: {name}")
        else:
            write_new_json(path, payload)
    cycle = SelfImprovementCycle(
        cycle_id=output.name, registry_hash=sha256_file(dataset / "manifest.json"),
        teacher_version="frozen_expert_review_2d_predictions_v1", parent_student_version=None,
        locked_external_test_hash=sha256_file(output / "locked_student_test_manifest.json"))
    transitions = [
        (CycleStage.TEACHER_COMPLETED, {"candidate_table": frozen / "cases.jsonl", "teacher_manifest": frozen / "summary.json"}),
        (CycleStage.ACQUISITION_FROZEN, {"acquisition_plan": output / "protocol.json", "train_partition_hash": dataset / "manifest.json"}),
        (CycleStage.EXPERT_REVIEW_COMPLETED, {"annotation_ledger": frozen / "review_store_snapshot.sqlite3",
                                           "review_time_report": output / "review_time_report.json"}),
        (CycleStage.STUDENT_TRAINED, {"student_checkpoint": checkpoint / "checkpoint.json", "training_manifest": output / "protocol.json"}),
        (CycleStage.EXPLORATORY_EVALUATED, {"evaluation_report": output / "cycle_summary.json",
                                         "held_out_groups_manifest": output / "held_out_groups_manifest.json"}),
        (CycleStage.RELEASE_REJECTED, {"release_decision": output / "release_decision.json",
                                     "model_lineage": checkpoint / "checkpoint.json"}),
    ]
    for stage, paths in transitions:
        cycle.advance(stage, artifact_hashes={key: sha256_file(path) for key, path in paths.items()},
                      notes="Existing labels only; research/shadow use, never a production accuracy claim")
    final = output / "cycle_lineage.json"
    if final.exists():
        if json.loads(final.read_text()) != cycle.manifest():
            raise ValueError("Cycle lineage changed on resume")
    else:
        write_new_json(final, cycle.manifest())


def train_cycle(args):
    dataset, output = contained_path(ROOT, args.dataset), contained_path(ROOT, args.output)
    if output.exists() and not args.resume:
        raise FileExistsError("Use a new cycle directory, or --resume the exact same protocol")
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    if not manifest["admission"]["development_retirement_authorized"]:
        raise ValueError("Cycle lacks explicit development-data retirement")
    if args.steps < 1 or args.checkpoint_every < 1 or args.threads < 1:
        raise ValueError("Training counts must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use the existing CPU runtime")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    protocol = {
        "format": "phyevo-expert-cycle-protocol-v1", "learning_dataset": str(dataset.relative_to(ROOT)),
        "learning_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "steps": args.steps, "seed": args.seed, "base_channels": args.base_channels,
        "learning_rate": args.learning_rate, "batch_size": args.batch_size, "patch_size": args.patch_size,
        "checkpoint_every": args.checkpoint_every, "device": args.device, "threads": args.threads,
        "inference_tile_size": 256, "code_hashes": code_hashes(),
        "folds": manifest["folds"], "variants": ["teacher_only", "teacher_plus_expert"],
        "threshold_selection": "none_fixed_candidate_and_area_budgets_no_validation_tuning",
        "negatives": "unknown_except_explicit_no_defect_teacher_background_is_a_pseudo_prior",
        "point_tolerance_source_pixels": 6, "unvalidated_geometries": manifest["unvalidated_geometries"],
        "original_physics_default_changed": False,
        "null_controls": ["uniform_spatial_null", "border_only_null_24px_decay"],
        "sampling": "geometry_then_acquisition_balanced_annotation_independent_40pct_border_crops",
        "loss_reduction": "equal_labeled_image_weight_not_equal_pixel_weight_across_images",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Cannot resume after protocol/code/data change; use a new output")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_new_json(protocol_path, protocol)
    protocol_hash = sha256_file(protocol_path)
    if args.gpus:
        if protocol["device"] != "cuda:0":
            raise ValueError("GPU workers use isolated CUDA_VISIBLE_DEVICES; set --device cuda:0")
        launch_gpu_workers(output, protocol, [int(v) for v in args.gpus.split(",")], resume=args.resume)
    reports = []
    for fold in protocol["folds"]:
        for variant in protocol["variants"]:
            run = output / fold["fold_id"] / variant
            if args.gpus:
                marker = json.loads((run / "worker_completed.json").read_text())
                if marker["protocol_sha256"] != protocol_hash:
                    raise ValueError("Worker completed a different protocol")
                checkpoint = contained_path(ROOT, marker["checkpoint"])
                if marker["checkpoint_sha256"] != sha256_file(checkpoint / "checkpoint.json"):
                    raise ValueError("Worker checkpoint changed")
                report = json.loads((run / "evaluation.json").read_text())
                if report["checkpoint_sha256"] != marker["checkpoint_sha256"]:
                    raise ValueError("Worker evaluated a different checkpoint")
            else:
                model, checkpoint = train_model(manifest["entries"], arrays, run, protocol, protocol_hash, fold, variant,
                                                resume=args.resume)
                report = evaluate_fold(manifest["entries"], arrays, model, run, fold, protocol, checkpoint)
            reports.append((variant, report))
    # Fitting all available usable labels yields a shadow/inference checkpoint,
    # never another measurement of generalization on those same fitted images.
    final_fold = final_training_fold(manifest["entries"])
    if args.gpus:
        marker = json.loads((output / "all_expert_labels" / "worker_completed.json").read_text())
        if marker["protocol_sha256"] != protocol_hash:
            raise ValueError("Final worker completed a different protocol")
        checkpoint = contained_path(ROOT, marker["checkpoint"])
        if marker["checkpoint_sha256"] != sha256_file(checkpoint / "checkpoint.json"):
            raise ValueError("Final worker checkpoint changed")
    else:
        model, checkpoint = train_model(manifest["entries"], arrays, output / "all_expert_labels",
                                        protocol, protocol_hash, final_fold, "teacher_plus_expert", resume=args.resume)
    final_path = output / "cycle_summary.json"
    if not final_path.exists():
        summary = {
            "format": "phyevo-expert-cycle-result-v1", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol_hash, "group_metrics": summarize(reports),
            "shadow_checkpoint": str(checkpoint.relative_to(ROOT)),
            "shadow_checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
            "release_decision": "retain_physics_default_expert_student_shadow_only",
            "formal_gold_calibration": "not_available_single_assisted_expert_labels",
            "locked_external_student_evaluation": "not_claimed",
            "independent_accuracy_claim_allowed": False, "extra_expert_annotation_required_to_run": False,
            "scope_limits": ["assisted labels", "incomplete negatives", "few independent acquisition groups",
                             "single seed", "interface bands are not exact centerlines",
                             "region has one source group; line has no accepted geometry"],
        }
        write_new_json(final_path, summary)
    close_exploratory_cycle(output, dataset, manifest, checkpoint)
    print(json.dumps({"cycle_summary": str(final_path), "shadow_checkpoint": str(checkpoint)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--frozen-round", type=Path, default=Path("outputs/expert_cycle_001/frozen_round"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--retire-development", action="store_true", help="Explicitly retire these sources from student independent test claims")
    train = commands.add_parser("train")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--steps", type=int, default=400)
    train.add_argument("--seed", type=int, default=23)
    train.add_argument("--base-channels", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--patch-size", type=int, default=128)
    train.add_argument("--checkpoint-every", type=int, default=100)
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--device", default="cpu")
    train.add_argument("--gpus", default=None, help="Comma-separated idle physical GPU indices; one worker per GPU")
    worker = commands.add_parser("worker", help="Internal isolated GPU worker using an already frozen protocol")
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--fold-id", required=True)
    worker.add_argument("--variant", required=True)
    worker.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = build_learning_dataset(ROOT, args.frozen_round, args.output, retire_development=args.retire_development)
        print(json.dumps({"cases": len(result["entries"]), "folds": len(result["folds"]),
                          "unvalidated_geometries": result["unvalidated_geometries"]}, indent=2))
    elif args.command == "worker":
        run_worker(args)
    else:
        train_cycle(args)


if __name__ == "__main__":
    main()
