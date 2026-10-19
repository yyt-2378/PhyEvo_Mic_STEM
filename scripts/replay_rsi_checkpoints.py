#!/usr/bin/env python3
"""Retrospective 400/800/1200 checkpoint replay, not a new continual task stream.

All previous datasets, annotations, checkpoints and scores stay read-only.
The outer-fold results are descriptive and cannot select a deployed checkpoint.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from tfree_detection.expert_student import predict_scores, restore_checkpoint, torch
from tfree_detection.expert_supervision import (
    contained_path, load_learning_dataset, sha256_file, verify_frozen_round, write_new_json,
)
from tfree_detection.rsi_evaluation import (
    EvaluationProtocol, aggregate_capabilities, evaluate_case, predict_methods,
)

ROOT = Path(__file__).resolve().parents[1]
STEPS = (400, 800, 1200)
SEEDS = (23, 41, 67)
SOURCE_FILES = (
    "scripts/replay_rsi_checkpoints.py", "src/tfree_detection/expert_student.py",
    "src/tfree_detection/expert_supervision.py", "src/tfree_detection/student.py",
    "src/tfree_detection/rsi_evaluation.py",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_checkpoint(meta, fold, seed, step, protocol_hash, manifest_hash):
    expected = {"seed": seed, "step": step, "variant": "teacher_plus_expert",
                "protocol_sha256": protocol_hash, "learning_manifest_sha256": manifest_hash,
                "train_case_numbers": fold["train_case_numbers"], "held_out_group": fold["held_out_group"]}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint identity or training inventory mismatch")
    if fold["held_out_group"] in meta["train_groups"]:
        raise ValueError("Held-out source group leaked into checkpoint")
    if set(fold["train_case_numbers"]) & set(fold["validation_case_numbers"]):
        raise ValueError("Held-out case leaked into checkpoint")


def make_protocol(dataset):
    manifest, _ = load_learning_dataset(ROOT, dataset)
    files, jobs = {}, []

    def remember(path, expected=None):
        path = contained_path(ROOT, path)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Input changed: {path.relative_to(ROOT)}")
        files[str(path.relative_to(ROOT))] = digest
        return digest

    manifest_hash = remember(dataset / "manifest.json")
    remember(dataset / "seal.json")
    frozen = contained_path(ROOT, manifest["frozen_bundle"])
    verify_frozen_round(ROOT, frozen)
    for name in ("seal.json", "cases.jsonl", "summary.json", "review_store_snapshot.sqlite3"):
        remember(frozen / name)
    for entry in manifest["entries"]:
        remember(contained_path(dataset, entry["plane"]), entry["plane_sha256"])
    entries = {e["case_number"]: e for e in manifest["entries"]}
    for seed in SEEDS:
        parent = contained_path(ROOT, f"outputs/expert_cycle_001/student_gpu_s{seed}_v1")
        original = read_json(parent / "protocol.json")
        phash = remember(parent / "protocol.json")
        if original["learning_manifest_sha256"] != manifest_hash or original["folds"] != manifest["folds"]:
            raise ValueError("Original training dataset/splits changed")
        for fold in manifest["folds"]:
            run = parent / fold["fold_id"] / "teacher_plus_expert"
            checkpoints = {}
            for step in STEPS:
                checkpoint = run / f"step-{step:06d}"
                meta = read_json(checkpoint / "checkpoint.json")
                validate_checkpoint(meta, fold, seed, step, phash, manifest_hash)
                if set(meta["train_groups"]) != {entries[n]["group"] for n in fold["train_case_numbers"]}:
                    raise ValueError("Training source inventory mismatch")
                checkpoints[str(step)] = {"path": str(checkpoint.relative_to(ROOT)),
                                          "sha256": remember(checkpoint / "checkpoint.json")}
                remember(checkpoint / "state.npz", meta["state_sha256"])
            evaluation = read_json(run / "evaluation.json")
            remember(run / "evaluation.json")
            if evaluation["checkpoint_sha256"] != checkpoints["1200"]["sha256"]:
                raise ValueError("Historical score checkpoint mismatch")
            if sorted(r["case_number"] for r in evaluation["cases"]) != sorted(fold["validation_case_numbers"]):
                raise ValueError("Historical score cases differ from outer fold")
            old_scores = []
            for row in evaluation["cases"]:
                path = contained_path(run, row["score_path"])
                if entries[row["case_number"]]["group"] != fold["held_out_group"]:
                    raise ValueError("Validation source mismatch")
                old_scores.append({"case_number": row["case_number"], "score_path": str(path.relative_to(ROOT)),
                                   "score_sha256": remember(path, row["score_sha256"])})
            jobs.append({"seed": seed, "fold": fold, "checkpoints": checkpoints, "scores_1200": old_scores})
    for name in SOURCE_FILES:
        remember(ROOT / name)
    return {
        "format": "phyevo-rsi-checkpoint-replay-v1", "dataset": str(dataset.relative_to(ROOT)),
        "steps": list(STEPS), "seeds": list(SEEDS), "jobs": jobs,
        "source_input_files_sha256": files, "evaluation": asdict(EvaluationProtocol()),
        "gate": {"minimum_equal_source_gain": 0.01, "maximum_source_loss": 0.01,
                 "require_each_seed_source_retention": True},
        "new_training_performed": False, "checkpoint_selection_for_release": False,
        "kind": "retrospective_single_optimization_trajectory_not_independent_continual_rounds",
        "not_new_external_test": True, "precision_or_f1_supported": False,
        "production_release_eligible": False, "default_physics_changed": False,
        "inference": {"device": "cuda:0", "tile_size": 256, "overlap": 64, "normalized": True},
        "full_resolution_source_pixel_scores": True,
    }


def verify_inputs(protocol):
    for name, expected in protocol["source_input_files_sha256"].items():
        if sha256_file(contained_path(ROOT, name)) != expected:
            raise ValueError(f"Frozen replay input changed: {name}")


def immutable_json(path, content):
    if path.exists():
        if read_json(path) != content:
            raise ValueError(f"Refusing changed existing result: {path}")
    else:
        write_new_json(path, content)


def score_record_valid(record, path, protocol_hash, checkpoint_hash):
    if (record["score_sha256"] != sha256_file(path)
            or record["protocol_sha256"] != protocol_hash
            or record["checkpoint_sha256"] != checkpoint_hash):
        raise ValueError("Prediction checksum or lineage mismatch")


def worker(args):
    output = contained_path(ROOT, args.output)
    protocol = read_json(output / "protocol.json")
    verify_inputs(protocol)
    job = next(j for j in protocol["jobs"] if j["seed"] == args.seed and j["fold"]["fold_id"] == args.fold_id)
    dataset = contained_path(ROOT, protocol["dataset"])
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    entries = {e["case_number"]: e for e in manifest["entries"]}
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise ValueError("Replay requires available CUDA")
    run = contained_path(ROOT, output / f"seed-{args.seed}" / args.fold_id)
    run.mkdir(parents=True, exist_ok=True)
    phash = sha256_file(output / "protocol.json")
    rows = []
    for step in STEPS[:2]:
        checkpoint = job["checkpoints"][str(step)]
        model, optimizer, rng, meta = restore_checkpoint(contained_path(ROOT, checkpoint["path"]), device="cuda:0")
        directory = contained_path(ROOT, run / f"step-{step:06d}")
        directory.mkdir(exist_ok=True)
        for number in job["fold"]["validation_case_numbers"]:
            path = directory / f"case-{number:03d}.npz"
            sidecar = path.with_suffix(".json")
            if path.exists() or sidecar.exists():
                if not args.resume or not path.is_file() or not sidecar.is_file():
                    raise ValueError("Incomplete or unapproved score reuse")
                record = read_json(sidecar)
                score_record_valid(record, path, phash, checkpoint["sha256"])
            else:
                score = predict_scores(model, arrays[number]["image"], entries[number]["geometry_index"],
                                       device="cuda:0", normalized=True, tile_size=256, overlap=64)
                if score.shape != tuple(entries[number]["shape"]) or not np.isfinite(score).all():
                    raise ValueError("Source-coordinate score invalid")
                with path.open("xb") as stream:
                    np.savez_compressed(stream, score=score)
                record = {"seed": args.seed, "step": step, "case_number": number,
                          "group": entries[number]["group"], "source_pixel_shape": list(score.shape),
                          "score_path": str(path.relative_to(ROOT)), "score_sha256": sha256_file(path),
                          "protocol_sha256": phash, "checkpoint_sha256": checkpoint["sha256"]}
                write_new_json(sidecar, record)
            rows.append(record)
        del model, optimizer, rng
        torch.cuda.empty_cache()
        print(f"seed {args.seed} {args.fold_id} step {step}: {len(job['fold']['validation_case_numbers'])} completed", flush=True)
    report = {"seed": args.seed, "fold": job["fold"], "protocol_sha256": phash, "scores": rows,
              "reused_1200_scores": job["scores_1200"]}
    immutable_json(run / "replay_manifest.json", report)
    marker = run / "worker_completed.json"
    if not marker.exists():
        write_new_json(marker, {"pid": os.getpid(), "finished_at_utc": now(), "protocol_sha256": phash,
                               "manifest_sha256": sha256_file(run / "replay_manifest.json"),
                               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                               "gpu_name": torch.cuda.get_device_name(0)})


def idle_gpus(indices):
    if not indices or len(indices) != len(set(indices)) or any(i not in {3, 4, 5, 6, 7} for i in indices):
        raise ValueError("Only unique explicitly reserved GPU indices 3 through 7 are allowed")
    status = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    observed = {int(i): (int(m), int(u)) for i, m, u in (line.split(",") for line in status.splitlines())}
    for gpu in indices:
        if gpu not in observed or observed[gpu][0] > 1024 or observed[gpu][1] > 5:
            raise ValueError(f"GPU {gpu} is occupied; refusing interference")


def launch(args):
    dataset, output = contained_path(ROOT, args.dataset), contained_path(ROOT, args.output)
    if output.exists() and not args.resume:
        raise FileExistsError("Existing replay requires exact --resume or a new output directory")
    if output.exists():
        protocol = read_json(output / "protocol.json")
        verify_inputs(protocol)
        if protocol["dataset"] != str(dataset.relative_to(ROOT)):
            raise ValueError("Resume dataset mismatch")
    else:
        protocol = make_protocol(dataset)
        output.mkdir(parents=True, exist_ok=False)
        write_new_json(output / "protocol.json", protocol)
    indices = [int(i) for i in args.gpus.split(",")]
    idle_gpus(indices)
    workers = contained_path(ROOT, output / "workers")
    workers.mkdir(exist_ok=True)
    temp_parent = contained_path(ROOT, ROOT / ".codex-tmp")
    temp_parent.mkdir(exist_ok=True)
    journal = workers / "process_events.jsonl"
    active, failures, jobs = {}, [], list(protocol["jobs"])

    def event(payload):
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at_utc": now(), **payload}) + "\n")

    while jobs or active:
        for gpu in indices:
            if gpu in active or not jobs:
                continue
            job = jobs.pop(0)
            seed, fold = job["seed"], job["fold"]["fold_id"]
            run = output / f"seed-{seed}" / fold
            if (run / "worker_completed.json").exists() and args.resume:
                marker = read_json(run / "worker_completed.json")
                if (marker["protocol_sha256"] != sha256_file(output / "protocol.json")
                        or marker["manifest_sha256"] != sha256_file(run / "replay_manifest.json")):
                    raise ValueError("Completed worker receipt mismatch")
                for row in read_json(run / "replay_manifest.json")["scores"]:
                    score_record_valid(row, contained_path(ROOT, row["score_path"]),
                                       marker["protocol_sha256"], job["checkpoints"][str(row["step"])]["sha256"])
                event({"event": "verified_completed_worker", "seed": seed, "fold": fold})
                continue
            idle_gpus([gpu])
            temp = tempfile.mkdtemp(prefix=f"replay-gpu{gpu}-", dir=temp_parent)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "TMPDIR": temp, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2"}
            cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--output", str(output),
                   "--seed", str(seed), "--fold-id", fold]
            if args.resume:
                cmd.append("--resume")
            log = (workers / f"seed-{seed}--{fold}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (process, log, seed, fold)
            event({"event": "started", "pid": process.pid, "gpu": gpu, "seed": seed, "fold": fold})
            print(f"GPU {gpu}: seed {seed} {fold}, PID {process.pid}", flush=True)
        for gpu, (process, log, seed, fold) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            event({"event": "finished", "pid": process.pid, "gpu": gpu, "seed": seed, "fold": fold, "returncode": code})
            print(f"GPU {gpu}: seed {seed} {fold}, exit {code}", flush=True)
            if code:
                failures.append([seed, fold, code])
            del active[gpu]
        if active:
            time.sleep(2)
    if failures:
        raise RuntimeError(f"Replay workers failed: {failures}; existing artifacts preserved")
    summarize(output)


def gate_transition(before, after, *, minimum_gain=0.01, maximum_loss=0.01):
    """Descriptive equal-acquisition macro gate; not deployment certification.

    Inputs are source -> seed -> positive-coverage scores. Never pool point
    counts and interface pixels; average dimensionless per-source coverages.
    """
    if not before or set(before) != set(after):
        raise ValueError("Gate requires identical nonempty capability strata")
    changes = []
    for group in sorted(before):
        if set(before[group]) != set(after[group]) or not before[group]:
            raise ValueError("Gate seed inventory mismatch")
        deltas = {str(seed): float(after[group][seed] - before[group][seed]) for seed in before[group]}
        if not np.isfinite(list(deltas.values())).all():
            raise ValueError("Gate metric must be finite")
        changes.append({"group": group, "mean_delta": float(np.mean(list(deltas.values()))), "delta_by_seed": deltas,
                        "worst_seed_delta": min(deltas.values())})
    gain = float(np.mean([r["mean_delta"] for r in changes]))
    losses = [r for r in changes if r["mean_delta"] < -maximum_loss - 1e-12]
    seed_losses = [r for r in changes if r["worst_seed_delta"] < -maximum_loss - 1e-12]
    passed = gain >= minimum_gain - 1e-12 and not losses and not seed_losses
    return {"equal_source_mean_gain": gain, "minimum_gain": minimum_gain, "maximum_source_loss": maximum_loss,
            "source_changes": changes, "mean_hides_source_regression": gain > 0 and bool(losses),
            "regressed_sources": [r["group"] for r in losses], "seed_unstable_sources": [r["group"] for r in seed_losses],
            "descriptive_metric_gate_passed": passed, "production_release_eligible": False,
            "decision": "would_pass_metric_guard_only" if passed else "reject_update_retain_incumbent"}


def summarize(output):
    protocol = read_json(output / "protocol.json")
    verify_inputs(protocol)
    dataset = contained_path(ROOT, protocol["dataset"])
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    entries = {e["case_number"]: e for e in manifest["entries"]}
    records = {r["case_number"]: r for r in verify_frozen_round(ROOT, contained_path(ROOT, manifest["frozen_bundle"]))}
    evaluation = EvaluationProtocol(**protocol["evaluation"])
    case_rows, by_step_seed = [], {(step, seed): [] for step in STEPS for seed in SEEDS}
    completed = []
    for job in protocol["jobs"]:
        run = output / f"seed-{job['seed']}" / job["fold"]["fold_id"]
        receipt = read_json(run / "worker_completed.json")
        report_path = run / "replay_manifest.json"
        if receipt["protocol_sha256"] != sha256_file(output / "protocol.json") or receipt["manifest_sha256"] != sha256_file(report_path):
            raise ValueError("Replay completion receipt mismatch")
        report = read_json(report_path)
        completed.append(receipt)
        rows = report["scores"] + [{**r, "step": 1200} for r in report["reused_1200_scores"]]
        for record in rows:
            number, step = record["case_number"], record["step"]
            path = contained_path(ROOT, record["score_path"])
            if sha256_file(path) != record["score_sha256"]:
                raise ValueError("Replay score changed")
            with np.load(path, allow_pickle=False) as archive:
                score = archive["score"]
            entry, plane = entries[number], arrays[number]
            predictions = predict_methods(plane["image"], plane["teacher"], records[number]["prediction"]["candidates"],
                                          score, score, entry["geometry"], protocol=evaluation)
            # The helper's teacher-only-student slot is unused: never present it
            # as physical baseline or a separately measured training variant.
            del predictions["student_teacher_only"]
            row = evaluate_case(entry, plane, predictions, protocol=evaluation)
            row.update({"step": step, "seed": job["seed"], "fold_id": job["fold"]["fold_id"],
                        "score_sha256": record["score_sha256"], "score_path": record["score_path"]})
            case_rows.append(row)
            by_step_seed[(step, job["seed"])].append(row)
    tables, maps = [], {}
    for (step, seed), rows in sorted(by_step_seed.items()):
        for stratum in aggregate_capabilities(rows)["strata"]:
            if stratum["dimension"] != "source_group":
                continue
            group = stratum["value"]
            for method, value in stratum["methods"].items():
                tables.append({"step": step, "seed": seed, "group": group, "geometry": stratum["geometry"],
                               "case_count": stratum["case_count"], "method": method, **value})
                maps.setdefault((step, method), {}).setdefault(group, {})[seed] = value["micro_positive_coverage"]
    transitions = []
    for method in ("student_expert", "hybrid_fixed_budget"):
        for first, last in zip(STEPS[:-1], STEPS[1:]):
            transitions.append({"method": method, "from_step": first, "to_step": last,
                                **gate_transition(maps[(first, method)], maps[(last, method)])})
    sequences = {}
    for method in ("student_expert", "hybrid_fixed_budget"):
        incumbent, label, sequence = maps[(400, "physics_budgeted")], "physics_budgeted", []
        for step in STEPS:
            gate = gate_transition(incumbent, maps[(step, method)])
            sequence.append({"candidate": f"{method}_step_{step}", "counterfactual_incumbent": label, **gate})
            if gate["descriptive_metric_gate_passed"]:
                incumbent, label = maps[(step, method)], f"{method}_step_{step}"
        sequences[method] = {"counterfactual_only_no_checkpoint_selected": True, "sequence": sequence,
                             "hypothetical_final_incumbent": label, "actual_production_detector": "original_physics_unchanged"}
    summary = {
        "kind": protocol["kind"], "completed_gpu_workers": len(completed), "gpu_names": sorted({r["gpu_name"] for r in completed}),
        "new_gpu_predictions": sum(r["step"] != 1200 for r in case_rows),
        "reused_historical_predictions": sum(r["step"] == 1200 for r in case_rows),
        "unique_oof_cases": len({r["case_number"] for r in case_rows}),
        "unique_main_eligible_cases": len({r["case_number"] for r in case_rows if r["main_comparison_eligible"]}),
        "source_group_count": len(maps[(400, "physics_budgeted")]), "capability_table": tables,
        "adjacent_checkpoint_transitions": transitions, "counterfactual_release_sequences": sequences,
        "actual_checkpoint_selection_or_promotion": False, "default_detector_changed": False,
        "new_expert_annotation_or_training": False, "input_hashes_reverified": True,
        "limitations": ["Same optimization trajectory, not successive independent learning tasks; not a continual-forgetting experiment.",
                        "Outer acquisition groups are retrospective developmental evidence, not checkpoint selection data.",
                        "Positive/unknown labels do not establish precision, false-positive risk, calibration or absence.",
                        "Source-average score averages dimensionless positive coverages, not pooled point and pixel counts.",
                        "Three seeds are training replicates, not independent acquisitions; one-case groups are fragile.",
                        "Passing this descriptive guard cannot authorize production release or guarantee future non-regression."],
    }
    immutable_json(output / "case_results.json", case_rows)
    immutable_json(output / "summary.json", summary)
    csv_path = output / "capability_trajectory.csv"
    if not csv_path.exists():
        columns = ["step", "seed", "group", "geometry", "case_count", "method", "matched_positive_count",
                   "annotated_positive_count", "micro_positive_coverage", "macro_case_positive_coverage"]
        with csv_path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(tables)
    verify_inputs(protocol)
    artifacts = {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*")
                 if p.is_file() and p.name != "seal.json" and "workers" not in p.relative_to(output).parts}
    immutable_json(output / "seal.json", {"format": "phyevo-rsi-replay-seal-v1", "artifacts_sha256": artifacts,
                                          "source_input_files_sha256": protocol["source_input_files_sha256"],
                                          "all_previous_inputs_preserved": True})
    print(json.dumps({k: summary[k] for k in ("completed_gpu_workers", "new_gpu_predictions", "reused_historical_predictions")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch_parser = sub.add_parser("run")
    launch_parser.add_argument("--dataset", default="outputs/expert_cycle_001/learning_dataset_v1")
    launch_parser.add_argument("--output", default="outputs/rsi_cycle_001/checkpoint_replay_v1")
    launch_parser.add_argument("--gpus", default="3,4,5,6,7")
    launch_parser.add_argument("--resume", action="store_true")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--output", required=True)
    worker_parser.add_argument("--seed", type=int, required=True)
    worker_parser.add_argument("--fold-id", required=True)
    worker_parser.add_argument("--resume", action="store_true")
    report_parser = sub.add_parser("summarize")
    report_parser.add_argument("--output", default="outputs/rsi_cycle_001/checkpoint_replay_v1")
    args = parser.parse_args()
    if args.command == "run":
        launch(args)
    elif args.command == "worker":
        worker(args)
    else:
        summarize(contained_path(ROOT, args.output))


if __name__ == "__main__":
    main()
