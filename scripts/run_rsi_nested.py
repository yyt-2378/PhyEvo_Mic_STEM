#!/usr/bin/env python3
"""Leave-two-acquisition-out learning for leakage-free developmental routing.

Reuses the sealed expert learner unchanged, never overwrites expert data or
existing runs. Each inner learner excludes BOTH its outer and inner groups.
This is nested developmental evidence, not a fresh locked external test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from scripts.run_expert_learning_cycle import train_model
from tfree_detection.expert_student import predict_scores, torch
from tfree_detection.expert_supervision import contained_path, load_learning_dataset, sha256_file, write_new_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "scripts/run_rsi_nested.py", "scripts/run_expert_learning_cycle.py",
    "src/tfree_detection/expert_student.py", "src/tfree_detection/expert_supervision.py",
    "src/tfree_detection/student.py", "src/tfree_detection/cycle.py",
)


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCE_PATHS}


def pair_folds(manifest):
    groups = sorted({f["held_out_group"] for f in manifest["folds"]})
    folds = []
    for index, pair in enumerate(combinations(groups, 2)):
        train = [e["case_number"] for e in manifest["entries"]
                 if e["human_known_pixels"] > 0 and e["group"] not in pair]
        valid = [e["case_number"] for e in manifest["entries"]
                 if e["group"] in pair and e["human_known_pixels"] > 0]
        if not train or not valid or set(train) & set(valid):
            raise ValueError("Invalid nested acquisition split")
        folds.append({"fold_id": f"pair-{index:02d}", "held_out_group": " + ".join(pair),
                      "excluded_groups": list(pair), "train_case_numbers": train,
                      "validation_case_numbers": valid})
    return folds


def make_protocol(manifest, dataset, *, steps=1200, seeds=(23, 41, 67)):
    return {
        "format": "phyevo-rsi-nested-development-v1",
        "learning_dataset": str(dataset.relative_to(ROOT)),
        "learning_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "source_hashes": source_hashes(), "folds": pair_folds(manifest), "seeds": list(seeds),
        "steps": steps, "batch_size": 32, "patch_size": 128, "base_channels": 16,
        "learning_rate": 0.001, "checkpoint_every": 400, "threads": 2,
        "device": "cuda:0", "inference_tile_size": 256,
        "variant": "teacher_plus_expert", "unvalidated_geometries": ["line", "region"],
        "student_architecture_changed": False,
        "evidence_kind": "nested_development_not_locked_external_test",
        "outer_protocol": "five acquisition groups; existing outer learners retained unchanged",
        "inner_protocol": "unordered pair exclusion is shared between symmetric outer/inner folds",
        "negative_semantics": "unmarked is unknown; teacher background is weak pseudo prior",
        "router_candidates": ["physics_budgeted", "student_expert", "hybrid_fixed_budget"],
        "router_scope": "geometry and observed teacher-empty status only; never material/group ID",
        "router_selection": "equal inner-acquisition weighting; min 2 groups; paired gain >=0.01; no inner-group loss >0.01; ties prefer physics then fixed hybrid then student",
        "router_fallback": "physics_budgeted when evidence insufficient; unknown/unsupported task uses original native physics at runtime",
        "evaluation": {"point_budget_per_512": 50, "point_tolerance_px": 6,
                       "area_budget_fraction": 0.1, "hybrid_teacher_fraction": 0.5},
        "null_controls": ["uniform_spatial", "border_24px", "border_32px"],
        "sensitivity": {"point_budgets": [20, 50, 100], "area_budgets": [0.05, 0.1, 0.2]},
        "claims": {"new_external_test": False, "precision_or_F1": False,
                   "autonomous_unbounded_improvement": False, "default_physics_changed": False},
    }


def read_protocol(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["source_hashes"] != source_hashes():
        raise ValueError("Producer code changed; cannot resume this immutable experiment")
    dataset = contained_path(ROOT, protocol["learning_dataset"])
    if sha256_file(dataset / "manifest.json") != protocol["learning_manifest_sha256"]:
        raise ValueError("Learning dataset changed")
    return protocol, dataset


def worker(args):
    output = contained_path(ROOT, args.output)
    protocol, dataset = read_protocol(output)
    if args.seed not in protocol["seeds"]:
        raise ValueError("Seed not in frozen protocol")
    fold = next(f for f in protocol["folds"] if f["fold_id"] == args.fold_id)
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    entries = manifest["entries"]
    trained = [e for e in entries if e["case_number"] in fold["train_case_numbers"]]
    if {e["group"] for e in trained} & set(fold["excluded_groups"]):
        raise ValueError("Outer or inner source group leaked into training")
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise ValueError("This registered experiment requires CUDA")
    run = contained_path(ROOT, output / f"seed-{args.seed}" / fold["fold_id"])
    phash = sha256_file(output / "protocol.json")
    model, checkpoint = train_model(entries, arrays, run, {**protocol, "seed": args.seed}, phash,
                                    fold, "teacher_plus_expert", resume=args.resume)
    score_dir = contained_path(ROOT, run / "held_out_scores")
    score_dir.mkdir(exist_ok=True)
    records = []
    for entry in entries:
        if entry["case_number"] not in fold["validation_case_numbers"]:
            continue
        if entry["group"] not in fold["excluded_groups"]:
            raise ValueError("Invalid validation source")
        path = score_dir / f"case-{entry['case_number']:03d}.npz"
        # Resume checks the registered array bytes; completed predictions need
        # not be recomputed merely because a scheduler observation was lost.
        score_manifest = path.with_suffix(".json")
        if path.exists() or score_manifest.exists():
            if not args.resume or not path.is_file() or not score_manifest.is_file():
                raise ValueError("Incomplete or unapproved prediction reuse")
            record = json.loads(score_manifest.read_text())
            if (record["score_sha256"] != sha256_file(path)
                    or record["checkpoint_sha256"] != sha256_file(checkpoint / "checkpoint.json")
                    or record["protocol_sha256"] != phash):
                raise ValueError("Nested prediction changed on resume")
        else:
            score = predict_scores(model, arrays[entry["case_number"]]["image"], entry["geometry_index"],
                                   device=protocol["device"], normalized=True, tile_size=256, overlap=64)
            with path.open("xb") as stream:
                np.savez_compressed(stream, score=score)
            record = {"case_number": entry["case_number"], "group": entry["group"],
                      "score_path": str(path.relative_to(ROOT)), "score_sha256": sha256_file(path),
                      "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
                      "protocol_sha256": phash}
            write_new_json(score_manifest, record)
        records.append(record)
    summary = {"seed": args.seed, "fold": fold, "protocol_sha256": phash,
               "checkpoint": str(checkpoint.relative_to(ROOT)),
               "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
               "geometry_support": sorted({e["geometry"] for e in trained}), "cases": records}
    target = run / "evaluation_manifest.json"
    if target.exists():
        if json.loads(target.read_text()) != summary:
            raise ValueError("Completed nested evaluation changed")
    else:
        write_new_json(target, summary)
    marker = run / "worker_completed.json"
    if not marker.exists():
        write_new_json(marker, {"pid": os.getpid(), "seed": args.seed, "fold_id": args.fold_id,
                               "protocol_sha256": phash, "evaluation_sha256": sha256_file(target),
                               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                               "device_name": torch.cuda.get_device_name(0)})


def idle_gpus(indices):
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("GPU list must be nonempty and unique")
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    found = {int(a): (int(b), int(c)) for a, b, c in (line.split(",") for line in output.splitlines())}
    for index in indices:
        if index not in found or found[index][0] > 1024 or found[index][1] > 5:
            raise ValueError(f"GPU {index} is occupied; no interference allowed")


def launch(args):
    dataset, output = contained_path(ROOT, args.dataset), contained_path(ROOT, args.output)
    manifest, _ = load_learning_dataset(ROOT, dataset)
    protocol = make_protocol(manifest, dataset, steps=args.steps)
    if output.exists() and not args.resume:
        raise FileExistsError("Use a new output, or explicitly resume exact protocol")
    if (output / "protocol.json").exists():
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Cannot resume different protocol")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_new_json(output / "protocol.json", protocol)
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise ValueError("Less than 15 GiB free; preserving existing data, refusing new training")
    indices = [int(i) for i in args.gpus.split(",")]
    idle_gpus(indices)
    workers = contained_path(ROOT, output / "workers")
    workers.mkdir(exist_ok=True)
    jobs = [(seed, fold["fold_id"]) for seed in protocol["seeds"] for fold in protocol["folds"]]
    active, failures = {}, []
    journal = workers / "process_events.jsonl"

    def event(payload):
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(), **payload}) + "\n")

    while jobs or active:
        for gpu in indices:
            if gpu in active or not jobs:
                continue
            seed, fold = jobs.pop(0)
            marker = output / f"seed-{seed}" / fold / "worker_completed.json"
            if marker.exists() and args.resume:
                saved = json.loads(marker.read_text())
                target = marker.parent / "evaluation_manifest.json"
                if (saved["protocol_sha256"] != sha256_file(output / "protocol.json")
                        or saved["evaluation_sha256"] != sha256_file(target)):
                    raise ValueError("Completed worker receipt changed")
                # Full score/state byte verification is left to worker restore
                # on explicit verification runs, not silently assumed here.
                event({"event": "already_completed", "seed": seed, "fold": fold})
                continue
            idle_gpus([gpu])
            temp = tempfile.mkdtemp(prefix=f"rsi-gpu{gpu}-", dir=ROOT / ".codex-tmp")
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
            event({"event": "finished", "pid": process.pid, "gpu": gpu, "seed": seed,
                   "fold": fold, "returncode": code})
            print(f"GPU {gpu}: seed {seed} {fold}, exit {code}", flush=True)
            if code:
                failures.append([seed, fold, code])
            del active[gpu]
        if active:
            time.sleep(2)
    if failures:
        raise RuntimeError(f"Preserved failed runs for diagnosis/resume: {failures}")
    final = output / "completed.json"
    if not final.exists():
        write_new_json(final, {"protocol_sha256": sha256_file(output / "protocol.json"),
                               "models": len(protocol["seeds"]) * len(protocol["folds"]),
                               "at_utc": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("train")
    run.add_argument("--dataset", type=Path, default=Path("outputs/expert_cycle_001/learning_dataset_v1"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--gpus", default="2,3,4,5,6,7")
    run.add_argument("--steps", type=int, default=1200)
    run.add_argument("--resume", action="store_true")
    work = sub.add_parser("worker")
    work.add_argument("--output", type=Path, required=True)
    work.add_argument("--seed", type=int, required=True)
    work.add_argument("--fold-id", required=True)
    work.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    (launch if args.command == "train" else worker)(args)


if __name__ == "__main__":
    main()
