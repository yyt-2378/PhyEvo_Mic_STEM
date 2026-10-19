#!/usr/bin/env python3
"""Immutable, source-held-out sampling/augmentation/weak-supervision ablations.

No held-out labels enter training, crops, augmentation or embedding banks.
Outer results are developmental only; selection must use pair-excluded inner
folds. The original learner, predictions and expert assets are never edited.
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

from scripts.run_rsi_nested import idle_gpus
from tfree_detection.expert_student import make_expert_student, restore_checkpoint, save_checkpoint, torch
from tfree_detection.expert_supervision import contained_path, load_learning_dataset, sha256_file, write_new_json
from tfree_detection.rsi_refinement import POLICIES, RefinedSampler, policy_dict, predict_with_embedding, refinement_loss

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/run_rsi_refinement.py", "src/tfree_detection/rsi_refinement.py",
    "scripts/run_rsi_nested.py", "scripts/run_expert_learning_cycle.py",
    "src/tfree_detection/expert_student.py", "src/tfree_detection/expert_supervision.py",
    "src/tfree_detection/student.py", "src/tfree_detection/cycle.py",
)


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCES}


def experiment_folds(manifest):
    entries = manifest["entries"]
    groups = sorted({f["held_out_group"] for f in manifest["folds"]})
    folds = []
    for stage, excluded_sets in (("outer", [(g,) for g in groups]), ("inner", combinations(groups, 2))):
        for index, excluded in enumerate(excluded_sets):
            folds.append({"fold_id": f"{stage}-{index:02d}", "stage": stage,
                          "excluded_groups": list(excluded),
                          "train_case_numbers": [e["case_number"] for e in entries if e["human_known_pixels"] > 0 and e["group"] not in excluded],
                          "validation_case_numbers": [e["case_number"] for e in entries if e["human_known_pixels"] > 0 and e["group"] in excluded]})
    ori, orii = "sigma3-orientation-i", "sigma3-orientation-ii"
    for source, target, name in ((ori, orii, "ORI-to-ORII"), (orii, ori, "ORII-to-ORI")):
        if source in groups and target in groups:
            folds.append({"fold_id": name, "stage": "directed", "excluded_groups": [g for g in groups if g != source],
                          "train_case_numbers": [e["case_number"] for e in entries if e["human_known_pixels"] > 0 and e["group"] == source],
                          "validation_case_numbers": [e["case_number"] for e in entries if e["human_known_pixels"] > 0 and e["group"] == target]})
    for fold in folds:
        selected = [e for e in entries if e["case_number"] in fold["train_case_numbers"]]
        if (not selected or not fold["validation_case_numbers"] or
                set(fold["train_case_numbers"]) & set(fold["validation_case_numbers"]) or
                {e["group"] for e in selected} & set(fold["excluded_groups"])):
            raise ValueError("Invalid source-held-out fold")
    return folds


def representatives(entries, per_source=3):
    """Deterministic training-only bank, chosen by case order, never outcomes."""
    result = []
    for group in sorted({e["group"] for e in entries}):
        selected = sorted([e for e in entries if e["group"] == group], key=lambda e: e["case_number"])
        for index in np.unique(np.linspace(0, len(selected)-1, min(per_source, len(selected)), dtype=int)):
            result.append(selected[int(index)])
    return result


def make_protocol(manifest, dataset):
    return {
        "format": "phyevo-rsi-refinement-development-v1",
        "learning_dataset": str(dataset.relative_to(ROOT)),
        "learning_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "source_hashes": source_hashes(), "folds": experiment_folds(manifest),
        "policies": {name: policy_dict(name) for name in POLICIES}, "seeds": [23, 41, 67],
        "steps": 1200, "checkpoints": [400, 800, 1200], "batch_size": 32,
        "patch_size": 128, "base_channels": 16, "learning_rate": .001,
        "gradient_clip": 5, "threads": 2, "device": "cuda:0", "inference_tile_size": 256,
        "augmentation_seed": "seed * 1000000 + step (independent of crop RNG)",
        "embedding_bank": "up to 3 order-spaced training cases per acquisition; no validation cases",
        "architecture_changed": False, "expert_labels_changed": False,
        "weak_roi_semantics": "experimental top-quarter positive bag plus explicit occupancy prior; unmarked is NOT expert-negative",
        "selection_protocol": "Within each outer exclusion, select policy/checkpoint by equal-acquisition inner validation only; never outer label or group-name routing. Retain physics fallback and report all candidates, losses and guard rejections.",
        "evaluation": {"point_budget_per_512": 50, "point_tolerance_px": 6, "point_min_distance_px": 4, "area_budget_fraction": .1},
        "feedback_candidates": ["physics_budgeted", "student_expert", "hybrid_fixed_budget", "student_reranked_physics_plus_rescue"],
        "embedding_role": "training-reference retrieval and experimental novelty gate; NOT calibrated OOD, chemical identity or strain",
        "claims": {"locked_external_test": False, "precision_or_F1": False, "default_detector_replaced": False,
                   "automatic_production_promotion": False, "guaranteed_continual_improvement": False},
    }


def read_protocol(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["source_hashes"] != source_hashes():
        raise ValueError("Producer changed; use a new experiment version")
    dataset = contained_path(ROOT, protocol["learning_dataset"])
    if sha256_file(dataset / "manifest.json") != protocol["learning_manifest_sha256"]:
        raise ValueError("Learning data changed")
    return protocol, dataset


def train(entries, arrays, run, protocol, protocol_hash, fold, policy_name, seed, *, resume):
    selected = [e for e in entries if e["case_number"] in fold["train_case_numbers"]]
    if set(fold["train_case_numbers"]) != {e["case_number"] for e in selected}:
        raise ValueError("Training inventory mismatch")
    policy = POLICIES[policy_name]
    sampler = RefinedSampler(selected, arrays, policy, excluded_groups=fold["excluded_groups"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_expert_student(protocol["base_channels"]).to(protocol["device"])
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=protocol["learning_rate"])
    if run.exists() and not resume:
        raise FileExistsError("Explicit resume required")
    run.mkdir(parents=True, exist_ok=True)
    start = 0
    identity = {"protocol_sha256": protocol_hash, "policy": policy_name, "seed": seed, "fold": fold}
    checkpoints = sorted(p for p in run.glob("step-*") if (p / "checkpoint.json").is_file())
    if checkpoints:
        model, optimizer, rng, meta = restore_checkpoint(checkpoints[-1], device=protocol["device"], resume=True)
        if any(meta[key] != value for key, value in identity.items()):
            raise ValueError("Checkpoint identity mismatch")
        start, sampler.audit = meta["step"], meta["sampler_audit"]
    metadata = {**identity, "base_channels": protocol["base_channels"], "learning_rate": protocol["learning_rate"],
                "learning_manifest_sha256": protocol["learning_manifest_sha256"],
                "train_case_numbers": fold["train_case_numbers"], "train_groups": sorted({e["group"] for e in selected}),
                "geometry_support": sorted({e["geometry"] for e in selected}),
                "torch_version": torch.__version__, "sampler_source_probabilities": sampler.probabilities(),
                "refinement_policy": policy_dict(policy_name)}
    began = time.monotonic()
    model.train()
    for step in range(start+1, protocol["steps"]+1):
        batch, _ = sampler.sample(rng, batch_size=protocol["batch_size"], patch_size=protocol["patch_size"],
                                  device=protocol["device"], augmentation_seed=seed*1000000+step)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = refinement_loss(model(batch["image"]), batch, policy)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite refinement loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip"])
        optimizer.step()
        if step == 1 or step % 100 == 0:
            event = {"step": step, "elapsed_seconds": time.monotonic()-began,
                     **{name: float(value.detach().cpu()) for name, value in parts.items()}}
            with (run / "training_events.jsonl").open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False)+"\n")
            print(f"{fold['fold_id']} {policy_name} {seed}: {step}/{protocol['steps']} loss={event['total']:.5f}", flush=True)
        if step in protocol["checkpoints"]:
            save_checkpoint(run / f"step-{step:06d}", model, optimizer, rng,
                            {**metadata, "step": step, "sampler_audit": sampler.audit})
    return selected


def score_diagnostics(score, plane):
    """Label-free output behavior; not a calibration claim."""
    dark = plane["image"] <= np.quantile(plane["image"], .1)
    count = max(1, int(np.ceil(score.size*.1)))
    top = np.argsort(-score.ravel(), kind="stable")[:count]
    return {"score_mean": float(score.mean()), "fraction_above_0_5": float((score > .5).mean()),
            "fraction_top10_in_darkest_decile": float(dark.ravel()[top].mean())}


def verify_run(run, protocol_hash):
    receipt = json.loads((run / "worker_completed.json").read_text())
    index = run / "evaluation_manifest.json"
    if receipt["protocol_sha256"] != protocol_hash or receipt["evaluation_sha256"] != sha256_file(index):
        raise ValueError("Completion receipt changed")
    result = json.loads(index.read_text())
    for record in result["checkpoints"]:
        for name, digest in record["files"].items():
            if sha256_file(contained_path(ROOT, name)) != digest:
                raise ValueError(f"Artifact changed: {name}")
    return result


def evaluate(entries, arrays, selected, run, protocol, protocol_hash, fold, policy, seed, *, resume):
    records = []
    valid = [e for e in entries if e["case_number"] in fold["validation_case_numbers"]]
    bank_entries = representatives(selected)
    if {e["case_number"] for e in bank_entries} & set(fold["validation_case_numbers"]):
        raise ValueError("Held-out input leaked into reference bank")
    for step in protocol["checkpoints"]:
        checkpoint = run / f"step-{step:06d}"
        model, _, _, meta = restore_checkpoint(checkpoint, device=protocol["device"])
        files = {str((checkpoint/name).relative_to(ROOT)): sha256_file(checkpoint/name) for name in ("checkpoint.json", "state.npz")}
        target = checkpoint / "held_out_scores"
        target.mkdir(exist_ok=True)
        cases = []
        for entry in valid:
            number = entry["case_number"]
            path = target / f"case-{number:03d}.npz"
            receipt = path.with_suffix(".json")
            if path.exists() or receipt.exists():
                if not resume or not path.is_file() or not receipt.is_file():
                    raise ValueError("Incomplete prediction retained for diagnosis")
                record = json.loads(receipt.read_text())
                if (record["score_sha256"] != sha256_file(path) or record["protocol_sha256"] != protocol_hash or
                        record["checkpoint_sha256"] != sha256_file(checkpoint/"checkpoint.json")):
                    raise ValueError("Prediction binding changed")
            else:
                score, embedding = predict_with_embedding(model, arrays[number]["image"], entry["geometry_index"], device=protocol["device"])
                with path.open("xb") as stream:
                    np.savez_compressed(stream, score=score, embedding=embedding)
                record = {"case_number": number, "group": entry["group"], "geometry": entry["geometry"],
                          "score_path": str(path.relative_to(ROOT)), "score_sha256": sha256_file(path),
                          "protocol_sha256": protocol_hash, "checkpoint_sha256": sha256_file(checkpoint/"checkpoint.json"),
                          "diagnostics": score_diagnostics(score, arrays[number])}
                write_new_json(receipt, record)
            cases.append(record)
            files[str(path.relative_to(ROOT))] = record["score_sha256"]
            files[str(receipt.relative_to(ROOT))] = sha256_file(receipt)
        bank_path = checkpoint / "training_embedding_bank.npz"
        bank_meta_path = checkpoint / "training_embedding_bank.json"
        if bank_path.exists() or bank_meta_path.exists():
            if not resume or not bank_path.is_file() or not bank_meta_path.is_file():
                raise ValueError("Incomplete embedding bank retained for diagnosis")
            bank_meta = json.loads(bank_meta_path.read_text())
            if (bank_meta["sha256"] != sha256_file(bank_path) or bank_meta["protocol_sha256"] != protocol_hash or
                    bank_meta["checkpoint_sha256"] != sha256_file(checkpoint/"checkpoint.json")):
                raise ValueError("Embedding bank binding changed")
        else:
            embeddings, bank_records = [], []
            for entry in bank_entries:
                score, feature = predict_with_embedding(model, arrays[entry["case_number"]]["image"], entry["geometry_index"], device=protocol["device"])
                embeddings.append(feature)
                bank_records.append({"case_number": entry["case_number"], "group": entry["group"], "geometry": entry["geometry"],
                                     "diagnostics": score_diagnostics(score, arrays[entry["case_number"]])})
            with bank_path.open("xb") as stream:
                np.savez_compressed(stream, embeddings=np.stack(embeddings))
            bank_meta = {"cases": bank_records, "sha256": sha256_file(bank_path), "protocol_sha256": protocol_hash,
                         "checkpoint_sha256": sha256_file(checkpoint/"checkpoint.json"), "selection_uses_validation": False}
            write_new_json(bank_meta_path, bank_meta)
        for path in (bank_path, bank_meta_path):
            files[str(path.relative_to(ROOT))] = sha256_file(path)
        records.append({"step": step, "checkpoint": str(checkpoint.relative_to(ROOT)), "cases": cases,
                        "sampler_audit": meta["sampler_audit"], "files": files})
    result = {"protocol_sha256": protocol_hash, "fold": fold, "seed": seed, "policy": policy, "checkpoints": records}
    path = run / "evaluation_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError("Evaluation changed")
    else:
        write_new_json(path, result)
    write_new_json(run / "worker_completed.json", {"protocol_sha256": protocol_hash,
                   "evaluation_sha256": sha256_file(path), "pid": os.getpid(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device_name": torch.cuda.get_device_name(0)})


def worker(args):
    output = contained_path(ROOT, args.output)
    protocol, dataset = read_protocol(output)
    if args.seed not in protocol["seeds"] or args.policy not in protocol["policies"]:
        raise ValueError("Unregistered seed/policy")
    fold = next(f for f in protocol["folds"] if f["fold_id"] == args.fold_id)
    run = contained_path(ROOT, output / args.policy / f"seed-{args.seed}" / args.fold_id)
    phash = sha256_file(output / "protocol.json")
    if (run / "worker_completed.json").exists():
        if not args.resume:
            raise FileExistsError("Completed run requires explicit verification/resume")
        verify_run(run, phash)
        return
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise ValueError("Registered experiment requires CUDA")
    selected = train(manifest["entries"], arrays, run, protocol, phash, fold, args.policy, args.seed, resume=args.resume)
    evaluate(manifest["entries"], arrays, selected, run, protocol, phash, fold, args.policy, args.seed, resume=args.resume)


def launch(args):
    output, dataset = contained_path(ROOT, args.output), contained_path(ROOT, args.dataset)
    if output == ROOT or output == dataset or dataset.is_relative_to(output):
        raise ValueError("Output must be a dedicated directory separate from input")
    manifest, _ = load_learning_dataset(ROOT, dataset)
    protocol = make_protocol(manifest, dataset)
    if output.exists():
        if not args.resume or json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Existing output is not an exact approved resume")
    else:
        output.mkdir(parents=True)
        write_new_json(output / "protocol.json", protocol)
    indices = [int(i) for i in args.gpus.split(",")]
    idle_gpus(indices)
    stages = args.stages.split(",")
    if not stages or not set(stages) <= {"outer", "inner", "directed"}:
        raise ValueError("Invalid experiment stage")
    jobs = [(seed, policy, fold["fold_id"]) for stage in stages for seed in protocol["seeds"]
            for policy in protocol["policies"] for fold in protocol["folds"] if fold["stage"] == stage]
    workers = output / "workers"
    workers.mkdir(exist_ok=True)
    phash = sha256_file(output / "protocol.json")
    active, failures = {}, []

    def event(data):
        with (workers / "process_events.jsonl").open("a") as stream:
            stream.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(), **data})+"\n")

    event({"event": "controller_started", "pid": os.getpid(), "stages": stages, "jobs": len(jobs)})
    while jobs or active:
        for gpu in indices:
            if gpu in active or not jobs:
                continue
            if shutil.disk_usage(ROOT).free < 15*1024**3:
                event({"event": "disk_guard", "pending_jobs": len(jobs)})
                failures.append(["disk_guard", len(jobs)])
                jobs.clear()
                break
            seed, policy, fold = jobs.pop(0)
            run = output / policy / f"seed-{seed}" / fold
            if (run / "worker_completed.json").exists() and args.resume:
                verify_run(run, phash)
                event({"event": "verified_completed", "seed": seed, "policy": policy, "fold": fold})
                continue
            idle_gpus([gpu])
            temp = tempfile.mkdtemp(prefix=f"refinement-gpu{gpu}-", dir=ROOT / ".codex-tmp")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "TMPDIR": temp, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2"}
            cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--output", str(output),
                   "--seed", str(seed), "--policy", policy, "--fold-id", fold]
            if args.resume:
                cmd.append("--resume")
            log = (workers / f"{policy}--seed-{seed}--{fold}.log").open("a")
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = process, log, seed, policy, fold
            event({"event": "started", "pid": process.pid, "gpu": gpu, "seed": seed, "policy": policy, "fold": fold})
            print(f"GPU {gpu}: {policy}, {seed}, {fold}, PID {process.pid}", flush=True)
        for gpu, (process, log, seed, policy, fold) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            event({"event": "finished", "pid": process.pid, "gpu": gpu, "returncode": code,
                   "seed": seed, "policy": policy, "fold": fold})
            print(f"GPU {gpu}: {policy}, {seed}, {fold}, exit {code}", flush=True)
            if code:
                failures.append([seed, policy, fold, code])
            del active[gpu]
        if active:
            time.sleep(2)
    if failures:
        raise RuntimeError(f"Failed work preserved, inspect before resume: {failures}")
    final = output / ("completed_" + "_".join(stages) + ".json")
    if not final.exists():
        write_new_json(final, {"protocol_sha256": phash, "stages": stages, "completed_at": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("train")
    run.add_argument("--dataset", type=Path, default=Path("outputs/expert_cycle_001/learning_dataset_v1"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--gpus", default="2,3,4,5,6,7")
    run.add_argument("--stages", default="outer,inner,directed")
    run.add_argument("--resume", action="store_true")
    work = sub.add_parser("worker")
    work.add_argument("--output", type=Path, required=True)
    work.add_argument("--seed", type=int, required=True)
    work.add_argument("--policy", required=True)
    work.add_argument("--fold-id", required=True)
    work.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    (launch if args.command == "train" else worker)(args)


if __name__ == "__main__":
    main()
