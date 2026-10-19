#!/usr/bin/env python3
"""Read-only independent mask/metric replay; writes only new verification files."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np
import torch

from tfree_detection.autonomous_program import identity, render
from tfree_detection.autonomous_rsi import scores, promotion, summarize
from tfree_detection.expert_supervision import contained_path, sha256_file, write_new_json

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def same_json(left, right):
    """Match the stored JSON domain exactly (tuples serialize as arrays)."""
    return json.loads(json.dumps(left, allow_nan=False)) == right


def runner():
    spec = importlib.util.spec_from_file_location("frozen_autonomous_runner", ROOT / "scripts/run_autonomous_rsi.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def integrity(out):
    module = runner(); p = module.checked(out)
    baseline = read(out / "baseline/manifest.json"); counts = {}
    for key in ("protected_files", "source_files"):
        for name, digest in baseline[key].items():
            require(sha256_file(contained_path(ROOT, name)) == digest, f"Baseline {key} changed: {name}")
        counts[key] = len(baseline[key])
    for name, digest in baseline["archives"].items():
        archive = out / "baseline" / name
        require(sha256_file(archive) == digest, "Baseline archive hash changed")
        with ZipFile(archive) as z:
            require(z.testzip() is None, "Corrupt recovery archive")
    files = 0
    for job in p["jobs"]:
        directory = out / "cache" / f'job-{job["index"]:02d}'
        receipt = read(directory / "complete.json")
        require(receipt["job"] == job, "Cache job changed")
        require({r["case_number"] for r in receipt["files"]} == set(job["train_cases"] + job["test_cases"]), "Incomplete cache")
        meta = read(contained_path(ROOT, job["checkpoint"]) / "checkpoint.json")
        require(meta["train_groups"] == job["train_groups"] and meta["train_cases"] == job["train_cases"], "Checkpoint split differs")
        require(not set(meta["train_groups"]) & {job["outer_group"], job["validation_group"]}, "Checkpoint source contamination")
        for row in receipt["files"]:
            require(sha256_file(directory / f'case-{row["case_number"]:03d}.npz') == row["sha256"], "Cached prediction changed")
            files += 1
    result = {"baseline_checks": counts, "recovery_archives_valid": True, "cache_jobs": len(p["jobs"]),
              "cache_score_files": files, "frozen_controller_sources": len(p["source_hashes"]),
              "protocol_sha256": sha256_file(out / "protocol.json"), "datasets_or_originals_modified": False}
    write_new_json(out / "integrity_verification.json", result)
    print(json.dumps(result), flush=True)


def independent_rows(jobs, data, states):
    rows = []
    for job in jobs:
        for n in job["test_cases"]:
            case = data[job["index"]][n]
            value = scores(case, states[str(job["index"])], full=True).detach().cpu().numpy()
            allowed = case.allowed.cpu().numpy(); human = case.human.cpu().numpy()
            require(np.isfinite(value).all(), "Nonfinite replay score")
            budget = int(np.ceil(.1 * value.size))
            indices = np.flatnonzero(allowed)
            selected = indices[np.argsort(-value.ravel()[indices], kind="stable")[:budget]]
            mask = np.zeros(value.shape, bool); mask.ravel()[selected] = True
            rows.append({"case_number": n, "group": case.group, "seed": job["seed"],
                         "coverage": int((mask & human).sum()) / max(1, int(human.sum())),
                         "selected_pixels": int(mask.sum()), "budget": budget})
    return rows


def replay(out, index):
    module = runner(); p = module.checked(out)
    seal = read(out / "search_seal.json")
    for name, digest in seal["files"].items():
        require(sha256_file(contained_path(out, name)) == digest, "Search seal mismatch")
    search = p["searches"][index]; directory = out / "searches" / f"search-{index:02d}"
    closed = read(directory / "closed.json")
    for name, digest in closed["files"].items():
        require(sha256_file(contained_path(directory, name)) == digest, "Trial history modified")
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    jobs = [j for j in p["jobs"] if j["stage"] == "inner" and j["outer_group"] == search["outer_group"]]
    data = {j["index"]: module.load_job(out, p, j) for j in jobs}
    states = {str(j["index"]): [] for j in jobs}
    baseline = independent_rows(jobs, data, states); parent = baseline; trials = 0; recomputed = 0
    require(baseline == read(directory / "baseline.json")["rows"], "Baseline replay mismatch")
    for iteration in range(1, p["rounds"] + 1):
        rd = directory / f"round-{iteration:02d}"
        proposal = read(rd / "proposals.json")
        require(len(proposal["pool"]) == p["pool_size"], "Proposal budget mismatch")
        require(set(proposal["evidence"]["groups"]) == set(p["groups"]) - {search["outer_group"]}, "Proposer sees wrong source set")
        fit_parents = states if search["arm"] == "feedback" else {str(j["index"]): [] for j in jobs}
        eligible = []
        for ci in range(p["candidates_per_round"]):
            trial = read(rd / f"candidate-{ci:02d}.json"); trials += 1
            for pi, node in enumerate(trial["candidate"]["programs"]):
                require((rd / f"candidate-{ci:02d}-feature-{pi:02d}.dsl.py").read_text() == render(node), "Generated source not bound to actual program")
            if trial["verdict"]["reason"] == "candidate_numerical_failure":
                require(not trial["verdict"]["accepted"], "Invalid trial promoted")
                continue
            for job in jobs:
                jj = str(job["index"]); stages = trial["states"][jj]
                require(stages[:-1] == fit_parents[jj], "Candidate did not inherit recorded parent")
                require(stages[-1]["train_groups"] == job["train_groups"], "Readout training source mismatch")
                require(stages[-1]["train_cases"] == sorted(job["train_cases"]), "Readout training cases mismatch")
                require(stages[-1]["programs"] == trial["candidate"]["programs"], "Different program trained")
                require(stages[-1]["steps"] == p["training_steps"], "Training budget changed")
            rows = independent_rows(jobs, data, trial["states"]); recomputed += len(rows)
            require(rows == trial["rows"], "Independent NumPy mask/metric replay mismatch")
            verdict = promotion(rows, parent, baseline, minimum_gain=p["minimum_macro_gain"],
                                source_regression=p["source_regression_tolerance"], seed_regression=p["seed_regression_tolerance"])
            require(same_json(verdict, trial["verdict"]), "Promotion decision mismatch")
            if verdict["accepted"]:
                eligible.append(trial)
        winner = max(eligible, key=lambda t: (t["verdict"]["macro_gain"], -t["candidate_index"])) if eligible else None
        decision = read(rd / "decision.json")
        require(decision["winner"] == (None if winner is None else winner["candidate_index"]), "Winner mismatch")
        if winner:
            states = winner["states"]; parent = winner["rows"]
        require(states == decision["states"] and parent == decision["rows"], "Recorded parent state mismatch")
    require(states == closed["states"], "Final lineage mismatch")
    dest = out / "verification"; dest.mkdir(exist_ok=True)
    result = {"search": search, "candidate_trials": trials, "metric_rows_independently_recomputed": recomputed,
              "all_masks_match_numpy_stable_topk": True, "parent_chain_verified": True,
              "source_exclusion_verified": True, "governor_replayed": True,
              "closed_sha256": sha256_file(directory / "closed.json")}
    write_new_json(dest / f"search-{index:02d}.json", result)
    print(json.dumps(result), flush=True)


def launch(out, gpus, attempt):
    p = runner().checked(out)
    read(out / "search_seal.json")
    status = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    devices = {int(line.split(",")[0]): [int(x.strip()) for x in line.split(",")[1:]] for line in status.splitlines()}
    require(len(set(gpus)) == len(gpus), "Repeated GPU")
    require(all(devices[g][0] <= 512 and devices[g][1] <= 5 for g in gpus), "Requested GPU occupied")
    if not attempt or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in attempt):
        raise ValueError("Invalid verification attempt identifier")
    dest = contained_path(out, "workers-verification-" + attempt); dest.mkdir(exist_ok=False)
    pending = [s["index"] for s in p["searches"]]; active = {}; failed = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            index = pending.pop(0); log = (dest / f"search-{index:02d}.log").open("x")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
                   "TMPDIR": str(ROOT / ".codex-tmp")}
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "replay", "--output", str(out),
                                     "--index", str(index)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (index, proc, log)
            with (dest / "processes.jsonl").open("a") as f:
                f.write(json.dumps({"pid": proc.pid, "gpu": gpu, "index": index}) + "\n")
            print("VERIFY START", index, gpu, proc.pid, flush=True)
        for gpu, (index, proc, log) in list(active.items()):
            if proc.poll() is not None:
                log.close(); del active[gpu]
                if proc.returncode:
                    failed.append(index)
                print("VERIFY FINISH", index, proc.returncode, flush=True)
        if active:
            time.sleep(2)
    write_new_json(dest / "execution.json", {"jobs": len(p["searches"]), "failed": failed})
    require(not failed, f"Verification failures retained: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["integrity", "replay", "launch"])
    parser.add_argument("--output", default="outputs/autonomous_rsi_20261015_v1"); parser.add_argument("--index", type=int)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--attempt", default="v2")
    a = parser.parse_args(); out = contained_path(ROOT, a.output)
    if a.command == "integrity":
        integrity(out)
    elif a.command == "replay":
        replay(out, a.index)
    else:
        launch(out, [int(g) for g in a.gpus.split(",")], a.attempt)
