#!/usr/bin/env python3
"""Frozen multi-round autonomous program synthesis with matched controls.

Stages: plan -> GPU cache -> independent development searches -> seal all search
lineages -> one-time outer audit. No outer audit is returned to the proposer.
Existing models, data, protocols and successful artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from tfree_detection.autonomous_program import identity, render
from tfree_detection.autonomous_rsi import (prepare_case, case_coverage, fit_candidate,
                                          propose, promotion, summarize)
from tfree_detection.expert_student import predict_scores, restore_checkpoint, torch
from tfree_detection.expert_supervision import contained_path, sha256_file, write_new_json
from tfree_detection.interface_release import training_healthy
from tfree_detection.interface_reliability import qualify_boundary_or_band
from tfree_detection.interface_runtime import choose_interface

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "outputs/autonomous_rsi_20261015_v1"
BASE = "outputs/interface_improvement_20261015_v1"
DATA = "outputs/expert_cycle_001/learning_dataset_v1"
CODE = ["src/tfree_detection/autonomous_program.py", "src/tfree_detection/autonomous_rsi.py",
        "scripts/run_autonomous_rsi.py", "scripts/freeze_autonomous_rsi_baseline.py",
        "src/tfree_detection/expert_student.py", "src/tfree_detection/expert_supervision.py",
        "src/tfree_detection/interface_release.py", "src/tfree_detection/interface_reliability.py",
        "src/tfree_detection/interface_runtime.py", "src/tfree_detection/interface_geometry.py",
        "src/tfree_detection/interface_student.py", "src/tfree_detection/config.py",
        "src/tfree_detection/student.py", "src/tfree_detection/tasks/interface.py"]
ARMS = ("feedback", "open_loop", "parameter_only")


def read(p):
    return json.loads(p.read_text())


def checked(out):
    p = read(out / "protocol.json")
    for name, digest in p["source_hashes"].items():
        if sha256_file(contained_path(ROOT, name)) != digest:
            raise ValueError("Frozen autonomous code changed; do not resume this study")
    if sha256_file(out / "baseline/manifest.json") != p["baseline_manifest_sha256"]:
        raise ValueError("Frozen baseline manifest changed")
    if sha256_file(ROOT / DATA / "manifest.json") != p["data_manifest_sha256"]:
        raise ValueError("Expert dataset manifest changed")
    if sha256_file(ROOT / BASE / "physical_audit.json") != p["physical_audit_sha256"] or sha256_file(ROOT / BASE / "stability/audit.json") != p["stability_audit_sha256"]:
        raise ValueError("Physical evidence manifest changed")
    return p


def plan(out):
    baseline = read(out / "baseline/manifest.json")
    for name, digest in baseline["protected_files"].items():
        if sha256_file(contained_path(ROOT, name)) != digest:
            raise ValueError("Protected original changed before planning")
    old = read(ROOT / BASE / "retention_v1/protocol.json")
    current = read(ROOT / BASE / "release_v2/results.json")
    jobs = []
    for j in old["jobs"]:
        decision = next(s for s in current["selectors"] if s["outer_group"] == j["outer_group"])
        step = decision["selected"]["step"]
        ckpt = ROOT / BASE / "retention_v1/runs" / f'job-{j["index"]:02d}' / f"step-{step:06d}"
        meta = read(ckpt / "checkpoint.json")
        if set(j["train_groups"]) & {j["outer_group"], j["validation_group"]}:
            raise ValueError("Original source exclusion failed")
        if sha256_file(ckpt / "state.npz") != meta["state_sha256"]:
            raise ValueError("Original checkpoint altered")
        run = read(ckpt.parent / "evaluation.json")
        jobs.append({**j, "checkpoint": str(ckpt.relative_to(ROOT)), "selected_step": step,
                     "checkpoint_sha256": sha256_file(ckpt / "checkpoint.json"),
                     "state_sha256": meta["state_sha256"], "admitted": training_healthy(run, step)})
    searches = [{"index": i * len(ARMS) + k, "outer_group": group, "arm": arm}
                for i, group in enumerate(old["groups"]) for k, arm in enumerate(ARMS)]
    protocol = {"format": "phyevo-autonomous-program-rsi-v1", "jobs": jobs, "searches": searches,
        "groups": old["groups"], "seeds": [23, 41, 67], "rounds": 3, "candidates_per_round": 6,
        "pool_size": 32, "training_steps": 120, "program_resolution": 384, "area_fraction": .1,
        "minimum_macro_gain": .002, "source_regression_tolerance": 1e-8, "seed_regression_tolerance": .01,
        "metric": "case-macro known-positive coverage; then seeds and sources macro averaged",
        "source_hashes": {name: sha256_file(ROOT / name) for name in CODE},
        "baseline_manifest_sha256": sha256_file(out / "baseline/manifest.json"),
        "data_manifest_sha256": sha256_file(ROOT / DATA / "manifest.json"),
        "physical_audit_sha256": sha256_file(ROOT / BASE / "physical_audit.json"),
        "stability_audit_sha256": sha256_file(ROOT / BASE / "stability/audit.json"),
        "synthesis_backend": "typed spatial program synthesis with gradient-trained residual readouts; no LLM",
        "mutable": ["composed spatial expression", "generated training hyperparameters", "residual coefficients"],
        "immutable": ["labels", "base models", "area budget", "validity", "evaluation", "promotion gate", "point path"],
        "development_feedback": "two remaining acquisition groups; pair-excluded base students; inner cross-source fits",
        "outer_audit": "read only after ALL search lineages are sealed; no audit feedback to search",
        "historical_base_and_physics_are_development_informed": True,
        "new_blind_external_test": False, "production_approved": False,
        "control_budget": "same proposal-pool count, trained-candidate count, steps, seeds and validation queries; measured wall times also reported",
        "point_branches": "protected unchanged historical replay; not newly trained"}
    write_new_json(out / "protocol.json", protocol)
    with ZipFile(out / "controller_source.zip", "x", compression=ZIP_DEFLATED) as z:
        for name in CODE:
            z.write(ROOT / name, name)
    print("PLANNED", len(jobs), "CACHE JOBS", len(searches), "SEARCHES", flush=True)


def plane(entry):
    path = ROOT / DATA / entry["plane"]
    if sha256_file(path) != entry["plane_sha256"]:
        raise ValueError("Expert input changed")
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def entries():
    return {e["case_number"]: e for e in read(ROOT / DATA / "manifest.json")["entries"]}


def cache_worker(out, index):
    p = checked(out); job = p["jobs"][index]
    target = out / "cache" / f"job-{index:02d}"
    target.mkdir(parents=True, exist_ok=False)
    ckpt = contained_path(ROOT, job["checkpoint"])
    if sha256_file(ckpt / "checkpoint.json") != job["checkpoint_sha256"] or sha256_file(ckpt / "state.npz") != job["state_sha256"]:
        raise ValueError("Bound base model changed")
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    model, _, _, _ = restore_checkpoint(ckpt, device="cuda:0")
    known = entries(); rows = []
    prior = read(ckpt.parent / "evaluation.json")
    old_records = {r["case_number"]: r for r in prior["records"] if r["step"] == job["selected_step"]}
    for n in sorted(set(job["train_cases"] + job["test_cases"])):
        data = plane(known[n])
        old = ckpt / "scores" / f"case-{n:03d}.npz"
        if n in old_records:
            if sha256_file(old) != old_records[n]["sha256"]:
                raise ValueError("Saved base score changed")
            with np.load(old, allow_pickle=False) as z:
                score = z["score"]
            origin = "existing_checksum_verified_score"
        else:
            score = predict_scores(model, data["image"], 4, device="cuda:0", normalized=True)
            origin = "frozen_checkpoint_inference"
        dest = target / f"case-{n:03d}.npz"
        with dest.open("xb") as f:
            np.savez_compressed(f, score=score)
        rows.append({"case_number": n, "sha256": sha256_file(dest), "origin": origin})
    write_new_json(target / "complete.json", {"job": job, "files": rows,
        "protocol_sha256": sha256_file(out / "protocol.json"), "gpu": torch.cuda.get_device_name(0)})
    print("CACHE DONE", index, len(rows), flush=True)


def load_job(out, p, job, *, permit_outer=False):
    if job["stage"] == "outer" and not permit_outer:
        raise ValueError("Outer data withheld from proposer")
    target = out / "cache" / f'job-{job["index"]:02d}'
    receipt = read(target / "complete.json")
    if receipt["job"] != job or receipt["protocol_sha256"] != sha256_file(out / "protocol.json"):
        raise ValueError("Cache binding changed")
    known = entries()
    guards = {r["case_number"]: r["guard"] for r in read(ROOT / BASE / "stability/audit.json")["records"]}
    physical_hashes = {r["case_number"]: r["sha256"] for r in read(ROOT / BASE / "physical_audit.json")["records"]}
    cases = {}
    for record in receipt["files"]:
        n = record["case_number"]
        if job["stage"] == "inner" and known[n]["group"] == job["outer_group"]:
            raise ValueError("Outer sample in development view")
        data = plane(known[n]); score_file = target / f"case-{n:03d}.npz"
        if sha256_file(score_file) != record["sha256"]:
            raise ValueError("Cache score changed")
        with np.load(score_file, allow_pickle=False) as z:
            student = z["score"]
        ph = ROOT / BASE / "physical" / f"case-{n:03d}.npz"
        if sha256_file(ph) != physical_hashes[n]:
            raise ValueError("Physical field changed")
        with np.load(ph, allow_pickle=False) as z:
            physical = {key: z[key] for key in z.files}
        guard = qualify_boundary_or_band(physical, guards[n])
        mask, decision = choose_interface(physical["score"], physical["valid"], guard, student,
            student_admitted=job["admitted"], budget=int(np.ceil(.1 * student.size)))
        is_physical = decision["branch"] != "student"
        raw = physical["score"] if is_physical else student
        allowed = (raw > 0) & (physical["valid"].astype(bool) if is_physical else np.ones(raw.shape, bool))
        case = prepare_case(n, known[n]["group"], data["image"], physical, student, raw, allowed, data["human"],
                            physical_route=is_physical, band=guard["hypothesis"] == "enclosed_texture_band",
                            device="cuda:0", resolution=p["program_resolution"])
        _, replay = case_coverage(case, [])
        if not np.array_equal(replay.cpu().numpy(), mask):
            raise ValueError("Normalized baseline order changed; cannot compare")
        cases[n] = case
    return cases


def evaluate_jobs(jobs, data, states, *, destination=None):
    rows = []
    for job in jobs:
        stages = states[str(job["index"])]
        for n in job["test_cases"]:
            case = data[job["index"]][n]
            coverage, mask = case_coverage(case, stages)
            row = {"case_number": n, "group": case.group, "seed": job["seed"],
                   "coverage": coverage, "selected_pixels": int(mask.sum()),
                   "budget": int(np.ceil(.1 * mask.numel()))}
            if destination is not None:
                dest = destination / f'case-{n:03d}-seed-{job["seed"]}.npz'
                with dest.open("xb") as f:
                    np.savez_compressed(f, mask=mask.cpu().numpy())
                row.update(prediction=str(dest.relative_to(destination)), sha256=sha256_file(dest))
            rows.append(row)
    return rows


def search_worker(out, index):
    p = checked(out); search = p["searches"][index]; outer = search["outer_group"]; arm = search["arm"]
    target = out / "searches" / f"search-{index:02d}"
    target.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    jobs = [j for j in p["jobs"] if j["stage"] == "inner" and j["outer_group"] == outer]
    if len(jobs) != 6:
        raise ValueError("Expected two source folds x three seeds")
    data = {j["index"]: load_job(out, p, j) for j in jobs}
    states = {str(j["index"]): [] for j in jobs}
    baseline_rows = evaluate_jobs(jobs, data, states); parent_rows = baseline_rows
    write_new_json(target / "baseline.json", {"rows": baseline_rows, "summary": summarize(baseline_rows)})
    history = []; lineage = []; start = time.monotonic()
    # Same initial random stream for matched arms. Feedback changes selection
    # and future grammar ancestry, not image/label access permissions.
    rng = np.random.default_rng(20261015 + p["groups"].index(outer))
    for iteration in range(1, p["rounds"] + 1):
        checked(out)
        round_dir = target / f"round-{iteration:02d}"; round_dir.mkdir()
        fit_parents = states if arm == "feedback" else {str(j["index"]): [] for j in jobs}
        critic_cases = []; critic_parents = []
        for job in jobs:
            if job["seed"] == p["seeds"][0]:
                for n in job["test_cases"]:
                    critic_cases.append(data[job["index"]][n]); critic_parents.append(fit_parents[str(job["index"])])
        candidates, proposal = propose(critic_cases, critic_parents, history, rng, arm=arm,
                                      count=p["candidates_per_round"], pool_size=p["pool_size"])
        write_new_json(round_dir / "proposals.json", proposal)
        trials = []
        for ci, candidate in enumerate(candidates):
            new_states = deepcopy(fit_parents)
            rows = []
            try:
                for job in jobs:
                    jj = str(job["index"])
                    training = [data[job["index"]][n] for n in job["train_cases"]]
                    if outer in {c.group for c in training} or job["validation_group"] in {c.group for c in training}:
                        raise ValueError("Candidate learner source leakage")
                    stage = fit_candidate(training, [fit_parents[jj]] * len(training), candidate,
                                          seed=job["seed"], steps=p["training_steps"])
                    new_states[jj].append(stage)
                rows = evaluate_jobs(jobs, data, new_states)
                verdict = promotion(rows, parent_rows, baseline_rows, minimum_gain=p["minimum_macro_gain"],
                                    source_regression=p["source_regression_tolerance"], seed_regression=p["seed_regression_tolerance"])
            except (FloatingPointError, ValueError) as error:
                # Numerical algorithm failures are rejected trials. Integrity,
                # leakage and malformed-evaluation errors abort the experiment.
                if not isinstance(error, FloatingPointError) and str(error) not in {
                    "Nonfinite or flat candidate ranking", "Non-finite synthesized feature"}:
                    raise
                verdict = {"accepted": False, "reason": "candidate_numerical_failure",
                           "error": str(error), "macro_gain": None}
            trial = {"candidate_index": ci, "candidate": candidate, "states": new_states, "rows": rows,
                     "verdict": verdict, "parent_lineage": list(lineage), "elapsed_seconds": time.monotonic() - start}
            write_new_json(round_dir / f"candidate-{ci:02d}.json", trial)
            for pi, program in enumerate(candidate["programs"]):
                with (round_dir / f"candidate-{ci:02d}-feature-{pi:02d}.dsl.py").open("x") as f:
                    f.write(render(program))
            trials.append(trial)
            print("TRIAL", index, iteration, ci, verdict["accepted"], verdict["macro_gain"], flush=True)
        good = [t for t in trials if t["verdict"]["accepted"]]
        winner = max(good, key=lambda t: (t["verdict"]["macro_gain"], -t["candidate_index"])) if good else None
        if winner is not None:
            states = winner["states"]; parent_rows = winner["rows"]
            if arm != "feedback":
                lineage = []
            lineage.append({"round": iteration, "candidate_index": winner["candidate_index"], "candidate": winner["candidate"]})
        for trial in trials:
            history.append({"round": iteration, "candidate_index": trial["candidate_index"],
                "programs": trial["candidate"]["programs"], "accepted": trial is winner,
                "macro_gain": trial["verdict"]["macro_gain"], "rejection": None if trial is winner else trial["verdict"]})
        write_new_json(round_dir / "decision.json", {"winner": None if winner is None else winner["candidate_index"],
            "lineage": lineage, "states": states, "rows": parent_rows, "history": history,
            "protocol_sha256": sha256_file(out / "protocol.json"), "outer_labels_accessed": False})
        print("ROUND CLOSED", index, iteration, "WINNER", None if winner is None else winner["candidate_index"], flush=True)
    files = {str(f.relative_to(target)): sha256_file(f) for f in target.rglob("*") if f.is_file()}
    write_new_json(target / "closed.json", {"search": search, "lineage": lineage, "states": states,
        "history": history, "rows": parent_rows, "files": files, "outer_labels_accessed": False,
        "human_roundwise_interventions": 0, "elapsed_seconds": time.monotonic() - start,
        "protocol_sha256": sha256_file(out / "protocol.json")})
    print("SEARCH DONE", index, len(lineage), flush=True)


def seal_searches(out):
    p = checked(out); hashes = {}
    for search in p["searches"]:
        dest = out / "searches" / f'search-{search["index"]:02d}' / "closed.json"
        record = read(dest)
        if record["search"] != search or record["outer_labels_accessed"]:
            raise ValueError("Search receipt mismatch")
        for name, digest in record["files"].items():
            if sha256_file(contained_path(dest.parent, name)) != digest:
                raise ValueError("Search modified after completion")
        hashes[str(dest.relative_to(out))] = sha256_file(dest)
    write_new_json(out / "search_seal.json", {"files": hashes, "protocol_sha256": sha256_file(out / "protocol.json"),
        "all_searches_closed_before_any_outer_audit": True})


def audit_worker(out, index):
    p = checked(out); seal = read(out / "search_seal.json")
    for name, digest in seal["files"].items():
        if sha256_file(contained_path(out, name)) != digest:
            raise ValueError("Search seal changed")
    search = p["searches"][index]
    closed = read(out / "searches" / f"search-{index:02d}" / "closed.json")
    target = out / "audit" / f"search-{index:02d}"; target.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    jobs = [j for j in p["jobs"] if j["stage"] == "outer" and j["outer_group"] == search["outer_group"]]
    data = {j["index"]: load_job(out, p, j, permit_outer=True) for j in jobs}
    states = {str(j["index"]): [] for j in jobs}
    base_rows = evaluate_jobs(jobs, data, states)
    for element in closed["lineage"]:
        for job in jobs:
            jj = str(job["index"]); training = [data[job["index"]][n] for n in job["train_cases"]]
            if search["outer_group"] in {c.group for c in training}:
                raise ValueError("Outer label entered refit")
            stage = fit_candidate(training, [states[jj]] * len(training), element["candidate"],
                                  seed=job["seed"], steps=p["training_steps"])
            states[jj].append(stage)
    predictions = target / "predictions"; predictions.mkdir()
    rows = evaluate_jobs(jobs, data, states, destination=predictions)
    write_new_json(target / "result.json", {"search": search, "lineage": closed["lineage"], "states": states,
        "baseline_rows": base_rows, "rows": rows, "baseline": summarize(base_rows), "summary": summarize(rows),
        "outer_results_returned_to_proposer": False, "search_seal_sha256": sha256_file(out / "search_seal.json"),
        "production_approved": False, "historical_baseline_development_informed": True})
    write_new_json(target / "complete.json", {"result_sha256": sha256_file(target / "result.json")})
    print("AUDIT DONE", index, summarize(rows), flush=True)


def launch(out, stage, gpus):
    p = checked(out)
    if len(gpus) != len(set(gpus)):
        raise ValueError("Repeated GPU")
    status = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    devices = {int(r.split(",")[0]): [int(v.strip()) for v in r.split(",")[1:]] for r in status.splitlines()}
    if any(devices[g][0] > 512 or devices[g][1] > 5 for g in gpus):
        raise RuntimeError("Requested GPU already busy; no existing process will be touched")
    logs = out / f"workers-{stage}"; logs.mkdir(exist_ok=False)
    jobs = p["jobs"] if stage == "cache" else p["searches"]
    pending = [j["index"] for j in jobs]; active = {}; failures = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            index = pending.pop(0); stream = (logs / f"job-{index:02d}.log").open("x")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
                   "TMPDIR": str(ROOT / ".codex-tmp")}
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), stage + "-worker", "--output", str(out),
                                     "--index", str(index)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[gpu] = (index, proc, stream)
            with (logs / "processes.jsonl").open("a") as f:
                f.write(json.dumps({"gpu": gpu, "pid": proc.pid, "job": index, "stage": stage}) + "\n")
            print("START", stage, index, "GPU", gpu, "PID", proc.pid, flush=True)
        for gpu, (index, proc, stream) in list(active.items()):
            if proc.poll() is not None:
                stream.close(); del active[gpu]
                print("FINISH", stage, index, "EXIT", proc.returncode, flush=True)
                if proc.returncode:
                    failures.append(index)
        if active:
            time.sleep(2)
    write_new_json(logs / "execution.json", {"stage": stage, "jobs": len(jobs), "failed": failures})
    if failures:
        raise RuntimeError(f"Failed workers retained: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "cache", "cache-worker", "search", "search-worker", "seal-searches", "audit", "audit-worker"])
    parser.add_argument("--output", default=DEFAULT); parser.add_argument("--index", type=int)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    args = parser.parse_args(); output = contained_path(ROOT, args.output)
    if args.command == "plan":
        plan(output)
    elif args.command == "seal-searches":
        seal_searches(output)
    elif args.command.endswith("-worker"):
        {"cache-worker": cache_worker, "search-worker": search_worker, "audit-worker": audit_worker}[args.command](output, args.index)
    else:
        launch(output, args.command, [int(g) for g in args.gpus.split(",")])
