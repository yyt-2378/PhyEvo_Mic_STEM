#!/usr/bin/env python3
"""Publish and execute a family-wide RSI admission; validate actual fallback.

The sealed experiment is read-only. Outputs must be new paths. Active mode does
not construct rejected programs; explicit shadow mode cannot own primary masks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from tfree_detection.autonomous_admission import (AdmittedRSIRuntime, load_admission,
    publish_admission, read, reference)
from tfree_detection.expert_supervision import contained_path, normalize_image, sha256_file, write_new_json
from tfree_detection.preprocessing import load_grayscale

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = "outputs/autonomous_rsi_20261015_v1"
DEFAULT_OUTPUT = "outputs/autonomous_rsi_admission_20261015_v1"


def publish(args):
    output = contained_path(ROOT, args.output)
    result = publish_admission(ROOT, args.study, args.release or str(Path(args.study) / "release_v1"),
                               output / "admission.json", arm=args.arm)
    print(json.dumps({"admission": str((output / 'admission.json').relative_to(ROOT)),
                      "selected_family": result["selected_family"], "gate": result["gate"]["reason"]}), flush=True)


def predict(args):
    if args.source is None or args.job_index is None:
        raise ValueError("A source and explicit configured job index are required")
    output = contained_path(ROOT, args.output)
    if output.exists():
        raise FileExistsError("New prediction output required")
    source = contained_path(ROOT, args.source)
    model = AdmittedRSIRuntime(ROOT, args.admission, args.job_index, mode=args.mode,
                               device=args.device, allow_experimental=True)
    image = load_grayscale(source, frame_index=args.frame_index, dataset_path=args.dataset_path)
    result = model.predict(normalize_image(image))
    output.mkdir(parents=True, exist_ok=False)
    arrays = output / "prediction.npz"
    with arrays.open("xb") as stream:
        np.savez_compressed(stream, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    write_new_json(output / "receipt.json", {**{k: v for k, v in result.items() if not isinstance(v, np.ndarray)},
        "source": reference(ROOT, source), "admission": reference(ROOT, args.admission),
        "prediction_sha256": sha256_file(arrays), "job_index": args.job_index,
        "frame_index": args.frame_index, "dataset_path": args.dataset_path})
    print(result["rsi_admission"], flush=True)


def worker(args):
    import torch
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    output = contained_path(ROOT, args.output); admission_path = contained_path(ROOT, args.admission)
    model = AdmittedRSIRuntime(ROOT, admission_path, args.job_index, device="cuda:0", allow_experimental=True)
    if model.gate["accepted"]:
        raise ValueError("This validation checks rejected-family preservation, not a claimed update gain")
    evidence = model.admission; study = ROOT / evidence["study"]
    protocol = read(study / "protocol.json")
    job = next(j for j in protocol["jobs"] if j["index"] == args.job_index and j["stage"] == "outer")
    baseline = read(study / "baseline/manifest.json")
    current = contained_path(ROOT, baseline["current_interface_release"])
    dataset = ROOT / "outputs/expert_cycle_001/learning_dataset_v1"
    entries = {e["case_number"]: e for e in read(dataset / "manifest.json")["entries"]}
    dest = output / "replay" / f"job-{args.job_index:02d}"; dest.mkdir(parents=True, exist_ok=False)
    records = []
    for n in job["test_cases"]:
        path = dataset / entries[n]["plane"]
        if sha256_file(path) != entries[n]["plane_sha256"]:
            raise ValueError("Expert-learning asset changed")
        with np.load(path, allow_pickle=False) as z:
            image = z["image"]
        prediction = model.predict(image)
        incumbent = model.base.predict(image)
        if model.candidate is not None or prediction["rsi_admission"]["candidate_evaluated"]:
            raise AssertionError("Rejected program was executed in active mode")
        if not np.array_equal(prediction["mask"], incumbent["mask"]):
            raise AssertionError("Active primary differs from live incumbent")
        with np.load(path, allow_pickle=False) as z:
            human = z["human"].astype(bool)
        old_path = current / "predictions" / f'case-{n:03d}-seed-{job["seed"]}.npz'
        with np.load(old_path, allow_pickle=False) as z:
            archived = z["mask"]
        saved = dest / f"case-{n:03d}.npz"
        with saved.open("xb") as stream:
            np.savez_compressed(stream, active=prediction["mask"], live_incumbent=incumbent["mask"])
        records.append({"case_number": n, "group": job["outer_group"], "seed": job["seed"],
            "active_equals_live_incumbent": True, "candidate_constructed": False,
            "archived_incumbent_differing_pixels": int((prediction["mask"] != archived).sum()),
            "coverage": int((prediction["mask"] & human).sum()) / max(1, int(human.sum())),
            "selected_pixels": int(prediction["mask"].sum()), "budget": prediction["budget"],
            "file": str(saved.relative_to(output)), "sha256": sha256_file(saved)})
        print("ACTIVE REPLAY", args.job_index, n, records[-1]["archived_incumbent_differing_pixels"], flush=True)
    # One explicit shadow check per configured model. Never choose examples by score.
    n = job["test_cases"][0]
    with np.load(dataset / entries[n]["plane"], allow_pickle=False) as z:
        image = z["image"]
    shadow = AdmittedRSIRuntime(ROOT, admission_path, args.job_index, mode="shadow", device="cuda:0", allow_experimental=True)
    paired = shadow.predict(image)
    with np.load(dest / f"case-{n:03d}.npz", allow_pickle=False) as z:
        active = z["active"]
    if not np.array_equal(paired["mask"], active) or "shadow_mask" not in paired:
        raise AssertionError("Explicit shadow did not preserve incumbent primary")
    shadow_file = dest / "shadow.npz"
    with shadow_file.open("xb") as stream:
        np.savez_compressed(stream, primary=paired["mask"], shadow=paired["shadow_mask"])
    write_new_json(dest / "result.json", {"job": job, "admission_sha256": sha256_file(admission_path),
        "records": records, "shadow": {"case_number": n, "primary_equals_active": True,
            "candidate_evaluated": paired["rsi_admission"]["candidate_evaluated"],
            "primary_shadow_differing_pixels": int((paired["mask"] != paired["shadow_mask"]).sum()),
            "file": str(shadow_file.relative_to(output)), "sha256": sha256_file(shadow_file)},
        "source_sha256": sha256_file(Path(__file__).resolve())})
    write_new_json(dest / "complete.json", {"result_sha256": sha256_file(dest / "result.json")})


def replay(args):
    output = contained_path(ROOT, args.output); a = load_admission(ROOT, args.admission)
    gpus = [int(g) for g in args.gpus.split(",")]
    if len(set(gpus)) != len(gpus) or not gpus:
        raise ValueError("Distinct explicit GPUs required")
    status = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    states = {int(line.split(",")[0]): [int(v.strip()) for v in line.split(",")[1:]] for line in status.splitlines()}
    if any(states[g][0] > 512 or states[g][1] > 5 for g in gpus):
        raise RuntimeError("Requested GPUs already occupied; existing processes untouched")
    logs = output / "workers"; logs.mkdir(exist_ok=False)
    pending = [b["job_index"] for b in a["bindings"]]; active = {}; failed = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            index = pending.pop(0); log = (logs / f"job-{index:02d}.log").open("x")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
                "TMPDIR": str(ROOT / ".codex-tmp")}
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
                "--admission", str(contained_path(ROOT, args.admission)), "--job-index", str(index)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (index, proc, log)
            with (logs / "processes.jsonl").open("a") as f:
                f.write(json.dumps({"pid": proc.pid, "gpu": gpu, "job_index": index}) + "\n")
            print("START", index, gpu, proc.pid, flush=True)
        for gpu, (index, proc, log) in list(active.items()):
            if proc.poll() is not None:
                log.close(); del active[gpu]
                if proc.returncode:
                    failed.append(index)
                print("FINISH", index, proc.returncode, flush=True)
        if active:
            time.sleep(2)
    write_new_json(logs / "execution.json", {"jobs": len(a["bindings"]), "failed": failed})
    if failed:
        raise RuntimeError(f"Failed validation workers retained: {failed}")


def verify(args):
    output = contained_path(ROOT, args.output); a = load_admission(ROOT, args.admission)
    if read(output / "workers/execution.json")["failed"]:
        raise ValueError("Incomplete GPU validation")
    rows = []; shadows = []
    for b in a["bindings"]:
        dest = output / "replay" / f'job-{b["job_index"]:02d}'
        r = read(dest / "result.json")
        if sha256_file(dest / "result.json") != read(dest / "complete.json")["result_sha256"]:
            raise ValueError("Replay receipt changed")
        if r["admission_sha256"] != sha256_file(contained_path(ROOT, args.admission)):
            raise ValueError("Replay used another admission")
        for row in r["records"]:
            path = contained_path(output, row["file"])
            if sha256_file(path) != row["sha256"]:
                raise ValueError("Saved active result changed")
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z["active"], z["live_incumbent"]) or int(z["active"].sum()) > row["budget"]:
                    raise ValueError("Live fallback result or budget mismatch")
            rows.append(row)
        s = r["shadow"]; path = contained_path(output, s["file"])
        if sha256_file(path) != s["sha256"]:
            raise ValueError("Shadow result changed")
        shadows.append(s)
    study = ROOT / a["study"]
    seal = read(study / "attempt_seal.json")
    for name, digest in seal["files"].items():
        if sha256_file(contained_path(study, name)) != digest:
            raise ValueError("Sealed autonomous experiment modified")
    baseline = read(study / "baseline/manifest.json")
    for key in ("protected_files", "source_files"):
        for name, digest in baseline[key].items():
            if sha256_file(contained_path(ROOT, name)) != digest:
                raise ValueError("Original best asset changed")
    summary = {"selected_family": a["selected_family"], "family_gate": a["gate"], "rows": rows,
        "active_planes_tested": len(rows), "all_active_equal_live_incumbent": all(r["active_equals_live_incumbent"] for r in rows),
        "rejected_candidate_constructions_in_active_mode": sum(int(r["candidate_constructed"]) for r in rows),
        "archived_offline_mask_differing_pixels": sum(r["archived_incumbent_differing_pixels"] for r in rows),
        "shadow_tests": shadows, "sealed_attempt_files_unchanged": len(seal["files"]),
        "original_best_assets_unchanged": len(baseline["protected_files"]),
        "original_source_files_unchanged": len(baseline["source_files"]),
        "claim": "runtime rejection and incumbent preservation verified; not a new performance gain or independent generalization result"}
    write_new_json(output / "verification.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "family_gate", "shadow_tests"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["publish", "predict", "replay", "worker", "verify"])
    parser.add_argument("--study", default=DEFAULT_STUDY); parser.add_argument("--release")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--admission", default=DEFAULT_OUTPUT + "/admission.json")
    parser.add_argument("--arm", default="feedback", choices=["feedback", "open_loop", "parameter_only"])
    parser.add_argument("--job-index", type=int); parser.add_argument("--mode", choices=["active", "shadow"], default="active")
    parser.add_argument("--source"); parser.add_argument("--frame-index", type=int); parser.add_argument("--dataset-path")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    args = parser.parse_args()
    {"publish": publish, "predict": predict, "replay": replay, "worker": worker, "verify": verify}[args.command](args)
