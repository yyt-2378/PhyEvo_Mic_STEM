#!/usr/bin/env python3
"""Versioned interface geometry/ranking experiments, no writes to old assets.

prepare: materialize input-only geometry evidence and audit legacy reproduction.
plan: freeze source/data hashes and source-held-out, matched-control GPU jobs.
worker/launch: train immutable specialists; all expert annotations stay read-only.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from tfree_detection.config import load_config
from tfree_detection.context import DetectionContext
from tfree_detection.expert_student import (F, expert_adaptation_loss, make_expert_student,
    predict_scores, save_checkpoint, torch)
from tfree_detection.expert_supervision import contained_path, load_learning_dataset, sha256_file, write_new_json
from tfree_detection.interface_geometry import ranked_support
from tfree_detection.interface_student import InterfaceSampler, physical_ranking_loss, score_health
from tfree_detection.rsi_refinement import RefinementPolicy
from tfree_detection.tasks.interface import PhaseBoundaryTask

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "outputs/interface_improvement_20261015_v1"
DATASET = "outputs/expert_cycle_001/learning_dataset_v1"
SOURCES = ["scripts/run_interface_improvement.py", "src/tfree_detection/interface_geometry.py",
           "src/tfree_detection/interface_student.py", "src/tfree_detection/tasks/interface.py",
           "src/tfree_detection/config.py", "src/tfree_detection/expert_student.py",
           "src/tfree_detection/expert_supervision.py", "src/tfree_detection/student.py",
           "src/tfree_detection/rsi_refinement.py", "configs/phase_boundary_geometry_aware.yaml",
           "configs/phase_boundary.yaml"]


def read(path):
    return json.loads(path.read_text())


def coverage(mask, human):
    return float(np.count_nonzero(mask & human) / max(1, np.count_nonzero(human)))


def prepare(output):
    if output.exists():
        raise FileExistsError("Use a new experiment output; no overwrite")
    manifest, arrays = load_learning_dataset(ROOT, ROOT / DATASET)
    output.mkdir(parents=True)
    cache = output / "physical"; cache.mkdir()
    records = []
    legacy_config = load_config(ROOT / "configs/phase_boundary.yaml")
    new_config = load_config(ROOT / "configs/phase_boundary_geometry_aware.yaml")
    for entry in manifest["entries"]:
        if entry["geometry"] != "interface":
            continue
        n = entry["case_number"]; plane = arrays[n]
        old = PhaseBoundaryTask().run(plane["image"], legacy_config, {}, DetectionContext())
        result = PhaseBoundaryTask().run(plane["image"], new_config, {}, DetectionContext())
        if not np.array_equal(old.mask, plane["teacher"]):
            raise ValueError(f"Legacy physical mask did not reproduce for case {n}")
        valid = result.evidence_maps["boundary_validity"] > .5
        budget = int(np.ceil(.1 * plane["image"].size))
        support = ranked_support(result.score_map, budget, valid=valid)
        path = cache / f"case-{n:03d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, score=result.score_map, mask=result.mask,
                legacy_score=old.score_map, legacy_mask=old.mask, valid=valid,
                labels=result.evidence_maps["domain_label"],
                variation=result.evidence_maps["reciprocal_signature_variation"])
        info = {"case_number": n, "group": entry["group"], "file": str(path.relative_to(output)),
            "sha256": sha256_file(path), "geometry": result.metadata["geometry_interpretation"],
            "legacy_mask_exact": True, "score_unchanged": bool(np.array_equal(old.score_map,result.score_map)),
            "native_coverage": coverage(old.mask, plane["human"]),
            "old_dense_at_budget": coverage(ranked_support(old.score_map,budget,valid=valid),plane["human"]),
            "new_dense_at_budget": coverage(support, plane["human"]),
            "selected_pixels": int(support.sum()), "budget": budget}
        records.append(info)
        print(n, info["geometry"]["hypothesis"], round(info["new_dense_at_budget"], 4), flush=True)
    write_new_json(output / "physical_audit.json", {"records": records,
        "source_hashes": {p: sha256_file(ROOT/p) for p in SOURCES},
        "data_manifest_sha256": sha256_file(ROOT/DATASET/"manifest.json"),
        "annotations_edited": False, "budget_fraction": .1,
        "interpretation": "post-hoc development evaluation; not a locked external test"})


def plan(output):
    audit = read(output/"physical_audit.json")
    manifest, _ = load_learning_dataset(ROOT, ROOT/DATASET)
    entries = [e for e in manifest["entries"] if e["geometry"] == "interface"]
    groups = sorted({e["group"] for e in entries})
    jobs = []
    for policy in ("control", "physical_rank"):
        for count in (1, 2):
            for train_groups in combinations(groups, count):
                for seed in (23, 41, 67):
                    jobs.append({"index": len(jobs), "policy": policy, "seed": seed,
                        "train_groups": list(train_groups),
                        "train_cases": [e["case_number"] for e in entries if e["group"] in train_groups],
                        "test_cases": [e["case_number"] for e in entries if e["group"] not in train_groups]})
    protocol = {"format": "phyevo-interface-improvement-v1", "jobs": jobs, "groups": groups,
        "source_hashes": {p: sha256_file(ROOT/p) for p in SOURCES},
        "physical_audit_sha256": sha256_file(output/"physical_audit.json"),
        "data_manifest_sha256": audit["data_manifest_sha256"],
        "steps": 800, "checkpoints": [200,400,800], "batch_size": 16, "patch_size":128,
        "base_channels": 16, "learning_rate": .001, "area_budget": .1,
        "selection": "for outer group G, inner predictions on H from a model trained on neither G nor H; equal-source means, 3 seeds; all-high checkpoints vetoed; no outer-label selection",
        "geometry_role": "input-only A|B|A topology; legacy path preserved otherwise",
        "training_targets": "physical variation ordering in unchanged weak ROI; qualified far low-variation weak references; not all unknown pixels negative",
        "specialist_scope": "interface only; point checkpoints and point code untouched",
        "claims": {"blind_external_test": False, "precision_or_iou": False,
                   "guaranteed_RSI": False, "expert_labels_changed": False}}
    write_new_json(output/"protocol.json", protocol)
    print("PLANNED", len(jobs), "jobs", flush=True)


def checked(output):
    p = read(output/"protocol.json")
    if p["source_hashes"] != {s: sha256_file(ROOT/s) for s in SOURCES}:
        raise ValueError("Producer changed after freeze; create a new protocol")
    if sha256_file(output/"physical_audit.json") != p["physical_audit_sha256"]:
        raise ValueError("Evidence audit changed")
    if sha256_file(ROOT/DATASET/"manifest.json") != p["data_manifest_sha256"]:
        raise ValueError("Learning manifest changed")
    return p


def worker(output, index):
    p = checked(output); job = p["jobs"][index]
    target = output/"runs"/f"job-{index:02d}"
    if target.exists():
        raise FileExistsError("Refusing to overwrite existing run")
    manifest, arrays = load_learning_dataset(ROOT, ROOT/DATASET)
    entries = {e["case_number"]: e for e in manifest["entries"]}
    training = [entries[n] for n in job["train_cases"]]
    if set(job["train_cases"]) & set(job["test_cases"]):
        raise ValueError("Source-held-out isolation violated")
    physical = {}
    for record in read(output/"physical_audit.json")["records"]:
        if record["case_number"] in job["train_cases"]:
            path = contained_path(output, record["file"])
            if sha256_file(path) != record["sha256"]:
                raise ValueError("Physical input changed")
            with np.load(path, allow_pickle=False) as archive:
                physical[record["case_number"]] = {k: archive[k] for k in archive.files}
    target.mkdir(parents=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.manual_seed(job["seed"])
    device = "cuda:0"
    model = make_expert_student(p["base_channels"]).to(device)
    optimizer = torch.optim.Adam([v for v in model.parameters() if v.requires_grad], lr=p["learning_rate"])
    rng = np.random.default_rng(job["seed"])
    sampler_policy = RefinementPolicy("interface_matched", source_power=0,
                                      positive_crop_probability=.75, photometric=True)
    sampler = InterfaceSampler(training, arrays, sampler_policy, physical,
        excluded_groups=set(p["groups"])-set(job["train_groups"]))
    began = time.monotonic(); evaluations = []
    for step in range(1, p["steps"]+1):
        model.train()
        batch, _ = sampler.sample(rng, batch_size=p["batch_size"], patch_size=p["patch_size"],
                                  device=device, augmentation_seed=job["seed"]*1000000+step)
        optimizer.zero_grad(set_to_none=True)
        output_batch = model(batch["image"])
        if job["policy"] == "physical_rank":
            smaller = F.interpolate(batch["image"], scale_factor=.5, mode="bilinear", align_corners=False)
            loss, parts = physical_ranking_loss(output_batch, batch, model(smaller))
        else:
            loss, parts = expert_adaptation_loss(output_batch, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if step == 1 or step % 100 == 0:
            event = {"step":step, "seconds":time.monotonic()-began,
                     **{k:float(v.detach().cpu()) for k,v in parts.items()}}
            with (target/"events.jsonl").open("a") as stream:
                stream.write(json.dumps(event,allow_nan=False)+"\n")
            print("JOB",index,job["policy"],step,"loss",round(event["total"],4),flush=True)
        if step in p["checkpoints"]:
            directory = target/f"step-{step:06d}"
            save_checkpoint(directory,model,optimizer,rng,{**job,"step":step,
                "base_channels":p["base_channels"],"learning_rate":p["learning_rate"],
                "protocol_sha256":sha256_file(output/"protocol.json"),"sampler_audit":sampler.audit,
                "annotation_checksum":p["data_manifest_sha256"]})
            scores = directory/"scores"; scores.mkdir()
            # Same deterministic order-spaced training diagnostics for both losses.
            reps=[]
            for group in job["train_groups"]:
                group_cases=[e["case_number"] for e in training if e["group"]==group]
                reps += [group_cases[int(i)] for i in np.unique(np.linspace(0,len(group_cases)-1,min(3,len(group_cases)),dtype=int))]
            for n in reps+job["test_cases"]:
                score = predict_scores(model,arrays[n]["image"],4,device=device,normalized=True)
                health = score_health(score)
                path = scores/f"case-{n:03d}.npz"
                with path.open("xb") as stream: np.savez_compressed(stream,score=score)
                budget=int(np.ceil(p["area_budget"]*score.size))
                evaluations.append({"step":step,"case_number":n,"group":entries[n]["group"],
                    "is_training_diagnostic":n in reps,"health":health,
                    "coverage":coverage(ranked_support(score,budget),arrays[n]["human"]),
                    "score_sha256":sha256_file(path)})
            model.train()
    write_new_json(target/"evaluation.json",{"job":job,"records":evaluations,
        "protocol_sha256":sha256_file(output/"protocol.json"),"elapsed_seconds":time.monotonic()-began,
        "sampler_audit":sampler.audit,"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__})
    write_new_json(target/"complete.json",{"evaluation_sha256":sha256_file(target/"evaluation.json")})


def launch(output, gpu_ids, indices):
    p=checked(output)
    status=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    state={int(r.split(',')[0]):[int(v.strip()) for v in r.split(',')[1:]] for r in status.splitlines()}
    if any(g not in state or state[g][0]>512 or state[g][1]>5 for g in gpu_ids):
        raise RuntimeError("Requested GPU occupied; no process will be interrupted")
    logdir=output/"workers";logdir.mkdir(exist_ok=True)
    pending=list(indices if indices else range(len(p["jobs"]))); active={}; failed=[]
    # Completed runs are reused only with checksum validation; partial ones stop.
    for index in list(pending):
        target=output/"runs"/f"job-{index:02d}"
        if (target/"complete.json").exists():
            if read(target/"complete.json")["evaluation_sha256"] != sha256_file(target/"evaluation.json"):
                raise ValueError("Completed evaluation changed")
            pending.remove(index)
        elif target.exists():
            raise RuntimeError(f"Partial run {index}; preserve it and inspect before resuming")
    while pending or active:
        for gpu in gpu_ids:
            if gpu not in active and pending:
                index=pending.pop(0)
                stream=(logdir/f"job-{index:02d}.log").open("x")
                env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"CUBLAS_WORKSPACE_CONFIG":":4096:8",
                    "OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","MKL_NUM_THREADS":"2",
                    "PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":str(ROOT/"src")+os.pathsep+str(ROOT)}
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"worker","--output",str(output),"--index",str(index)],
                                      cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active[gpu]=(index,proc,stream)
                print("START",index,"GPU",gpu,"PID",proc.pid,flush=True)
        for gpu,(index,proc,stream) in list(active.items()):
            if proc.poll() is not None:
                stream.close();del active[gpu]
                print("FINISH",index,"EXIT",proc.returncode,flush=True)
                if proc.returncode:failed.append(index)
        time.sleep(2)
    if failed:raise RuntimeError(f"Failed jobs retained: {failed}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["prepare","plan","worker","launch"])
    parser.add_argument("--output",default=DEFAULT)
    parser.add_argument("--index",type=int)
    parser.add_argument("--indices",default="")
    parser.add_argument("--gpus",default="1,2,3,4,5,6,7")
    args=parser.parse_args();output=contained_path(ROOT,args.output)
    if args.command=="prepare":prepare(output)
    elif args.command=="plan":plan(output)
    elif args.command=="worker":worker(output,args.index)
    else:launch(output,[int(g) for g in args.gpus.split(',')],
                [int(i) for i in args.indices.split(',') if i])


if __name__=="__main__":main()
