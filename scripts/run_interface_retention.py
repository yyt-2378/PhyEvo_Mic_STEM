#!/usr/bin/env python3
"""Warm-start interface readout adaptation with pair-excluded inner runs.

Historical base configurations are the disclosed, already test-informed best
collection. New step selection uses inner sources only; this does NOT turn the
base collection into a blind benchmark. No point checkpoint is altered.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np

from tfree_detection.expert_student import F,geometry_logits,predict_scores,restore_checkpoint,save_checkpoint,torch
from tfree_detection.expert_supervision import contained_path,load_learning_dataset,sha256_file,write_new_json
from tfree_detection.interface_geometry import ranked_support
from tfree_detection.interface_retention import RetentionSampler,freeze_feature_extractor,retention_loss
from tfree_detection.interface_student import score_health
from tfree_detection.rsi_refinement import RefinementPolicy

ROOT=Path(__file__).resolve().parents[1]
BASE="outputs/interface_improvement_20261015_v1"
DEFAULT=BASE+"/retention_v1"
DATASET="outputs/expert_cycle_001/learning_dataset_v1"
SOURCES=["scripts/run_interface_retention.py","src/tfree_detection/interface_retention.py",
         "src/tfree_detection/interface_student.py","src/tfree_detection/interface_geometry.py",
         "src/tfree_detection/expert_student.py","src/tfree_detection/expert_supervision.py",
         "src/tfree_detection/rsi_refinement.py","src/tfree_detection/student.py"]


def read(p):return json.loads(p.read_text())


def plan(output):
    if output.exists():raise FileExistsError("New output required")
    manifest,_=load_learning_dataset(ROOT,ROOT/DATASET)
    entries=[e for e in manifest["entries"] if e["geometry"]=="interface"]
    groups=sorted({e["group"] for e in entries})
    frozen=read(ROOT/"outputs/best_release_20261014_v1/manifest.json")
    oldprotocol=read(ROOT/"outputs/rsi_cycle_002/training_v1/protocol.json") if (ROOT/"outputs/rsi_cycle_002/training_v1/protocol.json").exists() else read(ROOT/"outputs/rsi_cycle_002/protocol.json")
    jobs=[]
    for outer in groups:
        for validation in [None]+[g for g in groups if g!=outer]:
            excluded=[outer] if validation is None else sorted([outer,validation])
            for seed in [23,41,67]:
                base=next(j for j in frozen["jobs"] if j["group"]==outer and j["seed"]==seed)
                fold=next(f for f in oldprotocol["folds"] if sorted(f["excluded_groups"])==sorted(excluded) and f["stage"] in ["inner","outer"])
                if validation is None:path=contained_path(ROOT,base["frozen_checkpoint"])
                else:path=ROOT/"outputs/rsi_cycle_002/training_v1"/base["policy"]/f"seed-{seed}"/fold["fold_id"]/f"step-{base['step']:06d}"
                meta=read(path/"checkpoint.json")
                if set(meta["train_groups"])&set(excluded):raise ValueError("Base checkpoint sees excluded source")
                if sha256_file(path/"state.npz")!=meta["state_sha256"]:raise ValueError("Base weight checksum mismatch")
                jobs.append({"index":len(jobs),"outer_group":outer,"validation_group":validation or outer,
                    "stage":"outer" if validation is None else "inner","seed":seed,
                    "base":str(path.relative_to(ROOT)),"base_checkpoint_sha256":sha256_file(path/"checkpoint.json"),
                    "base_state_sha256":meta["state_sha256"],"historical_policy":base["policy"],"historical_step":base["step"],
                    "train_groups":[g for g in groups if g not in excluded],
                    "train_cases":[e["case_number"] for e in entries if e["group"] not in excluded],
                    "test_cases":[e["case_number"] for e in entries if e["group"]==(validation or outer)]})
    output.mkdir(parents=True)
    p={"format":"phyevo-interface-retention-v1","jobs":jobs,"groups":groups,"steps":400,"checkpoints":[50,100,200,400],
       "base_channels":16,"batch_size":16,"patch_size":128,"learning_rate":.005,
       "source_hashes":{s:sha256_file(ROOT/s) for s in SOURCES},
       "physical_audit_sha256":sha256_file(ROOT/BASE/"physical_audit.json"),
       "data_manifest_sha256":sha256_file(ROOT/DATASET/"manifest.json"),
       "selection":"equal-source inner validation; 3 seeds; reject all-high/flat checkpoints; no outer-label selection",
       "historical_base_selection_was_test_informed":True,"production_approved":False,
       "initialization":"training-only low-variation reference median logit offset; NOT held-out calibration",
       "frozen_modules":"all except geometry_head","retention_weight":10}
    write_new_json(output/"protocol.json",p)
    with ZipFile(output/"source_snapshot.zip","x",compression=ZIP_DEFLATED) as archive:
        for s in SOURCES:archive.write(ROOT/s,s)
    print("PLANNED",len(jobs),flush=True)


def checked(output):
    p=read(output/"protocol.json")
    if p["source_hashes"]!={s:sha256_file(ROOT/s) for s in SOURCES}:raise ValueError("Retention source changed")
    if p["physical_audit_sha256"]!=sha256_file(ROOT/BASE/"physical_audit.json"):raise ValueError("Evidence changed")
    return p


def worker(output,index):
    p=checked(output);j=p["jobs"][index];target=output/"runs"/f"job-{index:02d}"
    if target.exists():raise FileExistsError("Existing run is protected")
    manifest,arrays=load_learning_dataset(ROOT,ROOT/DATASET)
    entries={e["case_number"]:e for e in manifest["entries"]};training=[entries[n] for n in j["train_cases"]]
    physical={}
    for r in read(ROOT/BASE/"physical_audit.json")["records"]:
        if r["case_number"] in j["train_cases"]:
            path=contained_path(ROOT/BASE,r["file"])
            if sha256_file(path)!=r["sha256"]:raise ValueError("Physical input hash mismatch")
            with np.load(path,allow_pickle=False) as z:physical[r["case_number"]]={k:z[k] for k in z.files}
    base=contained_path(ROOT,j["base"])
    if sha256_file(base/"checkpoint.json")!=j["base_checkpoint_sha256"]:raise ValueError("Base changed")
    device="cuda:0";torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(j["seed"])
    teacher,_,_,_=restore_checkpoint(base,device=device);teacher.eval()
    for v in teacher.parameters():v.requires_grad_(False)
    model,_,_,_=restore_checkpoint(base,device=device);freeze_feature_extractor(model)
    policy=RefinementPolicy("retention",positive_crop_probability=.75,photometric=True)
    sampler=RetentionSampler(training,arrays,policy,physical,excluded_groups=set(p["groups"])-set(j["train_groups"]))
    rng=np.random.default_rng(j["seed"]);init_rng=np.random.default_rng(j["seed"]+100000)
    reference_logits=[]
    with torch.no_grad():
        for step in range(8):
            batch,_=sampler.sample(init_rng,batch_size=p["batch_size"],patch_size=p["patch_size"],device=device,augmentation_seed=j["seed"]+step)
            z=geometry_logits(teacher(batch["image"]),batch["geometry_index"])
            selected=z[batch["physical_reference_prior"]>0]
            if selected.numel():reference_logits.append(selected.cpu().numpy())
        shift=float(np.median(np.concatenate(reference_logits)))+1. if reference_logits else 0.
        model.geometry_head.bias[4]-=shift
    optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=p["learning_rate"])
    target.mkdir(parents=True);records=[];start=time.monotonic()
    for step in range(1,p["steps"]+1):
        model.train();batch,_=sampler.sample(rng,batch_size=p["batch_size"],patch_size=p["patch_size"],device=device,augmentation_seed=j["seed"]*1000000+step)
        with torch.no_grad():old=teacher(batch["image"])
        out=model(batch["image"])
        small=model(F.interpolate(batch["image"],scale_factor=.5,mode="bilinear",align_corners=False))
        loss,parts=retention_loss(out,old,batch,small)
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite retention loss")
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if step==1 or step%50==0:
            event={"step":step,"seconds":time.monotonic()-start,**{k:float(v.detach().cpu()) for k,v in parts.items()}}
            with (target/"events.jsonl").open("a") as stream:stream.write(json.dumps(event,allow_nan=False)+"\n")
            print("JOB",index,step,round(event["total"],4),flush=True)
        if step in p["checkpoints"]:
            directory=target/f"step-{step:06d}"
            save_checkpoint(directory,model,optimizer,rng,{**j,"step":step,"base_channels":p["base_channels"],
                "learning_rate":p["learning_rate"],"protocol_sha256":sha256_file(output/"protocol.json"),
                "reference_initial_shift":shift,"geometry_support":["interface"],"learning_manifest_sha256":p["data_manifest_sha256"],
                "sampler_audit":sampler.audit,"train_case_numbers":j["train_cases"]})
            scores=directory/"scores";scores.mkdir()
            reps=[]
            for g in j["train_groups"]:
                nn=[e["case_number"] for e in training if e["group"]==g]
                reps.extend(nn[int(i)] for i in np.unique(np.linspace(0,len(nn)-1,min(3,len(nn)),dtype=int)))
            for n in reps+j["test_cases"]:
                score=predict_scores(model,arrays[n]["image"],4,device=device,normalized=True)
                path=scores/f"case-{n:03d}.npz"
                with path.open("xb") as stream:np.savez_compressed(stream,score=score)
                human=arrays[n]["human"];mask=ranked_support(score,int(np.ceil(.1*score.size)))
                records.append({"step":step,"case_number":n,"group":entries[n]["group"],"is_training_diagnostic":n in reps,
                    "coverage":float((mask&human).sum()/max(1,human.sum())),"health":score_health(score),"sha256":sha256_file(path)})
    # Encoder/decoder equality is a contractual preservation check, not a claim
    # that every candidate ranking stayed unchanged after readout adaptation.
    unchanged=all(torch.equal(v,teacher.state_dict()[k]) for k,v in model.state_dict().items() if not k.startswith("geometry_head."))
    if not unchanged:raise ValueError("Frozen feature extractor changed")
    write_new_json(target/"evaluation.json",{"job":j,"records":records,"feature_extractor_exact":unchanged,
        "protocol_sha256":sha256_file(output/"protocol.json"),"elapsed_seconds":time.monotonic()-start})
    write_new_json(target/"complete.json",{"evaluation_sha256":sha256_file(target/"evaluation.json")})


def launch(output,gpus):
    p=checked(output)
    status=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    states={int(r.split(',')[0]):[int(v.strip()) for v in r.split(',')[1:]] for r in status.splitlines()}
    if any(states[g][0]>512 or states[g][1]>5 for g in gpus):raise RuntimeError("GPU busy; not interrupting")
    logdir=output/"workers";logdir.mkdir(exist_ok=False)
    pending=list(range(len(p["jobs"])));active={};failed=[]
    while pending or active:
        for gpu in gpus:
            if gpu not in active and pending:
                index=pending.pop(0);stream=(logdir/f"job-{index:02d}.log").open("x")
                env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"CUBLAS_WORKSPACE_CONFIG":":4096:8","OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","MKL_NUM_THREADS":"2","PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":str(ROOT/"src")+os.pathsep+str(ROOT)}
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"worker","--output",str(output),"--index",str(index)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active[gpu]=(index,proc,stream);print("START",index,"GPU",gpu,"PID",proc.pid,flush=True)
        for gpu,(index,proc,stream) in list(active.items()):
            if proc.poll() is not None:
                stream.close();del active[gpu];print("FINISH",index,"EXIT",proc.returncode,flush=True)
                if proc.returncode:failed.append(index)
        time.sleep(2)
    if failed:raise RuntimeError(f"Failed jobs retained {failed}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=["plan","worker","launch"]);p.add_argument("--output",default=DEFAULT)
    p.add_argument("--gpus",default="1,2,3,4,5,6,7");p.add_argument("--index",type=int);a=p.parse_args();out=contained_path(ROOT,a.output)
    if a.command=="plan":plan(out)
    elif a.command=="worker":worker(out,a.index)
    else:launch(out,[int(g) for g in a.gpus.split(',')])
