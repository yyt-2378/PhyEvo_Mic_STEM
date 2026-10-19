#!/usr/bin/env python3
"""Source-excluded inner selection, then auditable outer development evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tfree_detection.expert_supervision import contained_path,load_learning_dataset,sha256_file,write_new_json
from tfree_detection.interface_runtime import choose_interface

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())


def training_healthy(run,step):
    rows=[r for r in run["records"] if r["step"]==step and r["is_training_diagnostic"]]
    return bool(rows) and all(r["health"]["eligible"] for r in rows)


def select(protocol,runs,outer_group,policy):
    candidates=[]
    for step in protocol["checkpoints"]:
        values=[];receipts=[];valid=True
        for group in protocol["groups"]:
            if group==outer_group:continue
            group_values=[]
            for seed in [23,41,67]:
                matches=[r for r in runs if r["job"]["policy"]==policy and r["job"]["seed"]==seed
                         and len(r["job"]["train_groups"])==1
                         and outer_group not in r["job"]["train_groups"] and group not in r["job"]["train_groups"]]
                if len(matches)!=1:raise ValueError("Inner split identity mismatch")
                run=matches[0]
                rows=[r for r in run["records"] if r["step"]==step and r["group"]==group and not r["is_training_diagnostic"]]
                ok=training_healthy(run,step) and all(r["health"]["eligible"] for r in rows)
                valid &= ok
                group_values.append(float(np.mean([r["coverage"] for r in rows])))
                receipts.append({"job":run["job"]["index"],"validation_group":group,
                                 "training_groups":run["job"]["train_groups"],"healthy":ok})
            values.append(float(np.mean(group_values)))
        candidates.append({"step":step,"inner_macro_coverage":float(np.mean(values)),
                           "eligible":bool(valid),"receipts":receipts})
    admissible=[v for v in candidates if v["eligible"]]
    best=max(admissible,key=lambda v:(v["inner_macro_coverage"],-v["step"])) if admissible else None
    return {"policy":policy,"outer_group":outer_group,"selected":best,
            "candidates":candidates,"uses_outer_labels":False}


def analyze(output):
    target=output/"analysis_v1";target.mkdir(exist_ok=False)
    protocol=read(output/"protocol.json")
    runs=[]
    for job in protocol["jobs"]:
        path=output/"runs"/f"job-{job['index']:02d}"/"evaluation.json"
        if sha256_file(path)!=read(path.parent/"complete.json")["evaluation_sha256"]:
            raise ValueError("Completed run changed")
        run=read(path)
        if run["job"]!=job or run["protocol_sha256"]!=sha256_file(output/"protocol.json"):
            raise ValueError("Run identity mismatch")
        runs.append(run)
    manifest,arrays=load_learning_dataset(ROOT,ROOT/"outputs/expert_cycle_001/learning_dataset_v1")
    entries={e["case_number"]:e for e in manifest["entries"]}
    guards={r["case_number"]:r["guard"] for r in read(output/"stability/audit.json")["records"]}
    selectors=[];rows=[]
    for group in protocol["groups"]:
        for policy in ["control","physical_rank"]:
            selector=select(protocol,runs,group,policy);selectors.append(selector)
            if selector["selected"] is None:continue
            step=selector["selected"]["step"]
            for run in runs:
                job=run["job"]
                if job["policy"]!=policy or len(job["train_groups"])!=2 or group in job["train_groups"]:continue
                admitted=training_healthy(run,step)
                for n in job["test_cases"]:
                    if entries[n]["group"]!=group:raise ValueError("Wrong outer group")
                    path=output/"runs"/f"job-{job['index']:02d}"/f"step-{step:06d}"/"scores"/f"case-{n:03d}.npz"
                    rec=next(r for r in run["records"] if r["step"]==step and r["case_number"]==n)
                    if sha256_file(path)!=rec["score_sha256"]:raise ValueError("Score changed")
                    with np.load(path,allow_pickle=False) as z:score=z["score"]
                    with np.load(output/"physical"/f"case-{n:03d}.npz",allow_pickle=False) as a:
                        mask,decision=choose_interface(a["score"],a["valid"],guards[n],score,
                            student_admitted=admitted,budget=int(np.ceil(.1*score.size)))
                    human=arrays[n]["human"]
                    rows.append({"case_number":n,"group":group,"policy":policy,"seed":job["seed"],"step":step,
                        "student_coverage":rec["coverage"],"routed_coverage":float((mask&human).sum()/max(1,human.sum())),
                        "decision":decision,"student_training_admitted":admitted,
                        "selected_pixels":int(mask.sum()),"budget":int(np.ceil(.1*score.size))})
    summary=[]
    for group in protocol["groups"]:
        for policy in ["control","physical_rank"]:
            rr=[r for r in rows if r["group"]==group and r["policy"]==policy]
            if rr:summary.append({"group":group,"policy":policy,"student":float(np.mean([r["student_coverage"] for r in rr])),
                "routed":float(np.mean([r["routed_coverage"] for r in rr])),"count":len(rr),
                "seeds":{str(s):float(np.mean([r["routed_coverage"] for r in rr if r["seed"]==s])) for s in [23,41,67]}})
    # Full learning curves remain visible; these are never used by the selector.
    curves=[]
    for run in runs:
        if len(run["job"]["train_groups"])!=2:continue
        for step in protocol["checkpoints"]:
            rr=[r for r in run["records"] if r["step"]==step and not r["is_training_diagnostic"]]
            curves.append({"job":run["job"],"step":step,"training_healthy":training_healthy(run,step),
                           "mean_coverage":float(np.mean([r["coverage"] for r in rr])),
                           "healthy_test_count":sum(r["health"]["eligible"] for r in rr),"test_count":len(rr)})
    result={"selectors":selectors,"rows":rows,"summary":summary,"outer_learning_curves":curves,
            "protocol_sha256":sha256_file(output/"protocol.json"),
            "analysis_sources":{p:sha256_file(ROOT/p) for p in ["scripts/analyze_interface_improvement.py","src/tfree_detection/interface_runtime.py"]},
            "metric":"known-positive expert-band coverage at 10% area; NOT accuracy/IoU/precision",
            "test_status":"already exposed development sources; not blind external generalization"}
    write_new_json(target/"results.json",result)
    print(json.dumps(summary,indent=2))
    for selector in selectors:
        best=selector["selected"]
        print(selector["outer_group"],selector["policy"],None if best is None else (best["step"],best["inner_macro_coverage"]))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="outputs/interface_improvement_20261015_v1")
    args=p.parse_args();analyze(contained_path(ROOT,args.output))
