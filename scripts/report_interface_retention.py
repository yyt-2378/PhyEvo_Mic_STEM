#!/usr/bin/env python3
"""Recompute all retained interface candidates and freeze research bindings."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tfree_detection.expert_supervision import contained_path,load_learning_dataset,sha256_file,write_new_json
from tfree_detection.interface_release import select_retention_step,training_healthy
from tfree_detection.interface_reliability import qualify_boundary_or_band
from tfree_detection.interface_runtime import choose_interface

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"outputs/interface_improvement_20261015_v1"
RUNTIME_SOURCES=["src/tfree_detection/interface_release.py","src/tfree_detection/interface_runtime.py",
    "src/tfree_detection/interface_reliability.py","src/tfree_detection/interface_geometry.py",
    "src/tfree_detection/interface_student.py","src/tfree_detection/tasks/interface.py",
    "src/tfree_detection/config.py","configs/phase_boundary_geometry_aware.yaml",
    "src/tfree_detection/expert_student.py","src/tfree_detection/student.py"]


def read(path):return json.loads(path.read_text())


def main():
    out=OUTPUT/"release_v2";out.mkdir(exist_ok=False)
    bindings=out/"bindings";bindings.mkdir();predictions=out/"predictions";predictions.mkdir()
    retention=OUTPUT/"retention_v1";protocol=read(retention/"protocol.json");runs=[]
    for job in protocol["jobs"]:
        path=retention/"runs"/f"job-{job['index']:02d}"/"evaluation.json"
        if sha256_file(path)!=read(path.parent/"complete.json")["evaluation_sha256"]:raise ValueError("Run modified")
        run=read(path)
        if run["job"]!=job or run["protocol_sha256"]!=sha256_file(retention/"protocol.json"):raise ValueError("Wrong run")
        runs.append(run)
    manifest,arrays=load_learning_dataset(ROOT,ROOT/"outputs/expert_cycle_001/learning_dataset_v1")
    entries={e["case_number"]:e for e in manifest["entries"]}
    guards={r["case_number"]:r["guard"] for r in read(OUTPUT/"stability/audit.json")["records"]}
    # Independently evaluate the deployed route on INNER sources only before
    # choosing any step. Failed OOD student outputs are never used as scores.
    for run in runs:
        j=run["job"]
        if j["stage"]!="inner":continue
        for record in run["records"]:
            if record["is_training_diagnostic"]:continue
            n=record["case_number"];step=record["step"]
            if record["group"]==j["outer_group"]:raise ValueError("Outer labels in inner validation")
            with np.load(OUTPUT/"physical"/f"case-{n:03d}.npz",allow_pickle=False) as z:physical={k:z[k] for k in z.files}
            path=retention/"runs"/f"job-{j['index']:02d}"/f"step-{step:06d}"/"scores"/f"case-{n:03d}.npz"
            if sha256_file(path)!=record["sha256"]:raise ValueError("Inner score modified")
            with np.load(path,allow_pickle=False) as z:score=z["score"]
            guard=qualify_boundary_or_band(physical,guards[n])
            mask,decision=choose_interface(physical["score"],physical["valid"],guard,score,
                student_admitted=training_healthy(run,step),budget=int(np.ceil(.1*score.size)))
            human=arrays[n]["human"]
            record["routed_coverage"]=float((mask&human).sum()/max(1,human.sum()))
            record["inner_route"]=decision
    selectors=[];rows=[];binding_rows=[]
    for group in protocol["groups"]:
        selector=select_retention_step(protocol,runs,group);selectors.append(selector)
        selection_path=out/f"selector-{protocol['groups'].index(group)}.json"
        write_new_json(selection_path,selector)
        if selector["selected"] is None:raise RuntimeError(f"No admissible retained checkpoint for {group}")
        step=selector["selected"]["step"]
        for run in runs:
            j=run["job"]
            if j["stage"]!="outer" or j["outer_group"]!=group:continue
            directory=retention/"runs"/f"job-{j['index']:02d}"/f"step-{step:06d}"
            binding={"format":"phyevo-interface-research-binding-v1","production_approved":False,
                "checkpoint":str(directory.relative_to(ROOT)),"checkpoint_sha256":sha256_file(directory/"checkpoint.json"),
                "selector":str(selection_path.relative_to(ROOT)),"selector_sha256":sha256_file(selection_path),
                "student_training_admitted":training_healthy(run,step),
                "runtime_source_hashes":{s:sha256_file(ROOT/s) for s in RUNTIME_SOURCES},
                "fold_role":"source-held-out evaluation family; never inferred from input filename"}
            binding_path=bindings/f"job-{j['index']:02d}.json";write_new_json(binding_path,binding)
            binding_rows.append({"job":j["index"],"group":group,"seed":j["seed"],"path":str(binding_path.relative_to(ROOT))})
            for n in j["test_cases"]:
                with np.load(OUTPUT/"physical"/f"case-{n:03d}.npz",allow_pickle=False) as z:physical={k:z[k] for k in z.files}
                record=next(r for r in run["records"] if r["step"]==step and r["case_number"]==n)
                path=directory/"scores"/f"case-{n:03d}.npz"
                if sha256_file(path)!=record["sha256"]:raise ValueError("Retained score modified")
                with np.load(path,allow_pickle=False) as z:score=z["score"]
                guard=qualify_boundary_or_band(physical,guards[n])
                budget=int(np.ceil(.1*score.size))
                mask,decision=choose_interface(physical["score"],physical["valid"],guard,score,
                    student_admitted=binding["student_training_admitted"],budget=budget)
                dest=predictions/f"case-{n:03d}-seed-{j['seed']}.npz"
                with dest.open("xb") as stream:np.savez_compressed(stream,mask=mask,student_score=score)
                human=arrays[n]["human"]
                rows.append({"case_number":n,"group":group,"seed":j["seed"],"step":step,"job":j["index"],
                    "coverage":float((mask&human).sum()/max(1,human.sum())),"student_coverage":record["coverage"],
                    "selected_pixels":int(mask.sum()),"budget":budget,"guard":guard,"decision":decision,
                    "prediction":str(dest.relative_to(out)),"sha256":sha256_file(dest)})
    old=read(ROOT/"outputs/best_release_20261014_v1/recomputed_cases.json")["rows"]
    summary=[]
    for group in protocol["groups"]:
        rr=[r for r in rows if r["group"]==group];baseline=[r for r in old if r["group"]==group and r["main_comparison_eligible"]]
        summary.append({"group":group,"cases":len({r['case_number'] for r in rr}),
            "legacy_physics":float(np.mean([r["methods"]["physics_budgeted"]["positive_coverage"] for r in baseline])),
            "legacy_student":float(np.mean([r["methods"]["student_expert"]["positive_coverage"] for r in baseline])),
            "legacy_hybrid":float(np.mean([r["methods"]["hybrid_fixed_budget"]["positive_coverage"] for r in baseline])),
            "new_student":float(np.mean([r["student_coverage"] for r in rr])),
            "new_routed":float(np.mean([r["coverage"] for r in rr])),
            "seed_means":{str(s):float(np.mean([r["coverage"] for r in rr if r["seed"]==s])) for s in [23,41,67]}})
    # Preserve old successful weights/data and recompute point summaries from
    # the checksum-protected replay; this task does not replace the point path.
    frozen=ROOT/"outputs/best_release_20261014_v1"
    checked_files=0
    for seal_name in ["seal.json","replay_seal.json"]:
        for name,digest in read(frozen/seal_name)["files"].items():
            if sha256_file(contained_path(frozen,name))!=digest:raise ValueError("Historical asset changed")
            checked_files+=1
    point=[]
    for group in sorted({r["group"] for r in old if r["geometry"]=="point"}):
        rr=[r for r in old if r["group"]==group and r["main_comparison_eligible"]]
        point.append({"group":group,"status":"unchanged_checkpoint_and_code_path",
            "student":float(np.mean([r["methods"]["student_expert"]["positive_coverage"] for r in rr])),
            "retained_combination":float(np.mean([r["methods"]["student_reranked_physics_plus_rescue"]["positive_coverage"] for r in rr]))})
    result={"summary":summary,"rows":rows,"bindings":binding_rows,"selectors":selectors,"point_regression":point,
        "historical_asset_hash_checks":checked_files,"selection_uses_outer_labels":False,
        "historical_base_and_geometry_rules_are_development_informed":True,
        "all_budgets_respected":all(r["selected_pixels"]<=r["budget"] for r in rows),
        "all_routed_students_healthy":all(r["decision"]["student_health"]["eligible"] for r in rows if r["decision"]["branch"]=="student"),
        "all_interface_framework_means_nonregressing_vs_old_combination":all(r["new_routed"]>=r["legacy_hybrid"] for r in summary),
        "metric":"known-positive coverage, mean per frame then 3-seed mean; NOT accuracy, IoU or precision",
        "note":"Pure historical student may still exceed a new routed result; report separately, never claim uniform student improvement.",
        "production_approved":False,"continuous_positive_RSI_proven":False,
        "source_hashes":{s:sha256_file(ROOT/s) for s in RUNTIME_SOURCES+["scripts/report_interface_retention.py"]}}
    write_new_json(out/"results.json",result)
    print(json.dumps({"summary":summary,"point_regression":point,"historical_checks":checked_files},indent=2))


if __name__=="__main__":main()
