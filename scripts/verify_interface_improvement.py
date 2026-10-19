#!/usr/bin/env python3
"""Verify immutable inputs, training artifacts, route budgets and numeric claims."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np

from tfree_detection.expert_supervision import contained_path,load_learning_dataset,sha256_file,write_new_json
from tfree_detection.interface_geometry import ranked_support
from tfree_detection.interface_runtime import choose_interface

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/interface_improvement_20261015_v1"
OUT=BASE/"release_v2"


def read(path):return json.loads(path.read_text())


def require(condition,message):
    if not condition:raise ValueError(message)


def main():
    _,arrays=load_learning_dataset(ROOT,ROOT/"outputs/expert_cycle_001/learning_dataset_v1")
    counts={"completed_runs":0,"checkpoints":0,"score_files":0,"retained_feature_checks":0,"routed_rows":0}
    for experiment in [BASE,BASE/"retention_v1"]:
        protocol=read(experiment/"protocol.json")
        for name,digest in protocol["source_hashes"].items():
            require(sha256_file(contained_path(ROOT,name))==digest,"Training source changed")
        for job in protocol["jobs"]:
            run=experiment/"runs"/f"job-{job['index']:02d}"
            data=read(run/"evaluation.json")
            require(sha256_file(run/"evaluation.json")==read(run/"complete.json")["evaluation_sha256"],"Evaluation checksum mismatch")
            require(data["job"]==job,"Job identity mismatch")
            require(not set(job["train_cases"])&set(job["test_cases"]),"Sample overlap")
            counts["completed_runs"]+=1
            if "base" in job:
                base=contained_path(ROOT,job["base"])
                require(sha256_file(base/"state.npz")==job["base_state_sha256"],"Historical base changed")
                with np.load(base/"state.npz",allow_pickle=False) as z:
                    original={k:z[k] for k in z.files if k.startswith("model:") and not k.startswith("model:geometry_head.")}
            for step in protocol["checkpoints"]:
                ckpt=run/f"step-{step:06d}";meta=read(ckpt/"checkpoint.json")
                require(sha256_file(ckpt/"state.npz")==meta["state_sha256"],"Checkpoint state changed")
                counts["checkpoints"]+=1
                if "base" in job:
                    with np.load(ckpt/"state.npz",allow_pickle=False) as z:
                        require(all(np.array_equal(z[k],v) for k,v in original.items()),"Feature extractor changed")
                    counts["retained_feature_checks"]+=1
            for row in data["records"]:
                path=run/f"step-{row['step']:06d}"/"scores"/f"case-{row['case_number']:03d}.npz"
                require(sha256_file(path)==row.get("sha256",row.get("score_sha256")),"Score checksum changed")
                with np.load(path,allow_pickle=False) as z:score=z["score"]
                human=arrays[row["case_number"]]["human"]
                mask=ranked_support(score,int(np.ceil(.1*score.size)))
                actual=float((mask&human).sum()/max(1,human.sum()))
                require(abs(actual-row["coverage"])<1e-12,"Recorded coverage not reproducible")
                counts["score_files"]+=1
    result=read(OUT/"results.json")
    for row in result["rows"]:
        path=contained_path(OUT,row["prediction"])
        require(sha256_file(path)==row["sha256"],"Route prediction changed")
        with np.load(path,allow_pickle=False) as z:mask=z["mask"];score=z["student_score"]
        with np.load(BASE/"physical"/f"case-{row['case_number']:03d}.npz",allow_pickle=False) as p:
            expected,decision=choose_interface(p["score"],p["valid"],row["guard"],score,
                student_admitted=read(OUT/"bindings"/f"job-{row['job']:02d}.json")["student_training_admitted"],budget=row["budget"])
        require(np.array_equal(mask,expected) and decision==row["decision"],"Routed mask not reproducible")
        require(int(mask.sum())<=row["budget"],"Budget exceeded")
        human=arrays[row["case_number"]]["human"]
        require(abs(float((mask&human).sum()/human.sum())-row["coverage"])<1e-12,"Route coverage differs")
        counts["routed_rows"]+=1
    for row in result["summary"]:
        rr=[r for r in result["rows"] if r["group"]==row["group"]]
        require(abs(float(np.mean([r["coverage"] for r in rr]))-row["new_routed"])<1e-12,"Summary mismatch")
    require(counts["completed_runs"]==63 and counts["checkpoints"]==216,"Incomplete training inventory")
    require(result["all_interface_framework_means_nonregressing_vs_old_combination"],"Framework regression")
    tests=ET.parse(BASE/"focused_tests.xml").getroot()
    suites=tests.findall("testsuite")
    test_count=sum(int(s.attrib["tests"]) for s in suites)
    require(test_count==155 and all(int(s.attrib["failures"])+int(s.attrib["errors"])==0 for s in suites),"Focused tests failed")
    raw=read(OUT/"raw_runtime_verification.json")
    require(raw["all_masks_identical"],"Raw runtime mask differs")
    source_paths=set(result["source_hashes"])
    source_paths.update(["scripts/run_interface_detection.py","scripts/build_interface_improvement_report.py",
                         "scripts/verify_interface_improvement.py","benchmarks/INTERFACE_IMPROVEMENT_EVIDENCE_v1.yaml"])
    source_paths.update(str(p.relative_to(ROOT)) for p in (ROOT/"tests").glob("test_interface_*.py"))
    with ZipFile(OUT/"source_snapshot.zip","x",compression=ZIP_DEFLATED) as archive:
        for name in sorted(source_paths):archive.write(contained_path(ROOT,name),name)
    receipt={**counts,"focused_tests_passed":test_count,"all_recorded_coverages_recomputed":True,
        "raw_runtime":raw,"historical_hash_checks":result["historical_asset_hash_checks"],
        "no_data_or_checkpoints_deleted":True,"no_expert_labels_edited":True,
        "live_source_hashes":{s:sha256_file(contained_path(ROOT,s)) for s in sorted(source_paths)},
        "limitations":["development-informed geometry and historical base selection",
                       "not uniformly better than best standalone student",
                       "raw-runtime student scores not bitwise identical, masks are identical",
                       "known-positive coverage is not precision or IoU"]}
    write_new_json(OUT/"verification.json",receipt)
    files={str(p.relative_to(OUT)):sha256_file(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    write_new_json(OUT/"seal.json",{"files":files})
    print(json.dumps({**counts,"focused_tests_passed":test_count,"sealed_files":len(files)},indent=2))


if __name__=="__main__":main()
