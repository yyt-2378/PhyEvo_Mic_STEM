#!/usr/bin/env python3
"""Read-only old inputs; immutable new consistency/selection/regression outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tfree_detection.config import load_config
from tfree_detection.expert_supervision import contained_path,load_learning_dataset,sha256_file,write_new_json
from tfree_detection.interface_runtime import physical_consistency

ROOT=Path(__file__).resolve().parents[1]


def stability(output):
    target=output/"stability";target.mkdir(exist_ok=False)
    manifest,arrays=load_learning_dataset(ROOT,ROOT/"outputs/expert_cycle_001/learning_dataset_v1")
    records=[]
    for entry in manifest["entries"]:
        if entry["geometry"]!="interface":continue
        n=entry["case_number"]
        result,guard=physical_consistency(arrays[n]["image"],load_config(ROOT/"configs/phase_boundary_geometry_aware.yaml"))
        with np.load(output/"physical"/f"case-{n:03d}.npz",allow_pickle=False) as cache:
            if not np.array_equal(result.score_map,cache["score"]):raise ValueError("Physical replay changed")
        record={"case_number":n,"group":entry["group"],"guard":guard}
        records.append(record)
        print(n,guard["eligible"],[round(v["top_budget_overlap"],3) for v in guard["views"]],flush=True)
    write_new_json(target/"audit.json",{"records":records,
        "source_hashes":{p:sha256_file(ROOT/p) for p in ["src/tfree_detection/interface_runtime.py","scripts/audit_interface_improvement.py"]}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["stability"])
    parser.add_argument("--output",default="outputs/interface_improvement_20261015_v1")
    args=parser.parse_args();stability(contained_path(ROOT,args.output))


if __name__=="__main__":main()
