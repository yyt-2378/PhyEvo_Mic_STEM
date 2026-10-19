#!/usr/bin/env python3
"""Create an immutable model binding or run optional refined feedback on raw input."""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from dataclasses import replace
import json
import os
from pathlib import Path

from tfree_detection.config import load_config
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.rsi_feedback_runtime import (
    create_feedback_binding, create_feedback_registry, detect_with_feedback, load_feedback_binding,
    resolve_context_binding, save_feedback_detection,
)
from tfree_detection.rsi_runtime import inside
from tfree_detection.expert_supervision import sha256_file, write_new_json


ROOT = Path(__file__).resolve().parents[1]


def bind_family(analysis, outer_fold, seed, output):
    """Materialize a fixed-seed family from an already sealed inner selector."""
    from scripts.analyze_rsi_refinement import load_analysis
    analysis, _, training, protocol, _ = load_analysis(inside(ROOT,analysis))
    receipt=json.loads((analysis/"result_seal.json").read_text())
    for name,digest in receipt["files"].items():
        if sha256_file(inside(analysis,name))!=digest:
            raise ValueError("Refinement analysis changed before family binding")
    fold=next((f for f in protocol["folds"] if f["fold_id"]==outer_fold and f["stage"]=="outer"),None)
    if fold is None or seed not in protocol["seeds"]:
        raise ValueError("Unregistered outer learner family or seed")
    selector=analysis/"selectors"/(outer_fold+".json")
    frozen=json.loads(selector.read_text())
    choices={(s["selected"]["policy"],s["selected"]["step"]) for s in frozen["scopes"].values() if s.get("selected")}
    choices.add(("control",400))
    output=inside(ROOT,output)
    output.mkdir(parents=True,exist_ok=False)
    bindings={}
    for policy,step in sorted(choices):
        checkpoint=training/policy/f"seed-{seed}"/outer_fold/f"step-{step:06d}"
        target=output/f"{policy}-{step}.json"
        create_feedback_binding(ROOT,checkpoint,training/"protocol.json",target,
            selector=selector,analysis_protocol=analysis/"analysis_protocol.json")
        bindings[f"{policy}::{step}"]=target
    create_feedback_registry(ROOT,selector,bindings,"control::400",output/"registry.json")
    return output/"registry.json"


def record_evidence(analysis, bindings, output):
    """Append actual refinement proposals to a new local research ledger."""
    from scripts.analyze_rsi_refinement import load_analysis
    from tfree_detection.rsi_policy import (ArtifactRef,CapabilityCard,CapabilityLedger,CapabilityScope,
        FeatureRange,PairedMetric,content_hash,evidence_contract,verify_artifacts)
    analysis,ap,_,tp,_=load_analysis(inside(ROOT,analysis))
    receipt=json.loads((analysis/"result_seal.json").read_text())
    for name,digest in receipt["files"].items():
        if sha256_file(inside(analysis,name))!=digest: raise ValueError("Analysis changed before evidence recording")
    bindings,output=inside(ROOT,bindings),inside(ROOT,output)
    rows=json.loads((analysis/"selected_outer_cases.json").read_text())
    grouped=defaultdict(list)
    for row in rows:
        if row["main_comparison_eligible"]:grouped[row["group"],row["seed"],row["geometry"],row["teacher_empty"]].append(row)
    output.mkdir(parents=True,exist_ok=False)
    ledger=CapabilityLedger(ROOT,output/"events")
    decisions=[]
    def ref(role,path):return ArtifactRef(role,str(path.relative_to(ROOT)),sha256_file(path))
    for (group,seed,geometry,empty),subset in sorted(grouped.items()):
        choice=subset[0]["selected_configuration"]
        if any(r["selected_configuration"]!=choice for r in subset):raise ValueError("Scope configuration is not frozen")
        fold=next(f for f in tp["folds"] if f["stage"]=="outer" and f["excluded_groups"]==[group])
        note={"outer_group":group,"seed":seed,"geometry":geometry,"teacher_empty":empty,
              "case_numbers":[r["case_number"] for r in subset],"selected_configuration":choice,
              "actual_branches":dict(Counter(r["selected_branch"] for r in subset)),"production_activated":False}
        if choice is None:
            decisions.append({**note,"reason":"inner_evidence_insufficient_physics_retained"})
            continue
        registry=bindings/fold["fold_id"]/f"seed-{seed}"/"registry.json"
        selected,_=resolve_context_binding(ROOT,registry,geometry=geometry,teacher_empty=empty)
        paths,meta,_,selector,_,_=load_feedback_binding(ROOT,selected)
        if any(r["checkpoint"]!=str(paths["checkpoint"].parent.relative_to(ROOT)) for r in subset):
            raise ValueError("Evidence metrics belong to a different checkpoint")
        scope_key=geometry+("::teacher_empty" if empty else "::teacher_nonempty")
        version=f"{fold['fold_id']}-seed-{seed}-{geometry}-empty-{int(empty)}"
        scope=CapabilityScope(geometry+f"-empty-{int(empty)}",
            ("atomic_point_defect" if geometry=="point" else "phase_boundary",),geometry,
            (FeatureRange("teacher_empty",int(empty),int(empty)),))
        metric_name="marked_point_recall" if geometry=="point" else "annotated_band_coverage"
        mean=lambda name:sum(r["methods"][name]["positive_coverage"] for r in subset)/len(subset)
        budget=content_hash(ap["evaluation"])
        metric=PairedMetric(metric_name,mean("physics_budgeted"),mean("nested_feedback"),True,budget,budget)
        card=CapabilityCard(version,scope,"student" if choice["branch"]=="student_expert" else "hybrid","development",
            (ref("checkpoint",paths["checkpoint"]),ref("protocol",analysis/"analysis_protocol.json"),
             ref("router",registry),ref("inner_selector",paths["selector"])),(metric,),(group,),tuple(meta["train_groups"]),
            frozen_before_evaluation=True,selection_method="nested_group_cv",
            selection_group_ids=tuple(selector["scopes"][scope_key]["inner_groups"]),
            protected_scope_ids=("point","interface"),
            caveats=("fixed_seed_not_ensemble","positive_unknown_labels","reused_development_cohort",
                     "feedback_method_and_embedding_fallback_defined_by_registry","no_complete_old_capability_certification"))
        evidence=output/(version+"-evidence.json")
        write_new_json(evidence,{**evidence_contract(card),**note,
            "actual_metric_source":ref("case_metrics",analysis/"selected_outer_cases.json").__dict__,
            "metric_aggregation":"macro_case_at_one_fixed_seed_after_optional_embedding_veto",
            "selection_freeze_semantics":"inner selector persisted before selected-system outer aggregation; component predictions and developmental diagnostics already existed; not a prospective locked test"})
        card=replace(card,artifacts=card.artifacts+(ref("evidence",evidence),))
        if not verify_artifacts(card,ROOT):raise ValueError("Refinement evidence contract differs")
        path=output/(version+".json");write_new_json(path,card.to_dict())
        event=ledger.record(card)
        if event["activated"]:raise AssertionError("Development refinement must not promote production")
        decisions.append({**note,"card":str(path.relative_to(ROOT)),"decision":event["decision"],
                          "reference":metric.reference,"candidate":metric.candidate})
    if ledger.active_cards():raise AssertionError("Unexpected active refinement card")
    summary={"format":"phyevo-refinement-evidence-ledger-v1","decisions":decisions,
             "events":len(ledger.events()),"production_activated":0,
             "old_ledgers_untouched":True,"default_detector":"native_physics"}
    write_new_json(output/"release_summary.json",summary)
    files={str(p.relative_to(output)):sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_new_json(output/"seal.json",{"files":files,"analysis_seal_sha256":sha256_file(analysis/"result_seal.json"),
        "source_hashes":{"scripts/run_rsi_feedback_detection.py":sha256_file(Path(__file__).resolve()),
                         "src/tfree_detection/rsi_feedback_runtime.py":sha256_file(ROOT/"src/tfree_detection/rsi_feedback_runtime.py")}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("bind")
    bind.add_argument("--checkpoint", required=True)
    bind.add_argument("--training-protocol", required=True)
    bind.add_argument("--output", required=True)
    bind.add_argument("--selector")
    bind.add_argument("--analysis-protocol")
    family = sub.add_parser("bind-family", help="Bind an inner-selected family, without using outer test scores")
    family.add_argument("--analysis",required=True)
    family.add_argument("--outer-fold",required=True)
    family.add_argument("--seed",type=int,required=True)
    family.add_argument("--output",required=True)
    record = sub.add_parser("record-evidence",help="Record research benefits, rejections and retention without promotion")
    record.add_argument("--analysis",required=True)
    record.add_argument("--bindings",required=True)
    record.add_argument("--output",required=True)
    detect = sub.add_parser("detect")
    detect.add_argument("--input", required=True)
    detect.add_argument("--config", required=True)
    detect.add_argument("--output", required=True)
    detect.add_argument("--binding",help="Single shadow binding or inner-selected context registry")
    detect.add_argument("--shadow", action="store_true")
    detect.add_argument("--experimental-research", action="store_true")
    detect.add_argument("--student-device", default="cpu")
    detect.add_argument("--skeleton")
    detect.add_argument("--exemplar")
    detect.add_argument("--dataset-path")
    stack = detect.add_mutually_exclusive_group()
    stack.add_argument("--frame-index", type=int)
    stack.add_argument("--projection", choices=("mean", "median", "max"))
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = inside(ROOT, args.output)
    if output.exists():
        raise FileExistsError("Choose a new feedback artifact path")
    if args.command == "bind-family":
        registry=bind_family(args.analysis,args.outer_fold,args.seed,output)
        print(json.dumps({"registry":str(registry)}))
        return
    if args.command == "record-evidence":
        summary=record_evidence(args.analysis,args.bindings,output)
        print(json.dumps({"events":summary["events"],"production_activated":0,"output":str(output)}))
        return
    if args.command == "bind":
        output.parent.mkdir(parents=True, exist_ok=True)
        create_feedback_binding(ROOT, args.checkpoint, args.training_protocol, output,
                                selector=args.selector, analysis_protocol=args.analysis_protocol)
        print(json.dumps({"binding": str(output)}))
        return
    config = load_config(inside(ROOT, args.config))
    if config.stage2.sam3.enabled:
        raise ValueError("Use physics-only config; no external worker provisioning in this command")
    result = detect_with_feedback(TrainingFreeDetectionFramework(config), inside(ROOT, args.input), project_root=ROOT,
                                  binding=args.binding, shadow=args.shadow, experimental_research=args.experimental_research,
                                  student_device=args.student_device, skeleton=args.skeleton, exemplar=args.exemplar,
                                  dataset_path=args.dataset_path, frame_index=args.frame_index, projection=args.projection)
    save_feedback_detection(result, config, output, project_root=ROOT)
    print(json.dumps({"output": str(output), "primary_branch": result.primary_branch,
                      "status": result.metadata["status"], "reason": result.metadata["reason"]}))


if __name__ == "__main__":
    main()
