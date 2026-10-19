#!/usr/bin/env python3
"""Independent byte/numeric checks for refinement baselines and runtime replay.

These checks establish reproducibility and preservation, not detection quality.
Completion of the scientific goal also requires full nested results and review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.run_rsi_refinement import read_protocol, verify_run
from tfree_detection.expert_supervision import contained_path, sha256_file, write_new_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def compare_archives(left, right, prefix="model:"):
    """Compare named arrays, never merely an archive's filename/metadata."""
    with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
        keys={k for k in a.files if k.startswith(prefix)}
        if not keys or keys!={k for k in b.files if k.startswith(prefix)}:
            raise ValueError("Archive tensor inventory mismatch")
        differences=[]
        for name in sorted(keys):
            if a[name].shape!=b[name].shape or a[name].dtype!=b[name].dtype or not np.array_equal(a[name],b[name],equal_nan=True):
                differences.append(name)
    return {"compared_arrays":len(keys),"different_arrays":differences,"all_arrays_equal":not differences}


def verify_baselines(training, output):
    training,output=contained_path(ROOT,training),contained_path(ROOT,output)
    if output.exists(): raise FileExistsError("Verification receipt must be new")
    protocol,_=read_protocol(training)
    receipts,inputs=[],{}
    for seed in protocol["seeds"]:
        old_run=ROOT/"outputs/expert_cycle_001"/f"student_gpu_s{seed}_v1"
        old_protocol=read(old_run/"protocol.json")
        for fold in protocol["folds"]:
            if fold["stage"]!="outer": continue
            old_fold=next(f for f in old_protocol["folds"] if f["held_out_group"]==fold["excluded_groups"][0])
            if fold["train_case_numbers"]!=old_fold["train_case_numbers"]:
                raise ValueError("Control training inventory changed")
            run=training/"control"/f"seed-{seed}"/fold["fold_id"]
            verify_run(run,sha256_file(training/"protocol.json"))
            for step in protocol["checkpoints"]:
                old=old_run/old_fold["fold_id"]/"teacher_plus_expert"/f"step-{step:06d}"
                new=run/f"step-{step:06d}"
                for directory in (old,new):
                    if read(directory/"checkpoint.json")["state_sha256"]!=sha256_file(directory/"state.npz"):
                        raise ValueError("Checkpoint state changed")
                    for name in ("checkpoint.json","state.npz"):
                        path=directory/name
                        inputs[str(path.relative_to(ROOT))]=sha256_file(path)
                models=compare_archives(old/"state.npz",new/"state.npz","model:")
                optimizers=compare_archives(old/"state.npz",new/"state.npz","optimizer:")
                if not models["all_arrays_equal"] or not optimizers["all_arrays_equal"]:
                    raise AssertionError(f"Baseline replay differs: seed {seed}, {fold['fold_id']}, step {step}")
                receipts.append({"seed":seed,"fold":fold["fold_id"],"step":step,"models":models,"optimizers":optimizers})
    output.parent.mkdir(parents=True,exist_ok=True)
    result={"format":"phyevo-refinement-baseline-replay-v1","checks":receipts,"input_files_sha256":inputs,
            "all_replayed_model_and_optimizer_arrays_equal":True,"scientific_goal_complete":False}
    write_new_json(output,result)
    return result


def verify_runtime(detection, old_physics, expected_prediction, output, *, require_exact=False):
    detection,old_physics,expected_prediction,output=[contained_path(ROOT,p) for p in (detection,old_physics,expected_prediction,output)]
    if output.exists(): raise FileExistsError("Verification receipt must be new")
    summary=read(detection/"rsi_detection.json")
    if summary["status"]=="physics_fallback" or not summary["physics_preserved"]:
        raise ValueError("Optional feedback did not complete successfully")
    checks,inputs={},{}
    for name in ("mask.npy","score_map.npy","probability_map.npy","reliability_map.npy","decision_map.npy","final_geometry_mask.npy"):
        a,b=old_physics/name,detection/"physics_original"/name
        if not np.array_equal(np.load(a,allow_pickle=False),np.load(b,allow_pickle=False),equal_nan=True):
            raise AssertionError(f"Original physics changed: {name}")
        checks[name]=True
        for p in (a,b): inputs[str(p.relative_to(ROOT))]=sha256_file(p)
    score_path=detection/"student_score.npz"
    with np.load(score_path,allow_pickle=False) as a,np.load(expected_prediction,allow_pickle=False) as b:
        score,expected=a["score"],b["score"]
        if score.shape!=expected.shape: raise ValueError("Student coordinate grid differs")
        exact=bool(np.array_equal(score,expected))
        maximum=float(np.max(np.abs(score-expected)))
        if require_exact and not exact: raise AssertionError(f"Same-backend student differs by {maximum}")
        embedding_equal=bool(np.array_equal(np.asarray(summary["query_embedding"]),b["embedding"]))
        if require_exact and not embedding_equal: raise AssertionError("Same-backend embedding differs")
    inputs[str(score_path.relative_to(ROOT))]=sha256_file(score_path)
    inputs[str(expected_prediction.relative_to(ROOT))]=sha256_file(expected_prediction)
    inputs[str((detection/"rsi_detection.json").relative_to(ROOT))]=sha256_file(detection/"rsi_detection.json")
    checkpoint=contained_path(ROOT,summary["checkpoint"])
    if sha256_file(checkpoint/"checkpoint.json")!=read(expected_prediction.with_suffix(".json"))["checkpoint_sha256"]:
        raise ValueError("Runtime and held-out prediction bind different checkpoints")
    from tfree_detection.rsi_evaluation import EvaluationProtocol
    from tfree_detection.rsi_feedback import feedback_predictions
    number=read(expected_prediction.with_suffix(".json"))["case_number"]
    dataset=ROOT/"outputs/expert_cycle_001/learning_dataset_v1"
    manifest=read(dataset/"manifest.json")
    if sha256_file(dataset/"manifest.json")!=read(dataset/"seal.json")["manifest_sha256"]:
        raise ValueError("Runtime reference dataset changed")
    entry=next(e for e in manifest["entries"] if e["case_number"]==number)
    plane_path=contained_path(ROOT,dataset/entry["plane"])
    if sha256_file(plane_path)!=entry["plane_sha256"]:raise ValueError("Runtime reference plane changed")
    records_path=contained_path(ROOT,Path(manifest["frozen_bundle"])/"cases.jsonl")
    record=next(r for r in map(json.loads,records_path.read_text().splitlines()) if r["case_number"]==number)
    with np.load(plane_path,allow_pickle=False) as plane:
        predictions=feedback_predictions(plane["image"],plane["teacher"],record["prediction"]["candidates"],
            expected,entry["geometry"],protocol=EvaluationProtocol(**summary["protocol"]))
    candidate_checks={}
    for method,prediction in predictions.items():
        ref=summary["candidate_artifacts"][method]
        path=contained_path(ROOT,ref["path"])
        if sha256_file(path)!=ref["sha256"]:raise ValueError("Runtime candidate archive changed")
        key,value=("points_xy",prediction.points_xy) if prediction.points_xy is not None else ("mask",prediction.mask)
        with np.load(path,allow_pickle=False) as actual:candidate_checks[method]=bool(np.array_equal(actual[key],value))
        if require_exact and not candidate_checks[method]:raise AssertionError(f"Runtime/offline candidate differs: {method}")
        inputs[str(path.relative_to(ROOT))]=sha256_file(path)
    for path in (dataset/"manifest.json",plane_path,records_path):inputs[str(path.relative_to(ROOT))]=sha256_file(path)
    result={"format":"phyevo-refinement-runtime-replay-v1","physical_arrays_unchanged":checks,
            "student_score_exact":exact,"student_score_max_absolute_difference":maximum,"exact_required":require_exact,
            "case_number":number,"candidate_arrays_equal_offline":candidate_checks,
            "embedding_exact":embedding_equal,
            "coordinate_shape":list(score.shape),"input_files_sha256":inputs,"runtime_status":summary["status"],
            "primary_branch":summary["primary_branch"],"scientific_goal_complete":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    write_new_json(output,result)
    return result


def verify_experiment(analysis, report, bindings, ledger, runtime_receipts, junit, output):
    """Re-read registered data/results and recompute selection and aggregation.

    This is artifact/algorithm verification. It does not certify material-wide
    safety or make the reused development cohort an independent locked test.
    """
    from scripts.analyze_rsi_refinement import load_analysis, macro_summary, metric_path
    from scripts.verify_rsi_cycle import Audit, disjoint_training, junit_counts, require
    from tfree_detection.rsi_feedback import fit_feedback_selector, selected_configuration
    from tfree_detection.rsi_feedback_runtime import resolve_context_binding
    from tfree_detection.rsi_policy import CapabilityCard,CapabilityLedger,verify_artifacts
    analysis,ap,training,tp,dataset=load_analysis(contained_path(ROOT,analysis))
    output=contained_path(ROOT,output)
    if output.exists(): raise FileExistsError("Experiment verification receipt must be new")
    audit=Audit(ROOT)
    audit.check(training/"protocol.json")
    audit.hashes(tp["source_hashes"])
    completed=audit.read(training/"completed_outer_inner_directed.json")
    require(completed["protocol_sha256"]==sha256_file(training/"protocol.json"),"Training completion binding differs")
    manifest=audit.read(dataset/"manifest.json")
    require(audit.check(dataset/"manifest.json")==tp["learning_manifest_sha256"],"Expert learning manifest changed")
    frozen=contained_path(ROOT,manifest["frozen_bundle"])
    audit.check(frozen/"seal.json",manifest["frozen_seal_sha256"])
    audit.hashes(audit.read(frozen/"seal.json")["files"],frozen)
    entries={e["case_number"]:e for e in manifest["entries"]}
    for entry in entries.values():
        audit.check(dataset/entry["plane"],entry["plane_sha256"])
        audit.check(entry["source_path"],entry["source_file_sha256"])
    audit.hashes(audit.read(analysis/"result_seal.json")["files"],analysis)
    inner=defaultdict(list)
    outer,directed=[],[]
    models,checkpoints,scores=0,0,0
    for policy in tp["policies"]:
        for seed in tp["seeds"]:
            for fold in tp["folds"]:
                run=training/policy/f"seed-{seed}"/fold["fold_id"]
                marker=audit.read(run/"worker_completed.json")
                require(marker["protocol_sha256"]==sha256_file(training/"protocol.json"),"Worker protocol differs")
                audit.check(run/"evaluation_manifest.json",marker["evaluation_sha256"])
                index=audit.read(run/"evaluation_manifest.json")
                require([c["step"] for c in index["checkpoints"]]==tp["checkpoints"],"Checkpoint inventory incomplete")
                for c in index["checkpoints"]:
                    audit.hashes(c["files"])
                    directory=contained_path(ROOT,c["checkpoint"])
                    meta=audit.read(directory/"checkpoint.json")
                    require(meta["policy"]==policy and meta["seed"]==seed and meta["fold"]==fold and meta["step"]==c["step"],"Model identity differs")
                    disjoint_training(entries,meta,fold["excluded_groups"],fold["train_case_numbers"])
                    require({r["case_number"] for r in c["cases"]}==set(fold["validation_case_numbers"]),"Score cohort incomplete")
                    bank=audit.read(directory/"training_embedding_bank.json")
                    require(bank["selection_uses_validation"] is False and {r["case_number"] for r in bank["cases"]}<=set(fold["train_case_numbers"]),"Embedding bank contains held-out examples")
                    scores+=len(c["cases"]);checkpoints+=1
                path=metric_path(analysis,policy,seed,fold)
                seal=audit.read(path.with_suffix(".seal.json"))
                audit.check(path,seal["sha256"]);audit.hashes(seal["inputs"])
                metrics=audit.read(path)
                require(metrics["analysis_protocol_sha256"]==sha256_file(analysis/"analysis_protocol.json"),"Metric protocol differs")
                require(len(metrics["rows"])==len(fold["validation_case_numbers"])*len(tp["checkpoints"]),"Metric row count differs")
                require({(r["case_number"],r["step"]) for r in metrics["rows"]}==
                    {(n,s) for n in fold["validation_case_numbers"] for s in tp["checkpoints"]},"Metric case/step inventory differs")
                for row in metrics["rows"]:
                    for method,value in row["methods"].items():
                        require(value["precision"] is None and value["f1"] is None,"Incomplete labels claimed precision/F1")
                        count=value.get("candidate_count",value.get("selected_pixels"))
                        if method!="physics_native" and count is not None:require(count<=value["budget"],"Candidate budget exceeded")
                    if fold["stage"]=="inner":inner[row["outer_group"]].append(row)
                    elif fold["stage"]=="outer":outer.append(row)
                    else:directed.append(row)
                models+=1
    selectors={}
    for fold in tp["folds"]:
        if fold["stage"]!="outer":continue
        group=fold["excluded_groups"][0]
        actual=audit.read(analysis/"selectors"/(fold["fold_id"]+".json"))
        require(fit_feedback_selector(inner[group],outer_group=group,**ap["selection"])==actual,"Refitted inner selector differs")
        selectors[group]=actual
    summary=audit.read(analysis/"summary.json")
    require(summary["outer_comparisons"]==macro_summary(outer),"Outer macro scores differ")
    require(summary["directed_same_material_comparisons"]==macro_summary(directed),"Directed scores differ")
    routed=audit.read(analysis/"selected_outer_cases.json")
    lookup={(r["group"],r["case_number"],r["seed"],r["policy"],r["step"]):r for r in outer}
    require(len(lookup)==len(outer),"Duplicate outer metrics")
    require({(r["case_number"],r["seed"]) for r in routed}==
        {(r["case_number"],r["seed"]) for r in outer} and
        len(routed)==len({(r["case_number"],r["seed"]) for r in routed}),"Routed case inventory differs")
    for row in routed:
        choice=selected_configuration(selectors[row["group"]],geometry=row["geometry"],teacher_empty=row["teacher_empty"])
        require(choice==row["selected_configuration"],"Routed configuration differs")
        policy,step=(choice["policy"],choice["step"]) if choice else ("control",400)
        original=lookup[row["group"],row["case_number"],row["seed"],policy,step]
        branch=choice["branch"] if choice and not (choice["embedding_veto"] and original["embedding_context"]["novelty_veto"]) else "physics_budgeted"
        require(branch==row["selected_branch"] and original["methods"][branch]==row["methods"]["nested_feedback"],"Routed branch or metric differs")
    require(summary["nested_selected_comparison"]==macro_summary(routed),"Routed aggregate differs")
    report=contained_path(ROOT,report)
    seal=audit.read(report/"result_seal.json")
    audit.hashes(seal["files"],report);audit.hashes(seal["inputs"])
    bindings=contained_path(ROOT,bindings)
    registries=list(bindings.glob("*/seed-*/registry.json"))
    require(len(registries)==len(selectors)*len(tp["seeds"]),"Missing model family registry")
    for registry in registries:
        audit.check(registry)
        for geometry in ("point","interface"):
            for empty in (False,True):
                path,_=resolve_context_binding(ROOT,registry,geometry=geometry,teacher_empty=empty)
                audit.check(path)
    ledger=contained_path(ROOT,ledger)
    ledger_seal=audit.read(ledger/"seal.json")
    audit.hashes(ledger_seal["files"],ledger);audit.hashes(ledger_seal["source_hashes"])
    require(ledger_seal["analysis_seal_sha256"]==sha256_file(analysis/"result_seal.json"),"Ledger uses different analysis")
    events=CapabilityLedger(ROOT,ledger/"events")
    decisions=audit.read(ledger/"release_summary.json")
    require(not events.active_cards() and decisions["production_activated"]==0,"Development promoted production")
    require(len(events.events())==decisions["events"],"Ledger event count differs")
    for decision in decisions["decisions"]:
        if "card" not in decision:continue
        card=CapabilityCard.from_dict(audit.read(decision["card"]))
        require(verify_artifacts(card,ROOT),"Ledger card evidence differs")
        subset=[r for r in routed if r["main_comparison_eligible"] and r["group"]==decision["outer_group"]
                and r["seed"]==decision["seed"] and r["geometry"]==decision["geometry"] and r["teacher_empty"]==decision["teacher_empty"]]
        for field,method in (("reference","physics_budgeted"),("candidate","nested_feedback")):
            value=sum(r["methods"][method]["positive_coverage"] for r in subset)/len(subset)
            require(value==decision[field] and value==getattr(card.metrics[0],field),"Ledger metric not tied to selected fixed-seed cases")
    probes=[]
    for path in runtime_receipts:
        value=audit.read(path);audit.hashes(value["input_files_sha256"])
        require(value["student_score_exact"] and value["embedding_exact"] and value["exact_required"],"GPU runtime not exact")
        require(all(value["physical_arrays_unchanged"].values()),"Runtime changed physics")
        require(all(value["candidate_arrays_equal_offline"].values()),"Runtime proposals differ from offline evaluation")
        probes.append(value["case_number"])
    require(set(probes)=={63,103,135},"Three raw-image runtime probes required")
    counts=junit_counts(contained_path(ROOT,junit));audit.check(junit)
    result={"format":"phyevo-refinement-experiment-verification-v1","models":models,"checkpoints":checkpoints,
        "score_records":scores,"expert_cases":len(entries),"refitted_selectors":len(selectors),
        "model_families":len(registries),"research_evidence_events":decisions["events"],"runtime_cases":probes,"tests":counts,
        "input_files_sha256":audit.files,"scientific_goal_complete":False,
        "scope":"artifact integrity, source isolation, recomputed selection and unchanged runtime physics; not external safety"}
    output.parent.mkdir(parents=True,exist_ok=True)
    write_new_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("baselines");p.add_argument("--training",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("runtime");p.add_argument("--detection",type=Path,required=True);p.add_argument("--old-physics",type=Path,required=True)
    p.add_argument("--expected-prediction",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--require-exact",action="store_true")
    p=sub.add_parser("experiment");p.add_argument("--analysis",type=Path,required=True);p.add_argument("--report",type=Path,required=True)
    p.add_argument("--bindings",type=Path,required=True);p.add_argument("--runtime-receipts",type=Path,nargs="+",required=True)
    p.add_argument("--ledger",type=Path,required=True)
    p.add_argument("--junit",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="baselines":result=verify_baselines(args.training,args.output)
    elif args.command=="runtime":
        result=verify_runtime(args.detection,args.old_physics,args.expected_prediction,args.output,require_exact=args.require_exact)
    else:result=verify_experiment(args.analysis,args.report,args.bindings,args.ledger,args.runtime_receipts,args.junit,args.output)
    print(json.dumps({"output":str(args.output),"format":result["format"]}))


if __name__=="__main__": main()
