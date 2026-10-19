"""Seal all related development searches before scoring any held-out source."""
from __future__ import annotations

import argparse
import fcntl
import json

from .archive import Archive
from .contracts import Protocol, file_hash, inside, write_new
from .executor import Executor
from .real_study import ROOT, ARMS, load_study, read
from .sandbox import Sandbox


def open_archive(directory):
    values = read(directory / "protocol.json")
    protocol = Protocol(**{k: tuple(v) if k.endswith("_groups") or k == "seeds" else v
                           for k, v in values.items()})
    return Archive(directory, protocol)


def assert_frozen(archive, specification):
    if archive.protocol.identity != specification["protocol_sha256"]:
        raise ValueError("Final-audit protocol differs from the joint seal")
    seals = archive.events("search_sealed")
    if not seals or seals[-1]["hash"] != specification["seal"]:
        raise ValueError("Missing or changed final-audit seal")
    if archive.incumbent() != specification["incumbent"]:
        raise ValueError("Task selection changed after final-audit planning")
    for event in archive.events():
        if event["seq"] <= seals[-1]["seq"]:
            continue
        if event["kind"] != "trial_recorded" or event["body"]["split"] != "audit":
            raise ValueError("Development activity continued after the search seal")
    for identity, code in specification["programs"].items():
        item = archive.candidate(identity)
        if item["code_hash"] != code["pipeline"] or item["policy_hash"] != code["policy"]:
            raise ValueError("Sealed candidate source changed")
        if item["proposal"].training_steps != 0:
            raise ValueError("Final audit must use a frozen inference-only program")


def seal_all(directory):
    directory = inside(ROOT, directory)
    target = directory / "final_audit_plan.json"
    if target.exists():
        print(json.dumps({"already_sealed": file_hash(target)}))
        return
    specifications = []
    for arm in ARMS:
        specifications.append((f"interface_live/{arm}", "three_arm_comparison", None))
    meta = read(directory / "meta_live/meta_result.json")
    continued = read(directory / "meta_live/continuation-result.json")
    if continued["policy"] != meta["selected_policy"]:
        raise ValueError("Research-strategy continuation was not completed")
    learning = read(directory / "learning_live/result.json")
    if not learning["checkpoint_reuse"] or not all(r["exact"] for r in learning["checkpoint_reuse"]):
        raise ValueError("Learning checkpoint reuse must be validated before final audit")
    specifications += [("meta_live/archive", "separate_meta_research", None),
                       ("learning_live/archive", "separate_learning_integration", learning["inference_candidate"])]
    records = []
    for name, scope, additional in specifications:
        archive = open_archive(directory / name)
        try:
            if any(t["split"] == "audit" for t in archive.trials(visible_only=False)):
                raise ValueError("Final audit was inspected before joint sealing")
            if scope == "three_arm_comparison" and len(archive.events("round_finished")) != archive.protocol.maximum_rounds:
                raise ValueError("Incomplete equal-budget search")
            baseline = archive.events("baseline_registered")[0]["body"]["candidate"]
            identities = list(dict.fromkeys([baseline, archive.incumbent()] + ([additional] if additional else [])))
            programs = {}
            for identity in identities:
                item = archive.candidate(identity)
                if item["proposal"].training_steps:
                    raise ValueError("Refuse to retrain during final audit; freeze an inference descendant first")
                programs[identity] = {"pipeline": item["code_hash"], "policy": item["policy_hash"]}
            seals = archive.events("search_sealed")
            seal = seals[-1]["hash"] if seals else archive.event("search_sealed", {
                "incumbent": archive.incumbent(), "scope": scope,
                "final_audit_candidates": identities, "reason": "all declared development requests terminal"})
            records.append({"archive": name, "scope": scope, "baseline": baseline,
                            "incumbent": archive.incumbent(), "programs": programs,
                            "protocol_sha256": archive.protocol.identity, "seal": seal})
        finally:
            archive.close()
    plan = {"archives": records, "all_sealed_before_first_audit": True,
            "model_requests_after_seal_allowed": False, "selection_from_audit_allowed": False,
            "trained_candidate_audited_even_if_development_rejected": True,
            "meta_result_sha256": file_hash(directory / "meta_live/meta_result.json"),
            "learning_result_sha256": file_hash(directory / "learning_live/result.json")}
    write_new(target, plan)
    print(json.dumps({"sealed_archives": len(records), "plan_sha256": file_hash(target)}))


def execute_audit(directory, gpu=None):
    directory = inside(ROOT, directory)
    plan = read(directory / "final_audit_plan.json")
    _, source_protocol, samples = load_study(directory / "interface_live")
    # Inspect every seal before executing the first candidate.
    for specification in plan["archives"]:
        archive = open_archive(directory / specification["archive"])
        try:
            assert_frozen(archive, specification)
            if archive.protocol.data_version != source_protocol.data_version:
                raise ValueError("Final-audit data contract changed")
        finally:
            archive.close()
    results = []
    for specification in plan["archives"]:
        archive = open_archive(directory / specification["archive"])
        try:
            executor = Executor(archive, Sandbox(ROOT, ROOT / ".venvs/sam2", gpu=gpu), samples)
            for identity in specification["programs"]:
                trial, measurement = executor.execute(identity, split="audit", audit_seal=specification["seal"])
                if "rows" not in measurement:
                    raise RuntimeError("A final-audit execution failed: " + trial)
                if len(measurement["executions"]) != len(archive.protocol.seeds):
                    raise ValueError("Unexpected training during frozen final audit")
                row = {"archive": specification["archive"], "scope": specification["scope"],
                       "candidate": identity, "baseline": identity == specification["baseline"],
                       "incumbent": identity == specification["incumbent"], "trial": trial,
                       "summary": measurement["summary"], "checkpoint": measurement["checkpoints"],
                       "actual_execution": measurement["actual_execution"], "selection_changed": False}
                results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            assert_frozen(archive, specification)
        finally:
            archive.close()
    target = directory / "final_audit_results.json"
    payload = {"plan_sha256": file_hash(directory / "final_audit_plan.json"), "results": results,
               "audit_used_for_selection": False, "training_during_audit": False,
               "metric": "known_positive_area_coverage_at_fixed_area"}
    if target.exists():
        if read(target) != payload:
            raise ValueError("Frozen audit results changed on replay")
    else:
        write_new(target, payload)
    print(json.dumps({"audit_complete": True, "executed_or_verified_cached_trials": len(results)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "execute"))
    parser.add_argument("directory")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    directory = inside(ROOT, args.directory)
    lock_path = inside(ROOT, directory / "final_audit.runner.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == "seal":
            seal_all(directory)
        else:
            execute_audit(directory, args.gpu)


if __name__ == "__main__":
    main()
