"""Additive real-microscopy study with a sealed external-model exchange.

Cached fields/checkpoints are verified, not rewritten. This first live study is
an interface-source transfer experiment; it is not labelled a new blind dataset
or a point-defect experiment. The broader engine is task-agnostic.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from .archive import Archive
from .contracts import Protocol, engine_fingerprint, file_hash, inside, verify_snapshot, write_new
from .controller import Controller
from .executor import Executor, Sample, sample_fingerprint
from .provider import FileExchangeModel, publish_response
from .sandbox import Sandbox
from .study_io import load_study as _load_study


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/expert_cycle_001/learning_dataset_v1"
HISTORICAL = ROOT / "outputs/autonomous_rsi_20261015_v1"
PHYSICAL = ROOT / "outputs/interface_improvement_20261015_v1"
ARMS = ("full", "generic", "no_memory")


def read(path):
    return json.loads(inside(ROOT, path).read_text())


def checked(path, expected):
    path = inside(ROOT, path)
    if file_hash(path) != expected:
        raise ValueError("Source checksum mismatch: " + str(path.relative_to(ROOT)))
    return path


def arrays(path):
    with np.load(inside(ROOT, path), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def exclusion_audit(job, base_meta, retained_meta, entries):
    withheld = {job["validation_group"], job["outer_group"]}
    all_numbers = set(base_meta["train_case_numbers"]) | set(retained_meta["train_case_numbers"])
    used_groups = {entries[n]["group"] for n in all_numbers}
    if used_groups & withheld:
        raise ValueError("Inherited model was trained on a withheld source")
    if not withheld <= set(base_meta["fold"]["excluded_groups"]):
        raise ValueError("Inherited base fold does not exclude both held-out sources")
    return {"all_inherited_training_groups": sorted(used_groups),
            "all_inherited_training_cases": sorted(all_numbers),
            "new_fit_groups": job["train_groups"], "new_fit_cases": job["train_cases"],
            "validation_group": job["validation_group"], "audit_group": job["outer_group"],
            "withheld_sources_absent_from_both_weight_training_stages": True}


def prepare(destination, *, job_index=6, rounds=3):
    from tfree_detection.expert_student import predict_scores, restore_checkpoint, torch
    from tfree_detection.interface_geometry import ranked_support
    from tfree_detection.interface_reliability import qualify_boundary_or_band
    from tfree_detection.interface_runtime import choose_interface

    destination = inside(ROOT, destination)
    if destination.exists():
        raise FileExistsError("Use a new study directory; never overwrite a study")
    before = read(ROOT / "outputs/rsi_program_evolution_20261016_v1/protected_before.json")
    verify_snapshot(ROOT, before)
    old = read(HISTORICAL / "protocol.json")
    job = old["jobs"][job_index]
    if job["stage"] != "inner":
        raise ValueError("A source-excluded development job is required")
    checked(DATA / "manifest.json", old["data_manifest_sha256"])
    checked(PHYSICAL / "physical_audit.json", old["physical_audit_sha256"])
    checked(PHYSICAL / "stability/audit.json", old["stability_audit_sha256"])
    entries = {e["case_number"]: e for e in read(DATA / "manifest.json")["entries"]}
    checkpoint = inside(ROOT, job["checkpoint"])
    checked(checkpoint / "checkpoint.json", job["checkpoint_sha256"])
    checked(checkpoint / "state.npz", job["state_sha256"])
    base = inside(ROOT, job["base"])
    checked(base / "checkpoint.json", job["base_checkpoint_sha256"])
    checked(base / "state.npz", job["base_state_sha256"])
    source_audit = exclusion_audit(job, read(base / "checkpoint.json"), read(checkpoint / "checkpoint.json"), entries)
    cache = HISTORICAL / "cache" / f"job-{job_index:02d}"
    complete = read(cache / "complete.json")
    if complete["job"] != job or complete["protocol_sha256"] != file_hash(HISTORICAL / "protocol.json"):
        raise ValueError("Historical cache is not bound to this checkpoint and protocol")
    cached = {r["case_number"]: r for r in complete["files"]}
    phys_records = {r["case_number"]: r for r in read(PHYSICAL / "physical_audit.json")["records"]}
    guards = {r["case_number"]: r["guard"] for r in read(PHYSICAL / "stability/audit.json")["records"]}
    audit_numbers = [n for n, e in entries.items() if e["group"] == job["outer_group"]
                     and e["geometry"] == "interface" and e["human_positive_pixels"] > 0]
    numbers = sorted(set(job["train_cases"] + job["test_cases"] + audit_numbers))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "samples").mkdir()
    # Trusted existing inference code, CPU and bounded threads; only new derived
    # study artifacts are written. No generated code runs outside Sandbox.
    torch.set_num_threads(2)
    model, _, _, _ = restore_checkpoint(checkpoint, device="cpu")
    records, samples = [], []
    for number in numbers:
        entry = entries[number]
        plane_path = checked(DATA / entry["plane"], entry["plane_sha256"])
        plane = arrays(plane_path)
        if number in cached:
            score_path = checked(cache / f"case-{number:03d}.npz", cached[number]["sha256"])
            student = arrays(score_path)["score"]
            origin = {"kind": "verified_cached_score", "path": str(score_path.relative_to(ROOT)),
                      "sha256": cached[number]["sha256"]}
        else:
            student = predict_scores(model, plane["image"], 4, device="cpu", normalized=True)
            origin = {"kind": "new_cpu_inference_same_source_excluded_checkpoint",
                      "checkpoint_sha256": job["checkpoint_sha256"], "state_sha256": job["state_sha256"]}
        ph_path = checked(PHYSICAL / "physical" / f"case-{number:03d}.npz", phys_records[number]["sha256"])
        ph = arrays(ph_path)
        guard = qualify_boundary_or_band(ph, guards[number])
        old_mask, route = choose_interface(ph["score"], ph["valid"], guard, student,
            student_admitted=job["admitted"], budget=math.ceil(.1 * student.size))
        physical_route = route["branch"] != "student"
        base_score = ph["score"] if physical_route else student
        support = (base_score > 0) & (ph["valid"].astype(bool) if physical_route else np.ones(student.shape, bool))
        if not np.array_equal(old_mask, ranked_support(base_score, math.ceil(.1 * student.size), valid=support)):
            raise ValueError("Current routed baseline replay mismatch")
        fields = {"image": plane["image"], "base": base_score, "support": support,
                  "valid": np.isfinite(plane["image"]), "physics": ph["score"], "student": student,
                  "physics_valid": ph["valid"].astype(bool), "variation": ph["variation"],
                  "domain_labels": ph["labels"]}
        sample = Sample(f"case-{number:03d}", entry["group"], "interface", fields,
                        plane["human"].astype(bool), plane["human_weight"], np.empty((0, 2)))
        name = f"samples/case-{number:03d}.npz"
        with (destination / name).open("xb") as f:
            np.savez_compressed(f, **fields, human=sample.human, known=sample.known, points_xy=sample.points_xy)
        records.append({"key": sample.key, "group": sample.group, "geometry": sample.geometry,
                        "file": name, "sha256": file_hash(destination / name),
                        "source_plane": str(plane_path.relative_to(ROOT)), "source_sha256": entry["plane_sha256"],
                        "physical_sha256": phys_records[number]["sha256"], "student_origin": origin,
                        "baseline_route": route["branch"], "baseline_mask_exact": True})
        samples.append(sample)
    protocol = Protocol(tuple(source_audit["all_inherited_training_groups"]),
                        (job["validation_group"],), (job["outer_group"],), seeds=(job["seed"],),
                        maximum_rounds=rounds, maximum_model_calls=rounds,
                        data_version=sample_fingerprint(samples), engine_version=engine_fingerprint())
    write_new(destination / "protocol.json", asdict(protocol))
    with ZipFile(destination / "engine_source.zip", "x", compression=ZIP_DEFLATED) as frozen:
        for source in sorted(Path(__file__).resolve().parent.glob("*.py")):
            frozen.write(source, str(source.relative_to(ROOT)))
    write_new(destination / "dataset.json", {"samples": records, "source_audit": source_audit,
        "data_version": protocol.data_version, "frozen_model_seed": job["seed"],
        "scope": "interface development transfer, not a new blind external test",
        "historical_fields_and_model_selection_development_informed": True,
        "all_new_training_labels_from_existing_experts": True,
        "current_fit_sources_are_subset_of_inherited_training_sources": True,
        "final_audit_not_scored_during_prepare": True})
    write_new(destination / "comparison.json", {"arms": ARMS, "model": "gpt-5.6-sol",
        "calls_per_arm": rounds, "maximum_training_steps_per_call_per_seed": protocol.maximum_training_steps,
        "same_initial_data_and_program": True, "output_token_limit_is_requested_not_provider_enforced": True,
        "independent_model_repeats": 1, "statistical_significance_claimed": False,
        "seal_all_arms_before_any_audit": True})
    write_new(destination / "protected_check.json", verify_snapshot(ROOT, before))
    print(json.dumps({"prepared": str(destination), "samples": len(samples), "sources": source_audit}, ensure_ascii=False))


def load_study(directory):
    # Historical callers retain this API; current callers import study_io directly.
    return _load_study(directory, root=ROOT)


def loop(directory, arm, *, gpu=None):
    if arm not in ARMS:
        raise ValueError("Unknown study arm")
    directory, protocol, samples = load_study(directory)
    target = directory / arm
    archive = Archive(target, protocol, create=not target.exists())
    model = FileExchangeModel(target / "exchange", model=read(directory / "comparison.json")["model"],
                              maximum_calls=protocol.maximum_model_calls, timeout=43200)
    executor = Executor(archive, Sandbox(ROOT, ROOT / ".venvs/sam2", gpu=gpu), samples)
    return Controller(archive, executor, model, arm=arm), archive, executor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "advance", "publish", "status", "seal", "audit"))
    parser.add_argument("directory")
    parser.add_argument("--arm", choices=ARMS, default="full")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--job", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--raw")
    parser.add_argument("--origin")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory, job_index=args.job, rounds=args.rounds)
        return
    if args.command == "publish":
        if not args.raw or not args.origin:
            parser.error("publish requires --raw and --origin")
        response_hash = publish_response(inside(ROOT, args.directory), raw_text=inside(ROOT, args.raw).read_text(),
                                         model="gpt-5.6-sol", origin=args.origin)
        print(json.dumps({"published": response_hash}))
        return
    controller, archive, executor = loop(args.directory, args.arm, gpu=args.gpu)
    try:
        if args.command == "advance":
            print(json.dumps(controller.step(), ensure_ascii=False), flush=True)
        elif args.command == "status":
            print(json.dumps({"incumbent": archive.incumbent(), "rounds": [x["body"] for x in archive.events("round_finished")],
                              "calls": len(archive.events("model_call_started"))}, ensure_ascii=False))
        elif args.command == "seal":
            if len(archive.events("round_finished")) != archive.protocol.maximum_rounds:
                raise ValueError("Complete the preregistered rounds before sealing")
            print(json.dumps({"seal": controller.seal()}))
        else:
            directory = inside(ROOT, args.directory)
            # Audit is withheld until every comparison arm has stopped searching.
            for arm in ARMS:
                other = Archive(directory / arm, archive.protocol)
                try:
                    if not other.events("search_sealed"):
                        raise ValueError("All comparison arms must be sealed before audit")
                finally:
                    other.close()
            seal = archive.events("search_sealed")[-1]["hash"]
            initial = archive.events("baseline_registered")[0]["body"]["candidate"]
            for candidate in dict.fromkeys((initial, archive.incumbent())):
                trial, result = executor.execute(candidate, split="audit", audit_seal=seal)
                print(json.dumps({"candidate": candidate, "trial": trial, "summary": result.get("summary")}, ensure_ascii=False))
    finally:
        archive.close()


if __name__ == "__main__":
    main()
