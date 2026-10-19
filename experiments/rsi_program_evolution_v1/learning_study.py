"""One bounded, model-generated learner and actual frozen-checkpoint reuse.

This is a separately declared integration experiment, NOT an extra round in
the three-arm equal-budget comparison. No existing checkpoint is overwritten.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
import fcntl
import json

import numpy as np

from .archive import Archive
from .contracts import Protocol, file_hash, inside, write_new
from .controller import Controller
from .evaluation import compare
from .executor import Executor
from .meta_study import MODEL, emit_request, latest_success, receive_response
from .provider import FileExchangeModel
from .real_study import ROOT, load_study, read
from .sandbox import Sandbox
from .training_trace import trace_check


def load(directory, gpu=None):
    directory = inside(ROOT, directory)
    config = read(directory / "study.json")
    _, source_protocol, samples = load_study(config["source"])
    protocol = replace(source_protocol, maximum_model_calls=1, maximum_rounds=1, maximum_training_steps=64)
    archive = Archive(directory / "archive", protocol)
    executor = Executor(archive, Sandbox(ROOT, ROOT / ".venvs/sam2", gpu=gpu), samples)
    return directory, config, archive, executor


def validate_learner(proposal, parent, protocol):
    proposal.validate(protocol)
    functions = {n.name for n in ast.parse(proposal.pipeline_code).body if isinstance(n, ast.FunctionDef)}
    if (proposal.action != "tune" or proposal.parents != (parent,) or proposal.target != "learning"
            or not 1 <= proposal.training_steps <= 64 or "fit" not in functions):
        raise ValueError("Expected a bounded trainable learning proposal")


def prepare(directory, source, meta):
    source, p, samples = load_study(source)
    directory, meta = inside(ROOT, directory), inside(ROOT, meta)
    if directory.exists():
        raise FileExistsError("Learning experiment requires a new destination")
    selected = read(meta / "meta_result.json")
    meta_protocol = Protocol(**{k: tuple(v) if k.endswith("_groups") or k == "seeds" else v
                                for k, v in read(meta / "archive/protocol.json").items()})
    previous = Archive(meta / "archive", meta_protocol)
    try:
        if any(t["split"] == "audit" for t in previous.trials(visible_only=False)):
            raise ValueError("Do not propose learning after final-audit inspection")
        parent = selected["selected_task_program"]
        lineage = {}
        def visit(identity):
            item = previous.candidate(identity)
            for ancestor in item["parents"]:
                visit(ancestor) if ancestor not in lineage else None
            lineage[identity] = item["proposal"]
        visit(parent)
        parent_tail = previous.verify()
    finally:
        previous.close()
    protocol = replace(p, maximum_model_calls=1, maximum_rounds=1, maximum_training_steps=64)
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / "study.json", {"source": str(source.relative_to(ROOT)),
        "parent_archive": str(meta.relative_to(ROOT)), "parent_event_tail": parent_tail, "parent": parent,
        "scope": "separate learner integration experiment, not part of equal-budget three-arm comparison",
        "search_decision": "researcher-requested coverage of the executable learning pathway",
        "maximum_model_calls": 1, "maximum_training_steps": 64, "final_audit_not_seen": True})
    archive = Archive(directory / "archive", protocol, create=True)
    executor = Executor(archive, Sandbox(ROOT, ROOT / ".venvs/sam2"), samples)
    try:
        for identity, proposal in lineage.items():
            assert archive.add(proposal) == identity
        trial, baseline = executor.execute(parent)
        if "rows" not in baseline:
            raise RuntimeError("Learning reference failed to execute")
        archive.select(parent, True, "frozen_learning_reference", trial)
        archive.event("baseline_registered", {"candidate": parent, "trial": trial})
        model = FileExchangeModel(directory / "unused_controller_transport", model=MODEL, maximum_calls=1)
        context = Controller(archive, executor, model).context(1)
        training = [s for s in samples if protocol.split(s.group) == "train"]
        context.update(required_action="tune", required_parents=[parent], required_target="learning",
            parent_programs=[asdict(lineage[parent])],
            task="Implement and test a small trainable microscopy student using existing expert positives, "
                 "then use the learned state in predict. This is a declared learning-path integration experiment, "
                 "not evidence that the preceding research policy itself selected training.",
            training_summary={"planes": len(training), "known_positive_pixels": sum(int(s.human.sum()) for s in training),
                              "unknown_pixels_are_negative": False},
            learning_contract={
                "architecture": "Choose a bounded small neural learner or trainable residual; use torch autograd "
                    "and real optimizer updates. Architecture, features, supervision, sampling, and conservative "
                    "combination with the existing pipeline are your design decisions.",
                "supervision": "Only fit sees samples[i]['human'] (bool positive mask) and ['known'] (nonnegative "
                    "annotation weight). Never treat unannotated pixels as confirmed negatives. Physics/student/base "
                    "fields are fallible soft priors, not expert ground truth. Inference sees none of the labels.",
                "runtime": "1 to 64 declared training_steps, exactly context['training_steps'] optimizer steps; "
                    "at most 180 seconds per sandbox stage, small batches and less than 2 GiB working GPU memory. "
                    "Use context['device']; CPU fallback must work. No downloads or repository imports.",
                "checkpoint": "fit returns only finite numeric numpy arrays. Save actual trainable tensors for "
                    "predict to restore, plus audit_initial_parameters and audit_final_parameters (flat vectors of "
                    "the same trainable parameters), audit_loss_trace (one actual loss per optimizer step), "
                    "audit_steps (one integer). Do not fabricate these arrays. predict must work when fit is "
                    "not rerun and only this checkpoint is inherited. Preserve any useful parent state explicitly.",
                "evaluation": "Use the fixed source-disjoint development protocol; keep no source IDs or fixed "
                    "coordinates in the model. The output support may expand inside physically valid pixels. "
                    "The original pipeline remains the incumbent unless the fixed admission test passes."})
        request_hash = emit_request(directory / "request", context, protocol)
        archive.event("learning_request_reserved", {"request_sha256": request_hash, "model": MODEL})
        write_new(directory / "prepared.json", {"parent": parent, "baseline_trial": trial,
                   "training_planes": len(training), "request_sha256": request_hash})
        print(json.dumps({"request": str(directory / "request"), "training_planes": len(training)}))
    finally:
        archive.close()


def execute(directory, gpu=None):
    directory, config, archive, executor = load(directory, gpu)
    try:
        if (directory / "result.json").exists():
            print(json.dumps(read(directory / "result.json")))
            return
        if archive.events("search_sealed"):
            raise ValueError("Learning search already sealed")
        proposal, model_receipt = receive_response(directory / "request", archive.protocol)
        validate_learner(proposal, config["parent"], archive.protocol)
        candidate = archive.add(proposal)
        trial, receipt = executor.execute(candidate)
        if "rows" not in receipt:
            write_new(directory / "failed_execution.json", {"candidate": candidate, "trial": trial,
                       "receipt": receipt, "model_receipt": model_receipt})
            print(json.dumps({"status": "execution_failed", "trial": trial}))
            return
        training = {}
        for seed, checkpoint in receipt["checkpoints"].items():
            path = inside(archive.root, checkpoint["path"])
            if file_hash(path) != checkpoint["sha256"]:
                raise ValueError("Training checkpoint changed")
            with np.load(path, allow_pickle=False) as z:
                state = {k: z[k] for k in z.files}
            training[seed] = trace_check(state, proposal.training_steps)
        parent = latest_success(archive, config["parent"])
        decision = compare(receipt["rows"], parent["receipt"]["rows"], parent["receipt"]["rows"], archive.protocol)
        # An exact inference-only descendant proves checkpoint reuse and prevents
        # re-fitting on the final audit. This wrapper is not an LLM discovery.
        inference = replace(proposal, parents=(candidate,), training_steps=0,
            observation="Trained checkpoint exists; verify exact reuse without further fitting",
            intervention="Run identical prediction code with inherited checkpoint and zero training steps",
            expected_effect="Same numeric predictions as trained candidate",
            falsifier="Changed checkpoint hash or any changed prediction values")
        replay = archive.add(inference)
        replay_trial, replay_receipt = executor.execute(replay)
        if "rows" not in replay_receipt or replay_receipt["checkpoints"] != receipt["checkpoints"]:
            raise ValueError("Checkpoint inheritance did not preserve the trained state")
        comparisons = []
        for before, after in zip(receipt["rows"], replay_receipt["rows"]):
            if (before["case"], before["seed"]) != (after["case"], after["seed"]):
                raise ValueError("Replay cohort changed")
            with np.load(inside(archive.root, before["prediction"]), allow_pickle=False) as a:
                with np.load(inside(archive.root, after["prediction"]), allow_pickle=False) as b:
                    exact = all(np.array_equal(a[k], b[k]) for k in ("score", "support"))
            if not exact:
                raise ValueError("Inherited inference changed numeric predictions")
            comparisons.append({"case": before["case"], "seed": before["seed"], "exact": True})
        archive.select(replay, decision["accepted"], decision, replay_trial)
        archive.remember(candidate, trial, {"hypothesis": proposal.hypothesis, "decision": decision},
                         {"split": "validation", "scope": config["scope"]})
        result = {"candidate": candidate, "trial": trial, "inference_candidate": replay,
            "inference_trial": replay_trial, "model_receipt": model_receipt,
            "training": training, "checkpoint_reuse": comparisons,
            "summary": receipt["summary"], "reference_summary": parent["receipt"]["summary"],
            "decision": decision, "incumbent": archive.incumbent(),
            "gpu": gpu, "new_annotation_requested": False,
            "integration_probe_not_equal_budget_control": True}
        write_new(directory / "result.json", result)
        archive.event("learning_finished", {"candidate": candidate, "inference_candidate": replay,
                                            "result_sha256": file_hash(directory / "result.json")})
        print(json.dumps(result, ensure_ascii=False))
    finally:
        archive.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("directory")
    parser.add_argument("--source")
    parser.add_argument("--meta")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    destination = inside(ROOT, args.directory)
    lock_path = inside(ROOT, destination.with_name(destination.name + ".runner.lock"))
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == "prepare":
            if not args.source or not args.meta:
                parser.error("prepare requires --source and --meta")
            prepare(args.directory, args.source, args.meta)
        else:
            execute(args.directory, args.gpu)


if __name__ == "__main__":
    main()
