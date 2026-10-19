"""Paired, executable research-policy evaluation over real development evidence.

This runner is deliberately outside the frozen eight-module task engine. It
does not edit that engine or the earlier three-arm experiment. Requests are
durable, nonblocking artifacts: a disconnected terminal neither loses a model
answer nor silently spends another proposal. Generated policies run in bwrap.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path

from .archive import Archive
from .contracts import Proposal, digest, file_hash, inside, verify_snapshot, write_new
from .controller import Controller
from .evaluation import compare, summarize
from .executor import Executor
from .provider import FileExchangeModel, PROPOSAL_SCHEMA, proposer_protocol
from .real_study import ARMS, ROOT, load_study, read
from .sandbox import Sandbox


MODEL = "gpt-5.6-sol"
KEYS = ("policy", "p0-control", "p0-candidate", "p1-control", "p1-candidate", "continuation")
INSTRUCTIONS = (
    "Return one Proposal JSON object matching response_schema. Use only this sealed request. "
    "Implement complete Python pipeline_code and policy_code, not pseudocode. Follow required_action, "
    "required_parents and required_target if supplied. predict(sample,state,context) returns a finite "
    "2D score and boolean support matching image shape, inside sample['valid']. fit(samples,context) "
    "returns numeric-array state; only fit receives human/known training labels. Unknown pixels are "
    "NOT negative labels. plan(context) returns action, parents and focus. Libraries: numpy/scipy/"
    "skimage/torch. No repository imports, network, raw images, full labels, historical conversations, "
    "case IDs or fixed annotation coordinates. Do not invent observations or results. "
    "Do not change the evaluator, source split, candidate budget or admission rule."
)


def emit_request(folder, context, protocol):
    """One durable proposal reservation; restarting does not create another call."""
    payload = {"model": MODEL, "context": context, "protocol": proposer_protocol(protocol),
               "response_schema": PROPOSAL_SCHEMA, "instructions": INSTRUCTIONS}
    path = folder / "request.json"
    if path.exists():
        if read(path) != json.loads(json.dumps(payload)):
            raise ValueError("Cannot change an already reserved proposal request")
        return file_hash(path)
    folder.mkdir(parents=True, exist_ok=False)
    write_new(path, payload)
    return file_hash(path)


def receive_response(folder, protocol):
    """Validate the same sealed raw-text transport used by the task experiment."""
    request = read(folder / "request.json")
    response_path = folder / "response.json"
    marker = read(folder / "response.ready.json")
    if file_hash(response_path) != marker["response_sha256"]:
        raise ValueError("Published model response changed")
    response = read(response_path)
    if response["request_sha256"] != file_hash(folder / "request.json"):
        raise ValueError("Response belongs to a different request")
    if response["model"] != request["model"] or not response.get("origin"):
        raise ValueError("Missing or mismatched dispatch provenance")
    if json.loads(response["raw_text"]) != response["proposal"]:
        raise ValueError("Model proposal differs from the original reply")
    proposal = Proposal.from_dict(response["proposal"]).validate(protocol)
    receipt = {"requested_model": request["model"], "origin": response["origin"],
               "request_sha256": file_hash(folder / "request.json"),
               "response_sha256": file_hash(response_path),
               "raw_text_sha256": digest(response["raw_text"]),
               "returned_model": response.get("provider_returned_model"),
               "usage": response.get("usage"), "proposal_sessions": 1}
    return proposal, receipt


def filtered_context(context, plan):
    """A generated policy may retrieve evidence, but cannot invent its records."""
    context = json.loads(json.dumps(context))
    for option, field in (("memory_ids", "memories"), ("pool_ids", "pool")):
        if option not in plan:
            continue
        ids = plan[option]
        available = {item["id"] for item in context[field]}
        if (not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
                or len(ids) != len(set(ids)) or not set(ids) <= available):
            raise ValueError("Research policy selected unknown/duplicate evidence: " + option)
        if option == "pool_ids" and not {context["incumbent"], *plan["parents"]} <= set(ids):
            raise ValueError("Pool retrieval must retain the incumbent and chosen parents")
        context[field] = [item for item in context[field] if item["id"] in set(ids)]
    context.update(research_plan=plan, required_action=plan["action"], required_parents=plan["parents"])
    return context


def latest_success(archive, candidate):
    return next(t for t in reversed(archive.trials(candidate))
                if t["split"] == "validation" and t["outcome"] == "ok")


def utility(archive, start, execution_ids, calls):
    reference = summarize(latest_success(archive, start)["receipt"]["rows"])
    gains = [0.0]
    by_id = {t["id"]: t for t in archive.trials()}
    for tid in execution_ids:
        trial = by_id[tid]
        if trial["split"] != "validation" or trial["outcome"] != "ok":
            raise ValueError("Utility requires actual successful validation descendants")
        outcome = summarize(trial["receipt"]["rows"])
        gains.append(sum(outcome[g]["mean"] - reference[g]["mean"] for g in reference) / len(reference))
    return max(gains) / calls


def assert_policy_only(proposal, baseline):
    if (proposal.target != "research_policy" or proposal.action != "tune"
            or proposal.parents != (baseline.identity,) or proposal.training_steps != 0
            or proposal.pipeline_code != baseline.pipeline_code):
        raise ValueError("Meta proposal must change research policy only, not the task pipeline")


def prepare(destination, source):
    source, old_protocol, samples = load_study(source)
    destination = inside(ROOT, destination)
    if destination.exists():
        raise FileExistsError("Use a new meta-study destination")
    protection = read(ROOT / "outputs/rsi_program_evolution_20261016_v1/protected_before.json")
    verify_snapshot(ROOT, protection)
    proposals, source_tails, starts = {}, {}, []
    for arm in ARMS:
        previous = Archive(source / arm, old_protocol)
        try:
            if len(previous.events("round_finished")) != old_protocol.maximum_rounds:
                raise ValueError("Finish all preregistered main-study rounds first")
            if any(t["split"] == "audit" for t in previous.trials(visible_only=False)):
                raise ValueError("Meta search must precede any final-audit inspection")
            source_tails[arm] = previous.verify()
            for row in previous.pool():
                proposals[row["id"]] = previous.candidate(row["id"])["proposal"]
            if arm == "full":
                starts.append(previous.events("baseline_registered")[0]["body"]["candidate"])
            if arm == "no_memory":
                starts.append(previous.incumbent())
        finally:
            previous.close()
    # Two starting states are selected before the new strategy is generated.
    # They are NOT two independent datasets or two independent model repeats.
    protocol = replace(old_protocol, maximum_rounds=1, maximum_model_calls=len(KEYS))
    destination.mkdir(parents=True, exist_ok=False)
    write_new(destination / "protocol.json", asdict(protocol))
    write_new(destination / "study.json", {
        "source": str(source.relative_to(ROOT)), "source_event_tails": source_tails,
        "source_protocol_sha256": old_protocol.identity, "starts": starts,
        "model": MODEL, "allowed_requests": KEYS,
        "proposal_budget": {"strategy_design": 1, "paired_descendants": 4, "continuation": 1},
        "per_side_budget": {"model_calls": 1, "training_steps": protocol.maximum_training_steps},
        "minimum_meta_gain_per_call": .002,
        "same_validation_source_two_start_states": True,
        "statistical_significance_claimed": False,
        "final_audit_feedback_allowed": False,
        "design_cost_counted_separately_from_paired_execution": True})
    archive = Archive(destination / "archive", protocol, create=True)
    executor = Executor(archive, Sandbox(ROOT, ROOT / ".venvs/sam2"), samples)
    try:
        for identity, proposal in proposals.items():
            assert archive.add(proposal) == identity
            tid, result = executor.execute(identity)
            if "rows" not in result:
                raise RuntimeError("Failed to replay a research starting state: " + identity)
            if identity == starts[0]:
                archive.select(identity, True, "meta_frozen_baseline", tid)
                archive.event("baseline_registered", {"candidate": identity, "trial": tid})
            else:
                base_rows = latest_success(archive, starts[0])["receipt"]["rows"]
                decision = compare(result["rows"], base_rows, base_rows, protocol)
                archive.remember(identity, tid, {"hypothesis": proposal.hypothesis,
                    "intervention": proposal.intervention, "measured_task_gains": decision["task_gains"],
                    "regressions": decision["regressions"]},
                    {"split": "validation", "protocol": protocol.identity, "arm": "shared_research_pool"})
        controller = Controller(archive, executor, FileExchangeModel(destination / "unused_controller_transport",
                                model=MODEL, maximum_calls=len(KEYS)), arm="full")
        contexts = []
        aliases = {g: f"source-{i+1}" for i, g in enumerate(sorted(protocol.train_groups + protocol.validation_groups))}
        def anonymous(value):
            if isinstance(value, dict):
                return {aliases.get(k, k): anonymous(v) for k, v in value.items()}
            if isinstance(value, list):
                return [anonymous(v) for v in value]
            return aliases.get(value, value) if isinstance(value, str) else value
        for start in starts:
            context = controller.context(4)
            context["incumbent"] = start
            context["diagnostic"] = anonymous(executor.diagnose(start))
            context["budget"]["remaining_proposal_sessions"] = 1
            contexts.append(context)
        write_new(destination / "paired_contexts.json", contexts)
        write_new(destination / "prepared.json", {"reexecuted_programs": len(proposals),
                   "event_tail": archive.verify(), "protection": verify_snapshot(ROOT, protection)})
    finally:
        archive.close()
    print(json.dumps({"prepared": str(destination), "reexecuted_programs": len(proposals), "starts": starts}))


def open_study(destination, gpu=None):
    destination = inside(ROOT, destination)
    config = read(destination / "study.json")
    _, source_protocol, samples = load_study(config["source"])
    protocol = replace(source_protocol, maximum_rounds=1, maximum_model_calls=len(KEYS))
    if read(destination / "protocol.json") != json.loads(json.dumps(asdict(protocol))):
        raise ValueError("Meta protocol differs from its frozen source contract")
    archive = Archive(destination / "archive", protocol)
    return destination, config, archive, Executor(archive, Sandbox(ROOT, ROOT / ".venvs/sam2", gpu=gpu), samples)


def queue(destination, key, gpu=None):
    destination, config, archive, executor = open_study(destination, gpu)
    try:
        if key not in KEYS:
            raise ValueError("Unregistered proposal slot")
        folder = destination / "requests" / key
        if (folder / "request.json").exists():
            print(json.dumps({"already_reserved": str(folder), "sha256": file_hash(folder / "request.json")}))
            return
        contexts = read(destination / "paired_contexts.json")
        if key == "policy":
            baseline = archive.candidate(config["starts"][0])["proposal"]
            context = json.loads(json.dumps(contexts[0]))
            context.update(required_action="tune", required_parents=[baseline.identity],
                required_target="research_policy", parent_programs=[asdict(baseline)],
                task="Improve the executable research strategy plan(context), NOT the microscopy scoring pipeline.",
                meta_contract={
                    "pipeline_code": "Copy the parent pipeline_code byte-for-byte; training_steps must be zero.",
                    "evaluation": "One fresh descendant proposal per strategy per starting state. Compare best "
                                  "nonnegative development gain per proposal session. Same model, shared archive, "
                                  "observations, task starting program, data and budgets. No audit results.",
                    "modifiable": "Research action, parent selection, specific experimental focus and optional "
                                  "conditional memory_ids/pool_ids retrieval from the supplied context.",
                    "constraints": "plan(context) must return action/parents/focus. Optional memory_ids/pool_ids "
                                   "must name real available records; pool_ids must retain incumbent and parents. "
                                   "Use evidence and measured effects rather than hardcoded program IDs. "
                                   "Do not change evaluator, budgets, labels, or inject results.",
                    "objective": "Discover more useful program changes per fixed proposal budget. "
                                 "Use both successful and failed measurements; give an executable, testable mechanism.",
                    "paired_starting_states": contexts})
        else:
            if key == "continuation":
                completed = read(destination / "meta_result.json")
                policy_id = completed["selected_policy"]
                start = completed["selected_task_program"]
                controller = Controller(archive, executor, FileExchangeModel(destination / "unused_controller_transport",
                                        model=MODEL, maximum_calls=len(KEYS)), arm="full")
                context = controller.context(5)
                # The selected task program is actually installed by finish().
                assert context["incumbent"] == start
            else:
                if (destination / "meta_result.json").exists():
                    raise ValueError("Paired search is already sealed")
                index, side = key.split("-")
                context = contexts[int(index[1:])]
                policy_id = (config["starts"][0] if side == "control" else
                             read(destination / "policy_result.json")["candidate"])
            plan, policy_receipt = executor.policy(policy_id, context)
            if plan is None:
                write_new(destination / (key + "-policy-failure.json"), policy_receipt)
                raise RuntimeError("Generated research policy failed in the sandbox")
            common_input = digest(context)
            context = filtered_context(context, plan)
            context["parent_programs"] = [asdict(archive.candidate(p)["proposal"]) for p in plan["parents"]]
            if not plan["parents"]:
                context["reference_program"] = asdict(archive.candidate(context["incumbent"])["proposal"])
            # Every side receives the exact same initial task program as a reference,
            # even if the policy chooses other parents from the shared candidate pool.
            context["starting_program"] = asdict(archive.candidate(context["incumbent"])["proposal"])
            write_new(destination / (key + "-plan.json"), {
                "policy": policy_id, "starting_program": context["incumbent"],
                "common_input_sha256": common_input, "context_sha256": digest(context),
                "plan": plan, "execution": policy_receipt})
        request_hash = emit_request(folder, context, archive.protocol)
        archive.event("meta_request_reserved", {"key": key, "request_sha256": request_hash, "model": MODEL})
        print(json.dumps({"request": str(folder), "request_sha256": request_hash}))
    finally:
        archive.close()


def evaluate_reply(destination, key, gpu=None):
    destination, config, archive, executor = open_study(destination, gpu)
    try:
        if key not in KEYS:
            raise ValueError("Unregistered proposal slot")
        output = destination / ("policy_result.json" if key == "policy" else key + "-result.json")
        if output.exists():
            print(json.dumps(read(output)))
            return
        folder = destination / "requests" / key
        proposal, model_receipt = receive_response(folder, archive.protocol)
        context = read(folder / "request.json")["context"]
        if proposal.action != context["required_action"] or list(proposal.parents) != context["required_parents"]:
            raise ValueError("Proposal did not implement its sealed research plan")
        if key == "policy":
            assert_policy_only(proposal, archive.candidate(config["starts"][0])["proposal"])
            candidate = archive.add(proposal)
            result = {"candidate": candidate, "model_receipt": model_receipt,
                      "not_yet_admitted": True, "task_pipeline_unchanged": True}
        else:
            if proposal.target == "research_policy":
                raise ValueError("Descendant must change a task program, not its evaluator strategy")
            candidate = archive.add(proposal)
            trial, measurement = executor.execute(candidate)
            plan = read(destination / (key + "-plan.json"))
            start = plan["starting_program"]
            ok = "rows" in measurement
            decision = (compare(measurement["rows"], latest_success(archive, start)["receipt"]["rows"],
                        latest_success(archive, config["starts"][0])["receipt"]["rows"], archive.protocol)
                        if ok else {"accepted": False, "error": measurement.get("error")})
            if not any(m["verified_by_trial"] == trial for m in archive.memories()):
                archive.remember(candidate, trial, {"hypothesis": proposal.hypothesis,
                    "intervention": proposal.intervention, "decision": decision},
                    {"split": "validation", "protocol": archive.protocol.identity, "meta_slot": key})
            result = {"candidate": candidate, "trial": trial, "status": "ok" if ok else "execution_failed",
                "summary": measurement.get("summary"), "decision": decision,
                "model": MODEL, "model_receipt": model_receipt,
                "input_sha256": plan["common_input_sha256"],
                "starting_program_sha256": archive.candidate(start)["code_hash"],
                "baseline_trial": latest_success(archive, start)["id"],
                "budget": config["per_side_budget"], "execution_ids": [trial] if ok else [],
                "gain_per_call": utility(archive, start, [trial], 1) if ok else 0.0,
                "policy": plan["policy"], "policy_execution": plan["execution"]}
            if key == "continuation":
                archive.select(candidate, decision["accepted"], decision, trial) if ok else None
        write_new(output, result)
        archive.event("meta_slot_finished", {"key": key, "result_sha256": file_hash(output), "candidate": candidate})
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        archive.close()


def verify_pair(archive, control, candidate):
    """Bind equal budgets to actual model replies and executed policy contexts."""
    for key in ("model", "input_sha256", "starting_program_sha256", "budget", "baseline_trial"):
        if control[key] != candidate[key]:
            raise ValueError("Unmatched paired research condition: " + key)
    origins = []
    for run in (control, candidate):
        receipt = run["model_receipt"]
        if receipt["requested_model"] != run["model"] or receipt["proposal_sessions"] != run["budget"]["model_calls"]:
            raise ValueError("Declared proposal budget is not backed by model receipts")
        if run["policy_execution"]["input_sha256"] != run["input_sha256"]:
            raise ValueError("Research policy saw a different evidence bundle")
        if not receipt.get("request_sha256") or not receipt.get("response_sha256") or not receipt.get("origin"):
            raise ValueError("Missing raw model provenance")
        origins.append(receipt["origin"])
        if run["status"] != "ok":
            raise ValueError("Failed descendant prevents policy promotion in this pilot")
        if archive.candidate(run["candidate"])["proposal"].training_steps > run["budget"]["training_steps"]:
            raise ValueError("Descendant exceeded the shared training budget")
    if len(set(origins)) != 2:
        raise ValueError("Paired descendants must use independent proposal sessions")


def finish(destination):
    destination, config, archive, _ = open_study(destination)
    try:
        if (destination / "meta_result.json").exists():
            print(json.dumps(read(destination / "meta_result.json")))
            return
        pairs = []
        for index in range(2):
            pair = {side: read(destination / f"p{index}-{side}-result.json") for side in ("control", "candidate")}
            for side, run in pair.items():
                key = f"p{index}-{side}"
                reply, transport = receive_response(destination / "requests" / key, archive.protocol)
                if reply.identity != run["candidate"] or transport != run["model_receipt"]:
                    raise ValueError("Paired result differs from its verified raw model reply")
                recorded = next(t for t in archive.trials() if t["id"] == run["trial"])
                if recorded["candidate"] != run["candidate"] or run["execution_ids"] != [run["trial"]]:
                    raise ValueError("Paired result points to another program's execution")
                plan = read(destination / (key + "-plan.json"))
                if plan["execution"] != run["policy_execution"] or plan["policy"] != run["policy"]:
                    raise ValueError("Paired result changed the executed research policy")
            verify_pair(archive, **pair)
            pairs.append(pair)
        candidate = read(destination / "policy_result.json")["candidate"]
        receipt = {"actual_execution": True, "pairs": pairs,
                   "strategy_design_proposal_sessions": 1,
                   "same_validation_source_two_starting_states": True}
        # Resume after a process stop between evaluation, admission and the final
        # report. Immutable trials/events are reused rather than duplicated.
        existing = [t for t in archive.trials(candidate) if t["split"] == "meta_validation"
                    and t["outcome"] == "ok" and t["receipt"] == receipt]
        trial = existing[-1]["id"] if existing else archive.record_trial(candidate, "meta_validation", "ok", receipt)
        admitted = [e for e in archive.events("policy_selection") if e["body"]["trial"] == trial]
        accepted = (admitted[-1]["body"]["accepted"] if admitted else
                    archive.select_policy(candidate, trial, minimum_gain=config["minimum_meta_gain_per_call"]))
        policy = archive.events("policy_selection")[-1]["body"]["incumbent"]
        # The next real round starts at the best measured task program, not the
        # candidate policy's pipeline (which deliberately stayed unchanged).
        choices = list(config["starts"]) + [p[s]["candidate"] for p in pairs for s in ("control", "candidate")]
        def value(cid):
            values = summarize(latest_success(archive, cid)["receipt"]["rows"])
            return sum(v["mean"] for v in values.values()) / len(values)
        selected = max(choices, key=value)
        selected_trial = latest_success(archive, selected)["id"]
        if not any(e["body"].get("reason") == "best_task_state_for_next_actual_round"
                   and e["body"]["candidate"] == selected for e in archive.events("selection")):
            archive.select(selected, True, "best_task_state_for_next_actual_round", selected_trial)
        result = {"candidate_policy": candidate, "meta_trial": trial, "accepted": accepted,
                  "selected_policy": policy, "selected_task_program": selected,
                  "policy_selection": archive.events("policy_selection")[-1]["body"],
                  "pairs": pairs, "proposal_sessions_so_far": 5,
                  "token_usage": None, "currency_cost": None,
                  "final_audit_used": False, "statistical_significance_claimed": False}
        write_new(destination / "meta_result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, ensure_ascii=False))
    finally:
        archive.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "queue", "evaluate", "finish"))
    parser.add_argument("directory")
    parser.add_argument("--source")
    parser.add_argument("--key", choices=KEYS, default="policy")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    destination = inside(ROOT, args.directory)
    lock_path = inside(ROOT, destination.with_name(destination.name + ".runner.lock"))
    # Separate CLI invocations share one append-only archive. Serialize its
    # event chain and job-folder allocation; model generation itself remains
    # parallel and never holds this local lock.
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == "prepare":
            if not args.source:
                parser.error("prepare requires --source")
            prepare(args.directory, args.source)
        elif args.command == "queue":
            queue(args.directory, args.key, args.gpu)
        elif args.command == "evaluate":
            evaluate_reply(args.directory, args.key, args.gpu)
        else:
            finish(args.directory)


if __name__ == "__main__":
    main()
