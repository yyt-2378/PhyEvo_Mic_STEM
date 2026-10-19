"""Resume an exact late response after a terminal transport timeout.

No replacement proposal, extra call or metric selection is permitted. A prior
timeout stays in the append-only ledger. Reconciliation is idempotent after a
completed event and refuses resumption if later research has already started.
"""
import argparse
import json
from pathlib import Path

from .archive import Archive
from .contracts import Proposal, digest, file_hash
from .evaluation import compare
from .executor import Executor
from .real_study import ROOT, load_study
from .sandbox import Sandbox


def reconcile(directory, arm, *, gpu=None):
    directory, protocol, samples = load_study(directory)
    archive = Archive(directory / arm, protocol)
    try:
        calls = archive.events("model_call_started")
        if not calls:
            raise ValueError("No dispatched model request")
        call = calls[-1]
        prior = [e for e in archive.events("late_response_executed") if e["body"]["call"] == call["hash"]]
        if prior:
            return prior[-1]["body"]
        finished = [e for e in archive.events("model_call_finished") if e["body"]["call"] == call["hash"]]
        if len(finished) != 1 or finished[0]["body"]["receipt"]["status"] != "external_model_timeout":
            raise ValueError("Only a terminal external-model timeout can be reconciled here")
        folder = archive.root / "exchange" / f"call-{len(calls):03d}"
        request = json.loads((folder / "request.json").read_text())
        marker = json.loads((folder / "response.ready.json").read_text())
        if marker["response_sha256"] != file_hash(folder / "response.json"):
            raise ValueError("Response publication hash changed")
        response = json.loads((folder / "response.json").read_text())
        if (file_hash(folder / "request.json") != response["request_sha256"] or
                response["request_sha256"] != finished[0]["body"]["receipt"]["request_sha256"] or
                digest(request["context"]) != digest(call["body"]["context"])):
            raise ValueError("Response does not match the dispatched request")
        if response["model"] != request["model"] or not response.get("origin"):
            raise ValueError("Response model/origin changed")
        if json.loads(response["raw_text"]) != response["proposal"]:
            raise ValueError("Response was edited by broker")
        proposal = Proposal.from_dict(response["proposal"]).validate(protocol)
        context = request["context"]
        if (proposal.action != context["required_action"] or list(proposal.parents) != context["required_parents"]
                or proposal.target == "research_policy"):
            raise ValueError("Proposal violates the original research plan")
        if archive.events("search_sealed") or archive.incumbent() != context["incumbent"]:
            raise ValueError("Research has advanced or sealed; cannot insert a late result retroactively")
        initial = archive.events("baseline_registered")[0]["body"]["candidate"]
        identity = archive.add(proposal)
        executor = Executor(archive, Sandbox(ROOT, ROOT / ".venvs/sam2", gpu=gpu), samples)
        trial, result = executor.execute(identity)
        decision = {"accepted": False, "reason": "execution_failed"}
        if "rows" in result:
            def rows(cid):
                return next(t["receipt"]["rows"] for t in reversed(archive.trials(cid))
                            if t["split"] == "validation" and t["outcome"] == "ok")
            decision = compare(result["rows"], rows(archive.incumbent()), rows(initial), protocol)
            archive.select(identity, decision["accepted"], decision, trial)
            archive.remember(identity, trial,
                {"hypothesis": proposal.hypothesis, "intervention": proposal.intervention,
                 "measured_task_gains": decision["task_gains"], "regressions": decision["regressions"]},
                {"split": "validation", "protocol": protocol.identity, "arm": arm})
        else:
            archive.remember(identity, trial, "Execution failed: " + result["error"][-1000:],
                             "Technical failure; not evidence against the scientific hypothesis")
        outcome = {"call": call["hash"], "round": call["body"]["round"], "candidate": identity,
                   "trial": trial, "status": "accepted" if decision["accepted"] else
                   ("retained_for_research" if "rows" in result else "execution_failed"),
                   "decision": decision, "incumbent": archive.incumbent(),
                   "source_response_sha256": file_hash(folder / "response.json"),
                   "original_timeout_preserved": True, "extra_model_calls": 0}
        archive.event("late_response_executed", outcome)
        return outcome
    finally:
        archive.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--arm", required=True, choices=("full", "generic", "no_memory"))
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.directory, args.arm, gpu=args.gpu), ensure_ascii=False))
