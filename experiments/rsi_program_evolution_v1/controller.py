"""One executable RSI loop: observation -> program -> experiment -> inheritance."""
from __future__ import annotations

from dataclasses import asdict
import json

from .contracts import digest, Proposal
from .evaluation import compare


BASELINE_PIPELINE = '''def predict(sample, state, context):
    return {"score": sample["base"], "support": sample["support"].astype(bool)}
'''

BASELINE_POLICY = '''def plan(context):
    incumbent = context["incumbent"]
    repaired = {p for x in context["pool"] if x.get("last_status") == "ok"
                and x.get("action") == "repair" for p in x.get("parents", [])}
    failures = [x for x in context["pool"] if x.get("last_status") in {"failed", "timeout"} and x["id"] not in repaired]
    if failures:
        return {"action": "repair", "parents": [failures[-1]["id"]],
                "focus": "Repair the observed execution failure, without changing evaluation."}
    pool = [x for x in context["pool"] if x.get("last_status") == "ok"]
    if len(pool) >= 2 and context["round"] % 3 == 0:
        other = next((x["id"] for x in reversed(pool) if x["id"] != incumbent), None)
        if other:
            return {"action": "evolve", "parents": [incumbent, other],
                    "focus": "Combine complementary mechanisms based on measured effects."}
    return {"action": "tune", "parents": [incumbent],
            "focus": "Identify a specific cause of the current residual errors and change the program."}
'''


def baseline_proposal():
    return Proposal("generate", (), "Frozen current analysis fields", "Existing baseline",
                    "Replay unchanged baseline ranking and support", "Exact baseline reproduction",
                    "Mismatch with frozen baseline output", BASELINE_PIPELINE, BASELINE_POLICY)


class Controller:
    def __init__(self, archive, executor, model, *, arm="full"):
        if arm not in {"full", "generic", "no_memory"}:
            raise ValueError("Unknown comparison arm")
        self.archive, self.executor, self.model, self.arm = archive, executor, model, arm
        self.protocol = archive.protocol
        registrations = archive.events("arm_registered")
        if registrations and registrations[0]["body"]["arm"] != arm:
            raise ValueError("Cannot change the experimental arm on resume")
        if registrations and registrations[0]["body"]["requested_model"] != getattr(model, "model", "test_double"):
            raise ValueError("Cannot change the requested model on resume")
        if not registrations:
            archive.event("arm_registered", {"arm": arm, "requested_model": getattr(model, "model", "test_double")})
        # Provider failures and interrupted requests consume budget too.
        calls = archive.events("model_call_started")
        finished = {e["body"]["call"] for e in archive.events("model_call_finished")}
        if any(e["hash"] not in finished for e in calls):
            raise ValueError("Unresolved prior model request; inspect its live process before retrying")
        model.calls = len(calls)

    def start(self):
        initial = self.archive.events("baseline_registered")
        if initial:
            return initial[0]["body"]["candidate"]
        identity = self.archive.add(baseline_proposal())
        trial, receipt = self.executor.execute(identity)
        if "rows" not in receipt:
            raise RuntimeError("Baseline execution failed; no search can start")
        self.archive.select(identity, True, "frozen_initial_baseline", trial)
        self.archive.event("baseline_registered", {"candidate": identity, "trial": trial})
        return identity

    def _pool_view(self):
        pool = []
        for entry in self.archive.pool():
            candidate = self.archive.candidate(entry["id"])
            trials = [t for t in entry["trials"] if t["split"] == "validation"]
            last = trials[-1] if trials else None
            pool.append({"id": entry["id"], "lineage": entry["lineage"],
                         "action": candidate["action"], "parents": candidate["parents"],
                         "last_status": last["outcome"] if last else "untried",
                         "summary": last["receipt"].get("summary", {}) if last else {},
                         "error": last["receipt"].get("error") if last else None})
        if self.arm == "no_memory":
            pool = [p for p in pool if p["id"] == self.archive.incumbent()]
        return pool

    def context(self, iteration):
        incumbent = self.archive.incumbent()
        diagnostic = self.executor.diagnose(incumbent)
        source_groups = sorted(self.protocol.train_groups + self.protocol.validation_groups)
        aliases = {g: f"source-{i+1}" for i, g in enumerate(source_groups)}
        # Replacement only of exact group identifiers, not arbitrary text/code.
        def anonymize(value):
            if isinstance(value, dict):
                return {aliases.get(k, k): anonymize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [anonymize(v) for v in value]
            return aliases.get(value, value) if isinstance(value, str) else value
        base = {"round": iteration, "incumbent": incumbent, "pool": self._pool_view(),
                "memories": self.archive.memories() if self.arm == "full" else [],
                "diagnostic": diagnostic,
                "budget": {"area_fraction": self.protocol.area_fraction,
                    "point_budget_per_512": self.protocol.point_budget_per_512,
                    "point_tolerance_px": self.protocol.point_tolerance_px,
                    "maximum_training_steps": self.protocol.maximum_training_steps},
                "field_contract": {
                    "image": "normalized raw 2D image", "base": "initial frozen score",
                    "support": "initial candidate support, NOT physical impossibility",
                    "valid": "physically analyzable pixels; output must be inside this mask",
                    "physics": "physical evidence score, not a target label",
                    "student": "source-excluded student score, not a target label",
                    "variation": "local physical texture variation when available",
                    "domain_labels": "input-derived domain map when available"},
                "task": "Improve this microscopy analysis program using existing evidence and labels. "
                        "Implement analysis, learning or routing changes; meta-policy evaluation is a separate stage."}
        if self.arm == "generic":
            # Same observations and operators; generic control has no archive-
            # directed plan or curated memory, rather than less raw evidence.
            base["pool"] = [p for p in base["pool"] if p["id"] == incumbent]
        return anonymize(base)

    def step(self):
        if self.archive.events("search_sealed"):
            raise ValueError("Search sealed; final-audit feedback cannot reopen this lineage")
        baseline = self.start()
        iteration = len(self.archive.events("round_finished")) + 1
        if iteration > self.protocol.maximum_rounds:
            raise ValueError("Round budget exhausted")
        context = self.context(iteration)
        incumbent = self.archive.incumbent()
        # A task-score gain does not authorize promoting an untested meta policy.
        policy_identity = baseline
        policy_versions = self.archive.events("policy_selection")
        if self.arm == "full" and policy_versions:
            policy_identity = policy_versions[-1]["body"]["incumbent"]
        if self.arm == "generic":
            plan = {"action": "tune", "parents": [incumbent],
                    "focus": "Improve the current program from the provided observations."}
            policy_receipt = {"kind": "fixed_generic_control"}
        else:
            plan, policy_receipt = self.executor.policy(policy_identity, context)
            if plan is None:
                self.archive.event("policy_failed", {"round": iteration, "receipt": policy_receipt})
                return {"status": "policy_failed"}
        context["research_plan"] = plan
        context["required_action"] = plan["action"]
        context["required_parents"] = plan["parents"]
        context["parent_programs"] = [asdict(self.archive.candidate(p)["proposal"]) for p in plan["parents"]]
        if not plan["parents"]:
            context["reference_program"] = asdict(self.archive.candidate(incumbent)["proposal"])
        self.archive.event("round_started", {"round": iteration, "incumbent": incumbent,
            "policy_candidate": policy_identity, "context_sha256": digest(context),
            "policy_receipt": policy_receipt, "plan": plan})
        calls = len(self.archive.events("model_call_started"))
        if calls >= self.protocol.maximum_model_calls:
            return {"status": "model_budget_exhausted"}
        call = self.archive.event("model_call_started", {"round": iteration, "context": context})
        proposal, receipt = self.model.propose(context, self.protocol)
        self.archive.event("model_call_finished", {"call": call, "receipt": receipt})
        if proposal is None:
            self.archive.event("round_finished", {"round": iteration, "status": "provider_failed"})
            return {"status": "provider_failed", "receipt": receipt}
        proposal.validate(self.protocol)
        if (proposal.action != plan["action"] or list(proposal.parents) != plan["parents"] or
                proposal.target == "research_policy"):
            self.archive.event("round_finished", {"round": iteration, "status": "proposal_plan_mismatch",
                "proposal": asdict(proposal)})
            return {"status": "proposal_plan_mismatch"}
        identity = self.archive.add(proposal)
        self.archive.event("proposal_executing", {"round": iteration, "candidate": identity,
            "request": call, "mechanism": proposal.hypothesis,
            "intervention": proposal.intervention, "evidence_ids": proposal.evidence_ids})
        trial, result = self.executor.execute(identity)
        if "rows" not in result:
            self.archive.remember(identity, trial, "Execution failed: " + result["error"][-1000:],
                                  "Technical failure; not evidence against the scientific hypothesis")
            self.archive.event("round_finished", {"round": iteration, "candidate": identity,
                "status": "execution_failed", "trial": trial, "incumbent": incumbent})
            return {"status": "execution_failed", "candidate": identity}
        def rows(cid):
            return next(t["receipt"]["rows"] for t in reversed(self.archive.trials(cid))
                        if t["split"] == "validation" and t["outcome"] == "ok")
        decision = compare(result["rows"], rows(incumbent), rows(baseline), self.protocol)
        self.archive.select(identity, decision["accepted"], decision, trial)
        # Exact measured experience, not an LLM assertion of having succeeded.
        self.archive.remember(identity, trial,
            {"hypothesis": proposal.hypothesis, "intervention": proposal.intervention,
             "measured_task_gains": decision["task_gains"], "regressions": decision["regressions"]},
            {"split": "validation", "protocol": self.protocol.identity, "arm": self.arm})
        outcome = {"round": iteration, "candidate": identity, "trial": trial,
                   "status": "accepted" if decision["accepted"] else "retained_for_research",
                   "decision": decision, "incumbent": self.archive.incumbent()}
        self.archive.event("round_finished", outcome)
        return outcome

    def seal(self):
        if self.archive.events("search_sealed"):
            return self.archive.events("search_sealed")[-1]["hash"]
        if not self.archive.incumbent():
            raise ValueError("No evaluated incumbent")
        return self.archive.event("search_sealed", {"incumbent": self.archive.incumbent(),
            "rounds": len(self.archive.events("round_finished")),
            "model_calls": len(self.archive.events("model_call_started"))})
