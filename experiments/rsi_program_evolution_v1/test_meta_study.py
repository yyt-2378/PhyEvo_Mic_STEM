"""Real meta-runner contract tests, including resumable model exchange."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from experiments.rsi_program_evolution_v1.contracts import digest, file_hash, write_new
from experiments.rsi_program_evolution_v1.meta_study import (
    MODEL, assert_policy_only, emit_request, filtered_context, receive_response, verify_pair)
from experiments.rsi_program_evolution_v1.provider import publish_response
from experiments.rsi_program_evolution_v1.test_core import proposal, protocol, sandbox_fixture, POLICY


def test_request_resume_never_spends_another_slot(tmp_path):
    folder = tmp_path / "call"
    first = emit_request(folder, {"observation": "finite evidence"}, protocol())
    assert emit_request(folder, {"observation": "finite evidence"}, protocol()) == first
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ValueError, match="already reserved"):
        emit_request(folder, {"observation": "changed evidence"}, protocol())


def test_deferred_reply_is_bound_to_request_and_raw_code(tmp_path):
    folder = tmp_path / "call"
    emit_request(folder, {}, protocol())
    p = proposal()
    publish_response(folder, raw_text=json.dumps(asdict(p)), model=MODEL, origin="/independent/model/session")
    answer, receipt = receive_response(folder, protocol())
    assert answer == p and receipt["proposal_sessions"] == 1
    assert receipt["usage"] is None and receipt["returned_model"] is None
    assert receive_response(folder, protocol()) == (answer, receipt)


def test_wrong_request_cannot_be_published_as_a_matched_pair(tmp_path):
    folder = tmp_path / "call"
    emit_request(folder, {}, protocol())
    response = {"request_sha256": "wrong", "model": MODEL, "origin": "test",
                "proposal": asdict(proposal()), "raw_text": json.dumps(asdict(proposal()))}
    write_new(folder / "response.json", response)
    write_new(folder / "response.ready.json", {"response_sha256": file_hash(folder / "response.json")})
    with pytest.raises(ValueError, match="different request"):
        receive_response(folder, protocol())


def test_policy_may_not_smuggle_in_a_better_task_program():
    parent = proposal()
    policy = replace(parent, action="tune", parents=(parent.identity,), target="research_policy")
    assert_policy_only(policy, parent)
    for changed in (replace(policy, training_steps=1), replace(policy, pipeline_code=parent.pipeline_code + "\n"),
                    replace(policy, target="analysis")):
        with pytest.raises(ValueError, match="policy only"):
            assert_policy_only(changed, parent)


def context():
    return {"incumbent": "a", "pool": [{"id": "a"}, {"id": "b"}],
            "memories": [{"id": "m1", "outcome": "ok"}, {"id": "m2", "outcome": "failed"}]}


def test_conditional_retrieval_selects_real_evidence_without_mutating_source():
    original = context()
    plan = {"action": "tune", "parents": ["a"], "focus": "test observed mechanism",
            "memory_ids": ["m1"], "pool_ids": ["a"]}
    result = filtered_context(original, plan)
    assert result["memories"] == [{"id": "m1", "outcome": "ok"}]
    assert result["pool"] == [{"id": "a"}]
    assert original == context()
    for changes in ({"memory_ids": ["invented"]}, {"memory_ids": ["m1", "m1"]}, {"pool_ids": ["b"]}):
        with pytest.raises(ValueError):
            filtered_context(original, {**plan, **changes})


def test_policy_runs_in_real_sandbox_and_changes_retrieval(tmp_path):
    inputs, program, sandbox = sandbox_fixture(tmp_path)
    (program / "policy.py").write_text(
        "def plan(context):\n"
        "    return {'action': 'tune', 'parents': [context['incumbent']], 'focus': 'retrieve measured failures', "
        "'memory_ids': [m['id'] for m in context['memories'] if m['outcome'] == 'failed']}\n")
    write_new(inputs / "request.json", {"mode": "policy", "context": context()})
    receipt = sandbox.run(program, inputs, tmp_path / "execution")
    assert receipt["status"] == "ok"
    plan = json.loads((tmp_path / "execution/result.json").read_text())
    assert filtered_context(context(), plan)["memories"] == [{"id": "m2", "outcome": "failed"}]
    assert receipt["network"] is False and receipt["repository_mounted"] is False


class TinyArchive:
    def candidate(self, identity):
        return {"proposal": proposal()}


def paired_runs():
    common = {"model": MODEL, "input_sha256": "shared", "starting_program_sha256": "program",
              "baseline_trial": "baseline", "budget": {"model_calls": 1, "training_steps": 400},
              "policy_execution": {"input_sha256": "shared"}, "status": "ok", "candidate": "child",
              "model_receipt": {"requested_model": MODEL, "proposal_sessions": 1,
                                "request_sha256": "request", "response_sha256": "reply", "origin": "one"}}
    other = deepcopy(common)
    other["model_receipt"]["origin"] = "two"
    return common, other


def test_budget_is_backed_by_independent_model_receipts():
    a, b = paired_runs()
    verify_pair(TinyArchive(), a, b)
    b["model_receipt"]["proposal_sessions"] = 2
    with pytest.raises(ValueError, match="budget"):
        verify_pair(TinyArchive(), a, b)
    a, b = paired_runs()
    b["model_receipt"]["origin"] = a["model_receipt"]["origin"]
    with pytest.raises(ValueError, match="independent"):
        verify_pair(TinyArchive(), a, b)


def test_different_policy_inputs_and_failed_execution_cannot_promote():
    a, b = paired_runs()
    b["policy_execution"]["input_sha256"] = "extra-info"
    with pytest.raises(ValueError, match="different evidence"):
        verify_pair(TinyArchive(), a, b)
    a, b = paired_runs()
    b["status"] = "execution_failed"
    with pytest.raises(ValueError, match="Failed descendant"):
        verify_pair(TinyArchive(), a, b)
