"""Deterministic injected-model tests, not evidence of LLM scientific discovery."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.rsi_program_evolution_v1.archive import Archive
from experiments.rsi_program_evolution_v1.contracts import Protocol, Proposal, engine_fingerprint
from experiments.rsi_program_evolution_v1.controller import Controller, BASELINE_POLICY
from experiments.rsi_program_evolution_v1.executor import Executor, Sample, sample_fingerprint
from experiments.rsi_program_evolution_v1.sandbox import Sandbox


ROOT = Path(__file__).resolve().parents[2]


def samples():
    cases = []
    for i, group in enumerate(("A", "B", "C")):
        shape = (32, 32)
        image = np.ones(shape, np.float32)
        human = np.zeros(shape, bool); human[:, 20:22] = True
        old = np.zeros(shape, np.float32); old[:, 8:10] = 1
        student = human.astype(np.float32)
        fields = {"image": image, "base": old, "support": old > 0,
                  "valid": np.ones(shape, bool), "physics": old, "student": student}
        cases.append(Sample(str(i), group, "interface", fields, human, human.astype(float), np.zeros((0, 2))))
    return cases


class InjectedModel:
    """Explicit test double: no claim that this hard-coded fix was discovered."""
    model = "test_double_no_llm"
    def __init__(self):
        self.calls = 0
        self.contexts = []

    def propose(self, context, protocol):
        self.contexts.append(context)
        self.calls += 1
        plan = context["research_plan"]
        code = ("def predict(sample, state, context):\n"
                f"    return {{'score': sample['student'] * {self.calls}, 'support': sample['valid']}}\n")
        # Deliberately invalid policy: task success must not promote this policy.
        malicious_policy = "def plan(context):\n    raise RuntimeError('unvalidated policy must not execute')\n"
        p = Proposal(plan["action"], tuple(plan["parents"]), "outside support", "support excludes target",
                     "change the candidate-generating program", "same-budget coverage gain", "no gain",
                     code, malicious_policy, (), 0, "analysis")
        return p, {"status": "ok", "provider": "injected_test_double", "usage": {"tokens": 0}}


def make_loop(tmp_path, *, create=True, model=None):
    cases = samples()
    p = Protocol(("A",), ("B",), ("C",), seeds=(23,), maximum_rounds=2,
                 data_version=sample_fingerprint(cases), engine_version=engine_fingerprint())
    archive = Archive(tmp_path / "study", p, create=create)
    executor = Executor(archive, Sandbox(ROOT, ROOT / ".venvs/sam2", timeout=30), cases)
    model = model or InjectedModel()
    return Controller(archive, executor, model), archive, executor, model


def test_two_round_actual_program_execution_and_resume(tmp_path):
    controller, archive, executor, model = make_loop(tmp_path)
    result = controller.step()
    assert result["status"] == "accepted"
    assert result["decision"]["task_gains"]["known_positive_area_coverage_at_fixed_area"] == 1
    inherited = archive.incumbent()
    assert archive.candidate(inherited)["proposal"].action == "tune"
    archive.close()
    resumed, archive, executor, model = make_loop(tmp_path, create=False)
    second = resumed.step()
    assert second["status"] == "retained_for_research"
    assert model.contexts[0]["incumbent"] == inherited
    assert model.contexts[0]["parent_programs"][0]["pipeline_code"] == archive.candidate(inherited)["proposal"].pipeline_code
    assert len(model.contexts[0]["memories"]) == 1
    assert archive.incumbent() == inherited
    assert len(archive.events("proposal_executing")) == 2
    assert all(t["receipt"]["actual_execution"] for t in archive.trials())
    # The invalid, unverified policy from a successful task candidate was not run.
    baseline = archive.events("baseline_registered")[0]["body"]["candidate"]
    assert all(e["body"]["policy_candidate"] == baseline for e in archive.events("round_started"))
    seal = resumed.seal()
    _, audit = executor.execute(archive.incumbent(), split="audit", audit_seal=seal)
    assert audit["rows"][0]["coverage"] == 1
    assert all(t["split"] != "audit" for t in archive.trials())
    with pytest.raises(ValueError, match="Search sealed"):
        resumed.step()
    archive.close()


def test_no_audit_before_seal_and_changed_input_refused(tmp_path):
    controller, archive, executor, _ = make_loop(tmp_path)
    baseline = controller.start()
    with pytest.raises(ValueError, match="sealed"):
        executor.execute(baseline, split="audit")
    executor.samples[0].human[0, 0] = True
    with pytest.raises(ValueError, match="arrays changed"):
        executor.execute(baseline)
    archive.close()


def test_training_checkpoint_is_created_and_inherited(tmp_path):
    controller, archive, executor, _ = make_loop(tmp_path)
    baseline = controller.start()
    code = '''import numpy as np
def fit(samples, context):
    return {"weight": np.array([1.0], dtype=np.float32)}
def predict(sample, state, context):
    return {"score": sample["student"] * state["weight"][0], "support": sample["valid"]}
'''
    trained = Proposal("tune", (baseline,), "test", "test", "fit checkpoint", "test", "test",
                       code, BASELINE_POLICY, training_steps=1, target="learning")
    one = archive.add(trained)
    _, first = executor.execute(one)
    assert first["rows"][0]["coverage"] == 1
    assert "23" in first["checkpoints"]
    two = archive.add(replace(trained, parents=(one,), training_steps=0, intervention="inherit state"))
    _, second = executor.execute(two)
    assert second["checkpoints"] == first["checkpoints"]
    assert second["rows"][0]["coverage"] == 1
    archive.close()


def test_failed_execution_drives_repair_and_retains_failure_evidence(tmp_path):
    class FailThenRepair(InjectedModel):
        def propose(self, context, protocol):
            answer, receipt = super().propose(context, protocol)
            if self.calls == 1:
                answer = replace(answer, pipeline_code=(
                    "def predict(sample, state, context):\n"
                    "    raise RuntimeError('synthetic interface failure')\n"))
            return answer, receipt

    controller, archive, _, model = make_loop(tmp_path, model=FailThenRepair())
    first = controller.step()
    assert first["status"] == "execution_failed"
    failed = first["candidate"]
    baseline = archive.incumbent()
    assert failed != baseline
    assert archive.trials(failed)[0]["outcome"] == "failed"
    second = controller.step()
    assert second["status"] == "accepted"
    repaired = archive.candidate(second["candidate"])
    assert repaired["action"] == "repair" and repaired["parents"] == [failed]
    assert model.contexts[1]["required_action"] == "repair"
    assert model.contexts[1]["required_parents"] == [failed]
    assert any(m["candidate"] == failed and m["outcome"] == "failed"
               for m in model.contexts[1]["memories"])
    assert len(archive.pool()) == 3
    assert archive.trials(failed)[0]["receipt"]["actual_execution"]
    archive.verify()
    archive.close()
