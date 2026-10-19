from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from experiments.rsi_program_evolution_v1.archive import Archive
from experiments.rsi_program_evolution_v1.contracts import Protocol, Proposal, digest, write_new
from experiments.rsi_program_evolution_v1.evaluation import evaluate, compare
from experiments.rsi_program_evolution_v1.sandbox import Sandbox


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = "def predict(sample, state, context):\n    return {'score': sample['image'], 'support': sample['valid']}\n"
POLICY = "def plan(context):\n    return {'action': 'generate', 'parents': [], 'focus': 'diagnose actual residuals'}\n"


def protocol():
    return Protocol(("A",), ("B",), ("C",), seeds=(23,))


def proposal(**changes):
    p = Proposal("generate", (), "missing support", "support excludes targets",
                 "change support", "coverage improves", "same-budget coverage does not improve",
                 PIPELINE, POLICY)
    return replace(p, **changes)


def trial_receipt(coverage=.5):
    return {"rows": [{"group": "B", "case": "b1", "seed": 23,
                      "coverage": coverage, "selected": 10, "budget": 10,
                      "metric": "known_positive_area_coverage_at_fixed_area"}]}


def test_source_splits_and_protocol_identity():
    p = protocol()
    assert p.split("C") == "audit"
    assert digest(asdict(p)) == p.identity
    with pytest.raises(ValueError, match="source-disjoint"):
        replace(p, audit_groups=("B",))
    with pytest.raises(ValueError, match="Duplicate"):
        replace(p, seeds=(23, 23))
    with pytest.raises(ValueError, match="Unknown pixels"):
        replace(p, label_semantics="dense_binary")


def test_proposal_is_program_not_named_recipe():
    p = proposal()
    assert p.validate(protocol()) is p
    assert Proposal.from_dict(json.loads(json.dumps(asdict(p)))) == p
    for bad in (replace(p, action="evolve"), replace(p, pipeline_code="x=1"),
                replace(p, hypothesis=""), replace(p, training_steps=10000)):
        with pytest.raises(ValueError):
            bad.validate(protocol())


def test_resume_and_rejected_parent_remains_searchable(tmp_path):
    a = Archive(tmp_path / "experiment", protocol(), create=True)
    base = a.add(proposal())
    t = a.record_trial(base, "validation", "ok", trial_receipt())
    a.select(base, True, "initial", t)
    child = a.add(proposal(action="tune", parents=(base,), hypothesis="alternative mechanism"))
    t2 = a.record_trial(child, "validation", "ok", trial_receipt(.4))
    a.select(child, False, "regression", t2)
    a.remember(child, t2, "Coverage regressed", "Validation only")
    tail = a.verify()
    a.close()
    b = Archive(tmp_path / "experiment", protocol())
    assert b.incumbent() == base
    assert child in {r["id"] for r in b.pool()}
    assert b.candidate(child)["lineage"] == base
    assert b.verify() == tail
    assert len(b.memories()) == 1
    b.close()


def test_crossover_records_both_parents_and_new_lineage(tmp_path):
    a = Archive(tmp_path / "experiment", protocol(), create=True)
    one, two = a.add(proposal()), a.add(proposal(hypothesis="independent route"))
    child = a.add(proposal(action="evolve", parents=(one, two)))
    assert a.candidate(child)["parents"] == [one, two]
    assert a.candidate(child)["lineage"] == child
    a.close()


def test_audit_results_never_enter_search_or_memory(tmp_path):
    a = Archive(tmp_path / "experiment", protocol(), create=True)
    base = a.add(proposal())
    t = a.record_trial(base, "audit", "ok", {"hidden": "must-not-be-returned"})
    assert a.trials(base) == []
    assert a.pool()[0]["trials"] == []
    with pytest.raises(ValueError):
        a.select(base, True, "test oracle", t)
    with pytest.raises(ValueError):
        a.remember(base, t, "hidden result", "any")
    a.close()


def test_archive_refuses_rewrite_and_protocol_drift(tmp_path):
    folder = tmp_path / "experiment"
    a = Archive(folder, protocol(), create=True)
    identity = a.add(proposal())
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        a.db.execute("DELETE FROM candidates WHERE id=?", (identity,))
    a.close()
    with pytest.raises(ValueError, match="Protocol changed"):
        Archive(folder, replace(protocol(), area_fraction=.2))


def test_fixed_budget_evaluation_and_invalid_support():
    shape = (32, 32)
    human = np.zeros(shape, bool); human[:, 10:12] = True
    sample = {"image": np.ones(shape), "valid": np.ones(shape, bool),
              "geometry": "interface", "human": human}
    prediction = {"score": human.astype(float), "support": np.ones(shape, bool)}
    out = evaluate(sample, prediction, protocol())
    assert out["coverage"] == 1 and out["selected"] <= out["budget"]
    assert not out["precision_measured"] and not out["unknown_as_negative"]
    sample["valid"][0, 0] = False
    with pytest.raises(ValueError, match="physically invalid"):
        evaluate(sample, prediction, protocol())


def test_selection_uses_matched_rows_and_retention():
    old = trial_receipt()["rows"]
    assert compare(trial_receipt(.6)["rows"], old, old, protocol())["accepted"]
    assert not compare(trial_receipt(.4)["rows"], old, old, protocol())["accepted"]
    with pytest.raises(ValueError, match="Duplicate"):
        compare(old + old, old, old, protocol())
    bad = trial_receipt(.9)["rows"]; bad[0]["selected"] = 11
    with pytest.raises(ValueError, match="metric/budget"):
        compare(bad, old, old, protocol())


def sandbox_fixture(tmp_path):
    # pytest --basetemp must be explicitly under this repository.
    tmp_path.resolve().relative_to(ROOT)
    inputs, program = tmp_path / "inputs", tmp_path / "program"
    inputs.mkdir(); program.mkdir()
    return inputs, program, Sandbox(ROOT, ROOT / ".venvs/sam2", timeout=30)


def test_real_os_sandbox_blocks_credentials_production_writes_and_network(tmp_path):
    inputs, program, sandbox = sandbox_fixture(tmp_path)
    write_new(inputs / "request.json", {"mode": "probe", "production_path": str(ROOT / "src")})
    output = tmp_path / "execution"
    receipt = sandbox.run(program, inputs, output)
    assert receipt["status"] == "ok", (output / "stderr.log").read_text()
    result = json.loads((output / "result.json").read_text())
    assert all(result[k] for k in ("credentials_absent", "host_home_absent", "production_absent",
                                   "input_readonly", "network_blocked"))
    assert receipt["repository_mounted"] is False


def test_real_candidate_executes_not_only_archived(tmp_path):
    inputs, program, sandbox = sandbox_fixture(tmp_path)
    (program / "pipeline.py").write_text(PIPELINE)
    data = {"image": np.arange(64).reshape(8, 8).astype(np.float32), "valid": np.ones((8, 8), bool)}
    with (inputs / "sample.npz").open("xb") as f:
        np.savez(f, **data)
    write_new(inputs / "request.json", {"mode": "predict", "seed": 23,
                                        "samples": [{"file": "sample.npz"}]})
    output = tmp_path / "execution"
    receipt = sandbox.run(program, inputs, output)
    assert receipt["status"] == "ok", (output / "stderr.log").read_text()
    with np.load(output / "prediction-0000.npz", allow_pickle=False) as z:
        assert np.array_equal(z["score"], data["image"])
    with pytest.raises(FileExistsError):
        sandbox.run(program, inputs, output)


def test_worker_refuses_inference_label_leak(tmp_path):
    inputs, program, sandbox = sandbox_fixture(tmp_path)
    (program / "pipeline.py").write_text(PIPELINE)
    with (inputs / "sample.npz").open("xb") as f:
        np.savez(f, image=np.ones((8, 8)), valid=np.ones((8, 8), bool), human=np.ones((8, 8)))
    write_new(inputs / "request.json", {"mode": "predict", "seed": 23,
                                        "samples": [{"file": "sample.npz"}]})
    output = tmp_path / "execution"
    receipt = sandbox.run(program, inputs, output)
    assert receipt["status"] == "failed"
    assert "Supervision in inference" in (output / "stderr.log").read_text()
