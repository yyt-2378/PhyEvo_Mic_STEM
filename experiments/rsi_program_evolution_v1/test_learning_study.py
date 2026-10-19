from dataclasses import replace

import numpy as np
import pytest

from experiments.rsi_program_evolution_v1.learning_study import trace_check, validate_learner
from experiments.rsi_program_evolution_v1.test_core import proposal, protocol


def state():
    return {"audit_initial_parameters": np.array([0., 1.]), "audit_final_parameters": np.array([.1, .8]),
            "audit_loss_trace": np.array([.5, .4]), "audit_steps": np.array([2])}


def test_training_trace_requires_actual_changed_parameters_and_step_counts():
    assert trace_check(state(), 2)["parameter_delta_l2"] > 0
    for changes in ({"audit_steps": np.array([1])}, {"audit_loss_trace": np.array([.5])},
                    {"audit_final_parameters": np.array([0., 1.])},
                    {"audit_final_parameters": np.array([np.nan, .8])}):
        with pytest.raises(ValueError):
            trace_check({**state(), **changes}, 2)


def test_training_trace_rejects_checkpoint_without_provenance():
    with pytest.raises(ValueError, match="trace"):
        trace_check({"weight": np.ones(3)}, 2)


def test_learning_slot_is_not_satisfied_by_an_untrained_score_rule():
    parent = proposal()
    p = replace(parent, action="tune", parents=(parent.identity,), target="learning", training_steps=2,
                pipeline_code=parent.pipeline_code + "\ndef fit(samples, context):\n    return {}\n")
    validate_learner(p, parent.identity, protocol())
    for changed in (replace(p, training_steps=0), replace(p, target="analysis"),
                    replace(p, pipeline_code=parent.pipeline_code)):
        with pytest.raises(ValueError):
            validate_learner(changed, parent.identity, protocol())
