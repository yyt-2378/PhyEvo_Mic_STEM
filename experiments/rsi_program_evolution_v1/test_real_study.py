"""Source exclusion checks include inherited weights, not only new fitting."""
import pytest

from experiments.rsi_program_evolution_v1.real_study import exclusion_audit


def test_inherited_model_sources_are_part_of_training_contract():
    job = {"validation_group": "B", "outer_group": "C", "train_groups": ["A"], "train_cases": [1]}
    base = {"train_case_numbers": [1, 2], "fold": {"excluded_groups": ["B", "C"]}}
    retained = {"train_case_numbers": [1]}
    entries = {1: {"group": "A"}, 2: {"group": "auxiliary"}}
    out = exclusion_audit(job, base, retained, entries)
    assert out["all_inherited_training_groups"] == ["A", "auxiliary"]
    assert out["new_fit_groups"] == ["A"]
    entries[2]["group"] = "B"
    with pytest.raises(ValueError, match="Inherited model"):
        exclusion_audit(job, base, retained, entries)


def test_both_held_out_sources_must_be_in_base_exclusions():
    job = {"validation_group": "B", "outer_group": "C", "train_groups": ["A"], "train_cases": [1]}
    with pytest.raises(ValueError, match="both held-out"):
        exclusion_audit(job, {"train_case_numbers": [1], "fold": {"excluded_groups": ["B"]}},
                        {"train_case_numbers": [1]}, {1: {"group": "A"}})
