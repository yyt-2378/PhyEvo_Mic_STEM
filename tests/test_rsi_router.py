import copy

import pytest

from tfree_detection.rsi_router import fit_router, select_branch


def evidence(group, case=1, *, student=0.4, hybrid=0.3, physics=0.2):
    return {"outer_group": "outer", "group": group, "seed": 23, "case_number": case,
            "model_train_groups": ["train"], "model_geometry_support": ["interface"],
            "main_comparison_eligible": True, "geometry": "interface", "teacher_empty": False,
            "methods": {b: {"positive_coverage": v} for b, v in
                        [("physics_budgeted", physics), ("student_expert", student), ("hybrid_fixed_budget", hybrid)]}}


def test_inner_evidence_selects_without_outer_labels_and_unknown_scope_falls_back():
    router = fit_router([evidence("inner-a"), evidence("inner-b", 2)], outer_group="outer")
    assert select_branch(router, geometry="interface", teacher_empty=False) == "student_expert"
    assert select_branch(router, geometry="point", teacher_empty=False) == "physics_budgeted"
    assert select_branch(router, geometry="interface", teacher_empty=True) == "physics_budgeted"


def test_one_independent_group_even_many_seeds_cannot_select_student():
    records = [dict(evidence("inner-a"), seed=s) for s in [23, 41, 67]]
    router = fit_router(records, outer_group="outer")
    assert select_branch(router, geometry="interface", teacher_empty=False) == "physics_budgeted"


def test_average_gain_cannot_hide_inner_acquisition_regression():
    router = fit_router([evidence("inner-a", student=0.99, hybrid=0.1),
                         evidence("inner-b", 2, student=0.1, hybrid=0.1)], outer_group="outer")
    assert select_branch(router, geometry="interface", teacher_empty=False) == "physics_budgeted"


@pytest.mark.parametrize("field,value", [("group", "outer"), ("outer_group", "wrong"),
                                          ("model_train_groups", ["outer"]),
                                          ("model_train_groups", ["inner-a"])])
def test_nested_leakage_is_rejected(field, value):
    record = evidence("inner-a")
    record[field] = value
    with pytest.raises(ValueError):
        fit_router([record], outer_group="outer")


def test_duplicate_or_nan_evidence_is_rejected():
    record = evidence("inner-a")
    with pytest.raises(ValueError, match="Duplicate"):
        fit_router([record, record], outer_group="outer")
    invalid = copy.deepcopy(record)
    invalid["methods"]["student_expert"]["positive_coverage"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        fit_router([invalid], outer_group="outer")


def test_untrained_geometry_is_not_treated_as_a_valid_student_capability():
    records = [evidence("inner-a"), evidence("inner-b", 2)]
    for r in records:
        r["model_geometry_support"] = ["point"]
    router = fit_router(records, outer_group="outer")
    assert select_branch(router, geometry="interface", teacher_empty=False) == "physics_budgeted"


def test_router_inference_never_uses_test_group_or_labels():
    router = fit_router([evidence("inner-a"), evidence("inner-b", 2)], outer_group="outer")
    with pytest.raises(TypeError):
        select_branch(router, geometry="interface", teacher_empty=False, group="outer")
    with pytest.raises(ValueError):
        select_branch(router, geometry="interface", teacher_empty="False")
