import numpy as np
import pytest

from scripts.analyze_rsi_refinement import macro_summary, save_once
from tfree_detection.rsi_evaluation import EvaluationProtocol, evaluate_case, predict_methods
from tfree_detection.rsi_feedback import feedback_predictions


def test_feedback_does_not_change_baseline_evaluation_semantics():
    rng = np.random.default_rng(17)
    image = rng.random((32,32))
    teacher = image > .7
    human = np.zeros_like(teacher)
    human[10:15,10:15] = True
    score = rng.random(image.shape)
    entry = {"case_number": 1, "group": "test", "geometry": "interface", "action": "correct_geometry"}
    plane = {"image": image, "teacher": teacher, "human": human}
    a = evaluate_case(entry, plane, predict_methods(image,teacher,[],score,score,"interface"))
    b = evaluate_case(entry, plane, feedback_predictions(image,teacher,[],score,"interface"))
    for key in b["methods"]:
        if key in a["methods"]:
            assert a["methods"][key] == b["methods"][key]


def test_macro_averages_cases_then_seeds_without_pooled_pixel_counts():
    rows = []
    for seed, values in ((23,[0,1]), (41,[.2,.4])):
        for value in values:
            rows.append({"main_comparison_eligible": True, "group": "a", "geometry": "interface",
                         "policy": "b", "step": 400, "seed": seed,
                         "methods": {"student_expert": {"positive_coverage": value}}})
    result = macro_summary(rows)[0]
    assert result["mean"] == pytest.approx(.4)
    assert result["seed_std"] == pytest.approx(.1)
    assert result["seed_std_is_not_independent_acquisition_uncertainty"]


def test_saved_selector_cannot_be_silently_replaced(tmp_path):
    target = tmp_path/"selector.json"
    save_once(target, {"chosen": "physics"})
    save_once(target, {"chosen": "physics"})
    with pytest.raises(ValueError, match="changed"):
        save_once(target, {"chosen": "student"})
