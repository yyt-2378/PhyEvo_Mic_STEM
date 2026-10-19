import numpy as np
import pytest

from scripts.report_rsi_refinement import GROUPS, directed_tables, exact_crop_hit_probability, label_budget_bounds, outer_tables, point_border_audit
from tfree_detection.rsi_evaluation import EvaluationProtocol


@pytest.mark.parametrize("shape,patch", [((7,9),3),((7,9),7),((7,9),12),((1,5),3)])
def test_crop_probability_matches_bruteforce_including_small_images(shape,patch):
    known=np.zeros(shape,bool)
    known[0,0]=True
    known[-1,-2]=True
    ny,nx=max(1,shape[0]-patch+1),max(1,shape[1]-patch+1)
    hits=np.array([[known[y:y+patch,x:x+patch].any() for x in range(nx)] for y in range(ny)])
    expected=.6*hits.mean()+.1*(hits[0].mean()+hits[-1].mean()+hits[:,0].mean()+hits[:,-1].mean())
    assert exact_crop_hit_probability(known,patch)["combined"] == pytest.approx(expected)


def test_empty_supervision_and_dense_supervision_crop_boundaries():
    assert exact_crop_hit_probability(np.zeros((16,16)),8)["combined"] == 0
    assert exact_crop_hit_probability(np.ones((16,16)),8)["combined"] == pytest.approx(1)
    with pytest.raises(ValueError):
        exact_crop_hit_probability(np.ones((3,3,3)),8)
    with pytest.raises(ValueError):
        exact_crop_hit_probability(np.ones((3,3)),0)


def test_report_keeps_best_by_outer_separate_from_actual_inner_selected_policy():
    data,nested=[],[]
    for group in GROUPS:
        for policy,step,method,value in [("control",1200,"student_expert",.2),
                                         ("control",400,"student_expert",.3),
                                         ("control",1200,"physics_budgeted",.1),
                                         ("source_sqrt",400,"student_expert",.8)]:
            data.append({"group":group,"policy":policy,"step":step,"method":method,"mean":value})
        nested.append({"group":group,"method":"nested_feedback","mean":.15})
    table=outer_tables({"outer_comparisons":data,"nested_selected_comparison":nested})
    for row in table:
        assert row["descriptive_outer_best_student"]["mean"] == .8
        assert row["nested_selected"] == .15
        assert row["same_checkpoint_gain_of_descriptive_best"] == pytest.approx(.5)
        assert row["descriptive_outer_best_control"]["step"]==400
        assert row["gain_over_control_best_checkpoint"]==pytest.approx(.5)
        assert row["descriptive_best_is_deployable_policy"] is False


def test_directed_table_pairs_same_policy_and_checkpoint_not_target_best():
    outer,directed=[],[]
    for group in ("sigma3-orientation-i","sigma3-orientation-ii"):
        for step in (400,800,1200):
            row={"group":group,"step":step,"policy":"control","method":"student_expert","mean":step/2000}
            outer.append(row)
            directed.extend([{**row,"mean":step/3000},{**row,"policy":"photometric","mean":.99}])
    result=directed_tables({"outer_comparisons":outer,"directed_same_material_comparisons":directed})
    assert len(result)==6
    for row in result:
        assert row["directed"]["step"]==row["mixed"]["step"]==row["step"]
        assert row["directed"]["policy"]=="control"
        assert row["directed_contains_zro2"] is False
        assert row["same_checkpoint_not_posthoc_best"] is True


def test_budget_bound_is_case_macro_and_not_relabelled_as_accuracy():
    base={"action":"correct_geometry","geometry":"interface","shape":[10,10],"group":"a"}
    manifest={"entries":[{**base,"human_positive_pixels":20},{**base,"human_positive_pixels":10},
        {**base,"action":"accept","human_positive_pixels":100}]}
    row=label_budget_bounds(manifest,EvaluationProtocol(area_budget_fraction=.1))["a"]
    assert row["mean_count_budget_upper_bound"]==.75
    assert row["cases"]==2
    assert row["diagnostic_only_not_routing_or_rescaled_accuracy"] is True


def test_point_border_diagnostic_counts_points_not_frames_and_excludes_accepted():
    base={"geometry":"point","action":"correct_geometry","shape":[100,100],"group":"a"}
    manifest={"entries":[{**base,"points_xy":[[0,50],[50,50]]},{**base,"points_xy":[[99,99]]},
                          {**base,"action":"accept","points_xy":[[50,50]]}]}
    row=point_border_audit(manifest)["a"]
    assert row["marked_points"]==3 and row["within_border"]==2
    assert row["fraction"]==pytest.approx(2/3)
    assert row["not_an_inference_feature"] is True
