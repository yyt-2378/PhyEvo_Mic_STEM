from scripts.report_expert_learning_cycle import aggregate_group_metrics, evidence_audit


def test_multiseed_statistics_do_not_pool_frames_as_independent_acquisitions():
    summaries = [{"group_metrics": [
        {"geometry": "point", "group": "a", "variant": "teacher_only", "cases": 100,
         "marked_point_recall_at_50": value},
        {"geometry": "point", "group": "b", "variant": "teacher_only", "cases": 1,
         "marked_point_recall_at_50": 0.9},
    ]} for value in [0.2, 0.4]]
    result = aggregate_group_metrics(summaries)
    assert len(result) == 2
    first = result[0]
    assert first["seeds"] == 2
    assert abs(first["marked_point_recall_at_50"]["mean"] - 0.3) < 1e-10
    assert first["marked_point_recall_at_50"]["values"] == [0.2, 0.4]


def test_descriptive_audit_counts_source_pixel_border_distance():
    manifest = {"entries": [{"case_number": 1, "human_positive_pixels": 2,
        "teacher_positive_pixels": 0, "geometry": "point", "action": "correct_geometry",
        "semantic_kind": "sparse_positive_points", "shape": [64, 64], "points_xy": [[2, 3], [32, 32]]}]}
    result = evidence_audit(manifest)
    assert result["explicit_points"] == 2
    assert result["points_within_24px_border"] == 1
    assert result["empty_teacher_positive_expert"] == {"point": 1}
