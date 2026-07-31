from tfree_detection.review_app_logic import (
    ReviewQueueFilter,
    ReviewView,
    filter_review_queue,
    merge_selection_geometry,
    remove_last_geometry_point,
    selection_geometry,
    view_policy,
)
from tfree_detection.review_dataset import ReviewSample


def _sample(
    sample_id: str,
    *,
    dataset: str = "guo",
    split: str = "train",
    priority: str = "high_unlabeled",
    eligible: bool = True,
) -> ReviewSample:
    return ReviewSample(
        sample_id=sample_id,
        dataset_id=dataset,
        source_path="image.png",
        source_format="png",
        source_partition="train",
        review_split=split,
        split_group_id=f"group-{sample_id}",
        acquisition_id=f"acquisition-{sample_id}",
        material="test",
        primary_task="atomic_point_defect",
        prediction_tasks=("atomic_point_defect",),
        geometry_hint="point",
        review_priority=priority,
        review_eligible=eligible,
    )


def test_queue_filters_locked_and_nonmatching_samples() -> None:
    samples = [
        _sample("a"),
        _sample("b", dataset="emseek", priority="reference_audit"),
        _sample("c", eligible=False),
    ]
    filtered = filter_review_queue(
        samples,
        ReviewQueueFilter(
            datasets=("guo",),
            priorities=("high_unlabeled",),
        ),
    )
    assert [item.sample_id for item in filtered] == ["a"]


def test_reference_label_cannot_be_revealed_before_primary_review() -> None:
    blocked = view_policy(
        ReviewView.ADJUDICATION,
        primary_review_exists=False,
        reference_label_available=True,
        prediction_available=True,
    )
    assert not blocked.show_reference_label
    assert not blocked.show_prediction
    assert not blocked.allow_decision

    revealed = view_policy(
        ReviewView.ADJUDICATION,
        primary_review_exists=True,
        reference_label_available=True,
        prediction_available=True,
    )
    assert revealed.show_reference_label
    assert revealed.show_prediction


def test_lasso_and_box_selections_become_source_pixel_geometry() -> None:
    lasso = selection_geometry(
        {
            "selection": {
                "lasso": [
                    {"x": [-2, 12, 12], "y": [3, 3, 99]}
                ]
            }
        },
        image_shape=(32, 24),
        geometry_hint="region",
    )
    assert lasso is not None
    assert lasso["geometry_type"] == "region_contour"
    assert lasso["points_xy"][0] == [0.0, 3.0]
    assert lasso["points_xy"][-1] == [12.0, 31.0]
    assert lasso["valid_for_hint"]

    box = selection_geometry(
        {"selection": {"box": [{"x": [4, 9], "y": [6, 11]}]}},
        image_shape=(32, 24),
        geometry_hint="interface",
    )
    assert box is not None
    assert box["selection_kind"] == "box"
    assert len(box["points_xy"]) == 4


def test_sequential_clicks_append_and_can_undo_control_points() -> None:
    first = selection_geometry(
        {"selection": {"points": [{"x": 3, "y": 4}]}},
        image_shape=(32, 24),
        geometry_hint="interface",
    )
    second = selection_geometry(
        {"selection": {"points": [{"x": 12, "y": 18}]}},
        image_shape=(32, 24),
        geometry_hint="interface",
    )
    assert first is not None and second is not None
    merged = merge_selection_geometry(
        first, second, geometry_hint="interface"
    )
    assert merged["points_xy"] == [[3.0, 4.0], [12.0, 18.0]]
    assert merged["valid_for_hint"]
    undone = remove_last_geometry_point(merged, geometry_hint="interface")
    assert undone is not None
    assert undone["points_xy"] == [[3.0, 4.0]]
    assert not undone["valid_for_hint"]
