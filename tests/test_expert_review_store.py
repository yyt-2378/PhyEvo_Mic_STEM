import dataclasses

import pytest

from tfree_detection.review_store import (
    REVIEW_PROJECT_ID,
    CaseReviewAction,
    ReviewStore,
    ReviewSubmission,
)


_DIGEST = "a" * 64


def _submission(**changes) -> ReviewSubmission:
    values = {
        "project_id": REVIEW_PROJECT_ID,
        "reviewer_id": "expert-a",
        "case_id": "case:synthetic-001",
        "dataset_id": "synthetic",
        "review_split": "train",
        "selected_sample_id": "sample-001",
        "applies_to_sample_ids": ("sample-001",),
        "geometry_hint": "region",
        "annotation_scope": "single_plane",
        "image_shape": (32, 48),
        "action": CaseReviewAction.ACCEPT_PREDICTION,
        "defect_labels": ("amorphous_region",),
        "geometry": None,
        "model_assistance_shown": True,
        "prediction_sha256": _DIGEST,
        "case_manifest_sha256": "b" * 64,
        "annotation_duration_seconds": 3.0,
        "notes": "first pass",
        "client_submission_id": "click-001",
    }
    values.update(changes)
    return ReviewSubmission(**values)


def test_current_projection_survives_reopen_and_history_is_append_only(
    tmp_path,
) -> None:
    path = tmp_path / "reviews.sqlite3"
    store = ReviewStore(path)
    first = store.save(_submission())
    corrected_geometry = {
        "coordinate_space": "source_pixels_xy",
        "image_shape_yx": [32, 48],
        "geometry_type": "region_contour",
        "selection_kind": "lasso",
        "points_xy": [[2.0, 3.0], [20.0, 4.0], [8.0, 22.0]],
        "valid_for_hint": True,
    }
    second = store.save(
        _submission(
            action=CaseReviewAction.CORRECT_GEOMETRY,
            geometry=corrected_geometry,
            notes="corrected boundary",
            client_submission_id="click-002",
        )
    )
    assert first.revision_number == 1
    assert second.revision_number == 2

    reopened = ReviewStore(path)
    current = reopened.get_current("expert-a", "case:synthetic-001")
    assert current is not None
    assert current.revision_id == second.revision_id
    assert current.action == "correct_geometry"
    assert len(reopened.list_current(reviewer_id="expert-a")) == 1
    assert [item.revision_number for item in reopened.history(
        "expert-a", "case:synthetic-001"
    )] == [1, 2]
    assert len(reopened.list_history(reviewer_id="expert-a")) == 2
    assert reopened.integrity_report()["valid"]


def test_retry_and_same_semantics_do_not_create_duplicate_revision(
    tmp_path,
) -> None:
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    submission = _submission()
    first = store.save(submission)
    same_click = store.save(submission)
    same_content_new_click = store.save(
        dataclasses.replace(
            submission,
            client_submission_id="click-retry",
            annotation_duration_seconds=9.0,
        )
    )
    assert same_click.revision_id == first.revision_id
    assert same_content_new_click.revision_id == first.revision_id
    assert len(store.history("expert-a", submission.case_id)) == 1


def test_store_rejects_locked_test_and_invalid_geometry(tmp_path) -> None:
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    with pytest.raises(ValueError, match="train/validation"):
        store.save(_submission(review_split="test"))

    with pytest.raises(ValueError, match="requires expert geometry"):
        store.save(
            _submission(
                action=CaseReviewAction.CORRECT_GEOMETRY,
                geometry=None,
            )
        )


def test_progress_counts_only_current_revisions(tmp_path) -> None:
    store = ReviewStore(tmp_path / "reviews.sqlite3")
    store.save(_submission())
    store.save(
        _submission(
            action=CaseReviewAction.NO_DEFECT,
            defect_labels=(),
            prediction_sha256=None,
            notes="empty",
            client_submission_id="click-002",
        )
    )
    progress = store.progress(
        ["case:synthetic-001", "case:synthetic-002"],
        reviewer_id="expert-a",
    )
    assert progress["completed_count"] == 1
    assert progress["remaining_count"] == 1
    assert progress["action_counts"] == {"no_defect": 1}
