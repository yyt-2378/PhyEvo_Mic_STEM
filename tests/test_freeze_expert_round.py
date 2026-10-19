import json
import sqlite3

import pytest

from scripts.freeze_expert_round import (
    file_sha256, project_path, readonly_database, snapshot_database,
    verified_revisions,
)
from tfree_detection.review_store import (
    CaseReviewAction, REVIEW_PROJECT_ID, ReviewStore, ReviewSubmission,
)


def submission(reviewer="expert-a", case_id="case:one"):
    return ReviewSubmission(
        project_id=REVIEW_PROJECT_ID, reviewer_id=reviewer, case_id=case_id,
        dataset_id="synthetic", review_split="train", selected_sample_id="sample-1",
        applies_to_sample_ids=("sample-1",), geometry_hint="point",
        annotation_scope="single_plane", image_shape=(8, 8),
        action=CaseReviewAction.CORRECT_GEOMETRY,
        defect_labels=("atomic_column_distortion",),
        geometry={
            "coordinate_space": "source_pixels_xy", "geometry_type": "point_set",
            "image_shape_yx": [8, 8], "points_xy": [[3.0, 4.0]],
        },
        model_assistance_shown=True, prediction_sha256="a" * 64,
        case_manifest_sha256="b" * 64, annotation_duration_seconds=1.0,
        notes="synthetic", client_submission_id=f"click:{reviewer}:{case_id}",
    )


def test_snapshot_includes_committed_wal_and_does_not_mutate_source(tmp_path):
    live = tmp_path / "live.sqlite3"
    store = ReviewStore(live)
    # Keep WAL alive while snapshotting, rather than testing a checkpointed DB.
    keeper = sqlite3.connect(live)
    keeper.execute("PRAGMA journal_mode=WAL")
    try:
        saved = store.save(submission())
        snapshot = tmp_path / "frozen.sqlite3"
        snapshot_database(live, snapshot)
        records, history, excluded = verified_revisions(snapshot, ("expert-a",))
        assert records[0].revision_id == saved.revision_id
        assert history == 1 and excluded == 0
        assert store.get_current("expert-a", "case:one") == saved
        assert len(store.list_history()) == 1
    finally:
        keeper.close()


def test_snapshot_refuses_overwrite(tmp_path):
    live = tmp_path / "live.sqlite3"
    ReviewStore(live).save(submission())
    snapshot = tmp_path / "frozen.sqlite3"
    snapshot_database(live, snapshot)
    original = file_sha256(snapshot)
    with pytest.raises(FileExistsError):
        snapshot_database(live, snapshot)
    assert file_sha256(snapshot) == original


def test_nonselected_legacy_record_is_preserved_but_not_selected(tmp_path):
    path = tmp_path / "reviews.sqlite3"
    store = ReviewStore(path)
    store.save(submission())
    store.save(submission("legacy"))
    records, history, excluded = verified_revisions(path, ("expert-a",))
    assert len(records) == 1 and history == 2 and excluded == 1
    assert len(store.list_current()) == 2


def test_overlapping_reviewers_are_not_silently_merged(tmp_path):
    path = tmp_path / "reviews.sqlite3"
    store = ReviewStore(path)
    store.save(submission())
    store.save(submission("expert-b"))
    with pytest.raises(ValueError, match="Overlapping reviewers"):
        verified_revisions(path, ("expert-a", "expert-b"))


def test_corrupt_payload_is_detected(tmp_path):
    # This is an isolated synthetic fixture, never the production ledger.
    path = tmp_path / "reviews.sqlite3"
    ReviewStore(path).save(submission())
    with sqlite3.connect(path) as connection:
        raw = connection.execute("SELECT payload_json FROM review_events").fetchone()[0]
        payload = json.loads(raw)
        payload["notes"] = "tampered"
        connection.execute("UPDATE review_events SET payload_json=?", (json.dumps(payload),))
    with pytest.raises(ValueError, match="Payload checksum mismatch"):
        verified_revisions(path, ("expert-a",))


def test_query_only_database_rejects_writes(tmp_path):
    path = tmp_path / "reviews.sqlite3"
    ReviewStore(path).save(submission())
    connection = readonly_database(path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("CREATE TABLE forbidden_test_write (value TEXT)")
    finally:
        connection.close()


def test_outside_workspace_path_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        project_path(tmp_path, tmp_path.parent / "outside.sqlite3")


def test_missing_or_duplicate_reviewer_selection_is_rejected(tmp_path):
    path = tmp_path / "reviews.sqlite3"
    ReviewStore(path).save(submission())
    for reviewers in [(), ("expert-a", "expert-a"), ("missing",)]:
        with pytest.raises(ValueError):
            verified_revisions(path, reviewers)
