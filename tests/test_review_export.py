import hashlib
import json

import numpy as np
from skimage.io import imread, imsave

from tfree_detection.review_export import (
    build_review_export,
    rasterize_review_geometry,
    review_training_mask,
)
from tfree_detection.review_cases import build_expert_review_cases
from tfree_detection.review_dataset import ReviewSample
from tfree_detection.review_prediction import PREDICTION_FORMAT, ReviewPrediction
from tfree_detection.review_store import ReviewRevision


def _revision(action: str, geometry=None) -> ReviewRevision:
    return ReviewRevision(
        revision_id="revision:test",
        revision_number=1,
        submitted_at_utc="2026-07-31T00:00:00+00:00",
        client_submission_id="click-test",
        payload_sha256="a" * 64,
        annotation_duration_seconds=1.0,
        submission={
            "project_id": "project",
            "reviewer_id": "expert",
            "case_id": "case:test",
            "image_shape": [32, 48],
            "geometry_hint": "region",
            "action": action,
            "geometry": geometry,
        },
    )


def test_region_line_and_points_rasterize_with_expected_support() -> None:
    region = rasterize_review_geometry(
        {"points_xy": [[2, 2], [20, 2], [20, 20], [2, 20]]},
        geometry_hint="region",
        image_shape=(32, 48),
    )
    line = rasterize_review_geometry(
        {"points_xy": [[2, 2], [20, 20]]},
        geometry_hint="interface",
        image_shape=(32, 48),
    )
    points = rasterize_review_geometry(
        {"points_xy": [[8, 8], [24, 20]]},
        geometry_hint="point",
        image_shape=(32, 48),
    )
    assert region[10, 10]
    assert line[10, 10]
    assert points[8, 8] and points[20, 24]
    assert region.sum() > line.sum() > 0


def test_only_trainable_decisions_produce_masks() -> None:
    prediction = np.zeros((32, 48), dtype=bool)
    prediction[4:8, 7:12] = True
    accepted = review_training_mask(
        _revision("accept_prediction"), prediction_mask=prediction
    )
    empty = review_training_mask(_revision("no_defect"))
    rejected = review_training_mask(_revision("reject_prediction"))
    assert np.array_equal(accepted, prediction)
    assert empty is not None and not np.any(empty)
    assert rejected is None


def test_accepted_prediction_exports_sealed_mask_and_provenance(tmp_path) -> None:
    sample = ReviewSample(
        sample_id="sample-001",
        dataset_id="synthetic",
        source_path="image.png",
        source_format="png",
        source_partition="train",
        review_split="train",
        split_group_id="group",
        acquisition_id="acquisition",
        material="synthetic",
        primary_task="amorphous_region",
        prediction_tasks=("amorphous_region",),
        geometry_hint="region",
    )
    case = build_expert_review_cases([sample])[0]
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[5:10, 8:16] = 255
    mask_path = tmp_path / "prediction-mask.png"
    imsave(str(mask_path), mask, check_contrast=False)
    mask_digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    prediction = ReviewPrediction(
        prediction_format=PREDICTION_FORMAT,
        sample_id=sample.sample_id,
        task=sample.primary_task,
        sample_sha256="a" * 64,
        config_path="configs/amorphous.yaml",
        config_sha256="b" * 64,
        pixel_sha256="c" * 64,
        mask_path=mask_path.relative_to(tmp_path).as_posix(),
        mask_sha256=mask_digest,
        image_shape=(32, 48),
        detected_area_pixels=40,
        detected_area_fraction=40 / (32 * 48),
        candidates=(),
    )
    revision = ReviewRevision(
        revision_id="revision:accepted",
        revision_number=1,
        submitted_at_utc="2026-07-31T00:00:00+00:00",
        client_submission_id="click-accepted",
        payload_sha256="d" * 64,
        annotation_duration_seconds=2.0,
        submission={
            "project_id": "project",
            "reviewer_id": "expert",
            "case_id": case.case_id,
            "dataset_id": sample.dataset_id,
            "review_split": "train",
            "selected_sample_id": sample.sample_id,
            "applies_to_sample_ids": list(case.sample_ids),
            "geometry_hint": "region",
            "annotation_scope": "single_plane",
            "image_shape": [32, 48],
            "action": "accept_prediction",
            "defect_labels": ["amorphous_region"],
            "geometry": None,
            "model_assistance_shown": True,
            "prediction_sha256": mask_digest,
            "case_manifest_sha256": "e" * 64,
            "notes": "accepted",
        },
    )
    summary = build_review_export(
        [revision],
        cases=[case],
        samples=[sample],
        predictions={(sample.sample_id, sample.primary_task): prediction},
        project_root=tmp_path,
        output_directory="export-v1",
    )
    assert summary["training_eligible_count"] == 1
    record = json.loads(
        (tmp_path / "export-v1/annotations.jsonl").read_text()
    )
    exported_mask = imread(tmp_path / record["derived_mask_path"])
    assert np.array_equal(exported_mask != 0, mask != 0)
    assert record["label_tier"] == "expert_single_silver"
