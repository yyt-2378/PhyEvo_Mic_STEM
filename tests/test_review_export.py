import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from skimage.io import imread, imsave

from tfree_detection.review_export import (
    build_review_export,
    rasterize_review_geometry,
    review_usage_policy,
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


def _source_asset_record(
    sample: ReviewSample, *, registry_sha256: str = "b" * 64
) -> dict:
    return {
        "registry_sha256": registry_sha256,
        "asset": {
            "relative_path": sample.source_path,
            "dataset_id": sample.dataset_id,
            "partition": sample.source_partition,
            "split_group_id": sample.split_group_id,
            "acquisition_id": sample.acquisition_id,
            "material": sample.material,
        },
    }


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
        source_review_samples=[sample],
        source_asset_records=[_source_asset_record(sample)],
        predictions={(sample.sample_id, sample.primary_task): prediction},
        project_root=tmp_path,
        output_directory="export-v1",
        case_manifest_sha256="e" * 64,
        review_queue_sha256="a" * 64,
        source_review_manifest_sha256="c" * 64,
        source_asset_manifest_sha256="f" * 64,
        source_registry_sha256="b" * 64,
    )
    assert summary["training_eligible_count"] == 1
    assert summary["compiled_mask_count"] == 1
    assert summary["training_excluded_count"] == 0
    assert summary["source_partition_counts"] == {"train": 1}
    assert summary["case_manifest_sha256"] == "e" * 64
    assert summary["review_queue_sha256"] == "a" * 64
    assert summary["source_review_manifest_sha256"] == "c" * 64
    assert summary["source_asset_manifest_sha256"] == "f" * 64
    assert summary["source_registry_sha256"] == "b" * 64
    record = json.loads(
        (tmp_path / "export-v1/annotations.jsonl").read_text()
    )
    exported_mask = imread(tmp_path / record["derived_mask_path"])
    assert np.array_equal(exported_mask != 0, mask != 0)
    assert record["label_tier"] == "expert_single_silver"
    assert record["source_partition"] == "train"
    assert record["training_eligible"] is True
    assert record["calibration_eligible"] is False
    assert record["locked_evaluation_eligible"] is False
    assert record["case_manifest_sha256"] == "e" * 64
    assert record["review_queue_sha256"] == "a" * 64
    assert record["source_review_manifest_sha256"] == "c" * 64
    assert record["source_asset_manifest_sha256"] == "f" * 64
    assert record["source_registry_sha256"] == "b" * 64


def test_review_split_cannot_promote_development_assets_to_training() -> None:
    development = ReviewSample(
        sample_id="development",
        dataset_id="synthetic",
        source_path="image.png",
        source_format="png",
        source_partition="development_test",
        review_split="train",
        split_group_id="development-group",
        acquisition_id="development-acquisition",
        material="synthetic",
        primary_task="amorphous_region",
        prediction_tasks=("amorphous_region",),
        geometry_hint="region",
    )
    validation = ReviewSample(
        sample_id="validation",
        dataset_id="synthetic",
        source_path="image.png",
        source_format="png",
        source_partition="train",
        review_split="validation",
        split_group_id="validation-group",
        acquisition_id="validation-acquisition",
        material="synthetic",
        primary_task="amorphous_region",
        prediction_tasks=("amorphous_region",),
        geometry_hint="region",
    )
    train = ReviewSample(
        sample_id="train",
        dataset_id="synthetic",
        source_path="image.png",
        source_format="png",
        source_partition="train",
        review_split="train",
        split_group_id="train-group",
        acquisition_id="train-acquisition",
        material="synthetic",
        primary_task="amorphous_region",
        prediction_tasks=("amorphous_region",),
        geometry_hint="region",
    )

    development_policy = review_usage_policy(
        development, compiled_mask_available=True
    )
    validation_policy = review_usage_policy(
        validation, compiled_mask_available=True
    )
    train_policy = review_usage_policy(train, compiled_mask_available=True)

    assert development_policy["training_eligible"] is False
    assert validation_policy["training_eligible"] is False
    assert train_policy["training_eligible"] is True
    assert development_policy["allowed_uses"] == ["expert_review_audit"]


def test_export_rejects_queue_partition_rewritten_after_source_manifest(
    tmp_path,
) -> None:
    source = ReviewSample(
        sample_id="source-linked",
        dataset_id="synthetic",
        source_path="image.png",
        source_format="png",
        source_partition="development_test",
        review_split="validation",
        split_group_id="source-group",
        acquisition_id="source-acquisition",
        material="synthetic",
        primary_task="amorphous_region",
        prediction_tasks=("amorphous_region",),
        geometry_hint="region",
    )
    tampered_queue_sample = replace(
        source,
        source_partition="train",
        review_split="train",
    )
    with pytest.raises(
        ValueError,
        match="differs from its source review manifest",
    ):
        build_review_export(
            [],
            cases=[],
            samples=[tampered_queue_sample],
            source_review_samples=[source],
            source_asset_records=[_source_asset_record(source)],
            predictions={},
            project_root=tmp_path,
            output_directory="must-not-exist",
            case_manifest_sha256="e" * 64,
            review_queue_sha256="a" * 64,
            source_review_manifest_sha256="c" * 64,
            source_asset_manifest_sha256="f" * 64,
            source_registry_sha256="b" * 64,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_development_review_exports_mask_but_not_training_label(tmp_path) -> None:
    sample = ReviewSample(
        sample_id="development-001",
        dataset_id="synthetic",
        source_path="image.png",
        source_format="png",
        source_partition="development_test",
        review_split="train",
        split_group_id="development-group",
        acquisition_id="development-acquisition",
        material="synthetic",
        primary_task="amorphous_region",
        prediction_tasks=("amorphous_region",),
        geometry_hint="region",
    )
    case = build_expert_review_cases([sample])[0]
    revision = ReviewRevision(
        revision_id="revision:development-no-defect",
        revision_number=1,
        submitted_at_utc="2026-08-11T00:00:00+00:00",
        client_submission_id="click-development-no-defect",
        payload_sha256="f" * 64,
        annotation_duration_seconds=3.0,
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
            "action": "no_defect",
            "defect_labels": [],
            "geometry": None,
            "model_assistance_shown": True,
            "prediction_sha256": None,
            "case_manifest_sha256": "e" * 64,
            "notes": "development audit",
        },
    )
    summary = build_review_export(
        [revision],
        cases=[case],
        samples=[sample],
        source_review_samples=[sample],
        source_asset_records=[_source_asset_record(sample)],
        predictions={},
        project_root=tmp_path,
        output_directory="development-export-v2",
        case_manifest_sha256="e" * 64,
        review_queue_sha256="a" * 64,
        source_review_manifest_sha256="c" * 64,
        source_asset_manifest_sha256="f" * 64,
        source_registry_sha256="b" * 64,
    )
    record = json.loads(
        (tmp_path / "development-export-v2/annotations.jsonl").read_text()
    )
    assert summary["compiled_mask_count"] == 1
    assert summary["training_eligible_count"] == 0
    assert summary["training_exclusion_counts"] == {
        "source_partition_or_review_split_forbids_training": 1
    }
    assert record["training_eligible"] is False
    assert record["source_partition"] == "development_test"
    assert record["allowed_uses"] == ["expert_review_audit"]
