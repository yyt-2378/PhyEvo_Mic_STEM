import json
from pathlib import Path

import numpy as np
import pytest

from tfree_detection.config import DetectorConfig
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.governance import CandidateRecord, DataPartition
from tfree_detection.review import ReviewAction, ReviewResponse
from tfree_detection.review_ui import (
    ReviewBundleItem,
    build_multisource_review_bundle,
    build_review_bundle,
    load_and_validate_review_responses,
)
from tfree_detection.synthetic import make_crystal_with_amorphous_region


def _result():
    image, _ = make_crystal_with_amorphous_region(size=128, seed=18)
    config = DetectorConfig()
    config.spectral.patch_size = 48
    config.spectral.stride = 12
    config.spectral.max_frequency = 0.25
    config.spectral.autocorrelation_min_lag = 5
    config.spectral.autocorrelation_max_lag = 18
    config.fusion.threshold_quantile = 0.88
    config.fusion.threshold_mad = 1.8
    config.postprocess.border_margin = 24
    config.postprocess.minimum_region_area = 80
    result = TrainingFreeDetectionFramework(config).detect(image)
    assert result.regions
    return result


def test_assisted_and_blinded_bundles_keep_distinct_artifacts(
    tmp_path: Path,
) -> None:
    result = _result()
    assisted_dir = tmp_path / "assisted"
    blind_dir = tmp_path / "blind"
    assisted = build_review_bundle(
        result,
        assisted_dir,
        show_model_assistance=True,
        dataset_id="synthetic",
        partition=DataPartition.DEVELOPMENT_TEST,
    )
    blinded = build_review_bundle(
        result,
        blind_dir,
        show_model_assistance=False,
        dataset_id="synthetic",
        partition=DataPartition.DEVELOPMENT_TEST,
    )
    assert assisted["candidate_count"] == blinded["candidate_count"]
    assert len(assisted["artifact_sha256"]) == 2 * assisted["candidate_count"]
    assert len(blinded["artifact_sha256"]) == blinded["candidate_count"]
    packets = json.loads(
        (blind_dir / "review_packets.json").read_text(encoding="utf-8")
    )
    for packet in packets:
        candidate = packet["candidate"]
        assert "teacher_score" not in candidate
        assert "defect_type" not in candidate
        assert "evidence_vector" not in candidate
        assert packet["overlay_artifact"] == packet["image_artifact"]
    assert "teacher_score" not in (blind_dir / "index.html").read_text(
        encoding="utf-8"
    )


def test_review_export_is_validated_against_assistance_policy(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    build_review_bundle(
        _result(),
        bundle,
        show_model_assistance=False,
        partition=DataPartition.TRAIN,
    )
    packet = json.loads(
        (bundle / "review_packets.json").read_text(encoding="utf-8")
    )[0]
    response = {
        "response_id": "response-1",
        "candidate_id": packet["candidate"]["candidate_id"],
        "reviewer_id": "expert-a",
        "action": "accept",
        "defect_labels": ["coherence-loss"],
        "annotation_duration_seconds": 12.4,
        "model_assistance_shown": False,
        "geometry_payload": None,
        "notes": "",
    }
    response_path = tmp_path / "responses.json"
    response_path.write_text(
        json.dumps([response]), encoding="utf-8"
    )
    loaded = load_and_validate_review_responses(
        response_path, bundle / "review_packets.json"
    )
    assert loaded[0].reviewer_id == "expert-a"

    response["model_assistance_shown"] = True
    mismatch = tmp_path / "mismatch.json"
    mismatch.write_text(json.dumps([response]), encoding="utf-8")
    with pytest.raises(ValueError, match="assistance condition"):
        load_and_validate_review_responses(
            mismatch, bundle / "review_packets.json"
        )


def test_adjusted_geometry_requires_a_label_and_geometry() -> None:
    with pytest.raises(ValueError, match="need defect labels"):
        ReviewResponse(
            response_id="r",
            candidate_id="c",
            reviewer_id="expert",
            action=ReviewAction.ADJUST_LINE,
            defect_labels=(),
            annotation_duration_seconds=1.0,
            model_assistance_shown=False,
            geometry_payload={"points_xy": [[1, 1], [2, 2]]},
        ).validate()
    with pytest.raises(ValueError, match="geometry_payload"):
        ReviewResponse(
            response_id="r",
            candidate_id="c",
            reviewer_id="expert",
            action=ReviewAction.ADJUST_LINE,
            defect_labels=("phase-slip",),
            annotation_duration_seconds=1.0,
            model_assistance_shown=False,
        ).validate()


def test_locked_candidates_cannot_enter_review_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Locked external-test"):
        build_review_bundle(
            _result(),
            tmp_path / "locked",
            show_model_assistance=False,
            partition=DataPartition.LOCKED_EXTERNAL_TEST,
        )


def test_multisource_blinded_bundle_hides_selection_branch(
    tmp_path: Path,
) -> None:
    items = []
    for index in range(2):
        candidate = CandidateRecord(
            candidate_id=f"opaque-{index}",
            task="atomic_point_defect",
            source_id=f"source-{index}",
            dataset_id="train-set",
            partition=DataPartition.TRAIN.value,
            geometry="point",
            violation_types=("teacher_disagreement",),
            defect_type="private_selection_branch",
            bbox_xyxy=(28, 28, 37, 37),
            centroid_xy=(32.0, 32.0),
            area_pixels=49,
            teacher_score=0.5,
            legacy_confidence=0.5,
            reliability=0.5,
            decision_state="uncertain",
            label_tier="unresolved",
            evidence_vector={
                "private_branch": {"value": float(index)}
            },
            quality_flags=(),
            provenance={"selection_branch": f"branch-{index}"},
        )
        image = np.linspace(0, 1, 64 * 64).reshape(64, 64)
        mask = np.zeros((64, 64), dtype=bool)
        mask[28:37, 28:37] = True
        items.append(
            ReviewBundleItem(
                candidate=candidate,
                normalized_image=image,
                mask=mask,
                source_provenance={"source_id": f"source-{index}"},
            )
        )
    manifest = build_multisource_review_bundle(
        items,
        tmp_path / "multisource",
        show_model_assistance=False,
    )
    assert manifest["candidate_count"] == 2
    assert manifest["source_count"] == 2
    packets = json.loads(
        (
            tmp_path / "multisource" / "review_packets.json"
        ).read_text(encoding="utf-8")
    )
    serialized = json.dumps(packets)
    assert "private_selection_branch" not in serialized
    assert "selection_branch" not in serialized
    assert "teacher_score" not in serialized
