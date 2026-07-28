import numpy as np
import pytest

from tfree_detection.governance import DataPartition
from tfree_detection.lineage import (
    MetricSnapshot,
    ModelLineageRecord,
    evaluate_release,
)
from tfree_detection.uncertainty import (
    MahalanobisOODDetector,
    OODCalibrationEmbedding,
    ReferenceEmbedding,
    UncertaintyComponents,
    ood_report,
)


def _reference(index: int, partition: DataPartition) -> ReferenceEmbedding:
    return ReferenceEmbedding(
        asset_id=f"asset-{index}",
        embedding=np.asarray([index * 0.05, -index * 0.03, 0.1]),
        partition=partition,
        split_group_id=f"group-{index}",
    )


def test_ood_fit_and_threshold_respect_partitions() -> None:
    with pytest.raises(ValueError, match="only train"):
        MahalanobisOODDetector().fit(
            [
                _reference(0, DataPartition.TRAIN),
                _reference(1, DataPartition.DEVELOPMENT_TEST),
            ]
        )
    references = [_reference(index, DataPartition.TRAIN) for index in range(8)]
    detector = MahalanobisOODDetector().fit(references)
    calibration = [
        OODCalibrationEmbedding(
            asset_id=f"cal-{index}",
            embedding=np.asarray([index * 0.04, -index * 0.025, 0.1]),
            partition=DataPartition.CALIBRATION,
            is_in_domain=True,
        )
        for index in range(5)
    ]
    detector.calibrate_threshold(calibration)
    scores = detector.ood_scores(
        np.asarray(
            [
                [0.1, -0.05, 0.1],
                [0.2, -0.1, 0.1],
                [4.0, 4.0, -3.0],
                [-4.0, 3.0, 4.0],
            ]
        )
    )
    report = ood_report(scores, [False, False, True, True])
    assert report["ood_auroc"] == pytest.approx(1.0)
    assert report["ood_average_precision"] == pytest.approx(1.0)


def test_uncertainty_components_remain_separable() -> None:
    components = UncertaintyComponents(0.2, 0.3, 0.4)
    assert components.to_dict()["conservative_union"] == pytest.approx(0.664)
    with pytest.raises(ValueError, match="must be finite"):
        UncertaintyComponents(1.1, 0.2, 0.3).conservative_union()


def _metrics(
    detection: float,
    brier: float,
    ece: float,
    ood: float,
    risk: float,
    *,
    locked_hash: str = "locked-v1",
    count: int = 20,
) -> MetricSnapshot:
    return MetricSnapshot(
        detection_score=detection,
        brier_score=brier,
        expected_calibration_error=ece,
        ood_auroc=ood,
        risk_at_target_coverage=risk,
        locked_external_test_hash=locked_hash,
        locked_external_sample_count=count,
    )


def test_release_requires_accuracy_calibration_ood_and_locked_test() -> None:
    accepted = ModelLineageRecord(
        model_version="student-0.2",
        parent_version="student-0.1",
        training_data_hash="train",
        pseudo_teacher_version="teacher-1",
        human_annotation_hash="gold",
        calibration_set_hash="cal",
        architecture_hash="architecture",
        metrics_before=_metrics(0.70, 0.18, 0.08, 0.82, 0.16),
        metrics_after=_metrics(0.72, 0.16, 0.07, 0.84, 0.14),
    )
    decision = evaluate_release(accepted)
    assert decision.accepted_for_release
    assert len(decision.lineage_hash) == 64

    regressed = ModelLineageRecord(
        **{
            **accepted.__dict__,
            "model_version": "student-bad",
            "metrics_after": _metrics(
                0.73,
                0.20,
                0.09,
                0.86,
                0.13,
                locked_hash="changed-test",
            ),
        }
    )
    rejected = evaluate_release(regressed)
    assert not rejected.accepted_for_release
    assert "locked_test_unchanged" in rejected.reasons
    assert "brier_non_regression" in rejected.reasons
