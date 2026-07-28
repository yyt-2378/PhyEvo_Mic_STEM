import numpy as np
import pytest

from tfree_detection.active_learning import (
    AcquisitionCandidate,
    select_acquisition_batch,
    teacher_student_disagreement,
)
from tfree_detection.config import DetectorConfig
from tfree_detection.digital_twin import make_counterfactual_twin
from tfree_detection.governance import DataPartition
from tfree_detection.student import (
    DEFAULT_STUDENT_EVIDENCE,
    EvidencePreservingStudent,
    evidence_preserving_loss,
    hard_pseudolabel_baseline_loss,
    stack_student_targets,
    student_targets_from_counterfactual_twin,
    student_target_from_result,
)
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.synthetic import make_crystal_with_amorphous_region


torch = pytest.importorskip("torch")


def _teacher_result():
    image, _ = make_crystal_with_amorphous_region(size=128, seed=8)
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
    return TrainingFreeDetectionFramework(config).detect(image)


def test_student_target_preserves_soft_evidence_and_reliable_normals() -> None:
    target = student_target_from_result(_teacher_result())
    target.validate()
    assert target.evidence_targets.shape[0] == len(DEFAULT_STUDENT_EVIDENCE)
    assert np.any(target.evidence_weights > 0)
    assert np.any(
        (target.geometry_labels == 0) & (target.supervision_weight > 0)
    )
    assert target.provenance["target_kind"] == (
        "evidence_preserving_distillation"
    )
    assert np.all(target.ood_available == 0)


def test_multihead_student_forward_and_loss_are_differentiable() -> None:
    target = student_target_from_result(_teacher_result())
    batch = stack_student_targets([target])
    model = EvidencePreservingStudent(base_channels=4)
    output = model(batch["image"])
    assert output.geometry_logits.shape == (1, 5, 128, 128)
    assert output.violation_logits.shape[1] == 10
    assert output.evidence_prediction.shape[1] == len(
        DEFAULT_STUDENT_EVIDENCE
    )
    assert output.embedding.ndim == 2

    total, components = evidence_preserving_loss(output, batch)
    hard = hard_pseudolabel_baseline_loss(
        output, batch["teacher_soft_mask"] >= 0.5
    )
    assert torch.isfinite(total)
    assert torch.isfinite(hard)
    assert {
        "geometry",
        "soft_pseudo_mask",
        "violation",
        "evidence",
        "reliability",
        "ood",
    } == set(components)
    total.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_counterfactual_pair_creates_normal_and_defect_gold_targets() -> None:
    pair = make_counterfactual_twin("amorphous_region", size=128, seed=12)
    normal, defect = student_targets_from_counterfactual_twin(pair)
    normal.validate()
    defect.validate()
    assert np.all(normal.geometry_labels == 0)
    assert np.any(defect.geometry_labels != 0)
    assert np.any(defect.evidence_weights > 0)
    assert normal.provenance["counterfactual_member"] == "normal"
    assert defect.provenance["counterfactual_member"] == "defect"
    assert (
        defect.provenance["experimental_accuracy_claim_allowed"] is False
    )


def _candidate(
    name: str,
    partition: DataPartition,
    disagreement: float,
    cost: float = 30,
) -> AcquisitionCandidate:
    return AcquisitionCandidate(
        candidate_id=name,
        partition=partition,
        student_uncertainty=disagreement * 0.8,
        teacher_student_disagreement=disagreement,
        ood_score=disagreement * 0.6,
        diversity_score=disagreement * 0.5,
        physics_violation=disagreement * 0.9,
        annotation_cost_seconds=cost,
    )


def test_active_selection_excludes_all_non_train_partitions() -> None:
    plan = select_acquisition_batch(
        [
            _candidate("train-high", DataPartition.TRAIN, 0.95),
            _candidate("train-low", DataPartition.TRAIN, 0.1),
            _candidate("calibration", DataPartition.CALIBRATION, 1.0),
            _candidate(
                "development", DataPartition.DEVELOPMENT_TEST, 1.0
            ),
            _candidate(
                "locked", DataPartition.LOCKED_EXTERNAL_TEST, 1.0
            ),
        ],
        annotation_budget_seconds=45,
        maximum_candidates=1,
    )
    assert [item.candidate_id for item in plan.selected] == ["train-high"]
    assert plan.excluded_by_partition == {
        "calibration": 1,
        "development_test": 1,
        "locked_external_test": 1,
    }


def test_disagreement_combines_masks_and_reliable_evidence() -> None:
    teacher_mask = np.asarray([[0.0, 1.0], [0.3, 0.8]])
    student_mask = np.asarray([[0.1, 0.7], [0.5, 0.2]])
    teacher_evidence = np.stack([teacher_mask, teacher_mask])
    student_evidence = np.stack([student_mask, student_mask])
    reliability = np.ones_like(teacher_evidence)
    combined = teacher_student_disagreement(
        teacher_mask,
        student_mask,
        teacher_evidence=teacher_evidence,
        student_evidence=student_evidence,
        evidence_reliability=reliability,
    )
    assert combined > 0
    assert combined <= 1
