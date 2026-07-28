import json

import numpy as np
import pytest

from tfree_detection.config import DetectorConfig
from tfree_detection.context import IdealSkeleton, SkeletonNode
from tfree_detection.evidence import (
    DECISION_STATE_CODES,
    DecisionState,
    EvidenceDescriptor,
    EvidenceDirection,
    EvidenceField,
    EvidenceRole,
    EvidenceScale,
    LabelTier,
    ViolationType,
)
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.synthetic import (
    make_atomic_lattice_with_point_defects,
    make_crystal_with_amorphous_region,
    make_lattice_with_stacking_fault,
    make_two_phase_interface,
)


def _four_task_results():
    amorphous_image, _ = make_crystal_with_amorphous_region(
        size=192, seed=4
    )
    amorphous = DetectorConfig()
    amorphous.spectral.patch_size = 64
    amorphous.spectral.stride = 12
    amorphous.spectral.max_frequency = 0.25
    amorphous.spectral.autocorrelation_min_lag = 5
    amorphous.spectral.autocorrelation_max_lag = 20
    amorphous.fusion.threshold_quantile = 0.90
    amorphous.fusion.threshold_mad = 2.0
    amorphous.fusion.support_z = 0.9
    amorphous.postprocess.border_margin = 32
    amorphous.postprocess.minimum_region_area = 200

    atomic_image, _, atomic_metadata = make_atomic_lattice_with_point_defects()
    atomic = DetectorConfig(task="atomic_point_defect")
    atomic.atomic.peak_min_distance = 7
    atomic.atomic.peak_threshold_quantile = 0.70
    atomic.atomic.site_anomaly_z = 3.0
    atomic.atomic.displacement_z = 2.5
    skeleton = IdealSkeleton(
        basis=np.asarray(atomic_metadata["basis_rows_xy"]),
        nodes=[
            SkeletonNode(
                site_id="A",
                fractional_coordinate=(0.0, 0.0),
                sublattice="A",
                expected_relative_intensity=1.0,
            )
        ],
    )

    stacking_image, _ = make_lattice_with_stacking_fault()
    stacking = DetectorConfig(task="stacking_fault")
    stacking.spectral.min_frequency = 0.02
    stacking.spectral.max_frequency = 0.30
    stacking.spectral.global_peak_count = 16
    stacking.line_defect.threshold_quantile = 0.94
    stacking.line_defect.threshold_mad = 2.4
    stacking.line_defect.minimum_region_area = 30

    interface_image, _ = make_two_phase_interface()
    interface = DetectorConfig(task="phase_boundary")
    interface.spectral.min_frequency = 0.025
    interface.spectral.max_frequency = 0.25
    interface.phase_boundary.stride = 10
    interface.phase_boundary.minimum_signature_separation = 0.65

    return [
        TrainingFreeDetectionFramework(amorphous).detect(amorphous_image),
        TrainingFreeDetectionFramework(atomic).detect(
            atomic_image, skeleton=skeleton
        ),
        TrainingFreeDetectionFramework(stacking).detect(stacking_image),
        TrainingFreeDetectionFramework(interface).detect(interface_image),
    ]


def test_all_tasks_emit_valid_unified_evidence_and_abstention_contract() -> None:
    expected_scale = {
        "amorphous_region": EvidenceScale.REGION,
        "atomic_point_defect": EvidenceScale.NODE,
        "stacking_fault": EvidenceScale.EDGE,
        "phase_boundary": EvidenceScale.REGION,
    }
    for result in _four_task_results():
        bundle = result.evidence_bundle
        assert bundle is not None
        bundle.validate()
        assert any(
            field.descriptor.scale == expected_scale[result.task]
            for field in bundle.fields.values()
        )
        assert result.reliability_map is not None
        assert result.decision_map is not None
        assert result.reliability_map.shape == result.image_shape
        assert result.decision_map.shape == result.image_shape
        assert np.min(result.reliability_map) >= 0
        assert np.max(result.reliability_map) <= 1
        assert set(np.unique(result.decision_map)).issubset(
            set(DECISION_STATE_CODES.values())
        )
        assert result.hypothesis_maps
        json.dumps(result.summary())
        for region in result.regions:
            assert f":{result.task}:" in region.candidate_id
            assert region.spatial_support
            assert region.violation_types
            assert region.label_tier in {
                LabelTier.SILVER.value,
                LabelTier.UNRESOLVED.value,
            }
            assert region.label_tier != LabelTier.GOLD.value
            assert "uncalibrated_confidence" in region.quality_flags
            assert region.provenance["source_kind"] == "physics_pseudo"


def test_explicit_invalid_border_is_not_reported_as_background() -> None:
    image, _ = make_lattice_with_stacking_fault()
    config = DetectorConfig(task="stacking_fault")
    config.spectral.min_frequency = 0.02
    config.spectral.max_frequency = 0.30
    config.spectral.global_peak_count = 16
    result = TrainingFreeDetectionFramework(config).detect(image)
    assert result.decision_map is not None
    invalid_code = DECISION_STATE_CODES[
        DecisionState.INVALID_MEASUREMENT.value
    ]
    assert result.decision_map[0, 0] == invalid_code
    assert result.decision_map[image.shape[0] // 2, image.shape[1] // 2] != (
        invalid_code
    )


def test_amorphous_reliability_is_window_validity_not_anomaly_vote() -> None:
    result = _four_task_results()[0]
    assert result.reliability_map is not None
    assert result.decision_map is not None
    invalid_code = DECISION_STATE_CODES[
        DecisionState.INVALID_MEASUREMENT.value
    ]
    center = (
        result.image_shape[0] // 2,
        result.image_shape[1] // 2,
    )
    assert result.reliability_map[center] > 0.9
    assert result.decision_map[center] != invalid_code
    assert result.reliability_map[0, 0] == 0
    assert result.decision_map[0, 0] == invalid_code


def test_evidence_field_rejects_shape_and_reliability_errors() -> None:
    descriptor = EvidenceDescriptor(
        name="node.test",
        scale=EvidenceScale.NODE,
        violation_type=ViolationType.OCCUPANCY,
        role=EvidenceRole.ANOMALY,
        direction=EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
    )
    field = EvidenceField(
        descriptor=descriptor,
        values=np.zeros((8, 8), dtype=np.float32),
        reliability=np.full((8, 8), 1.2, dtype=np.float32),
        valid_mask=np.ones((8, 8), dtype=bool),
        source_key="test",
    )
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        field.validate((8, 8))
    field.reliability.fill(1.0)
    with pytest.raises(ValueError, match="expected"):
        field.validate((9, 8))
