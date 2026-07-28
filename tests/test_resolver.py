import copy

import numpy as np

from tfree_detection.config import DetectorConfig
from tfree_detection.evidence import (
    DECISION_STATE_CODES,
    DecisionState,
    EvidenceBundle,
    EvidenceDescriptor,
    EvidenceDirection,
    EvidenceField,
    EvidenceRole,
    EvidenceScale,
    HypothesisScore,
    ReliabilityBundle,
    SpatialSupport,
    ViolationType,
    DefectHypothesis,
)
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.models import DetectionRegion, DetectionResult
from tfree_detection.resolver import (
    ResolverPolicy,
    attach_hypothesis_resolutions,
)
from tfree_detection.synthetic import (
    make_crystal_with_amorphous_region,
    make_two_phase_interface,
)


def test_framework_registers_multiple_hypotheses_without_changing_mask() -> None:
    image, _ = make_crystal_with_amorphous_region(size=192, seed=4)
    config = DetectorConfig()
    config.spectral.patch_size = 64
    config.spectral.stride = 12
    config.spectral.max_frequency = 0.25
    config.spectral.autocorrelation_min_lag = 5
    config.spectral.autocorrelation_max_lag = 20
    config.fusion.threshold_quantile = 0.90
    config.fusion.threshold_mad = 2.0
    config.fusion.support_z = 0.9
    config.postprocess.border_margin = 32
    config.postprocess.minimum_region_area = 200

    without_resolver = copy.deepcopy(config)
    without_resolver.resolver.enabled = False
    baseline = TrainingFreeDetectionFramework(without_resolver).detect(image)
    resolved = TrainingFreeDetectionFramework(config).detect(image)

    assert np.array_equal(baseline.mask, resolved.mask)
    assert np.allclose(baseline.score_map, resolved.score_map)
    assert set(resolved.hypothesis_maps) >= {
        "local_amorphous_region",
        "partial_disordering",
        "defocus_or_blur",
    }
    assert resolved.regions
    region = resolved.regions[0]
    assert region.resolved_family == "order_and_coherence_degradation"
    assert region.epistemic_state in {"probable", "unresolved"}
    assert len(region.hypothesis_ranking) >= 2
    assert region.defect_type == "local_amorphous_region"


def test_phase_detector_refuses_to_force_twin_vs_heterophase() -> None:
    image, _ = make_two_phase_interface()
    config = DetectorConfig(task="phase_boundary")
    config.spectral.min_frequency = 0.025
    config.spectral.max_frequency = 0.25
    config.phase_boundary.stride = 10
    config.phase_boundary.minimum_signature_separation = 0.65
    result = TrainingFreeDetectionFramework(config).detect(image)
    assert result.regions
    assert all(
        region.resolved_label == "interface_subtype_unresolved"
        for region in result.regions
    )
    assert all(
        region.decision_state == DecisionState.UNRESOLVED.value
        for region in result.regions
    )
    assert DECISION_STATE_CODES[DecisionState.UNRESOLVED.value] in set(
        np.unique(result.decision_map)
    )


def test_artifact_hypothesis_is_a_first_class_output() -> None:
    shape = (32, 32)
    mask = np.zeros(shape, dtype=bool)
    mask[8:24, 8:24] = True
    reliable = np.ones(shape, dtype=np.float32)
    descriptor = EvidenceDescriptor(
        name="region.sharpness_loss",
        scale=EvidenceScale.REGION,
        violation_type=ViolationType.ACQUISITION_DEGRADATION,
        role=EvidenceRole.ANOMALY,
        direction=EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
    )
    field = EvidenceField(
        descriptor=descriptor,
        values=mask.astype(np.float32) * 6.0,
        reliability=reliable,
        valid_mask=np.ones(shape, dtype=bool),
        source_key="sharpness",
    )
    reliability = ReliabilityBundle(
        reliability_map=reliable,
        valid_mask=np.ones(shape, dtype=bool),
        ood_map=np.zeros(shape, dtype=np.float32),
        state_map=np.zeros(shape, dtype=np.uint8),
        reliable_threshold=0.6,
        invalid_threshold=0.2,
    )
    artifact = DefectHypothesis(
        name="defocus_or_blur",
        violation_types=(ViolationType.ACQUISITION_DEGRADATION,),
        spatial_support=SpatialSupport.REGION,
        evidence_weights={"region.sharpness_loss": 1.0},
        family="acquisition_artifact_and_quality",
        resolution_kind="artifact",
    )
    score = np.zeros(shape, dtype=np.float32)
    score[mask] = 2.0
    bundle = EvidenceBundle(
        task="amorphous_region",
        image_shape=shape,
        fields={"region.sharpness_loss": field},
        reliability=reliability,
        hypotheses={
            artifact.name: HypothesisScore(
                hypothesis=artifact,
                energy_map=score,
                support_count_map=np.ones(shape, dtype=np.uint8),
                available_fields=["region.sharpness_loss"],
            )
        },
    )
    region = DetectionRegion(
        region_id=1,
        bbox_xyxy=(8, 8, 24, 24),
        centroid_xy=(16.0, 16.0),
        area_pixels=256,
        mean_score=2.0,
        max_score=2.0,
        confidence=0.9,
        confidence_level="high",
        evidence_groups=["sharpness"],
        candidate_id="source:amorphous_region:0001",
        decision_state=DecisionState.CONFIRMED_CANDIDATE.value,
    )
    result = DetectionResult(
        task="amorphous_region",
        image_shape=shape,
        normalized_image=np.zeros(shape, dtype=np.float32),
        score_map=score,
        probability_map=mask.astype(np.float32),
        mask=mask,
        regions=[region],
        feature_maps={},
        evidence_maps={},
        evidence_bundle=bundle,
        reliability_map=reliable,
        decision_map=np.where(
            mask,
            DECISION_STATE_CODES[
                DecisionState.CONFIRMED_CANDIDATE.value
            ],
            DECISION_STATE_CODES[DecisionState.BACKGROUND.value],
        ).astype(np.uint8),
    )
    attach_hypothesis_resolutions(
        result,
        ResolverPolicy(minimum_hypothesis_score=0.1),
    )
    assert region.resolved_label == "defocus_or_blur"
    assert region.epistemic_state == "artifact"
    assert region.decision_state == DecisionState.ARTIFACT.value
    assert DECISION_STATE_CODES[DecisionState.ARTIFACT.value] in set(
        np.unique(result.decision_map)
    )
