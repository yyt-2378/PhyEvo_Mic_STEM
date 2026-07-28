import numpy as np

from tfree_detection.atomic_consensus import (
    AtomicConsensusConfig,
    detect_atomic_consensus,
)
from tfree_detection.config import DetectorConfig
from tfree_detection.evidence import DecisionState, LabelTier


def _periodic_image(size: int = 160, spacing: int = 16) -> np.ndarray:
    y, x = np.indices((size, size), dtype=float)
    image = np.zeros((size, size), dtype=float)
    for row in range(8, size - 8, spacing):
        for column in range(8, size - 8, spacing):
            image += np.exp(
                -(
                    (x - column) ** 2 + (y - row) ** 2
                )
                / (2 * 1.6**2)
            )
    image /= np.max(image)
    return image


def test_atomic_consensus_preserves_agreement_and_disagreement_states() -> None:
    image = _periodic_image()
    image[76:85, 76:85] += 0.9
    config = DetectorConfig(task="atomic_point_defect")
    config.spectral.min_frequency = 0.03
    config.spectral.max_frequency = 0.20
    result = detect_atomic_consensus(
        image,
        config,
        consensus_config=AtomicConsensusConfig(
            matching_tolerance_fraction=0.5
        ),
    )
    summary = result.metadata["consensus_summary"]
    assert result.evidence_bundle is not None
    assert summary["consensus_count"] >= 1
    assert result.metadata["annotations_read"] is False
    assert "teacher_agreement" in result.evidence_maps
    consensus = [
        region
        for region in result.regions
        if region.provenance["teacher_consensus_state"] == "consensus"
    ]
    assert consensus
    assert all(
        region.decision_state
        == DecisionState.CONFIRMED_CANDIDATE.value
        for region in consensus
    )
    assert all(region.label_tier == LabelTier.SILVER.value for region in consensus)
    singles = [
        region
        for region in result.regions
        if region.provenance["teacher_consensus_state"] != "consensus"
    ]
    assert all(region.epistemic_state == "unresolved" for region in singles)


def test_atomic_consensus_rejects_non_atomic_task() -> None:
    image = _periodic_image()
    config = DetectorConfig(task="amorphous_region")
    try:
        detect_atomic_consensus(image, config)
    except ValueError as error:
        assert "atomic_point_defect" in str(error)
    else:
        raise AssertionError("Non-atomic task was accepted")
