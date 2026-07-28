import numpy as np

from tfree_detection.config import DetectorConfig
from tfree_detection.periodic_counterfactual import (
    PeriodicCounterfactualConfig,
    detect_periodic_counterfactual,
)


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


def test_periodic_counterfactual_localizes_broken_motif() -> None:
    image = _periodic_image()
    image[76:85, 76:85] += 0.9
    config = DetectorConfig(task="atomic_point_defect")
    config.spectral.min_frequency = 0.03
    config.spectral.max_frequency = 0.20
    result = detect_periodic_counterfactual(
        image,
        config,
        PeriodicCounterfactualConfig(candidate_quantile=0.99),
    )
    assert result.task == "atomic_point_defect"
    assert result.metadata["annotations_read"] is False
    assert len(result.metadata["translation_vectors_yx"]) >= 2
    assert result.evidence_bundle is not None
    distances = [
        np.linalg.norm(np.asarray(region.centroid_xy) - [80, 80])
        for region in result.regions
    ]
    assert distances and min(distances) <= 8


def test_counterfactual_config_rejects_invalid_policy() -> None:
    policy = PeriodicCounterfactualConfig(aggregation="unknown")
    try:
        policy.validate()
    except ValueError as error:
        assert "aggregation" in str(error)
    else:
        raise AssertionError("Invalid aggregation was accepted")
