from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from tfree_detection.diagnostic_v4 import (
    analyze_posthoc_v4,
    compose_posthoc_v4,
)
from tfree_detection.synthetic import make_two_phase_interface


def _crystal_with_border_touching_blur(
    size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    crystal = (
        np.cos(2 * np.pi * xx / 12.0)
        + 0.9 * np.cos(2 * np.pi * yy / 14.0 + 0.4)
        + 0.25 * np.cos(2 * np.pi * (xx / 12.0 + yy / 14.0))
    )
    truth = (xx >= 92) & (xx <= 164) & (yy <= 150)
    blurred = gaussian_filter(crystal, sigma=5.0, mode="reflect")
    image = np.where(truth, blurred, crystal)
    lower, upper = np.percentile(image, [0.5, 99.5])
    image = np.clip((image - lower) / (upper - lower), 0, 1)
    return np.round(image * 65535).astype(np.uint16), truth


def test_border_touching_coherence_band_is_retained_with_lower_edge_reliability() -> None:
    image, truth = _crystal_with_border_touching_blur()

    result = analyze_posthoc_v4(image)

    overlap = result.coherence_loss_mask & truth
    assert overlap.sum() / truth.sum() > 0.45
    assert np.any(result.coherence_loss_mask[0, 92:165])
    edge = float(np.mean(result.reliability_map[0, 92:165]))
    interior = float(np.mean(result.reliability_map[100, 92:165]))
    assert edge < interior
    assert edge == 0.25


def test_clean_crystal_is_not_reported_as_full_frame_coherence_loss() -> None:
    image, _ = _crystal_with_border_touching_blur()
    clean = image.copy()
    clean[:, 92:165] = clean[:, :73]

    result = analyze_posthoc_v4(clean)

    assert result.coherence_loss_mask.mean() < 0.20


def test_two_phase_diagnostic_keeps_region_and_boundary_geometry_separate() -> None:
    image, _ = make_two_phase_interface(size=256, seed=31)

    result = analyze_posthoc_v4(image)

    assert result.secondary_symmetry_region_mask.shape == image.shape
    assert result.structural_boundary_mask.shape == image.shape
    assert np.any(result.secondary_symmetry_region_mask)
    assert np.any(result.structural_boundary_mask)
    assert np.all(
        result.structural_boundary_mask
        <= result.secondary_symmetry_region_mask
    )
    assert (
        result.metadata["scientific_status"]
        == "posthoc_diagnostic_not_locked"
    )
    assert result.metadata["legacy_locked_prediction_replaced"] is False


def test_exploratory_domain_does_not_leak_into_conservative_primary_mask() -> None:
    image, _ = make_two_phase_interface(size=256, seed=31)
    result = analyze_posthoc_v4(image)
    historical = np.zeros(image.shape, dtype=bool)
    historical[12:20, 12:20] = True

    composite = compose_posthoc_v4(result, historical)

    assert np.array_equal(
        composite.secondary_symmetry_region,
        historical,
    )
    assert np.array_equal(
        composite.any_defect_region,
        result.coherence_loss_mask | historical,
    )
    assert np.array_equal(
        composite.augmented_secondary_symmetry_region,
        historical | result.secondary_symmetry_region_mask,
    )
    assert np.any(
        composite.augmented_secondary_symmetry_region
        & ~composite.secondary_symmetry_region
    )
