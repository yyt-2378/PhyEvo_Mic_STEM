import numpy as np
import pytest

from tfree_detection.atomic_site_inventory import (
    AtomicSiteInventoryConfig,
    centered_similarity_transform,
    detect_species_sites,
    fit_ordered_intensity_mixture,
)


def test_ordered_mixture_recovers_imbalanced_species_brightness() -> None:
    rng = np.random.default_rng(7)
    values = np.concatenate(
        [
            rng.normal(0.18, 0.025, 600),
            rng.normal(0.48, 0.025, 300),
            rng.normal(0.86, 0.025, 80),
        ]
    )

    mixture = fit_ordered_intensity_mixture(values)

    assert np.allclose(mixture.means, [0.18, 0.48, 0.86], atol=0.02)
    assert np.all(np.diff(mixture.means) > 0)
    assert np.all(mixture.separation_sigma > 8.0)


def test_ordered_mixture_rejects_too_few_sites() -> None:
    with pytest.raises(ValueError, match="Too few detected sites"):
        fit_ordered_intensity_mixture(np.linspace(0.1, 0.9, 11))


def test_centered_similarity_aligns_centers_and_vectors() -> None:
    angle = np.deg2rad(17.0)
    linear = 1.25 * np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    transform = centered_similarity_transform(
        linear, (80, 120), (180, 220)
    )
    native_center = np.asarray([59.5, 39.5])
    canvas_center = np.asarray([109.5, 89.5])

    assert np.allclose(
        transform(native_center[None, :])[0], canvas_center
    )
    observed = (
        transform((native_center + [8.0, 0.0])[None, :])[0]
        - canvas_center
    )
    assert np.allclose(
        observed, linear @ np.asarray([8.0, 0.0])
    )


def test_species_detector_preserves_brightness_order() -> None:
    shape = (128, 128)
    rows, columns = np.indices(shape, dtype=float)
    image = np.zeros(shape, dtype=float)
    amplitudes = (0.20, 0.50, 0.90)
    positions = (
        [
            (20 + 16 * i, 20 + 16 * j)
            for i in range(5)
            for j in range(3)
        ],
        [
            (24 + 16 * i, 72 + 16 * j)
            for i in range(5)
            for j in range(2)
        ],
        [(32 + 24 * i, 108) for i in range(3)],
    )
    for amplitude, group in zip(amplitudes, positions):
        for x, y in group:
            image += amplitude * np.exp(
                -(
                    (columns - x) ** 2
                    + (rows - y) ** 2
                )
                / (2 * 1.0**2)
            )
    coverage = np.ones(shape, dtype=bool)
    config = AtomicSiteInventoryConfig(
        detection_noise_z=0.1,
        minimum_sites_per_image=12,
    )

    detected = detect_species_sites(image, coverage, config)

    assert len(detected["points_xy"]) >= 28
    assert np.all(np.diff(detected["mixture"].means) > 0)
    assert (
        detected["mixture"].means[2]
        > 1.5 * detected["mixture"].means[1]
    )
