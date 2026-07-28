"""Deterministic synthetic data for framework smoke tests and benchmarks."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import disk


def make_crystal_with_amorphous_region(
    size: int = 320,
    angle_degrees: float = 0.0,
    noise_standard_deviation: float = 0.10,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a rotated periodic texture with one locally disordered ellipse."""

    if size < 128:
        raise ValueError("Synthetic benchmark size must be at least 128")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    angle = np.deg2rad(angle_degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotated_x = cosine * xx - sine * yy
    rotated_y = sine * xx + cosine * yy

    crystal = (
        1.00 * np.cos(2 * np.pi * (0.074 * rotated_y + 0.018 * rotated_x))
        + 0.85 * np.cos(2 * np.pi * (0.069 * rotated_y - 0.023 * rotated_x) + 0.7)
        + 0.35 * np.cos(2 * np.pi * (0.145 * rotated_y - 0.005 * rotated_x) + 1.1)
    )
    crystal /= np.std(crystal)

    center_x, center_y = 0.47 * size, 0.56 * size
    radius_x, radius_y = 0.16 * size, 0.12 * size
    hard_mask = (
        ((xx - center_x) / radius_x) ** 2
        + ((yy - center_y) / radius_y) ** 2
        <= 1
    )
    blend = gaussian_filter(hard_mask.astype(float), sigma=max(size / 100, 1.5))
    blend = np.clip(blend, 0, 1)

    disorder = gaussian_filter(rng.normal(size=(size, size)), sigma=1.8)
    disorder = (disorder - np.mean(disorder)) / (np.std(disorder) + 1e-12)
    image = (1 - blend) * crystal + blend * disorder
    illumination = 0.25 * (xx / size) - 0.18 * (yy / size)
    image = image + illumination + rng.normal(
        scale=noise_standard_deviation, size=image.shape
    )
    lower, upper = np.percentile(image, [0.5, 99.5])
    image = np.clip((image - lower) / (upper - lower), 0, 1)
    return np.round(image * 65535).astype(np.uint16), hard_mask


def _normalized_uint16(image: np.ndarray) -> np.ndarray:
    lower, upper = np.percentile(image, [0.5, 99.5])
    normalized = np.clip((image - lower) / max(upper - lower, 1e-12), 0, 1)
    return np.round(normalized * 65535).astype(np.uint16)


def make_atomic_lattice_with_point_defects(
    size: int = 256,
    spacing: int = 16,
    noise_standard_deviation: float = 0.025,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Render known vacancy, displacement, substitution, and interstitial sites."""

    if size < 128 or spacing < 8:
        raise ValueError("Point-defect synthetic geometry is too small")
    rng = np.random.default_rng(seed)
    coordinates = [
        [float(x), float(y)]
        for y in range(spacing, size - spacing + 1, spacing)
        for x in range(spacing, size - spacing + 1, spacing)
    ]
    center = np.asarray([size / 2, size / 2])
    ordered = sorted(
        range(len(coordinates)),
        key=lambda index: np.linalg.norm(np.asarray(coordinates[index]) - center),
    )
    vacancy_index = ordered[0]
    displacement_index = ordered[4]
    substitution_index = ordered[8]
    vacancy_xy = np.asarray(coordinates[vacancy_index])
    displacement_expected = np.asarray(coordinates[displacement_index])
    substitution_xy = np.asarray(coordinates[substitution_index])

    impulse = np.zeros((size, size), dtype=float)
    for index, (x, y) in enumerate(coordinates):
        if index == vacancy_index:
            continue
        amplitude = 0.38 if index == substitution_index else 1.0
        if index == displacement_index:
            x += 0.28 * spacing
            y -= 0.20 * spacing
        impulse[int(round(y)), int(round(x))] += amplitude
    interstitial_xy = np.asarray(
        [0.375 * size + spacing / 2, 0.6875 * size + spacing / 2]
    )
    impulse[
        int(round(interstitial_xy[1])),
        int(round(interstitial_xy[0])),
    ] += 0.9
    image = gaussian_filter(impulse, sigma=1.45, mode="reflect")
    image /= np.max(image) + 1e-12
    yy, xx = np.mgrid[:size, :size]
    image += 0.05 * xx / size - 0.04 * yy / size
    image += rng.normal(scale=noise_standard_deviation, size=image.shape)

    truth = np.zeros((size, size), dtype=bool)
    defect_points = [
        vacancy_xy,
        displacement_expected,
        substitution_xy,
        interstitial_xy,
    ]
    for x, y in defect_points:
        rows, columns = disk(
            (y, x), radius=max(3, int(round(0.30 * spacing))), shape=truth.shape
        )
        truth[rows, columns] = True
    metadata = {
        "basis_rows_xy": [[float(spacing), 0.0], [0.0, float(spacing)]],
        "defect_points_xy": [point.tolist() for point in defect_points],
        "defect_types": [
            "vacancy",
            "displaced",
            "intensity_substitution",
            "interstitial",
        ],
    }
    return _normalized_uint16(image), truth, metadata


def make_lattice_with_stacking_fault(
    size: int = 256,
    spacing: float = 15.0,
    angle_degrees: float = 7.0,
    noise_standard_deviation: float = 0.06,
    seed: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a translational phase slip with no change in lattice frequency."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    angle = np.deg2rad(angle_degrees)
    xr = np.cos(angle) * xx - np.sin(angle) * yy
    yr = np.sin(angle) * xx + np.cos(angle) * yy
    fault_y = 0.53 * size + 0.10 * (xx - size / 2)
    shift = 0.34 * spacing * (
        0.5 + 0.5 * np.tanh((yy - fault_y) / 1.6)
    )
    crystal = (
        np.cos(2 * np.pi * (xr - shift) / spacing)
        + 0.90 * np.cos(2 * np.pi * yr / (0.92 * spacing) + 0.4)
        + 0.28
        * np.cos(
            2
            * np.pi
            * ((xr - shift) / spacing + yr / (0.92 * spacing))
        )
    )
    image = crystal + 0.12 * xx / size
    image += rng.normal(scale=noise_standard_deviation, size=image.shape)
    truth = np.abs(yy - fault_y) <= 5
    return _normalized_uint16(image), truth


def make_two_phase_interface(
    size: int = 256,
    noise_standard_deviation: float = 0.07,
    seed: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two crystalline domains with different spacing and orientation."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    left = (
        np.cos(2 * np.pi * xx / 15.0)
        + 0.9 * np.cos(2 * np.pi * yy / 13.5 + 0.3)
        + 0.25 * np.cos(2 * np.pi * (xx / 15.0 + yy / 13.5))
    )
    angle = np.deg2rad(19.0)
    xr = np.cos(angle) * xx - np.sin(angle) * yy
    yr = np.sin(angle) * xx + np.cos(angle) * yy
    right = (
        np.cos(2 * np.pi * xr / 12.0 + 0.5)
        + 0.82 * np.cos(2 * np.pi * yr / 16.5)
        + 0.25 * np.cos(2 * np.pi * (xr / 12.0 - yr / 16.5))
    )
    interface_x = 0.51 * size + 0.08 * (yy - size / 2)
    blend = 0.5 + 0.5 * np.tanh((xx - interface_x) / 2.0)
    image = (1 - blend) * left + blend * right
    image += 0.08 * yy / size
    image += rng.normal(scale=noise_standard_deviation, size=image.shape)
    truth = np.abs(xx - interface_x) <= 7
    return _normalized_uint16(image), truth


def make_phase_shifted_symmetry_domain(
    size: int = 256,
    spacing: float = 15.0,
    phase_shift_fraction: float = 0.35,
    noise_standard_deviation: float = 0.04,
    seed: int = 47,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a same-orientation domain related by a lattice translation."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    boundary_x = 0.56 * size + 0.05 * (yy - size / 2)
    domain = xx >= boundary_x
    shift = phase_shift_fraction * spacing * domain
    xr = xx - shift
    crystal = (
        np.cos(2 * np.pi * xr / spacing)
        + 0.90 * np.cos(2 * np.pi * yy / (0.92 * spacing) + 0.35)
        + 0.35
        * np.cos(
            2 * np.pi * (xr / spacing + yy / (0.92 * spacing))
        )
    )
    image = crystal + 0.08 * yy / size
    image += rng.normal(scale=noise_standard_deviation, size=image.shape)
    interface = np.abs(xx - boundary_x) <= 6
    return _normalized_uint16(image), domain, interface


def make_clean_lattice(
    size: int = 256,
    spacing: float = 15.0,
    noise_standard_deviation: float = 0.04,
    seed: int = 53,
) -> np.ndarray:
    """Create a defect-free lattice for explicit no-candidate tests."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    image = (
        np.cos(2 * np.pi * xx / spacing)
        + 0.90 * np.cos(2 * np.pi * yy / (0.92 * spacing) + 0.35)
        + 0.35
        * np.cos(
            2 * np.pi * (xx / spacing + yy / (0.92 * spacing))
        )
    )
    image += 0.08 * yy / size
    image += rng.normal(scale=noise_standard_deviation, size=image.shape)
    return _normalized_uint16(image)
