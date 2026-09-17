"""Nested profile models used to test an optional projected site."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class ProfileModelComparison:
    delta_bic: float
    center_amplitude: float
    valid: bool


def _robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, float(np.std(values)), 1e-6)
    return (values - median) / scale, median, scale


def compare_optional_center_profile(
    image: np.ndarray,
    center_xy: tuple[float, float],
    endpoint_xy: tuple[tuple[float, float], tuple[float, float]],
    *,
    sigma_pixels: float,
    minimum_endpoint_separation_pixels: float,
) -> ProfileModelComparison:
    """Compare endpoint-only and endpoint-plus-center 1-D intensity models."""

    frame = np.asarray(image, dtype=float)
    if frame.ndim != 2 or not np.all(np.isfinite(frame)):
        return ProfileModelComparison(float("-inf"), 0.0, False)
    endpoints = np.asarray(endpoint_xy, dtype=float)
    center = np.asarray(center_xy, dtype=float)
    direction = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(direction))
    if length < minimum_endpoint_separation_pixels:
        return ProfileModelComparison(float("-inf"), 0.0, False)
    direction /= length
    endpoint_positions = (endpoints - center) @ direction
    lower = float(np.min(endpoint_positions) - 2.5 * sigma_pixels)
    upper = float(np.max(endpoint_positions) + 2.5 * sigma_pixels)
    sample_count = int(np.clip(np.ceil((upper - lower) * 4.0), 31, 81))
    coordinate = np.linspace(lower, upper, sample_count)
    xy = center[None, :] + coordinate[:, None] * direction[None, :]
    profile = map_coordinates(
        frame,
        [xy[:, 1], xy[:, 0]],
        order=1,
        mode="nearest",
    )
    standardized, _, _ = _robust_center_scale(np.asarray(profile, dtype=float))

    half_length = max(0.5 * length, 1e-8)
    normalized_coordinate = coordinate / half_length
    gaussian_sigma = float(sigma_pixels) / half_length

    def gaussian(position: float) -> np.ndarray:
        return np.exp(
            -0.5 * ((normalized_coordinate - position) / gaussian_sigma) ** 2
        )

    endpoint_normalized = endpoint_positions / half_length
    baseline = np.column_stack(
        (
            np.ones(sample_count),
            normalized_coordinate,
            gaussian(float(endpoint_normalized[0])),
            gaussian(float(endpoint_normalized[1])),
        )
    )
    center_peak = gaussian(0.0)
    expanded = np.column_stack((baseline, center_peak))
    try:
        base_coefficients, *_ = np.linalg.lstsq(
            baseline, standardized, rcond=None
        )
        full_coefficients, *_ = np.linalg.lstsq(
            expanded, standardized, rcond=None
        )
    except np.linalg.LinAlgError:
        return ProfileModelComparison(float("-inf"), 0.0, False)

    base_residual = standardized - baseline @ base_coefficients
    full_residual = standardized - expanded @ full_coefficients
    n = float(sample_count)
    base_rss = max(float(np.dot(base_residual, base_residual)), 1e-12)
    full_rss = max(float(np.dot(full_residual, full_residual)), 1e-12)
    base_bic = n * np.log(base_rss / n) + baseline.shape[1] * np.log(n)
    full_bic = n * np.log(full_rss / n) + expanded.shape[1] * np.log(n)
    center_amplitude = float(full_coefficients[-1])
    delta_bic = float(base_bic - full_bic)
    valid = bool(
        np.isfinite(delta_bic)
        and np.isfinite(center_amplitude)
        and center_amplitude > 0
        and np.all(full_coefficients[2:4] > 0)
    )
    if not valid:
        delta_bic = float("-inf")
    return ProfileModelComparison(delta_bic, center_amplitude, valid)
