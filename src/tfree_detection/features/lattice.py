"""Reusable reciprocal-lattice and complex-demodulation utilities."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


Peak = Tuple[float, float]


def select_reciprocal_basis(
    peaks_fy_fx: Sequence[Peak],
    minimum_angle_degrees: float = 18.0,
) -> tuple[Peak, Peak]:
    """Select a low-order, independent Friedel-half-plane peak pair."""

    candidates: list[tuple[int, Peak]] = []
    for rank, (fy, fx) in enumerate(peaks_fy_fx):
        fy, fx = float(fy), float(fx)
        if fy > 0 or (abs(fy) < 1e-12 and fx > 0):
            candidates.append((rank, (fy, fx)))
    if len(candidates) < 2:
        raise ValueError("At least two reciprocal-lattice directions are required")

    minimum_sine = float(np.sin(np.deg2rad(minimum_angle_degrees)))
    radii = np.asarray(
        [np.hypot(peak[0], peak[1]) for _, peak in candidates],
        dtype=float,
    )
    radius_scale = float(np.median(radii)) + 1e-12
    best: tuple[float, Peak, Peak] | None = None
    for index, (rank_1, first) in enumerate(candidates):
        first_xy = np.asarray([first[1], first[0]], dtype=float)
        for rank_2, second in candidates[index + 1 :]:
            second_xy = np.asarray([second[1], second[0]], dtype=float)
            sine = abs(float(np.linalg.det(np.stack([first_xy, second_xy]))))
            sine /= (
                np.linalg.norm(first_xy) * np.linalg.norm(second_xy) + 1e-12
            )
            if sine < minimum_sine:
                continue
            radius_cost = (
                np.linalg.norm(first_xy) + np.linalg.norm(second_xy)
            ) / radius_scale
            rank_cost = 0.08 * (rank_1 + rank_2)
            conditioning_cost = 0.4 / max(sine, 1e-6)
            cost = float(radius_cost + rank_cost + conditioning_cost)
            if best is None or cost < best[0]:
                best = (cost, first, second)
    if best is None:
        raise ValueError("No sufficiently independent reciprocal peak pair found")
    return best[1], best[2]


def direct_basis_from_peaks(peaks: Sequence[Peak]) -> np.ndarray:
    """Return direct basis vectors as columns of a 2x2 xy matrix."""

    if len(peaks) != 2:
        raise ValueError("Exactly two reciprocal peaks are required")
    reciprocal = np.asarray(
        [[peaks[0][1], peaks[0][0]], [peaks[1][1], peaks[1][0]]],
        dtype=float,
    )
    if abs(float(np.linalg.det(reciprocal))) < 1e-8:
        raise ValueError("Reciprocal basis is singular")
    return np.linalg.inv(reciprocal)


def complex_demodulation(
    image: np.ndarray,
    peaks: Sequence[Peak],
    frequency_sigma: float,
    derivative_sigma: float,
    padding: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract stable phase gradients and normalized carrier amplitudes."""

    padding = min(max(int(padding), 0), max(min(image.shape) // 3, 0))
    padded = np.pad(
        image, ((padding, padding), (padding, padding)), mode="reflect"
    )
    height, width = padded.shape
    frequency_y = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    frequency_x = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2(padded - np.mean(padded)))
    yy = np.arange(height)[:, None]
    xx = np.arange(width)[None, :]

    derivatives_x = []
    derivatives_y = []
    amplitudes = []
    for peak_y, peak_x in peaks:
        frequency_mask = np.exp(
            -0.5
            * (
                (frequency_x - peak_x) ** 2
                + (frequency_y - peak_y) ** 2
            )
            / frequency_sigma**2
        )
        wave = np.fft.ifft2(np.fft.ifftshift(spectrum * frequency_mask))
        wave *= np.exp(-2j * np.pi * (peak_x * xx + peak_y * yy))
        amplitude_squared = np.abs(wave) ** 2
        denominator = (
            gaussian_filter(
                amplitude_squared, derivative_sigma, mode="nearest"
            )
            + 1e-12
        )
        derivatives_x.append(
            gaussian_filter(
                np.imag(np.conj(wave) * np.gradient(wave, axis=1)),
                derivative_sigma,
                mode="nearest",
            )
            / denominator
        )
        derivatives_y.append(
            gaussian_filter(
                np.imag(np.conj(wave) * np.gradient(wave, axis=0)),
                derivative_sigma,
                mode="nearest",
            )
            / denominator
        )
        amplitudes.append(np.abs(wave))

    crop = (
        (slice(padding, -padding), slice(padding, -padding))
        if padding
        else (slice(None), slice(None))
    )
    qx = np.stack([values[crop] for values in derivatives_x])
    qy = np.stack([values[crop] for values in derivatives_y])
    cropped_amplitudes = np.stack([values[crop] for values in amplitudes])
    background_sigma = max(
        4.0, 1.0 / max(2 * np.pi * frequency_sigma, 1e-6)
    )
    relative_amplitudes = []
    for amplitude in cropped_amplitudes:
        background = gaussian_filter(
            amplitude, background_sigma, mode="reflect"
        )
        relative = amplitude / (background + 1e-12)
        relative /= np.median(relative) + 1e-12
        relative_amplitudes.append(relative)
    coherence = np.sqrt(
        np.maximum(relative_amplitudes[0], 0)
        * np.maximum(relative_amplitudes[1], 0)
    )
    return (
        qx,
        qy,
        np.asarray(cropped_amplitudes),
        np.clip(coherence, 0, 2),
    )

