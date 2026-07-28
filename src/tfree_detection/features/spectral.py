"""Local Fourier and Patterson descriptors for crystalline order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.ndimage import maximum_filter

from ..config import SpectralConfig
from ..models import FeatureGrid


Peak = Tuple[float, float]


def estimate_global_reciprocal_peaks(
    image: np.ndarray, config: SpectralConfig
) -> List[Peak]:
    """Estimate common reciprocal-lattice peaks in cycles/pixel.

    Peaks are deliberately used only as one evidence branch. A local rotation,
    strain, or phase boundary therefore cannot by itself force an amorphous
    prediction.
    """

    height, width = image.shape
    standardized = (image - np.mean(image)) / (np.std(image) + 1e-12)
    window = np.outer(np.hanning(height), np.hanning(width))
    power = np.abs(np.fft.fftshift(np.fft.fft2(standardized * window))) ** 2
    frequencies_y = np.fft.fftshift(np.fft.fftfreq(height))
    frequencies_x = np.fft.fftshift(np.fft.fftfreq(width))
    fy, fx = np.meshgrid(frequencies_y, frequencies_x, indexing="ij")
    radius = np.hypot(fx, fy)
    valid = (radius >= config.min_frequency) & (radius <= config.max_frequency)

    filter_size = max(
        3,
        int(round(config.global_peak_min_separation * min(height, width))),
    )
    if filter_size % 2 == 0:
        filter_size += 1
    local_max = maximum_filter(power, size=filter_size, mode="nearest")
    candidates = np.argwhere(valid & (power == local_max))
    if candidates.size == 0:
        return []
    strengths = power[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(strengths)[::-1]

    peaks: List[Peak] = []
    for candidate_index in order:
        row, column = candidates[candidate_index]
        peak = (float(frequencies_y[row]), float(frequencies_x[column]))
        if any(
            np.hypot(peak[0] - old[0], peak[1] - old[1])
            < config.global_peak_min_separation
            for old in peaks
        ):
            continue
        peaks.append(peak)
        if len(peaks) >= config.global_peak_count:
            break
    return peaks


@dataclass
class _PatchGeometry:
    window: np.ndarray
    band_mask: np.ndarray
    lattice_mask: np.ndarray
    lag_mask: np.ndarray
    radial_masks: Sequence[np.ndarray]


class SpectralFeatureExtractor:
    """Extract interpretable local order descriptors without fitted weights."""

    feature_names = (
        "spectral_entropy",
        "spectral_flatness",
        "spectral_peak_concentration",
        "radial_peak_concentration",
        "global_lattice_power",
        "patterson_order",
        "gradient_energy",
    )

    def __init__(self, config: SpectralConfig):
        self.config = config

    def extract(self, image: np.ndarray) -> FeatureGrid:
        peaks = estimate_global_reciprocal_peaks(image, self.config)
        geometry = self._build_geometry(peaks)
        centers_y = self._centers(image.shape[0])
        centers_x = self._centers(image.shape[1])
        feature_arrays = {
            name: np.empty((len(centers_y), len(centers_x)), dtype=np.float32)
            for name in self.feature_names
        }

        half = self.config.patch_size // 2
        padded = np.pad(image, ((half, half), (half, half)), mode="reflect")
        for row, center_y in enumerate(centers_y):
            for column, center_x in enumerate(centers_x):
                patch = padded[
                    center_y : center_y + self.config.patch_size,
                    center_x : center_x + self.config.patch_size,
                ]
                values = self._describe_patch(patch, geometry)
                for name, value in zip(self.feature_names, values):
                    feature_arrays[name][row, column] = value

        grid = FeatureGrid(
            values=feature_arrays,
            centers_y=centers_y,
            centers_x=centers_x,
            metadata={
                "global_reciprocal_peaks_fy_fx": [
                    [float(fy), float(fx)] for fy, fx in peaks
                ],
                "patch_size": self.config.patch_size,
                "stride": self.config.stride,
            },
        )
        grid.validate()
        return grid

    def _centers(self, length: int) -> np.ndarray:
        start = min(self.config.stride // 2, max(length - 1, 0))
        centers = np.arange(start, length, self.config.stride, dtype=np.int32)
        if centers.size == 0:
            centers = np.asarray([length // 2], dtype=np.int32)
        return centers

    def _build_geometry(self, peaks: Sequence[Peak]) -> _PatchGeometry:
        size = self.config.patch_size
        frequencies = np.fft.fftshift(np.fft.fftfreq(size))
        fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
        radius = np.hypot(fx, fy)
        band_mask = (radius >= self.config.min_frequency) & (
            radius <= self.config.max_frequency
        )
        lattice_mask = np.zeros((size, size), dtype=bool)
        for peak_y, peak_x in peaks:
            lattice_mask |= (
                np.hypot(fy - peak_y, fx - peak_x)
                <= self.config.global_peak_radius
            )
        lattice_mask &= band_mask
        if not np.any(lattice_mask):
            lattice_mask = band_mask.copy()

        yy, xx = np.ogrid[:size, :size]
        pixel_radius = np.hypot(yy - size // 2, xx - size // 2)
        lag_mask = (
            (pixel_radius >= self.config.autocorrelation_min_lag)
            & (pixel_radius <= self.config.autocorrelation_max_lag)
        )

        edges = np.arange(
            self.config.min_frequency,
            self.config.max_frequency + self.config.radial_bin_width,
            self.config.radial_bin_width,
        )
        if edges[-1] < self.config.max_frequency:
            edges = np.append(edges, self.config.max_frequency)
        radial_masks = [
            (radius >= lower) & (radius < upper)
            for lower, upper in zip(edges[:-1], edges[1:])
            if np.any((radius >= lower) & (radius < upper))
        ]
        return _PatchGeometry(
            window=np.outer(np.hanning(size), np.hanning(size)),
            band_mask=band_mask,
            lattice_mask=lattice_mask,
            lag_mask=lag_mask,
            radial_masks=radial_masks,
        )

    def _describe_patch(
        self, patch: np.ndarray, geometry: _PatchGeometry
    ) -> Tuple[float, ...]:
        standardized = (patch - np.mean(patch)) / (np.std(patch) + 1e-12)
        tapered = standardized * geometry.window
        fft = np.fft.fftshift(np.fft.fft2(tapered))
        power = np.abs(fft) ** 2
        band_values = power[geometry.band_mask]
        band_total = float(np.sum(band_values)) + 1e-12
        probabilities = band_values / band_total

        entropy = -float(
            np.sum(probabilities * np.log(probabilities + 1e-15))
        ) / np.log(max(len(probabilities), 2))
        flatness = float(
            np.exp(np.mean(np.log(band_values + 1e-12)))
            / (np.mean(band_values) + 1e-12)
        )
        top_count = max(
            4, int(round(self.config.peak_fraction * len(band_values)))
        )
        top_count = min(top_count, len(band_values))
        peak_concentration = float(
            np.partition(band_values, -top_count)[-top_count:].sum() / band_total
        )
        radial_power = np.asarray(
            [np.sum(power[mask]) for mask in geometry.radial_masks],
            dtype=np.float64,
        )
        radial_concentration = float(
            np.max(radial_power) / (np.sum(radial_power) + 1e-12)
        )
        lattice_power = float(np.sum(power[geometry.lattice_mask]) / band_total)

        unshifted_power = np.abs(np.fft.fft2(tapered)) ** 2
        autocorrelation = np.fft.fftshift(
            np.fft.ifft2(unshifted_power).real
        )
        center_value = float(
            autocorrelation[
                self.config.patch_size // 2, self.config.patch_size // 2
            ]
        )
        patterson_order = float(
            np.percentile(autocorrelation[geometry.lag_mask], 99.0)
            / (center_value + 1e-12)
        )

        gradient_energy = float(
            np.var(np.diff(standardized, axis=0))
            + np.var(np.diff(standardized, axis=1))
        )
        return (
            entropy,
            flatness,
            peak_concentration,
            radial_concentration,
            lattice_power,
            patterson_order,
            gradient_energy,
        )
