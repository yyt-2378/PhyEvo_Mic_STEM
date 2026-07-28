"""Robust, weight-free evidence normalization and rule-based fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import FusionConfig
from .models import FeatureGrid


@dataclass(frozen=True)
class EvidenceSpec:
    name: str
    direction: int
    weight: float
    group: str

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError("Evidence direction must be -1 or +1")
        if self.weight <= 0:
            raise ValueError("Evidence weight must be positive")


@dataclass
class FusionResult:
    score_grid: np.ndarray
    z_maps: Dict[str, np.ndarray]
    group_maps: Dict[str, np.ndarray]
    support_count_grid: np.ndarray
    threshold: float
    score_scale: float
    statistics: Dict[str, dict]


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    location = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - location)))
    if scale < 1e-8:
        q25, q75 = np.percentile(values, [25, 75])
        scale = max(float((q75 - q25) / 1.349), 1e-8)
    return location, scale


def fuse_evidence(
    grid: FeatureGrid,
    specs: Iterable[EvidenceSpec],
    config: FusionConfig,
) -> FusionResult:
    """Fuse one-sided robust z-scores using top-k reliable evidence."""

    specs = list(specs)
    if not specs:
        raise ValueError("At least one evidence specification is required")
    missing = [spec.name for spec in specs if spec.name not in grid.values]
    if missing:
        raise KeyError("Missing feature(s): " + ", ".join(missing))

    z_maps: Dict[str, np.ndarray] = {}
    weighted_maps: List[np.ndarray] = []
    statistics: Dict[str, dict] = {}
    for spec in specs:
        values = grid.values[spec.name].astype(np.float64)
        location, scale = _robust_location_scale(values)
        signed = spec.direction * (values - location) / scale
        z_score = np.clip(signed, 0.0, 12.0)
        if config.smooth_sigma_grid > 0:
            z_score = gaussian_filter(
                z_score, config.smooth_sigma_grid, mode="nearest"
            )
        z_maps[spec.name] = z_score.astype(np.float32)
        weighted_maps.append(z_score * spec.weight)
        statistics[spec.name] = {
            "median": location,
            "robust_scale": scale,
            "direction": spec.direction,
            "weight": spec.weight,
            "group": spec.group,
        }

    stacked = np.stack(weighted_maps, axis=-1)
    top_k = min(max(config.top_k, 1), stacked.shape[-1])
    top_values = np.partition(stacked, -top_k, axis=-1)[..., -top_k:]
    score = np.mean(top_values, axis=-1)

    groups = sorted({spec.group for spec in specs})
    group_maps: Dict[str, np.ndarray] = {}
    for group in groups:
        member_maps = [
            z_maps[spec.name] for spec in specs if spec.group == group
        ]
        group_maps[group] = np.max(np.stack(member_maps, axis=-1), axis=-1)
    support_count = np.sum(
        np.stack(
            [
                values >= config.support_z
                for values in group_maps.values()
            ],
            axis=-1,
        ),
        axis=-1,
    ).astype(np.int16)

    score_location, score_scale = _robust_location_scale(score)
    robust_threshold = score_location + config.threshold_mad * score_scale
    quantile_threshold = float(
        np.quantile(score, config.threshold_quantile)
    )
    threshold = max(robust_threshold, quantile_threshold)
    statistics["fusion"] = {
        "score_median": score_location,
        "score_robust_scale": score_scale,
        "robust_threshold": robust_threshold,
        "quantile_threshold": quantile_threshold,
        "selected_threshold": threshold,
        "top_k": top_k,
    }
    return FusionResult(
        score_grid=score.astype(np.float32),
        z_maps=z_maps,
        group_maps=group_maps,
        support_count_grid=support_count,
        threshold=float(threshold),
        score_scale=float(score_scale),
        statistics=statistics,
    )

