"""Automatic multi-reference selection for crystalline symmetry analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class PatchObservation:
    """One coarse local crystallographic observation."""

    center_yx: tuple[int, int]
    motif: np.ndarray
    motif_coverage: float
    coherence: float
    deformation: np.ndarray
    edge_support: float
    quality: float


@dataclass(frozen=True)
class ReferenceSelection:
    """Selected identity-consistent normal references and audit metadata."""

    indices: tuple[int, ...]
    medoid_index: int
    candidate_indices: tuple[int, ...]
    cluster_indices: tuple[int, ...]
    cluster_support_fraction: float
    identity_radius: float
    pairwise_identity_median: float
    ambiguous_identity: bool


def _identity_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first - np.mean(first)
    second = second - np.mean(second)
    first /= np.linalg.norm(first) + 1e-12
    second /= np.linalg.norm(second) + 1e-12
    best = 0.0
    for shift_y in (-1, 0, 1):
        for shift_x in (-1, 0, 1):
            correlation = float(
                np.sum(
                    first
                    * np.roll(
                        second,
                        shift=(shift_y, shift_x),
                        axis=(0, 1),
                    )
                )
            )
            best = max(best, abs(correlation))
    return float(np.clip(1.0 - best, 0.0, 2.0))


def select_reference_patches(
    observations: list[PatchObservation],
    *,
    maximum_references: int = 5,
    minimum_references: int = 3,
    candidate_limit: int = 96,
    quality_quantile: float = 0.60,
    identity_radius_multiplier: float = 2.5,
    minimum_identity_radius: float = 0.03,
) -> ReferenceSelection:
    """Choose spatially diverse references from the densest identity cluster.

    The selector never assigns a fixed pixel fraction as defective.  It first
    finds the most mutually identity-consistent set among high-quality patches,
    then chooses several spatially separated representatives from that set.
    """

    if len(observations) < minimum_references:
        raise ValueError("Too few patch observations for multi-reference selection")
    if not 0 <= quality_quantile < 1:
        raise ValueError("quality_quantile must be in [0, 1)")
    qualities = np.asarray(
        [item.quality for item in observations],
        dtype=float,
    )
    threshold = float(np.quantile(qualities, quality_quantile))
    candidates = np.flatnonzero(qualities >= threshold)
    if len(candidates) < minimum_references:
        candidates = np.argsort(qualities)[-minimum_references:]
    if len(candidates) > candidate_limit:
        order = np.argsort(qualities[candidates])[::-1][:candidate_limit]
        candidates = candidates[order]

    count = len(candidates)
    distances = np.zeros((count, count), dtype=float)
    for first in range(count):
        for second in range(first + 1, count):
            value = _identity_distance(
                observations[int(candidates[first])].motif,
                observations[int(candidates[second])].motif,
            )
            distances[first, second] = value
            distances[second, first] = value
    if count > 1:
        nearest = np.partition(distances + np.eye(count) * 10, 0, axis=1)[
            :, 0
        ]
        typical = float(np.median(nearest))
        pairwise_median = float(
            np.median(distances[np.triu_indices(count, k=1)])
        )
    else:  # pragma: no cover - minimum_references prevents this
        typical = minimum_identity_radius
        pairwise_median = 0.0
    radius = max(
        minimum_identity_radius,
        identity_radius_multiplier * typical,
    )
    support = np.sum(distances <= radius, axis=1)
    best_support = int(np.max(support))
    supported = np.flatnonzero(support == best_support)
    medoid_local = int(
        supported[
            np.argmin(
                [
                    np.median(
                        distances[index][distances[index] <= radius]
                    )
                    for index in supported
                ]
            )
        ]
    )
    cluster_local = np.flatnonzero(distances[medoid_local] <= radius)
    if len(cluster_local) < minimum_references:
        cluster_local = np.argsort(distances[medoid_local])[
            :minimum_references
        ]

    medoid_index = int(candidates[medoid_local])
    cluster_indices = [int(candidates[index]) for index in cluster_local]
    selected = [medoid_index]
    centers = np.asarray(
        [observations[index].center_yx for index in cluster_indices],
        dtype=float,
    )
    diagonal = float(
        np.linalg.norm(np.ptp(centers, axis=0))
    )
    diagonal = max(diagonal, 1.0)
    while (
        len(selected) < min(maximum_references, len(cluster_indices))
    ):
        best_index = None
        best_score = -np.inf
        for index in cluster_indices:
            if index in selected:
                continue
            local_position = int(np.where(candidates == index)[0][0])
            selected_positions = [
                int(np.where(candidates == item)[0][0])
                for item in selected
            ]
            if any(
                distances[local_position, selected_position] > radius
                for selected_position in selected_positions
            ):
                continue
            center = np.asarray(observations[index].center_yx, dtype=float)
            separation = min(
                np.linalg.norm(
                    center
                    - np.asarray(
                        observations[item].center_yx,
                        dtype=float,
                    )
                )
                for item in selected
            )
            identity_fit = distances[medoid_local, local_position]
            score = (
                separation / diagonal
                + 0.20 * observations[index].quality
                - 0.60 * identity_fit / max(radius, 1e-8)
            )
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)

    support_fraction = float(len(cluster_indices) / len(candidates))
    return ReferenceSelection(
        indices=tuple(selected),
        medoid_index=medoid_index,
        candidate_indices=tuple(int(item) for item in candidates),
        cluster_indices=tuple(cluster_indices),
        cluster_support_fraction=support_fraction,
        identity_radius=float(radius),
        pairwise_identity_median=pairwise_median,
        ambiguous_identity=support_fraction < 0.50,
    )
