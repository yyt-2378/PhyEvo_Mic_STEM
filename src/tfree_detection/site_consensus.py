"""One-to-one consensus and disagreement states for site detectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .site_matching import match_sites_lexicographic


@dataclass(frozen=True)
class SiteConsensus:
    consensus_xy: np.ndarray
    primary_only_xy: np.ndarray
    secondary_only_xy: np.ndarray
    primary_indices: np.ndarray
    secondary_indices: np.ndarray
    agreement_distances: np.ndarray
    tolerance_pixels: float

    def to_summary(self) -> dict[str, float | int | str]:
        return {
            "matching_version": "site_matching_v2",
            "consensus_count": int(len(self.consensus_xy)),
            "primary_only_count": int(len(self.primary_only_xy)),
            "secondary_only_count": int(len(self.secondary_only_xy)),
            "mean_agreement_distance_pixels": (
                float(np.mean(self.agreement_distances))
                if len(self.agreement_distances)
                else float("nan")
            ),
            "p95_agreement_distance_pixels": (
                float(np.percentile(self.agreement_distances, 95))
                if len(self.agreement_distances)
                else float("nan")
            ),
            "tolerance_pixels": float(self.tolerance_pixels),
        }


def fuse_site_consensus(
    primary_xy: np.ndarray,
    secondary_xy: np.ndarray,
    *,
    tolerance_pixels: float,
) -> SiteConsensus:
    """Match independent site predictions and preserve every disagreement."""

    if tolerance_pixels <= 0:
        raise ValueError("Consensus tolerance must be positive")
    primary = np.asarray(primary_xy, dtype=float).reshape(-1, 2)
    secondary = np.asarray(secondary_xy, dtype=float).reshape(-1, 2)
    matched = match_sites_lexicographic(
        primary,
        secondary,
        tolerance_pixels=tolerance_pixels,
    )
    rows = matched.predicted_indices
    columns = matched.truth_indices
    distances = matched.distances
    primary_unmatched = np.ones(len(primary), dtype=bool)
    secondary_unmatched = np.ones(len(secondary), dtype=bool)
    primary_unmatched[rows] = False
    secondary_unmatched[columns] = False
    consensus = (
        0.5 * (primary[rows] + secondary[columns])
        if len(rows)
        else np.empty((0, 2), dtype=float)
    )
    return SiteConsensus(
        consensus_xy=consensus,
        primary_only_xy=primary[primary_unmatched],
        secondary_only_xy=secondary[secondary_unmatched],
        primary_indices=rows,
        secondary_indices=columns,
        agreement_distances=distances,
        tolerance_pixels=float(tolerance_pixels),
    )
