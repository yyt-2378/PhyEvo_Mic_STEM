"""Lexicographic one-to-one site matching within a physical tolerance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SiteMatch:
    predicted_indices: np.ndarray
    truth_indices: np.ndarray
    distances: np.ndarray
    predicted_count: int
    truth_count: int
    tolerance_pixels: float

    @property
    def matched_count(self) -> int:
        return int(len(self.distances))

    @property
    def precision(self) -> float:
        return self.matched_count / max(self.predicted_count, 1)

    @property
    def recall(self) -> float:
        return self.matched_count / max(self.truth_count, 1)

    @property
    def f1(self) -> float:
        return (
            2
            * self.precision
            * self.recall
            / max(self.precision + self.recall, 1e-12)
        )


def match_sites_lexicographic(
    predicted_xy: np.ndarray,
    truth_xy: np.ndarray,
    *,
    tolerance_pixels: float,
) -> SiteMatch:
    """Maximize matches inside tolerance, then minimize localization error.

    A plain Hungarian assignment over every pair can minimize total distance
    while choosing fewer edges inside the tolerance. We first decompose the
    threshold graph into local connected components. Within each component,
    unmatched assignments receive a penalty larger than every possible change
    in the sum of valid distances. This makes cardinality the primary objective
    and distance the secondary objective.
    """

    if tolerance_pixels <= 0:
        raise ValueError("Site matching tolerance must be positive")
    predicted = np.asarray(predicted_xy, dtype=float).reshape(-1, 2)
    truth = np.asarray(truth_xy, dtype=float).reshape(-1, 2)
    if not len(predicted) or not len(truth):
        return SiteMatch(
            predicted_indices=np.empty(0, dtype=int),
            truth_indices=np.empty(0, dtype=int),
            distances=np.empty(0, dtype=float),
            predicted_count=len(predicted),
            truth_count=len(truth),
            tolerance_pixels=float(tolerance_pixels),
        )

    tree = cKDTree(truth)
    neighborhoods = tree.query_ball_point(
        predicted, tolerance_pixels
    )
    edge_rows = []
    edge_columns = []
    for row, columns in enumerate(neighborhoods):
        edge_rows.extend([row] * len(columns))
        edge_columns.extend(columns)
    if not edge_rows:
        return SiteMatch(
            predicted_indices=np.empty(0, dtype=int),
            truth_indices=np.empty(0, dtype=int),
            distances=np.empty(0, dtype=float),
            predicted_count=len(predicted),
            truth_count=len(truth),
            tolerance_pixels=float(tolerance_pixels),
        )
    adjacency = csr_matrix(
        (
            np.ones(len(edge_rows), dtype=np.uint8),
            (edge_rows, edge_columns),
        ),
        shape=(len(predicted), len(truth)),
    )
    zero_predicted = csr_matrix(
        (len(predicted), len(predicted)), dtype=np.uint8
    )
    zero_truth = csr_matrix(
        (len(truth), len(truth)), dtype=np.uint8
    )
    block = bmat(
        [
            [zero_predicted, adjacency],
            [adjacency.T, zero_truth],
        ],
        format="csr",
    )
    component_count, labels = connected_components(
        block, directed=False, return_labels=True
    )
    predicted_matches = []
    truth_matches = []
    match_distances = []
    for component in range(component_count):
        predicted_indices = np.flatnonzero(
            labels[: len(predicted)] == component
        )
        truth_indices = np.flatnonzero(
            labels[len(predicted) :] == component
        )
        if not len(predicted_indices) or not len(truth_indices):
            continue
        distances = np.linalg.norm(
            predicted[predicted_indices, None, :]
            - truth[None, truth_indices, :],
            axis=-1,
        )
        minimum_side = min(
            len(predicted_indices), len(truth_indices)
        )
        unmatched_penalty = (
            (minimum_side + 1) * tolerance_pixels
        )
        invalid_penalty = 3.0 * unmatched_penalty
        real_cost = np.where(
            distances <= tolerance_pixels,
            distances,
            invalid_penalty,
        )
        dummy_cost = np.full(
            (len(predicted_indices), len(predicted_indices)),
            unmatched_penalty,
            dtype=float,
        )
        cost = np.hstack((real_cost, dummy_cost))
        rows, columns = linear_sum_assignment(cost)
        real = columns < len(truth_indices)
        rows = rows[real]
        columns = columns[real]
        valid = distances[rows, columns] <= tolerance_pixels
        rows = rows[valid]
        columns = columns[valid]
        predicted_matches.extend(predicted_indices[rows].tolist())
        truth_matches.extend(truth_indices[columns].tolist())
        match_distances.extend(distances[rows, columns].tolist())
    order = np.argsort(predicted_matches)
    return SiteMatch(
        predicted_indices=np.asarray(
            predicted_matches, dtype=int
        )[order],
        truth_indices=np.asarray(truth_matches, dtype=int)[order],
        distances=np.asarray(match_distances, dtype=float)[order],
        predicted_count=len(predicted),
        truth_count=len(truth),
        tolerance_pixels=float(tolerance_pixels),
    )
