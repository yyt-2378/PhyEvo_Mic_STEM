"""Discrete crystallographic group alignment for folded lattice motifs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class GroupAlignment:
    """Best reference relation and the identity-versus-aligned residuals."""

    relation: str
    action: str
    shift_bins_yx: tuple[int, int]
    polarity: int
    identity_residual: float
    aligned_residual: float
    relation_gain: float


def _normalize_motif(values: np.ndarray) -> np.ndarray:
    motif = np.asarray(values, dtype=float)
    motif = motif - float(np.mean(motif))
    norm = float(np.linalg.norm(motif))
    if norm < 1e-10:
        return np.zeros_like(motif, dtype=np.float32)
    return (motif / norm).astype(np.float32)


def fold_periodic_motif(
    image: np.ndarray,
    reciprocal_basis_xy_rows: np.ndarray,
    center_yx: tuple[int, int],
    patch_radius: int,
    bins: int,
) -> tuple[np.ndarray, float]:
    """Fold a local patch into one fractional unit cell.

    Global coordinates are retained before applying the reciprocal basis, so
    a coherent sub-cell phase translation remains visible as a circular motif
    shift instead of being removed by local recentering.
    """

    values = np.asarray(image, dtype=float)
    reciprocal = np.asarray(reciprocal_basis_xy_rows, dtype=float)
    if values.ndim != 2:
        raise ValueError("fold_periodic_motif expects a 2-D image")
    if reciprocal.shape != (2, 2):
        raise ValueError("reciprocal basis must have shape (2, 2)")
    if patch_radius < 2 or bins < 4:
        raise ValueError("motif patch radius/bins are too small")
    center_y, center_x = (int(item) for item in center_yx)
    y1 = max(0, center_y - patch_radius)
    y2 = min(values.shape[0], center_y + patch_radius + 1)
    x1 = max(0, center_x - patch_radius)
    x2 = min(values.shape[1], center_x + patch_radius + 1)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    xy = np.stack((xx.ravel(), yy.ravel()), axis=1)
    fractional = np.mod(xy @ reciprocal.T, 1.0)
    first = np.floor(fractional[:, 0] * bins).astype(int) % bins
    second = np.floor(fractional[:, 1] * bins).astype(int) % bins
    flat_index = first * bins + second
    patch = values[y1:y2, x1:x2].ravel()
    sums = np.bincount(
        flat_index,
        weights=patch,
        minlength=bins * bins,
    ).astype(float)
    counts = np.bincount(
        flat_index,
        minlength=bins * bins,
    ).astype(float)
    coverage = float(np.mean(counts > 0))
    fallback = float(np.mean(patch)) if patch.size else 0.0
    motif = np.full(bins * bins, fallback, dtype=float)
    populated = counts > 0
    motif[populated] = sums[populated] / counts[populated]
    motif = motif.reshape(bins, bins)
    motif = gaussian_filter(motif, sigma=0.45, mode="wrap")
    return _normalize_motif(motif), coverage


def _actions(reference: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    return (
        ("identity", reference),
        ("rotation_90", np.rot90(reference, 1)),
        ("rotation_180", np.rot90(reference, 2)),
        ("rotation_270", np.rot90(reference, 3)),
        ("mirror_x", np.fliplr(reference)),
        ("mirror_y", np.flipud(reference)),
        ("mirror_diagonal", reference.T),
        ("mirror_antidiagonal", np.rot90(reference.T, 2)),
    )


def _canonical_shift(index: int, bins: int) -> int:
    return int(index - bins if index > bins // 2 else index)


def _circular_correlation(
    local: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    return np.fft.ifft2(
        np.fft.fft2(local) * np.conj(np.fft.fft2(reference))
    ).real


def _identity_correlation(
    local: np.ndarray,
    reference: np.ndarray,
    tolerance: int,
) -> float:
    candidates = []
    for shift_y in range(-tolerance, tolerance + 1):
        for shift_x in range(-tolerance, tolerance + 1):
            shifted = np.roll(
                reference,
                shift=(shift_y, shift_x),
                axis=(0, 1),
            )
            value = float(np.sum(local * shifted))
            candidates.extend((value, -value))
    return float(np.clip(max(candidates), -1.0, 1.0))


def _relation_from_action(
    action: str,
    shift_yx: tuple[int, int],
    bins: int,
    identity_tolerance: int,
) -> str:
    shift_y, shift_x = shift_yx
    if action.startswith("mirror"):
        return "mirror"
    if action.startswith("rotation"):
        return "rotation"
    if max(abs(shift_y), abs(shift_x)) <= identity_tolerance:
        return "identity"
    half = bins // 2
    if any(
        abs(abs(value) - half) <= 1
        for value in (shift_y, shift_x)
        if value
    ):
        return "sublattice_exchange"
    return "translation"


def align_group_motif(
    local_motif: np.ndarray,
    reference_motif: np.ndarray,
    *,
    identity_shift_tolerance_bins: int = 1,
) -> GroupAlignment:
    """Align one motif under translation, rotation, mirror, and polarity.

    Small circular shifts are treated as registration nuisance.  A larger
    winning shift, mirror, or rotation is retained as a non-identity relation
    even when it explains the local motif with a small residual.
    """

    local = _normalize_motif(local_motif)
    reference = _normalize_motif(reference_motif)
    if local.shape != reference.shape or local.ndim != 2:
        raise ValueError("Local and reference motifs must share a 2-D shape")
    if local.shape[0] != local.shape[1]:
        raise ValueError("Group alignment requires square motif grids")
    bins = int(local.shape[0])
    identity_correlation = _identity_correlation(
        local,
        reference,
        identity_shift_tolerance_bins,
    )
    best: tuple[float, str, int, int, int] | None = None
    for action, transformed in _actions(reference):
        correlation = _circular_correlation(local, transformed)
        for polarity in (1, -1):
            active = correlation * polarity
            flat_index = int(np.argmax(active))
            index_y, index_x = np.unravel_index(
                flat_index,
                active.shape,
            )
            value = float(active[index_y, index_x])
            candidate = (
                value,
                action,
                _canonical_shift(index_y, bins),
                _canonical_shift(index_x, bins),
                polarity,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:  # pragma: no cover - actions are always non-empty
        raise RuntimeError("No crystallographic group action was evaluated")
    correlation, action, shift_y, shift_x, polarity = best
    identity_residual = float(
        np.clip(1.0 - identity_correlation, 0.0, 2.0)
    )
    aligned_residual = float(np.clip(1.0 - correlation, 0.0, 2.0))
    relation = _relation_from_action(
        action,
        (shift_y, shift_x),
        bins,
        identity_shift_tolerance_bins,
    )
    return GroupAlignment(
        relation=relation,
        action=action,
        shift_bins_yx=(shift_y, shift_x),
        polarity=int(polarity),
        identity_residual=identity_residual,
        aligned_residual=aligned_residual,
        relation_gain=max(identity_residual - aligned_residual, 0.0),
    )
