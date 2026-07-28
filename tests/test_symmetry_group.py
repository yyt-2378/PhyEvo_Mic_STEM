"""Focused tests for reference-anchored crystallographic group relations."""

from __future__ import annotations

import numpy as np

from tfree_detection.symmetry import (
    PatchObservation,
    align_group_motif,
    select_reference_patches,
)


def _asymmetric_motif() -> np.ndarray:
    motif = np.zeros((8, 8), dtype=float)
    motif[1, 2] = 1.0
    motif[3, 6] = 0.7
    motif[6, 1] = -0.4
    return motif


def test_group_alignment_separates_identity_translation_rotation_and_mirror() -> None:
    reference = _asymmetric_motif()

    identity = align_group_motif(reference, reference)
    translated = align_group_motif(
        np.roll(reference, shift=(0, 2), axis=(0, 1)),
        reference,
    )
    rotated = align_group_motif(np.rot90(reference), reference)
    mirrored = align_group_motif(np.fliplr(reference), reference)

    assert identity.relation == "identity"
    assert identity.identity_residual < 1e-6
    assert translated.relation == "translation"
    assert translated.relation_gain > 0.5
    assert rotated.relation == "rotation"
    assert rotated.aligned_residual < 1e-6
    assert mirrored.relation == "mirror"
    assert mirrored.aligned_residual < 1e-6


def test_reference_selection_keeps_the_dominant_identity_cluster() -> None:
    reference = _asymmetric_motif()
    observations = []
    for index in range(8):
        if index < 6:
            motif = reference + (index + 1) * 1e-4
        else:
            motif = np.roll(reference, shift=(0, 3), axis=(0, 1))
        observations.append(
            PatchObservation(
                center_yx=(16 + 24 * (index // 4), 16 + 24 * (index % 4)),
                motif=motif,
                motif_coverage=1.0,
                coherence=0.9,
                deformation=np.zeros(4, dtype=float),
                edge_support=1.0,
                quality=1.0 - 0.01 * index,
            )
        )

    selected = select_reference_patches(
        observations,
        quality_quantile=0.0,
        maximum_references=4,
    )

    assert len(selected.indices) == 4
    assert set(selected.indices).issubset(set(range(6)))
    assert set(selected.cluster_indices).issubset(set(range(6)))
    assert selected.cluster_support_fraction >= 0.70
    assert not selected.ambiguous_identity
