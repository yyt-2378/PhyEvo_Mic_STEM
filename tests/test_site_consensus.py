import numpy as np
import pytest

from tfree_detection.site_consensus import fuse_site_consensus


def test_site_consensus_is_one_to_one_and_preserves_disagreement() -> None:
    primary = np.asarray([[0, 0], [0.5, 0], [10, 10], [30, 30]])
    secondary = np.asarray([[0.2, 0], [10.5, 10], [20, 20]])
    result = fuse_site_consensus(
        primary, secondary, tolerance_pixels=1.0
    )
    assert result.consensus_xy.shape == (2, 2)
    assert result.primary_only_xy.shape == (2, 2)
    assert result.secondary_only_xy.shape == (1, 2)
    assert len(set(result.primary_indices.tolist())) == 2
    assert len(set(result.secondary_indices.tolist())) == 2
    assert result.to_summary()["consensus_count"] == 2


def test_site_consensus_handles_empty_inputs_and_invalid_tolerance() -> None:
    result = fuse_site_consensus(
        np.empty((0, 2)),
        np.asarray([[1, 1]]),
        tolerance_pixels=2.0,
    )
    assert len(result.consensus_xy) == 0
    assert result.secondary_only_xy.tolist() == [[1.0, 1.0]]
    with pytest.raises(ValueError, match="positive"):
        fuse_site_consensus(
            np.empty((0, 2)),
            np.empty((0, 2)),
            tolerance_pixels=0,
        )


def test_site_consensus_maximizes_agreement_cardinality() -> None:
    primary = np.asarray(
        [
            [-1.3812213700073914, 0.08576778978006505],
            [-0.12199235134378883, -1.3129512625503732],
            [0.42398450741812477, 1.0578985154419702],
        ]
    )
    secondary = np.asarray(
        [
            [0.27882305431285204, -0.7197076567883304],
            [1.0196445630942264, 0.028487644564528125],
            [0.03266665339959918, 0.7590906231065335],
        ]
    )
    result = fuse_site_consensus(
        primary, secondary, tolerance_pixels=1.0
    )
    assert len(result.consensus_xy) == 2
    assert result.to_summary()["matching_version"] == "site_matching_v2"
