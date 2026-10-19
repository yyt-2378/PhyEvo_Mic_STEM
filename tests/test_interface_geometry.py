import numpy as np
import pytest

from tfree_detection.interface_geometry import interpret_domain_geometry, ranked_support


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("start", [24, 55])
@pytest.mark.parametrize("invert", [False, True])
def test_band_is_label_translation_and_rotation_independent(rotation, start, invert):
    labels = np.zeros((128, 128), np.int8)
    labels[:, start:start+24] = 1
    if invert:
        labels = 1-labels
    labels = np.rot90(labels, rotation)
    score, info = interpret_domain_geometry(labels, np.ones_like(labels, bool), separated=True)
    assert info["hypothesis"] == "enclosed_texture_band"
    band_label = 0 if invert else 1
    assert np.all(score[labels != band_label] == 0)
    mask = ranked_support(score, 1000)
    assert mask.sum() == 1000
    assert np.all(labels[mask] == band_label)


def test_bulk_boundary_is_not_a_band_and_low_separation_abstains():
    labels = np.zeros((128, 128), np.int8); labels[:, 80:] = 1
    valid = np.ones_like(labels, bool)
    score, info = interpret_domain_geometry(labels, valid, separated=True)
    assert info["hypothesis"] == "two_domain_boundary"
    assert not score.any()
    labels[:, 40:64] = 1
    _, info = interpret_domain_geometry(labels, valid, separated=False)
    assert info["hypothesis"] == "unresolved"


def test_unknown_vacuum_is_not_a_matching_flank():
    labels = np.zeros((128, 128), np.int8)
    labels[:, 40:64] = 1; labels[:, 64:] = -1
    score, info = interpret_domain_geometry(labels, labels >= 0, separated=True)
    assert info["hypothesis"] != "enclosed_texture_band"
    assert not score.any()


def test_band_preserves_validity_and_does_not_fill_zero_evidence():
    labels = np.zeros((128, 128), np.int8); labels[:, 45:70] = 1
    valid = np.ones_like(labels, bool); valid[:25] = False
    score, info = interpret_domain_geometry(labels, valid, separated=True)
    assert info["hypothesis"] == "enclosed_texture_band"
    mask = ranked_support(score, 100000)
    assert not mask[~valid].any()
    assert np.array_equal(mask, (score > 0) & valid)


def test_diagonal_band_topology():
    y, x = np.indices((128, 128))
    labels = (np.abs(x-y) < 12).astype(np.int8)
    _, info = interpret_domain_geometry(labels, np.ones_like(labels, bool), separated=True)
    assert info["hypothesis"] == "enclosed_texture_band"
