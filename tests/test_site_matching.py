import numpy as np

from tfree_detection.site_matching import match_sites_lexicographic


def test_matching_maximizes_cardinality_before_distance() -> None:
    predicted = np.asarray(
        [
            [-1.3812213700073914, 0.08576778978006505],
            [-0.12199235134378883, -1.3129512625503732],
            [0.42398450741812477, 1.0578985154419702],
        ]
    )
    truth = np.asarray(
        [
            [0.27882305431285204, -0.7197076567883304],
            [1.0196445630942264, 0.028487644564528125],
            [0.03266665339959918, 0.7590906231065335],
        ]
    )
    result = match_sites_lexicographic(
        predicted, truth, tolerance_pixels=1.0
    )
    assert result.matched_count == 2
    assert len(set(result.predicted_indices.tolist())) == 2
    assert len(set(result.truth_indices.tolist())) == 2
    assert np.all(result.distances <= 1.0)


def test_matching_is_one_to_one_and_handles_empty_inputs() -> None:
    result = match_sites_lexicographic(
        np.asarray([[0, 0], [0.1, 0]]),
        np.asarray([[0, 0]]),
        tolerance_pixels=1.0,
    )
    assert result.matched_count == 1
    assert result.precision == 0.5
    assert result.recall == 1.0
    empty = match_sites_lexicographic(
        np.empty((0, 2)),
        np.asarray([[0, 0]]),
        tolerance_pixels=1.0,
    )
    assert empty.matched_count == 0
    assert empty.precision == 0.0
    assert empty.recall == 0.0
