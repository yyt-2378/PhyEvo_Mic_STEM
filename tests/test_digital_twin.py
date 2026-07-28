import numpy as np
import pytest

from tfree_detection.digital_twin import (
    NuisanceParameters,
    SUPPORTED_TWIN_TASKS,
    make_counterfactual_twin,
)


@pytest.mark.parametrize("task", SUPPORTED_TWIN_TASKS)
def test_counterfactual_twins_change_structure_under_shared_nuisance(task) -> None:
    nuisance = NuisanceParameters(
        angle_degrees=17,
        gaussian_noise_standard_deviation=0.08,
        blur_sigma=0.6,
        illumination_x=0.12,
        illumination_y=-0.09,
        dose_electrons_per_normalized_unit=60,
    )
    pair = make_counterfactual_twin(
        task, size=128, seed=9, nuisance=nuisance
    )
    pair.validate()
    assert pair.normal_image.shape == (128, 128)
    assert pair.defect_image.dtype == np.uint16
    assert pair.manifest()["source_kind"] == "digital_twin"
    assert pair.metadata["shared_random_imaging_realization"] is True
    difference = np.abs(
        pair.defect_image.astype(float) - pair.normal_image.astype(float)
    )
    inside = float(np.mean(difference[pair.defect_mask]))
    outside = float(np.mean(difference[~pair.defect_mask]))
    assert inside > outside
    assert pair.metadata["experimental_accuracy_claim_allowed"] is False


def test_twin_pair_id_is_deterministic_and_nuisance_sensitive() -> None:
    first = make_counterfactual_twin("amorphous_region", seed=3)
    second = make_counterfactual_twin("amorphous_region", seed=3)
    changed = make_counterfactual_twin(
        "amorphous_region",
        seed=3,
        nuisance=NuisanceParameters(angle_degrees=12),
    )
    assert first.pair_id == second.pair_id
    assert np.array_equal(first.normal_image, second.normal_image)
    assert np.array_equal(first.defect_image, second.defect_image)
    assert changed.pair_id != first.pair_id
