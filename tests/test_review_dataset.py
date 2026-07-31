from pathlib import Path

import numpy as np
import pytest
from skimage.io import imsave

from tfree_detection.review_dataset import (
    LabelKind,
    ReviewSample,
    ReviewSplit,
    load_review_label,
    split_annotation_and_reference_pools,
    split_for_group,
    validate_review_samples,
)


def _sample(**overrides) -> ReviewSample:
    values = {
        "sample_id": "sample-a",
        "dataset_id": "emseek_remaining_v1",
        "source_path": "image.png",
        "source_format": "png",
        "source_partition": "train",
        "review_split": ReviewSplit.TRAIN.value,
        "split_group_id": "group-a",
        "acquisition_id": "acquisition-a",
        "material": "WSe2",
        "primary_task": "atomic_point_defect",
        "prediction_tasks": ("atomic_point_defect",),
        "geometry_hint": "point",
    }
    values.update(overrides)
    return ReviewSample(**values)


def test_split_policy_keeps_known_acquisitions_together() -> None:
    assert split_for_group(
        "guo_2021_superscan",
        "guo-superscan-single-acquisition",
        "development_test",
    ) == ReviewSplit.TRAIN
    assert split_for_group(
        "dennler_2021_hrstem_defects",
        "dennler-archive",
        "locked_external_test",
    ) == ReviewSplit.TEST


def test_manifest_rejects_group_leakage() -> None:
    train = _sample()
    validation = _sample(
        sample_id="sample-b",
        review_split=ReviewSplit.VALIDATION.value,
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_review_samples([train, validation])


def test_locked_test_cannot_be_assisted_review() -> None:
    sample = _sample(
        source_partition="locked_external_test",
        review_split=ReviewSplit.TEST.value,
        review_eligible=True,
    )
    with pytest.raises(ValueError, match="cannot enter"):
        sample.validate()


def test_pixel_label_and_empty_control_load_with_image_shape(
    tmp_path: Path,
) -> None:
    image = np.zeros((24, 24), dtype=np.uint8)
    label = image.copy()
    label[5:9, 7:11] = 255
    imsave(tmp_path / "image.png", image, check_contrast=False)
    imsave(tmp_path / "label.png", label, check_contrast=False)
    pixel = _sample(
        label_kind=LabelKind.PIXEL_MASK.value,
        label_path="label.png",
    )
    loaded = load_review_label(pixel, tmp_path, image.shape)
    assert loaded is not None
    assert int(np.count_nonzero(loaded)) == 16

    empty = _sample(
        sample_id="empty",
        label_kind=LabelKind.IMPLICIT_EMPTY_MASK.value,
    )
    empty_mask = load_review_label(empty, tmp_path, image.shape)
    assert empty_mask is not None
    assert not np.any(empty_mask)


def test_existing_pixel_labels_never_enter_annotation_pool() -> None:
    unlabeled = _sample(sample_id="unlabeled")
    source_prior = _sample(
        sample_id="prior", label_kind=LabelKind.SOURCE_PRIOR_ONLY.value
    )
    pixel = _sample(
        sample_id="pixel",
        label_kind=LabelKind.PIXEL_MASK.value,
        label_path="label.png",
    )
    empty = _sample(
        sample_id="empty",
        label_kind=LabelKind.IMPLICIT_EMPTY_MASK.value,
    )
    annotation, reference = split_annotation_and_reference_pools(
        [unlabeled, source_prior, pixel, empty]
    )
    assert [item.sample_id for item in annotation] == [
        "unlabeled",
        "prior",
    ]
    assert [item.sample_id for item in reference] == ["pixel", "empty"]
