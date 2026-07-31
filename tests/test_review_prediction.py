import numpy as np
import pytest

from tfree_detection.review_dataset import ReviewSample
from tfree_detection.review_prediction import (
    PREDICTION_FORMAT,
    ReviewPrediction,
    comparison_overlay,
    label_support,
    normalize_for_display,
)


def _sample() -> ReviewSample:
    return ReviewSample(
        sample_id="sample",
        dataset_id="synthetic",
        source_path="image.png",
        source_format="png",
        source_partition="train",
        review_split="train",
        split_group_id="group",
        acquisition_id="acquisition",
        material="synthetic",
        primary_task="amorphous_region",
        prediction_tasks=("amorphous_region",),
        geometry_hint="region",
    )


def test_comparison_overlay_keeps_label_and_prediction_distinct() -> None:
    image = np.arange(64 * 64, dtype=float).reshape(64, 64)
    label = np.zeros((64, 64, 3), dtype=np.uint8)
    label[8:24, 8:24, 1] = 255
    prediction = np.zeros((64, 64), dtype=bool)
    prediction[32:48, 32:48] = True
    overlay = comparison_overlay(
        image, label=label, prediction_mask=prediction
    )
    assert overlay.shape == (64, 64, 3)
    assert np.max(overlay[8:24, 8:24, 1]) == 255
    assert overlay[40, 40, 0] > overlay[40, 40, 2]


def test_label_support_accepts_rgb_palette_and_empty_mask() -> None:
    palette = np.zeros((16, 16, 3), dtype=np.uint8)
    palette[3, 4] = (0, 128, 255)
    support = label_support(palette)
    assert support is not None
    assert int(support.sum()) == 1
    assert not np.any(label_support(np.zeros((16, 16), dtype=np.uint8)))


def test_display_normalization_rejects_no_shape_mismatch() -> None:
    image = np.linspace(0, 1, 32 * 32).reshape(32, 32)
    assert normalize_for_display(image).dtype == np.uint8
    with pytest.raises(ValueError, match="shape"):
        comparison_overlay(
            image,
            label=None,
            prediction_mask=np.zeros((16, 16), dtype=bool),
        )


def test_abstention_cannot_claim_positive_geometry() -> None:
    with pytest.raises(ValueError, match="positive geometry"):
        ReviewPrediction(
            prediction_format=PREDICTION_FORMAT,
            sample_id="sample",
            task="atomic_point_defect",
            sample_sha256="a" * 64,
            config_path="configs/atomic_point_defect.yaml",
            config_sha256="b" * 64,
            pixel_sha256="c" * 64,
            mask_path="mask.png",
            mask_sha256="d" * 64,
            image_shape=(32, 32),
            detected_area_pixels=1,
            detected_area_fraction=1 / 1024,
            candidates=(),
            status="abstained",
            abstention_reason="insufficient lattice sites",
        ).validate()
