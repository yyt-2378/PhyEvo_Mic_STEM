import numpy as np

from tfree_detection.config import PreprocessConfig
from tfree_detection.models import DetectionRegion, DetectionResult
from tfree_detection.preprocessing import robust_normalize
from tfree_detection.temporal_support import (
    TemporalSupportConfig,
    analyze_temporal_support_from_stack,
)


def _result(image: np.ndarray) -> DetectionResult:
    normalized, _ = robust_normalize(image, PreprocessConfig())
    shape = image.shape
    mask = np.zeros(shape, dtype=bool)
    mask[24:40, 24:40] = True
    score = mask.astype(np.float32)
    return DetectionResult(
        task="stacking_fault",
        image_shape=shape,
        normalized_image=normalized,
        score_map=score,
        probability_map=score,
        mask=mask,
        regions=[
            DetectionRegion(
                region_id=1,
                bbox_xyxy=(24, 24, 40, 40),
                centroid_xy=(32.0, 32.0),
                area_pixels=256,
                mean_score=1.0,
                max_score=1.0,
                confidence=0.8,
                confidence_level="medium",
                evidence_groups=["phase_jump"],
            )
        ],
        feature_maps={},
        evidence_maps={},
    )


def test_temporal_support_registers_frames_without_filtering_candidates() -> None:
    y, x = np.indices((64, 64), dtype=float)
    reference = (
        np.sin(2 * np.pi * x / 8)
        + np.cos(2 * np.pi * y / 11)
        + 2 * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / 30)
    )
    stack = np.stack(
        [
            np.roll(reference, shift=(dy, dx), axis=(0, 1))
            for dy, dx in ((1, 0), (0, 1), (0, 0), (-1, 0), (0, -1))
        ]
    )
    result = _result(stack[2])
    report = analyze_temporal_support_from_stack(
        stack,
        result,
        PreprocessConfig(),
        reference_frame_index=2,
        config=TemporalSupportConfig(maximum_sampled_frames=5),
    )
    assert len(report.accepted_frame_indices) == 5
    assert report.metadata["labels_read"] is False
    assert report.candidate_support[0]["hard_candidate_filter_applied"] is False
    assert report.temporal_consistency_map.shape == (64, 64)
