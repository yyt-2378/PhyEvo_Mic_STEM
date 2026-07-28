import numpy as np

from tfree_detection.line_topology import attach_line_topology
from tfree_detection.models import DetectionRegion, DetectionResult


def test_line_topology_describes_line_without_changing_mask() -> None:
    shape = (64, 64)
    mask = np.zeros(shape, dtype=bool)
    mask[30:34, 8:56] = True
    before = mask.copy()
    result = DetectionResult(
        task="stacking_fault",
        image_shape=shape,
        normalized_image=np.zeros(shape, dtype=np.float32),
        score_map=mask.astype(np.float32),
        probability_map=mask.astype(np.float32),
        mask=mask,
        regions=[
            DetectionRegion(
                region_id=1,
                bbox_xyxy=(8, 30, 56, 34),
                centroid_xy=(32.0, 32.0),
                area_pixels=int(mask.sum()),
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
    enriched = attach_line_topology(result)
    topology = enriched.regions[0].provenance["line_topology"]
    assert np.array_equal(enriched.mask, before)
    assert topology["endpoint_count"] == 2
    assert topology["linearity"] > 0.9
    assert topology["topology_state"] == "simple_coherent_line"
    assert "topology_skeleton" in enriched.evidence_maps
