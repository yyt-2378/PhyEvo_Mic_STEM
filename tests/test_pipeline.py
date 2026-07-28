from pathlib import Path

import numpy as np

from tfree_detection.config import DetectorConfig
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.reporting import save_result
from tfree_detection.synthetic import make_crystal_with_amorphous_region


def benchmark_config() -> DetectorConfig:
    config = DetectorConfig()
    config.spectral.patch_size = 64
    config.spectral.stride = 12
    config.spectral.min_frequency = 0.035
    config.spectral.max_frequency = 0.25
    config.spectral.autocorrelation_min_lag = 5
    config.spectral.autocorrelation_max_lag = 20
    config.fusion.threshold_quantile = 0.90
    config.fusion.threshold_mad = 2.0
    config.fusion.support_z = 0.9
    config.postprocess.border_margin = 32
    config.postprocess.morphology_radius = 3
    config.postprocess.minimum_region_area = 300
    config.postprocess.minimum_hole_area = 150
    config.validate()
    return config


def test_rotated_synthetic_amorphous_region_is_localized() -> None:
    image, truth = make_crystal_with_amorphous_region(
        size=256, angle_degrees=19.0, seed=4
    )
    result = TrainingFreeDetectionFramework(benchmark_config()).detect(image)
    truth_scores = result.score_map[truth]
    normal_scores = result.score_map[~truth]

    assert np.median(truth_scores) > np.percentile(normal_scores, 80)
    assert np.any(result.mask & truth)
    assert len(result.regions) >= 1
    nearest = min(
        np.hypot(
            region.centroid_xy[0] - 0.47 * image.shape[1],
            region.centroid_xy[1] - 0.56 * image.shape[0],
        )
        for region in result.regions
    )
    assert nearest < 0.18 * image.shape[0]


def test_intensity_affine_transform_is_invariant() -> None:
    image, _ = make_crystal_with_amorphous_region(size=224, seed=7)
    config = benchmark_config()
    framework = TrainingFreeDetectionFramework(config)
    first = framework.detect(image.astype(np.float64))
    second = framework.detect(image.astype(np.float64) * 3.2 + 1700.0)
    assert np.allclose(first.score_map, second.score_map, atol=2e-4)
    assert np.array_equal(first.mask, second.mask)


def test_report_contract(tmp_path: Path) -> None:
    image, _ = make_crystal_with_amorphous_region(size=192, seed=2)
    config = benchmark_config()
    result = TrainingFreeDetectionFramework(config).detect(image)
    artifacts = save_result(result, config, tmp_path)
    for path in artifacts.values():
        assert Path(path).is_file()
    assert (tmp_path / "summary.json").read_text(encoding="utf-8").startswith("{")

