#!/usr/bin/env python
"""Run three rotation/noise cases and report pixel-level overlap metrics."""

from __future__ import annotations

import json

import numpy as np

from tfree_detection.config import DetectorConfig
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.synthetic import make_crystal_with_amorphous_region


def make_config() -> DetectorConfig:
    config = DetectorConfig()
    config.spectral.patch_size = 64
    config.spectral.stride = 12
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
    config.output.save_feature_maps = False
    config.validate()
    return config


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    intersection = int(np.sum(prediction & truth))
    union = int(np.sum(prediction | truth))
    return {
        "iou": intersection / max(union, 1),
        "precision": intersection / max(int(np.sum(prediction)), 1),
        "recall": intersection / max(int(np.sum(truth)), 1),
    }


def main() -> None:
    framework = TrainingFreeDetectionFramework(make_config())
    cases = [
        {"angle": 0.0, "noise": 0.08, "seed": 1},
        {"angle": 17.0, "noise": 0.10, "seed": 2},
        {"angle": 33.0, "noise": 0.14, "seed": 3},
    ]
    results = []
    for case in cases:
        image, truth = make_crystal_with_amorphous_region(
            size=256,
            angle_degrees=case["angle"],
            noise_standard_deviation=case["noise"],
            seed=case["seed"],
        )
        detection = framework.detect(image)
        results.append(
            {
                **case,
                **metrics(detection.mask, truth),
                "regions": len(detection.regions),
            }
        )
    payload = {
        "neural_training_used": False,
        "supervised_labels_used": False,
        "per_image_statistical_adaptation": True,
        "cases": results,
        "mean_iou": float(np.mean([case["iou"] for case in results])),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
