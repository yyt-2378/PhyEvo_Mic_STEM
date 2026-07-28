#!/usr/bin/env python
"""Run deterministic, exact-mask benchmarks for every implemented task."""

from __future__ import annotations

import json

import numpy as np

from tfree_detection.config import DetectorConfig
from tfree_detection.context import IdealSkeleton, SkeletonNode
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.synthetic import (
    make_atomic_lattice_with_point_defects,
    make_crystal_with_amorphous_region,
    make_lattice_with_stacking_fault,
    make_two_phase_interface,
)


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    intersection = int(np.sum(prediction & truth))
    union = int(np.sum(prediction | truth))
    return {
        "iou": intersection / max(union, 1),
        "precision": intersection / max(int(np.sum(prediction)), 1),
        "recall": intersection / max(int(np.sum(truth)), 1),
    }


def amorphous_case() -> dict:
    image, truth = make_crystal_with_amorphous_region(
        size=256, angle_degrees=19, seed=4
    )
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
    result = TrainingFreeDetectionFramework(config).detect(image)
    return {"task": config.task, **metrics(result.mask, truth)}


def atomic_case() -> dict:
    image, truth, metadata = make_atomic_lattice_with_point_defects()
    config = DetectorConfig(task="atomic_point_defect")
    config.atomic.peak_min_distance = 7
    config.atomic.peak_threshold_quantile = 0.70
    config.atomic.site_anomaly_z = 3.0
    config.atomic.displacement_z = 2.5
    skeleton = IdealSkeleton(
        basis=np.asarray(metadata["basis_rows_xy"]),
        nodes=[
            SkeletonNode(
                site_id="A",
                fractional_coordinate=(0.0, 0.0),
                sublattice="A",
                expected_relative_intensity=1.0,
            )
        ],
    )
    result = TrainingFreeDetectionFramework(config).detect(
        image, skeleton=skeleton
    )
    return {
        "task": config.task,
        **metrics(result.mask, truth),
        "detected_sites": len(result.regions),
    }


def stacking_case() -> dict:
    image, truth = make_lattice_with_stacking_fault()
    config = DetectorConfig(task="stacking_fault")
    config.spectral.min_frequency = 0.02
    config.spectral.max_frequency = 0.30
    config.spectral.global_peak_count = 16
    config.line_defect.threshold_quantile = 0.94
    config.line_defect.threshold_mad = 2.4
    config.line_defect.minimum_region_area = 30
    result = TrainingFreeDetectionFramework(config).detect(image)
    return {"task": config.task, **metrics(result.mask, truth)}


def phase_boundary_case() -> dict:
    image, truth = make_two_phase_interface()
    config = DetectorConfig(task="phase_boundary")
    config.spectral.min_frequency = 0.025
    config.spectral.max_frequency = 0.25
    config.phase_boundary.stride = 10
    config.phase_boundary.minimum_signature_separation = 0.65
    result = TrainingFreeDetectionFramework(config).detect(image)
    return {
        "task": config.task,
        **metrics(result.mask, truth),
        "domain_fractions": result.metadata["domain_fractions"],
        "signature_separation": result.metadata[
            "prototype_separation_over_within_dispersion"
        ],
    }


def main() -> None:
    cases = [
        amorphous_case(),
        atomic_case(),
        stacking_case(),
        phase_boundary_case(),
    ]
    print(
        json.dumps(
            {
                "neural_training_used": False,
                "supervised_labels_used": False,
                "per_image_statistical_adaptation": True,
                "ground_truth": "deterministic_synthetic_masks",
                "tasks": cases,
                "mean_iou": float(np.mean([case["iou"] for case in cases])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
