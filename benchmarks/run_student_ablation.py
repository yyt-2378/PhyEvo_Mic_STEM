#!/usr/bin/env python
"""Small deterministic student ablation on held-out synthetic conditions.

This is a software/learning-signal benchmark, not evidence of cross-material
generalization.  Real conclusions require the empty Gold and locked partitions
in datasets/partition_registry.yaml to be populated.
"""

from __future__ import annotations

import copy
import json
import random

import numpy as np
import torch

from tfree_detection.config import DetectorConfig
from tfree_detection.digital_twin import NuisanceParameters, make_counterfactual_twin
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.student import (
    EvidencePreservingStudent,
    StudentLossWeights,
    evidence_preserving_loss,
    hard_pseudolabel_baseline_loss,
    stack_student_targets,
    student_targets_from_counterfactual_twin,
    student_target_from_result,
)
from tfree_detection.synthetic import make_crystal_with_amorphous_region


def _config() -> DetectorConfig:
    config = DetectorConfig()
    config.spectral.patch_size = 48
    config.spectral.stride = 12
    config.spectral.max_frequency = 0.25
    config.spectral.autocorrelation_min_lag = 5
    config.spectral.autocorrelation_max_lag = 18
    config.fusion.threshold_quantile = 0.88
    config.fusion.threshold_mad = 1.8
    config.fusion.support_z = 0.8
    config.postprocess.border_margin = 24
    config.postprocess.morphology_radius = 2
    config.postprocess.minimum_region_area = 80
    config.postprocess.minimum_hole_area = 40
    return config


def _dataset(cases: list[tuple[int, float, float]]):
    framework = TrainingFreeDetectionFramework(_config())
    targets = []
    truth = []
    for seed, angle, noise in cases:
        image, mask = make_crystal_with_amorphous_region(
            size=128,
            angle_degrees=angle,
            noise_standard_deviation=noise,
            seed=seed,
        )
        targets.append(student_target_from_result(framework.detect(image)))
        truth.append(mask)
    return targets, np.stack(truth)


def _candidate_probability(model, image):
    with torch.no_grad():
        return torch.sigmoid(model(image).candidate_logits).cpu().numpy()[:, 0]


def _metrics(probability: np.ndarray, truth: np.ndarray) -> dict:
    prediction = probability >= 0.5
    intersection = int(np.sum(prediction & truth))
    union = int(np.sum(prediction | truth))
    predicted = int(np.sum(prediction))
    actual = int(np.sum(truth))
    return {
        "iou": intersection / max(union, 1),
        "dice": 2 * intersection / max(predicted + actual, 1),
        "precision": intersection / max(predicted, 1),
        "recall": intersection / max(actual, 1),
    }


def main() -> None:
    torch.manual_seed(23)
    np.random.seed(23)
    random.seed(23)
    torch.set_num_threads(1)

    training_cases = [
        (1, 0.0, 0.07),
        (2, 8.0, 0.09),
        (3, 18.0, 0.11),
        (4, 28.0, 0.13),
        (5, 36.0, 0.10),
        (6, 44.0, 0.12),
    ]
    test_cases = [(31, 13.0, 0.08), (32, 23.0, 0.12), (33, 41.0, 0.14)]
    train_targets, _ = _dataset(training_cases)
    counterfactual_pairs = []
    for seed, angle, noise in training_cases[:3]:
        pair = make_counterfactual_twin(
            "amorphous_region",
            size=128,
            seed=seed + 100,
            nuisance=NuisanceParameters(
                angle_degrees=angle,
                gaussian_noise_standard_deviation=noise,
                blur_sigma=0.7,
            ),
        )
        counterfactual_pairs.append(pair)
        train_targets.extend(student_targets_from_counterfactual_twin(pair))
    test_targets, test_truth = _dataset(test_cases)
    train_batch = stack_student_targets(train_targets)
    test_batch = stack_student_targets(test_targets)

    evidence_model = EvidencePreservingStudent(base_channels=8)
    hard_model = EvidencePreservingStudent(base_channels=8)
    hard_model.load_state_dict(copy.deepcopy(evidence_model.state_dict()))
    evidence_optimizer = torch.optim.Adam(
        evidence_model.parameters(), lr=2e-3
    )
    hard_optimizer = torch.optim.Adam(hard_model.parameters(), lr=2e-3)
    loss_weights = StudentLossWeights(
        geometry=0.0,
        soft_pseudo_mask=1.0,
        violation=0.01,
        evidence=0.02,
        reliability=0.01,
        ood=0.0,
    )
    epochs = 50
    for _ in range(epochs):
        evidence_optimizer.zero_grad()
        evidence_output = evidence_model(train_batch["image"])
        evidence_loss, _ = evidence_preserving_loss(
            evidence_output, train_batch, loss_weights
        )
        evidence_loss.backward()
        evidence_optimizer.step()

        hard_optimizer.zero_grad()
        hard_output = hard_model(train_batch["image"])
        hard_loss = hard_pseudolabel_baseline_loss(
            hard_output,
            train_batch["teacher_soft_mask"] >= 0.5,
            train_batch["supervision_weight"],
        )
        hard_loss.backward()
        hard_optimizer.step()

    evidence_metrics = _metrics(
        _candidate_probability(evidence_model, test_batch["image"]),
        test_truth,
    )
    hard_metrics = _metrics(
        _candidate_probability(hard_model, test_batch["image"]),
        test_truth,
    )
    print(
        json.dumps(
            {
                "benchmark": "evidence_student_vs_hard_pseudolabel",
                "scope": (
                    "held-out synthetic rotation/noise conditions; "
                    "not cross-material or external validation"
                ),
                "random_seed": 23,
                "training_cases": len(training_cases),
                "counterfactual_pairs": len(counterfactual_pairs),
                "test_cases": len(test_cases),
                "epochs": epochs,
                "evidence_preserving_student": evidence_metrics,
                "hard_pseudolabel_student": hard_metrics,
                "iou_difference": (
                    evidence_metrics["iou"] - hard_metrics["iou"]
                ),
                "release_claim_allowed": False,
                "blocking_real_data_requirements": [
                    "independent Gold train/calibration assets",
                    "prospectively locked external material/lab assets",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
