from pathlib import Path

import numpy as np
import pytest

from tfree_detection.config import DINOConfig, DetectorConfig, PhaseAnalysisConfig
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.stage2.dino import verify_regions_with_dino
from tfree_detection.synthetic import make_crystal_with_amorphous_region


PROJECT_ROOT = Path(__file__).parents[1]
DINO_CHECKPOINT = (
    PROJECT_ROOT
    / "vendor"
    / "models"
    / "dino-vits8"
    / "dino_deitsmall8_pretrain.pth"
)


def _stage2_config() -> DetectorConfig:
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
    config.stage2.enabled = True
    config.stage2.phase.fft_padding = 32
    config.stage2.phase.amplitude_background_sigma = 32
    config.stage2.phase.transition_inner_radius = 4
    config.stage2.phase.transition_outer_radius = 20
    config.stage2.phase.reference_inner_radius = 28
    config.stage2.phase.reference_outer_radius = 48
    config.stage2.dino.enabled = False
    config.validate()
    return config


def test_phase_stage_produces_deformation_contract() -> None:
    image, _ = make_crystal_with_amorphous_region(
        size=256, angle_degrees=19, seed=4
    )
    result = TrainingFreeDetectionFramework(_stage2_config()).detect(image)
    stage2 = result.metadata["stage2"]
    assert stage2["enabled"]
    assert len(stage2["phase"]["direct_basis_lengths_pixels"]) == 2
    assert stage2["regions"]
    for name in (
        "gpa_strain_magnitude",
        "gpa_lattice_coherence",
        "gpa_exx",
        "gpa_eyy",
        "gpa_exy",
        "gpa_validity",
    ):
        assert result.evidence_maps[name].shape == image.shape
        assert np.all(np.isfinite(result.evidence_maps[name]))
    validity = result.evidence_maps["gpa_validity"]
    assert validity[0, 0] == 0
    assert validity[image.shape[0] // 2, image.shape[1] // 2] == 1


@pytest.mark.skipif(
    not DINO_CHECKPOINT.is_file(), reason="optional DINO checkpoint is absent"
)
def test_frozen_dino_verifier_runs_without_training() -> None:
    image, mask = make_crystal_with_amorphous_region(size=160, seed=3)
    phase = PhaseAnalysisConfig(
        transition_inner_radius=4,
        transition_outer_radius=12,
        reference_inner_radius=18,
        reference_outer_radius=30,
    )
    dino = DINOConfig(
        enabled=True,
        local_model_directory=str(DINO_CHECKPOINT.parent),
        microscopy_patch_size=64,
        batch_size=2,
        samples_per_zone=2,
    )
    result = verify_regions_with_dino(
        image.astype(np.float32) / 65535.0,
        mask,
        phase,
        dino,
    )
    assert result.metadata["weights_frozen"] is True
    assert result.metadata["training_or_fitting_used"] is False
    assert result.metadata["embedding_dimension"] == 384
    assert result.region_metrics
    assert result.region_metrics[0]["cosine_distance"]["reference"]["count"] == 2
