from pathlib import Path

import pytest

from tfree_detection.config import load_config
from tfree_detection.context import IdealSkeleton


def test_example_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "amorphous.yaml"
    config = load_config(path)
    assert config.task == "amorphous_region"
    assert config.spectral.patch_size == 128
    assert config.postprocess.exclude_border_touching


def test_stage2_example_config_enables_phase_and_dino() -> None:
    path = Path(__file__).parents[1] / "configs" / "amorphous_stage2.yaml"
    config = load_config(path)
    assert config.stage2.enabled
    assert config.stage2.phase.enabled
    assert config.stage2.dino.enabled
    assert config.stage2.dino.samples_per_zone == 16


def test_unknown_configuration_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("spectral:\n  nonexistent: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(path)


def test_example_skeleton_contract_loads() -> None:
    path = Path(__file__).parents[1] / "schemas" / "skeleton.example.json"
    skeleton = IdealSkeleton.from_json(path)
    assert skeleton.basis.shape == (2, 2)
    assert len(skeleton.nodes) == 2
    assert skeleton.edges[0].kind == "coordination"
