import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from skimage.io import imread, imsave

from tfree_detection.locked_evaluation import (
    create_protocol_seal,
    generate_locked_predictions,
    score_locked_predictions,
    verify_protocol_seal,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    (root / "configs").mkdir(parents=True)
    (root / "datasets").mkdir()
    (root / "benchmarks").mkdir()
    (root / "src").mkdir()
    archive = root / "datasets" / "locked.zip"
    archive.write_bytes(b"checksum-frozen-archive")
    config = root / "configs" / "amorphous.yaml"
    config.write_text("task: amorphous_region\n", encoding="utf-8")
    frozen_source = root / "src" / "detector.py"
    frozen_source.write_text("VERSION = 1\n", encoding="utf-8")
    registry = {
        "registry_version": "test-1",
        "split_unit": "explicit_group",
        "assets": [
            {
                "asset_id": "locked-asset",
                "dataset_id": "synthetic",
                "relative_path": "datasets/locked.zip",
                "partition": "locked_external_test",
                "split_group_id": "locked-group",
                "acquisition_id": "acquisition",
                "sample_id": "sample",
                "lab_id": "lab",
                "microscope_id": "scope",
                "material": "test",
                "tasks": ["amorphous_region"],
                "checksum_algorithm": "sha256",
                "checksum": _sha256(archive),
                "eligible_for_active_learning": False,
            }
        ],
    }
    registry_path = root / "datasets" / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    protocol = {
        "protocol_format": "phyevo-mic-locked-protocol-v1",
        "protocol_id": "synthetic-locked-v1",
        "asset_id": "locked-asset",
        "registry_path": "datasets/registry.yaml",
        "archive_sha256": _sha256(archive),
        "dataset_layout": {
            "defective_images": "tree/defective/images",
            "defective_labels": "tree/defective/labels",
            "normal_images": "tree/normal/images",
            "expected_defective_images": 1,
            "expected_normal_images": 1,
        },
        "detectors": [
            {
                "id": "amorphous_region",
                "config_path": "configs/amorphous.yaml",
            }
        ],
        "targets": [
            {
                "id": "amorphous_composite",
                "role": "primary",
                "truth_classes": [2],
                "prediction": {
                    "operation": "single",
                    "members": ["amorphous_region"],
                },
            }
        ],
        "metrics": {"boundary_tolerance_pixels": 1.0},
        "frozen_files": [
            "configs/amorphous.yaml",
            "src/detector.py",
        ],
        "prohibited_actions": ["no threshold tuning"],
        "interpretation_limits": ["synthetic protocol fixture"],
    }
    protocol_path = root / "benchmarks" / "protocol.yaml"
    protocol_path.write_text(
        yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8"
    )
    seal_path = root / "benchmarks" / "protocol_seal.json"
    return protocol_path, seal_path, archive


def _write_dataset(root: Path) -> Path:
    dataset = root / "datasets" / "unpacked"
    defective_images = dataset / "tree" / "defective" / "images"
    defective_labels = dataset / "tree" / "defective" / "labels"
    normal_images = dataset / "tree" / "normal" / "images"
    defective_images.mkdir(parents=True)
    defective_labels.mkdir(parents=True)
    normal_images.mkdir(parents=True)
    image = np.zeros((20, 20), dtype=np.uint8)
    image[6:14, 6:14] = 200
    label = np.zeros((20, 20), dtype=np.uint8)
    label[2:5, 2:5] = 127
    label[6:14, 6:14] = 255
    imsave(defective_images / "defect.tif", image, check_contrast=False)
    imsave(defective_labels / "defect.tif", label, check_contrast=False)
    imsave(
        normal_images / "normal.tif",
        np.full((20, 20), 20, dtype=np.uint8),
        check_contrast=False,
    )
    return dataset


class _FakeFramework:
    def detect(self, source: Path):
        image = np.asarray(imread(str(source)))
        mask = image > 100
        shape = tuple(mask.shape)
        return SimpleNamespace(
            mask=mask,
            image_shape=shape,
            reliability_map=np.ones(shape, dtype=np.float32),
            decision_map=np.zeros(shape, dtype=np.uint8),
        )


def test_locked_prediction_is_sealed_before_label_scoring(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    protocol, seal, _ = _write_fixture(root)
    created = create_protocol_seal(
        protocol_path=protocol,
        seal_path=seal,
        project_root=root,
        forbidden_unpacked_path="datasets/unpacked",
    )
    assert created["state_at_seal"]["locked_labels_unpacked"] is False
    verified_protocol, _ = verify_protocol_seal(
        protocol_path=protocol,
        seal_path=seal,
        project_root=root,
    )
    assert verified_protocol.protocol_id == "synthetic-locked-v1"
    dataset = _write_dataset(root)
    predictions = root / "benchmarks" / "predictions-a"
    generated = generate_locked_predictions(
        protocol_path=protocol,
        seal_path=seal,
        project_root=root,
        dataset_root=dataset,
        output_directory=predictions,
        detector_factory=lambda _detector_id, _config: _FakeFramework(),
    )
    assert generated["manifest"]["counts"]["successful_predictions"] == 2
    assert generated["manifest"]["label_access"]["label_pixels_read"] is False

    score = score_locked_predictions(
        protocol_path=protocol,
        seal_path=seal,
        project_root=root,
        dataset_root=dataset,
        prediction_directory=predictions,
        output_directory=root / "benchmarks" / "score-a",
        ledger_path=root / "benchmarks" / "unblinding.jsonl",
    )
    metrics = score["targets"]["amorphous_composite"]
    assert metrics["micro_pixel_metrics"]["iou"] == 1.0
    assert (
        metrics["normal_controls"]["mean_false_positive_area_fraction"]
        == 0.0
    )
    assert score["dataset"]["raw_label_values_sorted"] == [0.0, 127.0, 255.0]


def test_unblinding_ledger_rejects_replacement_predictions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    protocol, seal, _ = _write_fixture(root)
    create_protocol_seal(
        protocol_path=protocol,
        seal_path=seal,
        project_root=root,
        forbidden_unpacked_path="datasets/unpacked",
    )
    dataset = _write_dataset(root)
    ledger = root / "benchmarks" / "unblinding.jsonl"
    for suffix in ("a", "b"):
        predictions = root / "benchmarks" / f"predictions-{suffix}"
        generate_locked_predictions(
            protocol_path=protocol,
            seal_path=seal,
            project_root=root,
            dataset_root=dataset,
            output_directory=predictions,
            detector_factory=lambda _detector_id, _config: _FakeFramework(),
        )
        if suffix == "a":
            score_locked_predictions(
                protocol_path=protocol,
                seal_path=seal,
                project_root=root,
                dataset_root=dataset,
                prediction_directory=predictions,
                output_directory=root / "benchmarks" / "score-a",
                ledger_path=ledger,
            )
        else:
            with pytest.raises(RuntimeError, match="already unblinded"):
                score_locked_predictions(
                    protocol_path=protocol,
                    seal_path=seal,
                    project_root=root,
                    dataset_root=dataset,
                    prediction_directory=predictions,
                    output_directory=root / "benchmarks" / "score-b",
                    ledger_path=ledger,
                )


def test_protocol_seal_detects_frozen_file_change(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    protocol, seal, _ = _write_fixture(root)
    create_protocol_seal(
        protocol_path=protocol,
        seal_path=seal,
        project_root=root,
        forbidden_unpacked_path="datasets/unpacked",
    )
    (root / "src" / "detector.py").write_text(
        "VERSION = 2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Frozen implementation changed"):
        verify_protocol_seal(
            protocol_path=protocol,
            seal_path=seal,
            project_root=root,
        )
