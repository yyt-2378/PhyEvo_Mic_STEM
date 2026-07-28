from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import yaml

from tfree_detection.config import DetectorConfig, load_config
from tfree_detection.context import IdealSkeleton, SkeletonNode
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.preprocessing import list_emd_datasets, load_grayscale
from tfree_detection.synthetic import (
    make_atomic_lattice_with_point_defects,
    make_lattice_with_stacking_fault,
    make_two_phase_interface,
)
from tfree_detection.tasks.atomic import _infer_auto_sublattices


def _overlap(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sum(prediction & truth)) / max(
        int(np.sum(prediction)), 1
    )


def test_all_multitask_configs_load() -> None:
    root = Path(__file__).resolve().parents[1]
    for filename, task in [
        ("atomic_point_defect.yaml", "atomic_point_defect"),
        ("stacking_fault.yaml", "stacking_fault"),
        ("phase_boundary.yaml", "phase_boundary"),
    ]:
        assert load_config(root / "configs" / filename).task == task


def test_framework_registers_five_physical_tasks() -> None:
    framework = TrainingFreeDetectionFramework(DetectorConfig())
    assert framework.available_tasks == (
        "amorphous_region",
        "atomic_point_defect",
        "phase_boundary",
        "stacking_fault",
        "symmetry_relation",
    )


def test_atomic_point_defects_are_localized_with_skeleton() -> None:
    image, truth, metadata = make_atomic_lattice_with_point_defects()
    config = DetectorConfig(task="atomic_point_defect")
    config.spectral.min_frequency = 0.02
    config.spectral.max_frequency = 0.38
    config.atomic.peak_min_distance = 7
    config.atomic.peak_threshold_quantile = 0.70
    config.atomic.site_anomaly_z = 3.0
    config.atomic.displacement_z = 2.5
    config.validate()
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
    assert len(result.regions) >= 3
    assert _overlap(result.mask, truth) >= 0.45
    detected_types = {region.defect_type for region in result.regions}
    assert "vacancy_or_low_intensity_column" in detected_types
    assert "displaced_atom_column" in detected_types


def test_stacking_fault_phase_slip_is_localized() -> None:
    image, truth = make_lattice_with_stacking_fault()
    config = DetectorConfig(task="stacking_fault")
    config.spectral.min_frequency = 0.02
    config.spectral.max_frequency = 0.30
    config.spectral.global_peak_count = 16
    config.line_defect.threshold_quantile = 0.94
    config.line_defect.threshold_mad = 2.4
    config.line_defect.minimum_region_area = 30
    config.validate()
    result = TrainingFreeDetectionFramework(config).detect(image)
    assert np.median(result.score_map[truth]) > np.percentile(
        result.score_map[~truth], 85
    )
    assert np.any(result.mask & truth)
    assert _overlap(result.mask, truth) >= 0.35
    expected_support = int(
        np.ceil(
            3.0
            / (
                2
                * np.pi
                * config.line_defect.frequency_filter_sigma
            )
        )
    )
    assert result.metadata["valid_border_margin_pixels"] >= expected_support


def test_two_phase_interface_is_traced() -> None:
    image, truth = make_two_phase_interface()
    config = DetectorConfig(task="phase_boundary")
    config.spectral.min_frequency = 0.025
    config.spectral.max_frequency = 0.25
    config.phase_boundary.patch_size = 64
    config.phase_boundary.stride = 10
    config.phase_boundary.minimum_signature_separation = 0.65
    config.validate()
    result = TrainingFreeDetectionFramework(config).detect(image)
    assert result.metadata["genuine_two_phase"]
    assert np.any(result.mask & truth)
    assert _overlap(result.mask, truth) >= 0.35


def test_ndata1_zip_loader(tmp_path: Path) -> None:
    target = tmp_path / "sample.ndata1"
    buffer = BytesIO()
    np.save(buffer, np.arange(64 * 64, dtype=np.float32).reshape(64, 64))
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("data.npy", buffer.getvalue())
    loaded = load_grayscale(target)
    assert loaded.shape == (64, 64)
    assert loaded.dtype == np.float32


def test_ndata1_stack_requires_explicit_frame_or_projection(
    tmp_path: Path,
) -> None:
    target = tmp_path / "stack.ndata1"
    stack = np.stack(
        [
            np.full((64, 64), 1.0, dtype=np.float32),
            np.full((64, 64), 3.0, dtype=np.float32),
            np.full((64, 64), 5.0, dtype=np.float32),
        ]
    )
    buffer = BytesIO()
    np.save(buffer, stack)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("data.npy", buffer.getvalue())

    with pytest.raises(ValueError, match="choose frame_index or projection"):
        load_grayscale(target)
    assert np.all(load_grayscale(target, frame_index=1) == 3.0)
    assert np.all(load_grayscale(target, projection="mean") == 3.0)
    with pytest.raises(ValueError, match="either frame_index or projection"):
        load_grayscale(target, frame_index=0, projection="mean")


def test_emd_loader_refuses_to_guess_detector_channel(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    target = tmp_path / "multi-channel.emd"
    paths = (
        "Data/Image/channel-a/Data",
        "Data/Image/channel-b/Data",
    )
    with h5py.File(target, "w") as archive:
        archive.create_dataset(
            paths[0],
            data=np.stack(
                [
                    np.full((64, 64), 2, dtype=np.uint16),
                    np.full((64, 64), 4, dtype=np.uint16),
                ],
                axis=2,
            ),
        )
        archive.create_dataset(
            paths[1],
            data=np.full((64, 64), 7, dtype=np.uint16),
        )
    listing = list_emd_datasets(target)
    assert [item["dataset_path"] for item in listing] == list(paths)
    assert listing[0]["frame_axis"] == 2
    with pytest.raises(ValueError, match="multiple image datasets"):
        load_grayscale(target, projection="mean")
    projected = load_grayscale(
        target,
        dataset_path=paths[0],
        projection="mean",
    )
    assert projected.shape == (64, 64)
    assert np.all(projected == 3)
    single = load_grayscale(target, dataset_path=paths[1])
    assert np.all(single == 7)


def test_emd_listing_decodes_detector_and_scale_metadata(
    tmp_path: Path,
) -> None:
    import json

    h5py = pytest.importorskip("h5py")
    target = tmp_path / "metadata.emd"
    path = "Data/Image/channel-a"
    payload = {
        "Instrument": {"InstrumentModel": "TestScope"},
        "Optics": {
            "AccelerationVoltage": 200000,
            "IlluminationMode": "Probe",
        },
        "BinaryResult": {
            "Detector": "HAADF",
            "DetectorIndex": "6",
            "PixelSize": {"width": "1e-11", "height": "2e-11"},
            "PixelUnitX": "m",
            "PixelUnitY": "m",
        },
    }
    encoded = json.dumps(payload).encode("utf-8")
    metadata = np.zeros((1024, 2), dtype=np.uint8)
    metadata[: len(encoded), :] = np.frombuffer(
        encoded, dtype=np.uint8
    )[:, None]
    with h5py.File(target, "w") as archive:
        archive.create_dataset(
            f"{path}/Data",
            data=np.zeros((64, 64, 2), dtype=np.uint16),
        )
        archive.create_dataset(f"{path}/Metadata", data=metadata)

    listing = list_emd_datasets(target)
    assert listing[0]["metadata_status"] == "parsed"
    assert listing[0]["detector_name"] == "HAADF"
    assert listing[0]["instrument_model"] == "TestScope"
    assert listing[0]["acceleration_voltage_v"] == 200000.0
    assert listing[0]["pixel_size_x"] == 1e-11
    assert listing[0]["detector_stable_across_stack"] is True


def test_auto_sublattice_inference_finds_two_fractional_modes() -> None:
    basis = np.asarray([[16.0, 0.0], [0.0, 16.0]])
    points = []
    for first in range(16):
        for second in range(16):
            cell = basis @ np.asarray([first, second], dtype=float)
            points.append(cell)
            points.append(cell + basis @ np.asarray([0.5, 0.25]))
    inferred = _infer_auto_sublattices(
        np.asarray(points),
        basis,
        (256, 256),
        maximum_sublattices=4,
        histogram_bins=32,
    )
    assert inferred.shape == (2, 2)
    relative = np.minimum(inferred[1], 1 - inferred[1])
    assert np.allclose(sorted(relative), [0.25, 0.5], atol=0.05)


def test_public_dataset_manifest_has_pinned_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (root / "datasets" / "manifest.yaml").read_text(encoding="utf-8")
    )
    by_id = {item["id"]: item for item in payload["datasets"]}
    dataset = by_id["guo_2021_atomic_defects"]
    assert dataset["doi"] == "10.5281/zenodo.5520169"
    assert dataset["license"] == "CC-BY-4.0"
    assert dataset["labels"]["pixel_ground_truth"] is False
    assert len(dataset["files"]) == 4
    for entry in dataset["files"]:
        assert len(entry["md5"]) == 32
        assert entry["bytes"] > 1_000_000

    superscan = next(
        item
        for item in dataset["files"]
        if item["name"] == "Recording_of_SuperScan_img_stack.ndata1"
    )
    assert superscan["bytes"] == 104862360
    assert "temporal_defect_evolution" in superscan["tasks"]

    sigma3 = by_id["al_sigma3_twin_boundaries"]
    assert sigma3["doi"] == "10.5281/zenodo.6414955"
    assert sigma3["license"] == "CC-BY-4.0"
    assert sigma3["labels"]["pixel_ground_truth"] is False
    assert len(sigma3["files"]) == 5

    locked = by_id["dennler_2021_hrstem_defects"]
    assert locked["doi"] == "10.5281/zenodo.5792917"
    assert locked["license"] == "CC-BY-4.0"
    assert locked["labels"]["pixel_ground_truth"] is True
    assert locked["labels"]["source"] == "microscopy_expert"
    assert locked["labels"]["labeled_defective_images"] == 31
    assert len(locked["files"][0]["sha256"]) == 64
