from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from tfree_detection.governance import DataAsset, DataPartition, DataRegistry
from tfree_detection.sample_manifest import (
    build_sample_records,
    inspect_asset_file,
    summarize_sample_records,
)


def test_ndata_inspection_reads_header_and_safe_acquisition_metadata(
    tmp_path: Path,
) -> None:
    target = tmp_path / "movie.ndata1"
    buffer = BytesIO()
    np.save(buffer, np.zeros((3, 32, 48), dtype=np.float32))
    metadata = {
        "is_sequence": True,
        "title": "test movie",
        "metadata": {
            "instrument": {"high_tension": 60000.0},
            "hardware_source": {
                "hardware_source_name": "SuperScan",
                "channel_name": "a",
            },
            "scan": {"fov_nm": 8.0, "scan_size": [48, 32]},
        },
    }
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("data.npy", buffer.getvalue())
        archive.writestr("metadata.json", json.dumps(metadata))

    inspected = inspect_asset_file(target)
    assert inspected["shape"] == [3, 32, 48]
    assert inspected["frame_axis"] == 0
    assert inspected["metadata"]["acceleration_voltage_v"] == 60000.0
    assert inspected["metadata"]["hardware_source_name"] == "SuperScan"


def test_sample_manifest_preserves_partition_and_requires_stack_selection(
    tmp_path: Path,
) -> None:
    target = tmp_path / "movie.ndata1"
    buffer = BytesIO()
    np.save(buffer, np.zeros((2, 32, 32), dtype=np.uint16))
    with ZipFile(target, "w") as archive:
        archive.writestr("data.npy", buffer.getvalue())
    import hashlib

    registry = DataRegistry(
        registry_version="test-v1",
        split_unit="explicit_group",
        assets=[
            DataAsset(
                asset_id="movie",
                dataset_id="test",
                relative_path="movie.ndata1",
                partition=DataPartition.DEVELOPMENT_TEST,
                split_group_id="acquisition",
                acquisition_id="acquisition",
                sample_id="sample",
                lab_id="lab",
                tasks=("atomic_point_defect",),
                checksum=hashlib.sha256(target.read_bytes()).hexdigest(),
            )
        ],
    )
    records = build_sample_records(registry, tmp_path)
    assert records[0]["asset"]["partition"] == "development_test"
    assert records[0]["readiness"]["frame_selection_required"] is True
    assert records[0]["readiness"]["can_run_without_selection"] is False
    interface = records[0]["data_interface"]
    assert interface["interface_format"] == "tfree-data-interface-v1"
    assert interface["access_mode"] == "select_plane"
    assert interface["adapter_status"] == "selector_required"
    assert interface["canonical_output"]["kind"] == "image_2d"
    assert interface["required_selectors"][0]["parameters"] == [
        "frame_index",
        "projection",
    ]
    summary = summarize_sample_records(records, registry=registry)
    assert summary["asset_count"] == 1
    assert summary["dataset_count"] == 1
    assert summary["total_registered_bytes"] == target.stat().st_size
    assert summary["material_counts"] == {"unknown": 1}
    assert summary["task_counts"] == {"atomic_point_defect": 1}
    assert summary["frame_selection_required"] == 1
    assert summary["adapter_status_counts"] == {"selector_required": 1}
    assert summary["ready_for_detector_now"] == 0


def test_generic_hdf5_is_inventoried_without_guessing_fourdstem_axes(
    tmp_path: Path,
) -> None:
    import pytest

    h5py = pytest.importorskip("h5py")
    target = tmp_path / "calibration.h5"
    with h5py.File(target, "w") as archive:
        group = archive.create_group("calibration")
        group.create_dataset("gain", data=np.ones((4, 5)))

    inspected = inspect_asset_file(target)
    assert inspected["inspection_status"] == "ok"
    assert inspected["format"] == "hdf5"
    assert inspected["schema"] == "generic_hdf5_axes_not_inferred"
    assert inspected["dataset_count"] == 1
    assert inspected["datasets"][0]["path"] == "calibration/gain"
