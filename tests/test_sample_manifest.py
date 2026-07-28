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
    summary = summarize_sample_records(records, registry=registry)
    assert summary["asset_count"] == 1
    assert summary["frame_selection_required"] == 1
