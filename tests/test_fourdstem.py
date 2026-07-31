import json
import hashlib

import numpy as np
import pytest
import tifffile

from tfree_detection.fourdstem import (
    derive_fast_mapping_maps,
    inspect_fast_mapping_h5,
)
from tfree_detection.governance import (
    DataAsset,
    DataPartition,
    DataRegistry,
)
from tfree_detection.sample_manifest import (
    build_sample_records,
    inspect_asset_file,
)


h5py = pytest.importorskip("h5py")


def _write_cube(path, *, shape=(5, 7, 12, 12)):
    yy, xx = np.indices(shape[2:])
    radius = np.hypot(
        yy - (shape[2] - 1) / 2,
        xx - (shape[3] - 1) / 2,
    )
    cube = np.empty(shape, dtype=np.float32)
    for row in range(shape[0]):
        for column in range(shape[1]):
            width = 1.0 if column < shape[1] // 2 else 2.5
            cube[row, column] = np.exp(-(radius / width) ** 2)
            cube[row, column, row % shape[2], column % shape[3]] += 0.5
    with h5py.File(path, "w") as archive:
        archive.create_dataset("X4D", data=cube)
        coordinates = archive.create_group("coordinates")
        coordinates.create_dataset("x_real", data=np.arange(shape[0]))
        coordinates.create_dataset("y_real", data=np.arange(shape[1]))
        coordinates.create_dataset(
            "x_reciprocal", data=np.arange(shape[2])
        )
        coordinates.create_dataset(
            "y_reciprocal", data=np.arange(shape[3])
        )
        coordinates.create_dataset(
            "beamstop", data=np.zeros(shape[2:], dtype=np.uint8)
        )
        archive.create_dataset(
            "metadata",
            data=json.dumps({"instrument": "synthetic-test"}),
        )


def test_inspect_fast_mapping_h5_records_explicit_axes(tmp_path):
    source = tmp_path / "cube.h5"
    _write_cube(source)
    inspection = inspect_fast_mapping_h5(source)
    assert inspection["format"] == "hdf5_4dstem"
    assert inspection["shape"] == [5, 7, 12, 12]
    assert inspection["axis_semantics"] == [
        "real_x",
        "real_y",
        "reciprocal_x",
        "reciprocal_y",
    ]
    assert inspect_asset_file(source)["schema"] == "werbrouck_fast_mapping_v1"


def test_derive_fast_mapping_maps_preserves_scan_grid(tmp_path):
    source = tmp_path / "cube.h5"
    _write_cube(source)
    maps = derive_fast_mapping_maps(
        source, chunk_rows=2, radial_bins=3, angular_bins=8
    )
    assert set(maps) == {
        "integrated_intensity",
        "diffraction_entropy",
        "radial_first_moment",
        "diffraction_peak_fraction",
        "signature_pc1",
        "signature_anomaly",
    }
    assert all(values.shape == (5, 7) for values in maps.values())
    assert all(np.all(np.isfinite(values)) for values in maps.values())
    assert float(np.std(maps["signature_pc1"])) > 0


def test_fast_mapping_schema_refuses_axis_guessing(tmp_path):
    source = tmp_path / "ambiguous.h5"
    with h5py.File(source, "w") as archive:
        archive.create_dataset("X4D", data=np.ones((4, 4, 8, 8)))
    with pytest.raises(ValueError, match="missing"):
        inspect_fast_mapping_h5(source)


def test_derived_virtual_detector_is_an_image_not_an_annotation(tmp_path):
    source = tmp_path / "cube.h5"
    _write_cube(source)
    derived = tmp_path / "standardized_v1" / "signature_pc1.tif"
    derived.parent.mkdir()
    tifffile.imwrite(derived, np.ones((24, 32), dtype=np.float32))

    def checksum(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    registry = DataRegistry(
        registry_version="test-4dstem",
        split_unit="explicit_group",
        assets=[
            DataAsset(
                asset_id="raw",
                dataset_id="fourdstem",
                relative_path="cube.h5",
                partition=DataPartition.DEVELOPMENT_TEST,
                split_group_id="acquisition",
                acquisition_id="acquisition",
                sample_id="sample",
                lab_id="lab",
                tasks=("virtual_detector",),
                checksum=checksum(source),
            ),
            DataAsset(
                asset_id="derived",
                dataset_id="fourdstem",
                relative_path="standardized_v1/signature_pc1.tif",
                partition=DataPartition.DEVELOPMENT_TEST,
                split_group_id="acquisition",
                acquisition_id="acquisition",
                sample_id="sample",
                lab_id="lab",
                tasks=("phase_boundary",),
                checksum=checksum(derived),
                parent_asset_id="raw",
            ),
        ],
    )
    records = build_sample_records(registry, tmp_path)
    by_id = {record["asset"]["asset_id"]: record for record in records}
    assert by_id["raw"]["asset"]["role"] == "image"
    assert by_id["raw"]["readiness"]["can_run_without_selection"] is False
    raw_interface = by_id["raw"]["data_interface"]
    assert raw_interface["access_mode"] == "derive_evidence_map"
    assert raw_interface["adapter_status"] == "derivation_required"
    assert raw_interface["canonical_output"]["axes"] == [
        "scan_x",
        "scan_y",
    ]
    assert raw_interface["required_selectors"][0]["name"] == "evidence_map"
    assert by_id["derived"]["asset"]["role"] == "image"
    assert (
        by_id["derived"]["readiness"]["can_run_without_selection"] is True
    )
    assert by_id["derived"]["data_interface"]["access_mode"] == "direct_2d"
    assert by_id["derived"]["data_interface"]["ready_for_detector_now"] is True
