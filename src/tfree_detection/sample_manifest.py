"""Deterministic per-asset inspection for executable dataset manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
from zipfile import BadZipFile, ZipFile

import numpy as np
from skimage.io import imread

from .fourdstem import inspect_fast_mapping_h5
from .governance import DataAsset, DataRegistry
from .preprocessing import list_emd_datasets


SAMPLE_MANIFEST_FORMAT = "tfree-sample-manifest-v2"
DATA_INTERFACE_FORMAT = "tfree-data-interface-v1"
DIRECT_IMAGE_FORMATS = {
    "bmp",
    "jpeg",
    "jpg",
    "png",
    "tif",
    "tiff",
}
FAST_MAPPING_OUTPUTS = (
    "integrated_intensity",
    "diffraction_entropy",
    "radial_first_moment",
    "diffraction_peak_fraction",
    "signature_pc1",
    "signature_anomaly",
)
DIRECT_DETECTOR_TASKS = {
    "amorphous_region",
    "atomic_point_defect",
    "stacking_fault",
    "phase_boundary",
}
TASK_ADAPTER_CANDIDATES = {
    "twin_boundary": "phase_boundary",
    "interface_defect": "phase_boundary",
    "grain_boundary": "phase_boundary",
    "atomic_column_localization": "atomic_point_defect",
    "atomic_site_anomaly": "atomic_point_defect",
    "irradiation_damage_region": "amorphous_region",
    "artifact_or_damage_disambiguation": "amorphous_region",
}


def _npy_header(stream: Any) -> Dict[str, Any]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
            stream
        )
    else:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
            stream
        )
    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "fortran_order": bool(fortran_order),
    }


def _safe_ndata_metadata(values: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = values.get("metadata", {})
    instrument = (
        metadata.get("instrument", {}) if isinstance(metadata, dict) else {}
    )
    scan = metadata.get("scan", {}) if isinstance(metadata, dict) else {}
    hardware = (
        metadata.get("hardware_source", {})
        if isinstance(metadata, dict)
        else {}
    )
    return {
        "title": values.get("title"),
        "is_sequence": bool(values.get("is_sequence", False)),
        "spatial_calibrations": values.get("spatial_calibrations"),
        "acceleration_voltage_v": instrument.get("high_tension"),
        "hardware_source_name": hardware.get("hardware_source_name"),
        "detector_channel_id": hardware.get("channel_id"),
        "detector_channel_name": hardware.get("channel_name"),
        "pixel_time_us": hardware.get("pixel_time_us"),
        "scan_fov_nm": scan.get("fov_nm"),
        "scan_rotation_degrees": scan.get("rotation_deg"),
        "scan_size": scan.get("scan_size"),
    }


def _inspect_ndata(path: Path) -> Dict[str, Any]:
    try:
        with ZipFile(path) as archive:
            data_names = [
                name
                for name in archive.namelist()
                if name == "data.npy" or name.endswith("/data.npy")
            ]
            if len(data_names) != 1:
                raise ValueError(
                    "NData archive must contain exactly one data.npy"
                )
            with archive.open(data_names[0]) as stream:
                header = _npy_header(stream)
            metadata_names = [
                name
                for name in archive.namelist()
                if name == "metadata.json" or name.endswith("/metadata.json")
            ]
            metadata: Dict[str, Any] = {}
            if len(metadata_names) == 1:
                values = json.loads(
                    archive.read(metadata_names[0]).decode("utf-8")
                )
                if isinstance(values, dict):
                    metadata = {
                        key: value
                        for key, value in _safe_ndata_metadata(values).items()
                        if value is not None
                    }
    except BadZipFile as error:
        raise ValueError(f"Invalid NData archive: {path}") from error
    shape = header["shape"]
    return {
        "inspection_status": "ok",
        "format": "ndata1",
        **header,
        "frame_axis": 0 if len(shape) == 3 else None,
        "frame_count": int(shape[0]) if len(shape) == 3 else 1,
        "metadata": metadata,
    }


def _inspect_archive(path: Path) -> Dict[str, Any]:
    try:
        with ZipFile(path) as archive:
            files = [item for item in archive.infolist() if not item.is_dir()]
    except BadZipFile as error:
        raise ValueError(f"Invalid ZIP archive: {path}") from error
    suffixes = Counter(
        Path(item.filename).suffix.lower() or "<none>" for item in files
    )
    return {
        "inspection_status": "ok",
        "format": "zip",
        "member_count": len(files),
        "uncompressed_bytes": int(sum(item.file_size for item in files)),
        "compressed_bytes": int(sum(item.compress_size for item in files)),
        "encrypted_member_count": int(
            sum(bool(item.flag_bits & 0x1) for item in files)
        ),
        "member_suffix_counts": dict(sorted(suffixes.items())),
    }


def _inspect_image(path: Path) -> Dict[str, Any]:
    image = np.asarray(imread(str(path)))
    return {
        "inspection_status": "ok",
        "format": path.suffix.lower().lstrip("."),
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "frame_axis": None,
        "frame_count": 1,
    }


def _inspect_generic_hdf5(path: Path) -> Dict[str, Any]:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Generic HDF5 inspection requires the optional 'fourdstem' "
            "dependency (h5py)"
        ) from error

    datasets = []
    with h5py.File(path, "r") as archive:
        root_keys = sorted(str(key) for key in archive.keys())

        def visitor(name: str, value: Any) -> None:
            if not isinstance(value, h5py.Dataset):
                return
            datasets.append(
                {
                    "path": name,
                    "shape": [int(item) for item in value.shape],
                    "dtype": str(value.dtype),
                    "stored_bytes": int(value.id.get_storage_size()),
                }
            )

        archive.visititems(visitor)
    return {
        "inspection_status": "ok",
        "format": "hdf5",
        "schema": "generic_hdf5_axes_not_inferred",
        "root_keys": root_keys,
        "dataset_count": len(datasets),
        "datasets": datasets[:64],
        "dataset_listing_truncated": len(datasets) > 64,
    }


def inspect_asset_file(path: str | Path) -> Dict[str, Any]:
    """Inspect shape/container metadata without changing source data."""

    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".ndata1":
        return _inspect_ndata(target)
    if suffix == ".emd":
        datasets = list_emd_datasets(target)
        return {
            "inspection_status": "ok",
            "format": "emd",
            "channel_count": len(datasets),
            "channels": datasets,
        }
    if suffix in {".h5", ".hdf5"}:
        try:
            return inspect_fast_mapping_h5(target)
        except ValueError:
            return _inspect_generic_hdf5(target)
    if suffix == ".zip":
        return _inspect_archive(target)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return _inspect_image(target)
    if suffix in {".json", ".yaml", ".yml", ".md", ".txt", ".osc"}:
        return {
            "inspection_status": "metadata_only",
            "format": suffix.lstrip("."),
        }
    return {
        "inspection_status": "unsupported_format",
        "format": suffix.lstrip(".") or "no_extension",
    }


def _asset_role(asset: DataAsset) -> str:
    relative = Path(asset.relative_path)
    if "calibration" in relative.stem.lower():
        return "metadata"
    if any(
        part.lower() in {"label", "labels", "annotation", "annotations"}
        for part in relative.parts
    ):
        return "annotation"
    if relative.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".ndata1",
        ".emd",
        ".h5",
        ".hdf5",
        ".dm3",
        ".dm4",
        ".mib",
        ".raw",
    }:
        return "image"
    if relative.suffix.lower() == ".zip":
        return "container"
    return "metadata"


def _readiness(
    asset: DataAsset, role: str, inspection: Mapping[str, Any]
) -> Dict[str, Any]:
    direct_tasks = sorted(DIRECT_DETECTOR_TASKS.intersection(asset.tasks))
    adapter_candidates = {
        task: TASK_ADAPTER_CANDIDATES[task]
        for task in asset.tasks
        if task in TASK_ADAPTER_CANDIDATES
    }
    executable_tasks = sorted(
        set(direct_tasks).union(adapter_candidates.values())
    )
    channel_selection = (
        inspection.get("format") == "emd"
        and int(inspection.get("channel_count", 0)) > 1
    )
    frame_selection = False
    if inspection.get("format") == "ndata1":
        frame_selection = int(inspection.get("frame_count", 1)) > 1
    elif inspection.get("format") == "emd":
        frame_selection = any(
            int(channel.get("shape", [1])[-1]) > 1
            and channel.get("frame_axis") is not None
            for channel in inspection.get("channels", [])
        )
    elif inspection.get("format") == "hdf5_4dstem":
        frame_selection = True
    reasons = []
    if role != "image":
        reasons.append("not_a_direct_image_asset")
    if not executable_tasks:
        reasons.append("no_directly_registered_detector_task")
    if channel_selection:
        reasons.append("explicit_detector_channel_required")
    if frame_selection:
        if inspection.get("format") == "hdf5_4dstem":
            reasons.append("explicit_4dstem_virtual_detector_required")
        else:
            reasons.append("explicit_frame_or_projection_required")
    if inspection.get("inspection_status") != "ok":
        reasons.append("inspection_not_detector_ready")
    return {
        "direct_detector_tasks": direct_tasks,
        "task_adapter_candidates": adapter_candidates,
        "executable_detector_tasks": executable_tasks,
        "channel_selection_required": channel_selection,
        "frame_selection_required": frame_selection,
        "can_run_without_selection": (
            role == "image"
            and bool(executable_tasks)
            and not channel_selection
            and not frame_selection
            and inspection.get("inspection_status") == "ok"
        ),
        "blocking_reasons": reasons,
    }


def _canonical_output(
    kind: str,
    axes: Iterable[str],
    *,
    dtype: str = "float32",
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "axes": list(axes),
        "dtype": dtype,
    }


def build_data_interface(
    role: str,
    inspection: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> Dict[str, Any]:
    """Describe one asset through the stable, non-destructive data contract.

    The contract records how a source can become a canonical detector input.
    It never guesses unknown axes and never rewrites the source asset.
    """

    source_format = str(inspection.get("format", "unknown"))
    interface: Dict[str, Any] = {
        "interface_format": DATA_INTERFACE_FORMAT,
        "access_mode": "adapter_required",
        "adapter": "none",
        "adapter_status": "adapter_required",
        "required_selectors": [],
        "canonical_output": _canonical_output(
            "none", [], dtype="not_applicable"
        ),
        "source_policy": "read_only_checksum_anchored",
        "ready_for_detector_now": False,
    }
    if role == "annotation":
        interface.update(
            {
                "access_mode": "annotation_reference",
                "adapter": "dataset_label_adapter_registry",
                "adapter_status": "reference_only",
                "canonical_output": _canonical_output(
                    "annotation_reference",
                    ["y", "x"]
                    if source_format in DIRECT_IMAGE_FORMATS
                    else [],
                    dtype="source_dependent",
                ),
            }
        )
        return interface
    if role == "metadata":
        interface.update(
            {
                "access_mode": "metadata_only",
                "adapter": "metadata_inventory",
                "adapter_status": "metadata_only",
                "canonical_output": _canonical_output(
                    "metadata", [], dtype="source_dependent"
                ),
            }
        )
        return interface
    if role == "container":
        interface.update(
            {
                "access_mode": "container_adapter_required",
                "adapter": "source_specific_container_adapter",
                "adapter_status": "adapter_required",
                "canonical_output": _canonical_output(
                    "container_members", [], dtype="source_dependent"
                ),
            }
        )
        return interface
    if role != "image":
        return interface

    if source_format in DIRECT_IMAGE_FORMATS:
        interface.update(
            {
                "access_mode": "direct_2d",
                "adapter": "tfree_detection.preprocessing.load_grayscale",
                "adapter_status": "ready",
                "canonical_output": _canonical_output(
                    "image_2d", ["y", "x"]
                ),
                "ready_for_detector_now": bool(
                    readiness.get("can_run_without_selection", False)
                ),
            }
        )
        return interface

    if source_format == "ndata1":
        frame_count = int(inspection.get("frame_count", 1))
        selectors = []
        if frame_count > 1:
            selectors.append(
                {
                    "name": "plane_selection",
                    "parameters": ["frame_index", "projection"],
                    "constraint": "exactly_one",
                    "allowed_projection_values": ["mean", "median", "max"],
                }
            )
        interface.update(
            {
                "access_mode": (
                    "select_plane" if selectors else "direct_2d"
                ),
                "adapter": "tfree_detection.preprocessing.load_grayscale",
                "adapter_status": (
                    "selector_required" if selectors else "ready"
                ),
                "required_selectors": selectors,
                "canonical_output": _canonical_output(
                    "image_2d", ["y", "x"]
                ),
                "ready_for_detector_now": bool(
                    readiness.get("can_run_without_selection", False)
                ),
            }
        )
        return interface

    if source_format == "emd":
        selectors = []
        channels = list(inspection.get("channels", []))
        if int(inspection.get("channel_count", 0)) > 1:
            selectors.append(
                {
                    "name": "dataset_path",
                    "constraint": "exactly_one",
                    "allowed_values": [
                        str(channel["dataset_path"])
                        for channel in channels
                        if channel.get("dataset_path")
                    ],
                }
            )
        if any(channel.get("frame_axis") is not None for channel in channels):
            selectors.append(
                {
                    "name": "plane_selection",
                    "parameters": ["frame_index", "projection"],
                    "constraint": "exactly_one",
                    "allowed_projection_values": ["mean", "median", "max"],
                }
            )
        interface.update(
            {
                "access_mode": (
                    "select_channel_and_plane"
                    if len(selectors) > 1
                    else "select_plane"
                    if selectors
                    else "direct_2d"
                ),
                "adapter": "tfree_detection.preprocessing.load_grayscale",
                "adapter_status": (
                    "selector_required" if selectors else "ready"
                ),
                "required_selectors": selectors,
                "canonical_output": _canonical_output(
                    "image_2d", ["y", "x"]
                ),
                "ready_for_detector_now": bool(
                    readiness.get("can_run_without_selection", False)
                ),
            }
        )
        return interface

    if source_format == "hdf5_4dstem":
        interface.update(
            {
                "access_mode": "derive_evidence_map",
                "adapter": (
                    "tfree_detection.fourdstem.derive_fast_mapping_maps"
                ),
                "adapter_status": "derivation_required",
                "required_selectors": [
                    {
                        "name": "evidence_map",
                        "constraint": "exactly_one",
                        "allowed_values": list(FAST_MAPPING_OUTPUTS),
                    }
                ],
                "canonical_output": _canonical_output(
                    "evidence_map_2d", ["scan_x", "scan_y"]
                ),
            }
        )
        return interface

    return interface


def build_sample_records(
    registry: DataRegistry, project_root: str | Path
) -> list[Dict[str, Any]]:
    root = Path(project_root).resolve()
    records = []
    for asset in sorted(registry.assets, key=lambda item: item.asset_id):
        target = (root / asset.relative_path).resolve()
        target.relative_to(root)
        role = _asset_role(asset)
        inspection = inspect_asset_file(target)
        readiness = _readiness(asset, role, inspection)
        records.append(
            {
                "manifest_format": SAMPLE_MANIFEST_FORMAT,
                "registry_version": registry.registry_version,
                "registry_sha256": registry.content_hash(),
                "asset": {
                    **asset.to_dict(),
                    "role": role,
                    "bytes": target.stat().st_size,
                },
                "inspection": inspection,
                "readiness": readiness,
                "data_interface": build_data_interface(
                    role, inspection, readiness
                ),
            }
        )
    return records


def summarize_sample_records(
    records: Iterable[Mapping[str, Any]],
    *,
    registry: DataRegistry,
) -> Dict[str, Any]:
    materialized = list(records)
    by_partition = Counter(
        record["asset"]["partition"] for record in materialized
    )
    by_dataset = Counter(
        record["asset"]["dataset_id"] for record in materialized
    )
    by_role = Counter(record["asset"]["role"] for record in materialized)
    by_format = Counter(
        record["inspection"]["format"] for record in materialized
    )
    by_material = Counter(
        record["asset"]["material"] for record in materialized
    )
    by_task = Counter(
        task
        for record in materialized
        for task in record["asset"]["tasks"]
    )
    dataset_bytes = Counter()
    partition_bytes = Counter()
    for record in materialized:
        size = int(record["asset"]["bytes"])
        dataset_bytes[record["asset"]["dataset_id"]] += size
        partition_bytes[record["asset"]["partition"]] += size
    by_access_mode = Counter(
        record["data_interface"]["access_mode"] for record in materialized
    )
    by_adapter_status = Counter(
        record["data_interface"]["adapter_status"]
        for record in materialized
    )
    by_output_kind = Counter(
        record["data_interface"]["canonical_output"]["kind"]
        for record in materialized
    )
    return {
        "manifest_format": SAMPLE_MANIFEST_FORMAT,
        "data_interface_format": DATA_INTERFACE_FORMAT,
        "registry_version": registry.registry_version,
        "registry_sha256": registry.content_hash(),
        "asset_count": len(materialized),
        "dataset_count": len(by_dataset),
        "total_registered_bytes": sum(
            int(record["asset"]["bytes"]) for record in materialized
        ),
        "partition_counts": dict(sorted(by_partition.items())),
        "partition_bytes": dict(sorted(partition_bytes.items())),
        "dataset_counts": dict(sorted(by_dataset.items())),
        "dataset_bytes": dict(sorted(dataset_bytes.items())),
        "material_counts": dict(sorted(by_material.items())),
        "task_counts": dict(sorted(by_task.items())),
        "role_counts": dict(sorted(by_role.items())),
        "format_counts": dict(sorted(by_format.items())),
        "access_mode_counts": dict(sorted(by_access_mode.items())),
        "adapter_status_counts": dict(sorted(by_adapter_status.items())),
        "canonical_output_counts": dict(sorted(by_output_kind.items())),
        "channel_selection_required": sum(
            bool(record["readiness"]["channel_selection_required"])
            for record in materialized
        ),
        "frame_selection_required": sum(
            bool(record["readiness"]["frame_selection_required"])
            for record in materialized
        ),
        "runnable_without_selection": sum(
            bool(record["readiness"]["can_run_without_selection"])
            for record in materialized
        ),
        "ready_for_detector_now": sum(
            bool(record["data_interface"]["ready_for_detector_now"])
            for record in materialized
        ),
    }


def sha256_records(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()
