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

from .governance import DataAsset, DataRegistry
from .preprocessing import list_emd_datasets


SAMPLE_MANIFEST_FORMAT = "tfree-sample-manifest-v1"
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
    if asset.parent_asset_id is not None:
        return "annotation"
    relative = Path(asset.relative_path)
    if relative.parent.name.lower() in {"label", "labels", "annotation"}:
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
    reasons = []
    if role != "image":
        reasons.append("not_a_direct_image_asset")
    if not executable_tasks:
        reasons.append("no_directly_registered_detector_task")
    if channel_selection:
        reasons.append("explicit_detector_channel_required")
    if frame_selection:
        reasons.append("explicit_frame_or_projection_required")
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
                "readiness": _readiness(asset, role, inspection),
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
    return {
        "manifest_format": SAMPLE_MANIFEST_FORMAT,
        "registry_version": registry.registry_version,
        "registry_sha256": registry.content_hash(),
        "asset_count": len(materialized),
        "partition_counts": dict(sorted(by_partition.items())),
        "dataset_counts": dict(sorted(by_dataset.items())),
        "role_counts": dict(sorted(by_role.items())),
        "format_counts": dict(sorted(by_format.items())),
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
