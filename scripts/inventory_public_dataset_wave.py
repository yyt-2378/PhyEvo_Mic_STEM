#!/usr/bin/env python
"""Inventory source containers in place without extracting large payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import h5py
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets" / "public"
DATASET_IDS = (
    "zhang_2026_wse2_3d_defects_coordinates",
    "strauch_2021_srtio3_high_resolution_4dstem",
    "mills_2022_irradiated_al_4dstem",
    "wang_2026_polymer_lamella_4dstem_rt",
)


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zip_inventory(path: Path) -> dict[str, Any]:
    try:
        with ZipFile(path) as archive:
            members = [
                {
                    "member_path": info.filename,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "is_directory": info.is_dir(),
                }
                for info in archive.infolist()
            ]
    except BadZipFile as error:
        raise ValueError(f"Invalid ZIP archive: {path}") from error
    logical_bytes = sum(
        item["uncompressed_bytes"]
        for item in members
        if not item["is_directory"]
    )
    ratio = logical_bytes / max(path.stat().st_size, 1)
    return {
        "container": "zip",
        "member_count": len(members),
        "logical_uncompressed_bytes": logical_bytes,
        "logical_to_archive_ratio": ratio,
        "bulk_extraction_policy": (
            "do_not_bulk_extract"
            if ratio >= 50.0
            else "extract_only_with_adapter_and_space_check"
        ),
        "members": members,
    }


def _hdf5_inventory(path: Path) -> dict[str, Any]:
    datasets = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset):
                datasets.append(
                    {
                        "path": name,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "stored_bytes": int(value.id.get_storage_size()),
                    }
                )

        handle.visititems(visitor)
    return {
        "container": "hdf5",
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def _dm4_inventory(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(16)
    return {
        "container": "dm4",
        "header_hex": header.hex(),
        "adapter_status": "container_verified_parser_not_yet_available",
    }


def _container_inventory(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return _zip_inventory(path)
    if suffix in {".h5", ".hdf5"}:
        return _hdf5_inventory(path)
    if suffix == ".dm4":
        return _dm4_inventory(path)
    return {"container": suffix.lstrip(".") or "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/manifest.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/standardized/public_dataset_wave_v1.json"
        ),
    )
    arguments = parser.parse_args()
    manifest_path = _inside(arguments.manifest)
    output = _inside(arguments.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a versioned inventory")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in manifest["datasets"]}

    datasets = []
    for dataset_id in DATASET_IDS:
        dataset = by_id[dataset_id]
        files = []
        for entry in dataset["files"]:
            path = _inside(DATA_ROOT / dataset_id / entry["name"])
            if (
                not path.is_file()
                or path.stat().st_size != int(entry["bytes"])
            ):
                raise ValueError(f"Missing or incomplete source: {path}")
            if _digest(path, "md5") != str(entry["md5"]).lower():
                raise ValueError(f"MD5 mismatch: {path}")
            files.append(
                {
                    "name": entry["name"],
                    "source_name": entry.get("source_name", entry["name"]),
                    "relative_path": path.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    "bytes": path.stat().st_size,
                    "md5": entry["md5"],
                    "sha256": _digest(path, "sha256"),
                    "role": entry.get("role", "metadata"),
                    "tasks": entry.get("tasks", []),
                    "inventory": _container_inventory(path),
                }
            )
        datasets.append(
            {
                "dataset_id": dataset_id,
                "title": dataset["title"],
                "source_revision": dataset["source_revision"],
                "license": dataset["license"],
                "file_count": len(files),
                "source_bytes": sum(item["bytes"] for item in files),
                "files": files,
            }
        )
    payload = {
        "inventory_format": "tfree-public-dataset-wave-v1",
        "source_file_policy": "verified_in_place_never_overwritten",
        "bulk_archive_policy": (
            "archive members are inventoried without extraction; containers "
            "with logical/archive ratio >= 50 are never bulk-extracted"
        ),
        "dataset_count": len(datasets),
        "file_count": sum(item["file_count"] for item in datasets),
        "source_bytes": sum(item["source_bytes"] for item in datasets),
        "datasets": datasets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": output.relative_to(PROJECT_ROOT).as_posix(),
                "dataset_count": payload["dataset_count"],
                "file_count": payload["file_count"],
                "source_bytes": payload["source_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
