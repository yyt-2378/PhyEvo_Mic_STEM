#!/usr/bin/env python
"""Add verified compact-wave sources and missing 4D-STEM maps to a registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from tfree_detection.governance import (
    DataAsset,
    DataPartition,
    DataRegistry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets" / "public"
MAP_TASKS = {
    "integrated_intensity": ("virtual_detector", "phase_boundary"),
    "diffraction_entropy": ("amorphous_region", "phase_boundary"),
    "radial_first_moment": (
        "reciprocal_space_anomaly",
        "phase_boundary",
    ),
    "diffraction_peak_fraction": (
        "reciprocal_space_anomaly",
        "amorphous_region",
    ),
    "signature_pc1": ("phase_boundary",),
    "signature_anomaly": (
        "amorphous_region",
        "reciprocal_space_anomaly",
    ),
}
WAVE_SPECS: dict[str, dict[str, str]] = {
    "zhang_2026_wse2_3d_defects_coordinates": {
        "group_id": "wse2-3d-raw-1657-and-coordinate-archive",
        "acquisition_id": "wse2-3d-study-plus-raw-1657",
        "sample_id": "twisted-bilayer-wse2-published-study",
        "lab_id": "zhang_2026_archive",
        "microscope_id": "EMPAD_published_metadata",
        "material": "twisted_bilayer_WSe2",
    },
    "strauch_2021_srtio3_high_resolution_4dstem": {
        "group_id": "srtio3-high-resolution-single-acquisition",
        "acquisition_id": "strauch-2021-srtio3-20200518-165148",
        "sample_id": "strauch-2021-srtio3-crystal",
        "lab_id": "strauch_2021_archive",
        "microscope_id": "FEI_Titan_80_300_Medipix_Merlin",
        "material": "SrTiO3",
    },
    "mills_2022_irradiated_al_4dstem": {
        "group_id": "irradiated-al-raw-and-calibration",
        "acquisition_id": "mills-2022-irradiated-al",
        "sample_id": "mills-2022-irradiated-al-sample",
        "lab_id": "mills_2022_archive",
        "microscope_id": "FEI_TitanX_Gatan_Orius_830",
        "material": "irradiated_Al",
    },
    "wang_2026_polymer_lamella_4dstem_rt": {
        "group_id": "polymer-lamella-room-temperature-acquisition",
        "acquisition_id": "wang-2026-polymer-4dstem-rt",
        "sample_id": "wang-2026-polymer-lamella-sample",
        "lab_id": "wang_2026_archive",
        "microscope_id": "published_300kV_4dstem_instrument",
        "material": "polymer_lamellar_clusters",
    },
}


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


def _safe_asset_token(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "-"
        for character in value
    ).strip("-")


def _append_fast_4dstem(
    assets: list[DataAsset],
    standardization: dict[str, Any],
) -> int:
    existing_ids = {asset.asset_id for asset in assets}
    additions = 0
    for acquisition in standardization["acquisitions"]:
        acquisition_id = str(acquisition["acquisition_id"])
        group_id = f"{acquisition_id}-single-published-acquisition"
        raw_path = _inside(Path(acquisition["relative_path"]))
        if _digest(raw_path, "sha256") != acquisition["sha256"]:
            raise ValueError(f"Raw source checksum mismatch: {raw_path}")
        parent_id = f"fast4dstem:{acquisition_id}:raw"
        if parent_id in existing_ids:
            continue
        assets.append(
            DataAsset(
                asset_id=parent_id,
                dataset_id="werbrouck_2026_fast_4dstem_phase",
                relative_path=raw_path.relative_to(PROJECT_ROOT).as_posix(),
                partition=DataPartition.DEVELOPMENT_TEST,
                split_group_id=group_id,
                acquisition_id=acquisition_id,
                sample_id=str(acquisition["sample_id"]),
                lab_id="werbrouck_2026_archive",
                tasks=(
                    "amorphous_region",
                    "phase_boundary",
                    "virtual_detector",
                ),
                checksum=str(acquisition["sha256"]),
                material=str(acquisition["material"]),
                microscope_id="published_4dstem_instrument",
                eligible_for_active_learning=False,
            )
        )
        existing_ids.add(parent_id)
        additions += 1
        for derived in acquisition["derived"]:
            map_name = str(derived["map_name"])
            if map_name not in MAP_TASKS:
                continue
            derived_path = _inside(Path(derived["relative_path"]))
            if _digest(derived_path, "sha256") != derived["sha256"]:
                raise ValueError(
                    f"Derived checksum mismatch: {derived_path}"
                )
            asset_id = f"fast4dstem:{acquisition_id}:{map_name}"
            assets.append(
                DataAsset(
                    asset_id=asset_id,
                    dataset_id="werbrouck_2026_fast_4dstem_phase",
                    relative_path=derived_path.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    partition=DataPartition.DEVELOPMENT_TEST,
                    split_group_id=group_id,
                    acquisition_id=acquisition_id,
                    sample_id=str(acquisition["sample_id"]),
                    lab_id="werbrouck_2026_archive",
                    tasks=MAP_TASKS[map_name],
                    checksum=str(derived["sha256"]),
                    material=str(acquisition["material"]),
                    microscope_id="published_4dstem_instrument",
                    parent_asset_id=parent_id,
                    eligible_for_active_learning=False,
                )
            )
            existing_ids.add(asset_id)
            additions += 1
    return additions


def _append_wave(
    assets: list[DataAsset],
    manifest: dict[str, Any],
) -> int:
    by_id = {item["id"]: item for item in manifest["datasets"]}
    existing_ids = {asset.asset_id for asset in assets}
    additions = 0
    for dataset_id, spec in WAVE_SPECS.items():
        dataset = by_id[dataset_id]
        for entry in dataset["files"]:
            path = _inside(DATA_ROOT / dataset_id / entry["name"])
            if (
                not path.is_file()
                or path.stat().st_size != int(entry["bytes"])
            ):
                raise ValueError(f"Missing or incomplete source: {path}")
            if _digest(path, "md5") != str(entry["md5"]).lower():
                raise ValueError(f"MD5 mismatch: {path}")
            asset_id = (
                f"public:{dataset_id}:"
                f"{_safe_asset_token(str(entry['name']))}"
            )
            if asset_id in existing_ids:
                raise ValueError(f"Duplicate asset id: {asset_id}")
            assets.append(
                DataAsset(
                    asset_id=asset_id,
                    dataset_id=dataset_id,
                    relative_path=path.relative_to(PROJECT_ROOT).as_posix(),
                    partition=DataPartition.DEVELOPMENT_TEST,
                    split_group_id=spec["group_id"],
                    acquisition_id=spec["acquisition_id"],
                    sample_id=spec["sample_id"],
                    lab_id=spec["lab_id"],
                    tasks=tuple(str(item) for item in entry["tasks"]),
                    checksum=_digest(path, "sha256"),
                    material=spec["material"],
                    microscope_id=spec["microscope_id"],
                    eligible_for_active_learning=False,
                )
            )
            existing_ids.add(asset_id)
            additions += 1
    return additions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("datasets/partition_registry_v5.yaml"),
    )
    parser.add_argument(
        "--fast-standardization",
        type=Path,
        default=Path("datasets/standardized/fast_4dstem_phase_v2.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/manifest.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/partition_registry_v6.yaml"),
    )
    parser.add_argument("--registry-version", default="6.0.0")
    arguments = parser.parse_args()
    previous_path = _inside(arguments.previous)
    standardization_path = _inside(arguments.fast_standardization)
    manifest_path = _inside(arguments.manifest)
    output = _inside(arguments.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a registry version")

    previous = DataRegistry.from_yaml(previous_path)
    assets = list(previous.assets)
    standardization = json.loads(
        standardization_path.read_text(encoding="utf-8")
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    fast_additions = _append_fast_4dstem(assets, standardization)
    wave_additions = _append_wave(assets, manifest)
    registry = DataRegistry(
        registry_version=arguments.registry_version,
        split_unit=previous.split_unit,
        assets=assets,
    )
    registry.validate()
    verification = registry.verify_files(PROJECT_ROOT)
    output.write_text(
        yaml.safe_dump(
            {
                "registry_version": registry.registry_version,
                "split_unit": registry.split_unit,
                "assets": [asset.to_dict() for asset in registry.assets],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": output.relative_to(PROJECT_ROOT).as_posix(),
                "previous_asset_count": len(previous.assets),
                "fast_4dstem_added_asset_count": fast_additions,
                "public_wave_added_asset_count": wave_additions,
                "asset_count": len(registry.assets),
                "verified_file_count": verification["asset_count"],
                "registry_sha256": registry.content_hash(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
