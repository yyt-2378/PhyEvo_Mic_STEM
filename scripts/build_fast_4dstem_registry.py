#!/usr/bin/env python
"""Append raw and derived fast 4D-STEM assets to a new registry version."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from tfree_detection.governance import (
    DataAsset,
    DataPartition,
    DataRegistry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("datasets/partition_registry_v4.yaml"),
    )
    parser.add_argument(
        "--standardization",
        type=Path,
        default=Path(
            "datasets/standardized/fast_4dstem_phase_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/partition_registry_v5.yaml"),
    )
    arguments = parser.parse_args()
    previous_path = _inside(arguments.previous)
    standardization_path = _inside(arguments.standardization)
    output = _inside(arguments.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a registry version")

    previous = DataRegistry.from_yaml(previous_path)
    if any(
        asset.dataset_id == "werbrouck_2026_fast_4dstem_phase"
        for asset in previous.assets
    ):
        raise ValueError("Previous registry already contains the new dataset")
    manifest = json.loads(
        standardization_path.read_text(encoding="utf-8")
    )
    additions = []
    for acquisition in manifest["acquisitions"]:
        acquisition_id = str(acquisition["acquisition_id"])
        group_id = f"{acquisition_id}-single-published-acquisition"
        raw_path = _inside(Path(acquisition["relative_path"]))
        if _sha256(raw_path) != acquisition["sha256"]:
            raise ValueError(f"Raw source checksum mismatch: {raw_path}")
        parent_id = f"fast4dstem:{acquisition_id}:raw"
        additions.append(
            DataAsset(
                asset_id=parent_id,
                dataset_id="werbrouck_2026_fast_4dstem_phase",
                relative_path=raw_path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
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
        for derived in acquisition["derived"]:
            map_name = str(derived["map_name"])
            if map_name not in MAP_TASKS:
                continue
            derived_path = _inside(Path(derived["relative_path"]))
            if _sha256(derived_path) != derived["sha256"]:
                raise ValueError(
                    f"Derived checksum mismatch: {derived_path}"
                )
            additions.append(
                DataAsset(
                    asset_id=f"fast4dstem:{acquisition_id}:{map_name}",
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
    registry = DataRegistry(
        registry_version="5.0.0",
        split_unit=previous.split_unit,
        assets=[*previous.assets, *additions],
    )
    registry.validate()
    registry.verify_files(PROJECT_ROOT)
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
                "added_asset_count": len(additions),
                "asset_count": len(registry.assets),
                "registry_sha256": registry.content_hash(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
