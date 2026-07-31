#!/usr/bin/env python
"""Inventory the WSe2 coordinate archive without extracting or relabeling it."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MD5 = "f32c5556f8f9251a6f342b239cd37a80"


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


def _parse_coordinate_text(text: str) -> tuple[list[str], np.ndarray, list[str]]:
    species = []
    coordinates = []
    status = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 4:
            continue
        try:
            xyz = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
        except ValueError:
            continue
        species.append(tokens[0])
        coordinates.append(xyz)
        status.append(tokens[4] if len(tokens) >= 5 else "not_provided")
    if not coordinates:
        raise ValueError("Coordinate member contains no parseable x/y/z rows")
    return species, np.asarray(coordinates, dtype=np.float64), status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "datasets/public/zhang_2026_wse2_3d_defects_coordinates/"
            "coordinates.zip"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/standardized/wse2_3d_coordinate_inventory_v1.json"
        ),
    )
    arguments = parser.parse_args()
    archive_path = _inside(arguments.archive)
    output = _inside(arguments.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a versioned inventory")
    if _digest(archive_path, "md5") != EXPECTED_MD5:
        raise ValueError("Coordinate archive MD5 mismatch")

    members = []
    with ZipFile(archive_path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir() or not info.filename.endswith(".txt"):
                continue
            raw = archive.read(info)
            species, coordinates, status = _parse_coordinate_text(
                raw.decode("utf-8")
            )
            filename = Path(info.filename).name
            if filename.startswith("1657_"):
                acquisition_id = "1657"
                provenance = "experimental_ptychographic_reconstruction"
            elif filename.startswith("1815_"):
                acquisition_id = "1815"
                provenance = "experimental_ptychographic_reconstruction"
            else:
                acquisition_id = "simulation"
                provenance = "simulation_or_simulation_refinement"
            members.append(
                {
                    "member_path": info.filename,
                    "member_bytes": info.file_size,
                    "member_crc32": f"{info.CRC:08x}",
                    "acquisition_id": acquisition_id,
                    "provenance": provenance,
                    "coordinate_unit": "angstrom",
                    "coordinate_count": int(len(coordinates)),
                    "species_counts": dict(sorted(Counter(species).items())),
                    "status_counts": dict(sorted(Counter(status).items())),
                    "bounds_xyz_angstrom": {
                        "minimum": np.min(coordinates, axis=0).tolist(),
                        "maximum": np.max(coordinates, axis=0).tolist(),
                    },
                    "semantics": (
                        "author-reconstructed atomic coordinates with an "
                        "optional Good/Poor/Vacancy status; author status is "
                        "development evidence, not two-expert Gold"
                    ),
                }
            )
    payload = {
        "inventory_format": "tfree-wse2-3d-coordinate-inventory-v1",
        "dataset_id": "zhang_2026_wse2_3d_defects_coordinates",
        "source_doi": "10.5281/zenodo.18260255",
        "license": "CC-BY-4.0",
        "archive_relative_path": archive_path.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _digest(archive_path, "sha256"),
        "member_count": len(members),
        "experimental_member_count": sum(
            item["provenance"].startswith("experimental")
            for item in members
        ),
        "simulation_member_count": sum(
            item["provenance"].startswith("simulation")
            for item in members
        ),
        "coordinate_rows_total": sum(
            item["coordinate_count"] for item in members
        ),
        "members": members,
        "admission_note": (
            "The archive is inventoried in place. Experimental and simulation "
            "coordinates remain distinct; no source row is overwritten."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
