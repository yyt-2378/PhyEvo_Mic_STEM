#!/usr/bin/env python
"""Create deterministic 2-D evidence maps from published 4D-STEM HDF5 cubes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from tfree_detection.fourdstem import (
    derive_fast_mapping_maps,
    inspect_fast_mapping_h5,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACQUISITIONS = (
    {
        "slug": "tio2",
        "file": "TiO2_dataset.h5",
        "acquisition_id": "werbrouck-tio2-sio2",
        "sample_id": "werbrouck-tio2-on-sio2",
        "material": "TiO2_on_SiO2",
        "expected_md5": "45bb27a45a8c9568a7896d8cec8faa13",
        "expected_bytes": 142469777,
    },
    {
        "slug": "nmc-lgps",
        "file": "NMC_LGPS_dataset.h5",
        "acquisition_id": "werbrouck-nmc-lgps",
        "sample_id": "werbrouck-nmc-lgps-interface",
        "material": "NMC_LGPS_interface",
        "expected_md5": "c3d63ca5a29e5e471245626a65a139d0",
        "expected_bytes": 159258737,
    },
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


def _display(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros(values.shape, dtype=np.uint8)
    lower, upper = np.percentile(finite, [1.0, 99.0])
    if upper <= lower:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def _save_overview(
    maps: dict[str, np.ndarray], output: Path, material: str
) -> None:
    order = (
        ("integrated_intensity", "Integrated intensity"),
        ("diffraction_entropy", "Diffraction entropy"),
        ("radial_first_moment", "Radial first moment"),
        ("diffraction_peak_fraction", "Peak fraction"),
        ("signature_pc1", "Polar-signature PC1"),
        ("signature_anomaly", "Signature anomaly"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12, 5.4))
    for axis, (key, title) in zip(axes.ravel(), order):
        image = axis.imshow(maps[key], cmap="magma", aspect="equal")
        axis.set_title(title)
        axis.set_xlabel("real-y scan index")
        axis.set_ylabel("real-x scan index")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"{material}: label-free 4D-STEM standardization", fontsize=14
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "datasets/public/werbrouck_2026_fast_4dstem_phase"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "datasets/public/werbrouck_2026_fast_4dstem_phase/"
            "standardized_v1"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/standardized/fast_4dstem_phase_v1.json"
        ),
    )
    parser.add_argument("--chunk-rows", type=int, default=2)
    parser.add_argument(
        "--acquisition",
        choices=("tio2", "nmc-lgps", "all"),
        default="tio2",
        help=(
            "Fixed pre-outcome acquisition selection; default is the smaller "
            "published HDF5 file"
        ),
    )
    arguments = parser.parse_args()
    dataset_dir = _inside(arguments.dataset_dir)
    output_root = _inside(arguments.output_root)
    manifest_path = _inside(arguments.manifest)
    if output_root.exists() or manifest_path.exists():
        raise FileExistsError(
            "Refusing to overwrite standardized data or its versioned manifest"
        )

    selected_specs = [
        spec
        for spec in ACQUISITIONS
        if arguments.acquisition == "all"
        or spec["slug"] == arguments.acquisition
    ]
    verified: list[tuple[dict[str, Any], Path]] = []
    for spec in selected_specs:
        source = dataset_dir / str(spec["file"])
        if not source.is_file():
            raise FileNotFoundError(f"Source acquisition is missing: {source}")
        if source.stat().st_size != int(spec["expected_bytes"]):
            raise ValueError(f"Byte-size mismatch: {source}")
        if _digest(source, "md5") != spec["expected_md5"]:
            raise ValueError(f"MD5 mismatch: {source}")
        verified.append((spec, source))

    output_root.mkdir(parents=True, exist_ok=False)
    acquisitions = []
    for spec, source in verified:
        acquisition_dir = output_root / str(spec["acquisition_id"])
        acquisition_dir.mkdir()
        inspection = inspect_fast_mapping_h5(source)
        maps = derive_fast_mapping_maps(
            source,
            chunk_rows=arguments.chunk_rows,
            radial_bins=4,
            angular_bins=12,
        )
        derived = []
        for name, values in maps.items():
            target = acquisition_dir / f"{name}.tif"
            tifffile.imwrite(
                target,
                np.asarray(values, dtype=np.float32),
                metadata={
                    "axes": "YX",
                    "provenance": "derived_from_published_4d_stem",
                },
            )
            derived.append(
                {
                    "map_name": name,
                    "relative_path": target.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "bytes": target.stat().st_size,
                    "sha256": _digest(target, "sha256"),
                    "detector_ready": name
                    in {"signature_pc1", "signature_anomaly"},
                }
            )
        preview = acquisition_dir / "standardization_overview.png"
        _save_overview(maps, preview, str(spec["material"]))
        derived.append(
            {
                "map_name": "standardization_overview",
                "relative_path": preview.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "bytes": preview.stat().st_size,
                "sha256": _digest(preview, "sha256"),
                "detector_ready": False,
            }
        )
        source_entry = {
            **spec,
            "relative_path": source.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _digest(source, "sha256"),
            "inspection": inspection,
            "derived": derived,
        }
        metadata_path = acquisition_dir / "provenance.json"
        metadata_path.write_text(
            json.dumps(source_entry, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        acquisitions.append(source_entry)

    payload = {
        "manifest_format": "tfree-fast-4dstem-standardization-v1",
        "dataset_id": "werbrouck_2026_fast_4dstem_phase",
        "source_doi": "10.5281/zenodo.20366781",
        "source_license": "CC-BY-4.0",
        "source_code": (
            "https://github.com/awwerbro/fast_mapping_4DSTEM"
        ),
        "axis_semantics_source": "author_notebooks",
        "axis_semantics": [
            "real_x",
            "real_y",
            "reciprocal_x",
            "reciprocal_y",
        ],
        "selection": {
            "requested_acquisition": arguments.acquisition,
            "default_rule": (
                "smallest_published_HDF5_by_expected_bytes; fixed before "
                "detector execution and independent of outcomes"
            ),
            "complete_source_file_count": len(acquisitions),
            "published_source_file_count": len(ACQUISITIONS),
        },
        "derivation": {
            "supervised_training_used": False,
            "external_weights_used": False,
            "fixed_radial_bins": 4,
            "fixed_angular_bins": 12,
            "maps": {
                "integrated_intensity": "sum over both reciprocal axes",
                "diffraction_entropy": (
                    "normalized Shannon entropy of each diffraction pattern"
                ),
                "radial_first_moment": (
                    "intensity-weighted normalized reciprocal radius"
                ),
                "diffraction_peak_fraction": (
                    "maximum detector-pixel intensity divided by total"
                ),
                "signature_pc1": (
                    "first deterministic SVD coordinate of fixed polar-bin "
                    "Hellinger signatures"
                ),
                "signature_anomaly": (
                    "robust RMS distance from the median polar signature"
                ),
            },
        },
        "evaluation_status": (
            "development evidence only; no two-expert pixel Gold"
        ),
        "acquisitions": acquisitions,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "standardized": True,
                "acquisitions": len(acquisitions),
                "derived_assets": sum(
                    len(item["derived"]) for item in acquisitions
                ),
                "manifest": manifest_path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
