#!/usr/bin/env python
"""Run the 2-D training-free boundary branch on standardized 4D-STEM maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from tfree_detection.config import load_config
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.reporting import save_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        "--manifest",
        type=Path,
        default=Path(
            "datasets/standardized/fast_4dstem_phase_v1.json"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase_boundary_4dstem.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/public_fast_4dstem_v3"),
    )
    parser.add_argument(
        "--gallery-copy",
        type=Path,
        default=Path("docs/assets/fast_4dstem_test_gallery_v3.png"),
    )
    arguments = parser.parse_args()
    manifest_path = _inside(arguments.manifest)
    config_path = _inside(arguments.config)
    output = _inside(arguments.output)
    gallery_copy = _inside(arguments.gallery_copy)
    if output.exists() or gallery_copy.exists():
        raise FileExistsError("Refusing to overwrite a versioned test output")
    output.mkdir(parents=True, exist_ok=False)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    framework = TrainingFreeDetectionFramework(load_config(config_path))
    cases = []
    row_count = len(manifest["acquisitions"])
    figure, axes = plt.subplots(
        row_count,
        4,
        figsize=(13, 2.6 * row_count + 1.0),
        squeeze=False,
    )
    for row, acquisition in enumerate(manifest["acquisitions"]):
        paths = {
            item["map_name"]: _inside(Path(item["relative_path"]))
            for item in acquisition["derived"]
            if item["map_name"] != "standardization_overview"
        }
        target = paths["signature_pc1"]
        result = framework.detect(target, line_topology=True)
        run_dir = output / "runs" / acquisition["acquisition_id"]
        artifacts = save_result(result, framework.config, run_dir)
        cases.append(
            {
                "acquisition_id": acquisition["acquisition_id"],
                "sample_id": acquisition["sample_id"],
                "material": acquisition["material"],
                "source_sha256": acquisition["sha256"],
                "input_map": target.relative_to(PROJECT_ROOT).as_posix(),
                "input_map_sha256": _sha256(target),
                "task": result.task,
                "candidate_fraction": float(np.mean(result.mask)),
                "region_count": len(result.regions),
                "genuine_two_phase": bool(
                    result.metadata.get("genuine_two_phase", False)
                ),
                "prototype_separation_over_within": float(
                    result.metadata.get(
                        "prototype_separation_over_within_dispersion",
                        0.0,
                    )
                ),
                "line_topology_enabled": True,
                "artifacts": artifacts,
                "gold_status": (
                    "no two-expert pixel Gold; smoke/development evidence only"
                ),
            }
        )
        panels = (
            ("integrated_intensity", "Virtual integrated image"),
            ("diffraction_entropy", "Diffraction entropy"),
            ("signature_pc1", "Reciprocal signature PC1"),
        )
        for column, (map_name, title) in enumerate(panels):
            values = np.asarray(tifffile.imread(paths[map_name]))
            axes[row, column].imshow(values, cmap="gray", aspect="equal")
            axes[row, column].set_title(title)
        axes[row, 3].imshow(
            result.normalized_image, cmap="gray", aspect="equal"
        )
        if np.any(result.mask):
            axes[row, 3].contour(
                result.mask.astype(float),
                levels=[0.5],
                colors=["#00e5ff"],
                linewidths=1.2,
            )
        axes[row, 3].set_title(
            f"Boundary candidates\n{len(result.regions)} regions"
        )
        axes[row, 0].set_ylabel(acquisition["material"])
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Experimental 4D-STEM → fixed evidence maps → training-free boundary branch",
        fontsize=14,
    )
    figure.tight_layout()
    gallery = output / "fast_4dstem_test_gallery.png"
    figure.savefig(gallery, dpi=200, bbox_inches="tight")
    gallery_copy.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(gallery_copy, dpi=200, bbox_inches="tight")
    plt.close(figure)
    summary = {
        "report_format": "tfree-fast-4dstem-smoke-test-v1",
        "manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "config_sha256": _sha256(config_path),
        "training_used": False,
        "external_weights_used": False,
        "evaluation_type": "development_smoke_and_qualitative_localization",
        "accuracy_claim_permitted": False,
        "cases": cases,
        "gallery": gallery.relative_to(PROJECT_ROOT).as_posix(),
        "gallery_copy": gallery_copy.relative_to(PROJECT_ROOT).as_posix(),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
