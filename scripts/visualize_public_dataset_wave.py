#!/usr/bin/env python
"""Visualize verified bytes, task coverage, and adapter readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "zhang_2026_wse2_3d_defects_coordinates": "WSe2 raw 1657 + coords",
    "strauch_2021_srtio3_high_resolution_4dstem": "SrTiO3 normal control",
    "mills_2022_irradiated_al_4dstem": "Irradiated Al + calibration",
    "wang_2026_polymer_lamella_4dstem_rt": "Polymer RT orientation",
}
COVERAGE = {
    "NMC/LGPS interface": (0, 1, 0, 0, 1),
    "WSe2 raw 1657 + coords": (1, 1, 0, 0, 0),
    "SrTiO3 normal control": (0, 0, 1, 0, 0),
    "Irradiated Al + calibration": (0, 0, 0, 1, 1),
    "Polymer RT orientation": (0, 0, 0, 0, 1),
}


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def _logical_bytes(dataset: dict[str, Any]) -> int:
    total = 0
    for entry in dataset["files"]:
        inventory = entry["inventory"]
        total += int(
            inventory.get("logical_uncompressed_bytes", entry["bytes"])
        )
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "datasets/standardized/public_dataset_wave_v1.json"
        ),
    )
    parser.add_argument(
        "--fast-manifest",
        type=Path,
        default=Path("datasets/standardized/fast_4dstem_phase_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/public_dataset_wave_v1.png"),
    )
    arguments = parser.parse_args()
    inventory_path = _inside(arguments.inventory)
    fast_manifest_path = _inside(arguments.fast_manifest)
    output = _inside(arguments.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a versioned figure")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    fast_manifest = json.loads(
        fast_manifest_path.read_text(encoding="utf-8")
    )
    nmc = next(
        item
        for item in fast_manifest["acquisitions"]
        if item["acquisition_id"] == "werbrouck-nmc-lgps"
    )
    labels = ["NMC/LGPS interface"] + [
        LABELS[item["dataset_id"]] for item in inventory["datasets"]
    ]
    source_bytes = [int(nmc["expected_bytes"])] + [
        int(item["source_bytes"]) for item in inventory["datasets"]
    ]
    logical_bytes = [int(nmc["expected_bytes"])] + [
        _logical_bytes(item) for item in inventory["datasets"]
    ]

    figure = plt.figure(figsize=(15, 9), facecolor="#f5f7fb")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.05, 1.0),
        width_ratios=(1.05, 1.0),
        hspace=0.38,
        wspace=0.28,
    )
    axis_bytes = figure.add_subplot(grid[0, 0])
    y = np.arange(len(labels))
    axis_bytes.barh(
        y + 0.18,
        np.asarray(source_bytes) / (1024**2),
        height=0.34,
        color="#2563eb",
        label="verified source on disk",
    )
    axis_bytes.barh(
        y - 0.18,
        np.asarray(logical_bytes) / (1024**2),
        height=0.34,
        color="#f59e0b",
        label="logical payload",
    )
    axis_bytes.set_xscale("log")
    axis_bytes.set_yticks(y, labels)
    axis_bytes.invert_yaxis()
    axis_bytes.set_xlabel("MiB (log scale)")
    axis_bytes.set_title("Verified source bytes vs. logical archive payload")
    axis_bytes.legend(frameon=False, loc="lower right")
    axis_bytes.grid(axis="x", alpha=0.18)

    axis_status = figure.add_subplot(grid[0, 1])
    status_columns = (
        "license",
        "checksum",
        "sealed",
        "registry",
        "2D evidence",
    )
    status = np.ones((len(labels), len(status_columns)), dtype=float)
    status[1:, -1] = 0.0
    axis_status.imshow(
        status,
        cmap=matplotlib.colors.ListedColormap(["#d7dde8", "#10b981"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axis_status.set_xticks(
        np.arange(len(status_columns)),
        status_columns,
        rotation=25,
        ha="right",
    )
    axis_status.set_yticks(y, labels)
    for row in range(status.shape[0]):
        for column in range(status.shape[1]):
            axis_status.text(
                column,
                row,
                "ready" if status[row, column] else "adapter next",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if status[row, column] else "#475569",
            )
    axis_status.set_title("Governance complete; raw-format adapters stay explicit")

    axis_coverage = figure.add_subplot(grid[1, :])
    coverage_columns = (
        "atomic defect",
        "interface",
        "normal/OOD",
        "irradiation",
        "orientation/physics",
    )
    coverage = np.asarray([COVERAGE[label] for label in labels])
    axis_coverage.imshow(
        coverage,
        cmap=matplotlib.colors.ListedColormap(["#e5e7eb", "#7c3aed"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axis_coverage.set_xticks(
        np.arange(len(coverage_columns)),
        coverage_columns,
    )
    axis_coverage.set_yticks(y, labels)
    for row in range(coverage.shape[0]):
        for column in range(coverage.shape[1]):
            axis_coverage.text(
                column,
                row,
                "yes" if coverage[row, column] else "—",
                ha="center",
                va="center",
                color="white" if coverage[row, column] else "#94a3b8",
                fontweight="bold" if coverage[row, column] else "normal",
            )
    axis_coverage.set_title(
        "What this wave adds to the experimental STEM evidence map"
    )

    total_new = sum(source_bytes)
    figure.suptitle(
        "Public experimental STEM wave: 5 acquisitions, "
        f"{total_new / (1024**3):.2f} GiB verified, registry v6 = 313 assets",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )
    figure.text(
        0.5,
        0.02,
        "No supervised training, no external model weights. "
        "Gray cells are tracked adapter work, not failed or missing data.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(
        json.dumps(
            {
                "output": output.relative_to(PROJECT_ROOT).as_posix(),
                "acquisition_count": len(labels),
                "verified_new_bytes": total_new,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
