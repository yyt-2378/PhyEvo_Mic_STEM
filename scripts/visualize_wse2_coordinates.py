#!/usr/bin/env python
"""Visualize author coordinate-status labels without implying image Gold."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTAL_MEMBERS = (
    "1657_refined_coordinates.txt",
    "1815_refined_coordinates.txt",
)
COLORS = {"Good": "#718096", "Poor": "#f6ad55", "Vacancy": "#e53e3e"}


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def _read(text: str) -> tuple[np.ndarray, np.ndarray]:
    coordinates = []
    status = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 5:
            continue
        coordinates.append([float(tokens[1]), float(tokens[2])])
        status.append(tokens[4])
    return np.asarray(coordinates), np.asarray(status)


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
        default=Path("docs/assets/wse2_coordinate_status_gallery.png"),
    )
    arguments = parser.parse_args()
    archive_path = _inside(arguments.archive)
    output = _inside(arguments.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a versioned gallery")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    summaries = []
    with ZipFile(archive_path) as archive:
        for axis, member in zip(axes, EXPERIMENTAL_MEMBERS):
            coordinates, status = _read(
                archive.read(member).decode("utf-8")
            )
            counts = Counter(status.tolist())
            for name in ("Good", "Poor", "Vacancy"):
                selected = status == name
                size = 7 if name == "Good" else (12 if name == "Poor" else 28)
                zorder = 1 if name == "Good" else (2 if name == "Poor" else 3)
                axis.scatter(
                    coordinates[selected, 0],
                    coordinates[selected, 1],
                    s=size,
                    c=COLORS[name],
                    label=f"{name}: {int(np.sum(selected))}",
                    alpha=0.72 if name == "Good" else 0.95,
                    linewidths=0,
                    zorder=zorder,
                )
            axis.set_title(f"Acquisition {member[:4]}")
            axis.set_xlabel("x (Å)")
            axis.set_ylabel("y (Å)")
            axis.set_aspect("equal")
            axis.legend(loc="upper right", fontsize=8, frameon=True)
            summaries.append(
                {
                    "acquisition_id": member[:4],
                    "coordinate_count": int(len(coordinates)),
                    "status_counts": dict(sorted(counts.items())),
                }
            )
    figure.suptitle(
        "WSe₂ author-reconstructed coordinate status (development evidence, not Gold)",
        fontsize=13,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(
        json.dumps(
            {
                "output": output.relative_to(PROJECT_ROOT).as_posix(),
                "acquisitions": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
