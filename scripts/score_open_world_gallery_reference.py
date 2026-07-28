#!/usr/bin/env python3
"""Score the fixed EMSeek gallery sample against its author point reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.io import imread

from tfree_detection.config import load_config
from tfree_detection.preprocessing import load_grayscale, robust_normalize
from tfree_detection.site_matching import match_sites_lexicographic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE = PROJECT_ROOT / "datasets/public/emseek_wse2_atomic_defects/image/1.png"
LABEL = PROJECT_ROOT / "datasets/public/emseek_wse2_atomic_defects/label/1.png"
CONFIG = PROJECT_ROOT / "configs/atomic_point_defect.yaml"


def _within_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes project root: {resolved}") from error
    return resolved


def _truth_points() -> np.ndarray:
    image = np.asarray(imread(LABEL))
    mask = np.any(image[..., :3] != 0, axis=-1)
    components, count = ndimage.label(mask)
    centers_yx = np.asarray(
        ndimage.center_of_mass(mask, components, range(1, count + 1)),
        dtype=float,
    )
    return centers_yx[:, ::-1]


def score(gallery: Path) -> tuple[Path, Path]:
    gallery = _within_project(gallery)
    summary_path = (
        gallery / "runs/emseek_wse2_image_1/summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    predicted = np.asarray(
        [region["centroid_xy"] for region in payload["regions"]],
        dtype=float,
    ).reshape(-1, 2)
    truth = _truth_points()
    tolerance = float(payload["metadata"]["matching_tolerance_pixels"])
    match = match_sites_lexicographic(
        predicted,
        truth,
        tolerance_pixels=tolerance,
    )

    metrics_path = gallery / "development_reference_check.json"
    figure_path = gallery / "wse2_image1_reference_alignment.png"
    existing = [path for path in (metrics_path, figure_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite: " + ", ".join(str(path) for path in existing)
        )
    metrics = {
        "dataset": "EMSeek WSe2",
        "file_id": "image-1",
        "scope": "single fixed development-reference image",
        "matching_version": "site_matching_v2",
        "tolerance_pixels": tolerance,
        "predicted_sites": match.predicted_count,
        "author_reference_sites": match.truth_count,
        "matched_sites": match.matched_count,
        "reference_precision": match.precision,
        "reference_recall": match.recall,
        "reference_f1": match.f1,
        "mean_localization_error_pixels": (
            float(np.mean(match.distances))
            if match.distances.size
            else None
        ),
        "p95_localization_error_pixels": (
            float(np.percentile(match.distances, 95))
            if match.distances.size
            else None
        ),
        "interpretation": (
            "Development-reference agreement only. The author point mask is not "
            "promoted to calibrated Gold and provides no defect-subtype labels."
        ),
        "training_or_fitting_performed": False,
    }
    with metrics_path.open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    config = load_config(CONFIG)
    normalized, _ = robust_normalize(load_grayscale(IMAGE), config.preprocess)
    matched_predicted = np.zeros(len(predicted), dtype=bool)
    matched_truth = np.zeros(len(truth), dtype=bool)
    matched_predicted[match.predicted_indices] = True
    matched_truth[match.truth_indices] = True

    figure, axis = plt.subplots(figsize=(9.2, 9.2))
    axis.imshow(normalized, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.scatter(
        predicted[~matched_predicted, 0],
        predicted[~matched_predicted, 1],
        s=14,
        facecolors="none",
        edgecolors="#ff3cac",
        linewidths=0.65,
        label=f"prediction only ({int((~matched_predicted).sum())})",
    )
    axis.scatter(
        truth[~matched_truth, 0],
        truth[~matched_truth, 1],
        s=22,
        marker="x",
        color="#ffd60a",
        linewidths=0.9,
        label=f"author point only ({int((~matched_truth).sum())})",
    )
    axis.scatter(
        predicted[matched_predicted, 0],
        predicted[matched_predicted, 1],
        s=20,
        facecolors="none",
        edgecolors="#00d084",
        linewidths=0.8,
        label=f"one-to-one match ({match.matched_count})",
    )
    axis.set_title(
        "EMSeek WSe2 image 1 — development-reference agreement\n"
        f"P={match.precision:.3f}, R={match.recall:.3f}, "
        f"F1={match.f1:.3f}; tolerance={tolerance:.2f} px",
        fontsize=12,
    )
    axis.legend(
        loc="lower left",
        fontsize=8,
        framealpha=0.82,
        borderpad=0.45,
    )
    axis.set_axis_off()
    figure.text(
        0.5,
        0.018,
        (
            "Single fixed development image; author points are a reference, not "
            "calibrated Gold. No defect-subtype accuracy is implied."
        ),
        ha="center",
        fontsize=8,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return metrics_path, figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gallery",
        type=Path,
        default=Path("outputs/open_world_dataset_gallery_v04"),
    )
    return parser.parse_args()


def main() -> None:
    metrics, figure = score(parse_args().gallery)
    print(metrics.relative_to(PROJECT_ROOT))
    print(figure.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
