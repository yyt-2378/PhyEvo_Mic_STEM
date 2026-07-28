#!/usr/bin/env python
"""Create descriptive figures for the final Class1 symmetry diagnostic."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "outputs/class1_symmetry_final_v2"
PREDICTIONS = ROOT / "predictions"
METRICS = ROOT / "score/class1_metrics.json"
ARCHIVE = (
    PROJECT_ROOT
    / "datasets/public/dennler_2021_hrstem_defects/"
    "hrstem_defects_dataset.zip"
)
OUTPUT = ROOT / "figures"


def _read_image(values: bytes) -> np.ndarray:
    image = np.asarray(tifffile.imread(BytesIO(values)))
    if image.ndim == 3:
        image = np.mean(image[..., :3].astype(float), axis=-1)
    low, high = np.percentile(image, [1, 99])
    return np.clip((image - low) / max(high - low, 1e-8), 0, 1)


def _read_truth(
    archive: ZipFile,
    record: dict,
) -> np.ndarray:
    if record["category"] == "normal":
        return np.zeros(tuple(record["image_shape"]), dtype=bool)
    label_path = record["source_relative_path"].replace(
        "/images/", "/labels/"
    )
    label = np.asarray(
        tifffile.imread(BytesIO(archive.read(label_path)))
    )
    if label.ndim == 3:
        label = label[..., 0]
    return np.asarray(label == 127, dtype=bool)


def _overlay(
    image: np.ndarray,
    masks: list[tuple[np.ndarray, str]],
) -> np.ndarray:
    rgb = np.repeat(image[..., None], 3, axis=-1)
    colors = {
        "red": np.asarray([1.0, 0.12, 0.10]),
        "cyan": np.asarray([0.05, 0.95, 1.0]),
        "yellow": np.asarray([1.0, 0.92, 0.10]),
        "magenta": np.asarray([1.0, 0.10, 0.92]),
        "green": np.asarray([0.1, 1.0, 0.25]),
    }
    for mask, color_name in masks:
        boundary = find_boundaries(mask, mode="thick")
        rgb[boundary] = colors[color_name]
    return rgb


def metric_figure(payload: dict) -> None:
    names = [
        "historical_class1",
        "symmetry_domain_region",
        "multigeometry_class1",
        "guided_expansion",
    ]
    labels = [
        "Historical\nClass1",
        "New domain\nregion",
        "New multi-\ngeometry",
        "Guided\nexpansion",
    ]
    metrics = ["iou", "f1", "precision", "recall"]
    colors = ["#77839a", "#14a89b", "#e09b2d", "#4b6fe2"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    x = np.arange(len(names))
    width = 0.18
    for index, metric in enumerate(metrics):
        values = [
            payload["targets"][name]["micro_pixel_metrics"][metric]
            for name in names
        ]
        axes[0].bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=metric.upper(),
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 0.85)
    axes[0].set_ylabel("Pixel metric")
    axes[0].set_title("Consumed Dennler Class1: post-hoc diagnostic")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(axis="y", alpha=0.25)

    normal_fp = [
        payload["targets"][name][
            "normal_control_mean_false_positive_area_fraction"
        ]
        for name in names
    ]
    axes[1].bar(labels, normal_fp, color=colors)
    axes[1].set_ylabel("Mean false-positive area on normal images")
    axes[1].set_ylim(0, max(normal_fp) * 1.35)
    axes[1].set_title("Normal-control cost")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(normal_fp):
        axes[1].text(
            index,
            value + max(normal_fp) * 0.04,
            f"{100 * value:.2f}%",
            ha="center",
            fontsize=9,
        )
    fig.text(
        0.5,
        0.01,
        "Labels were previously unblinded; these are diagnostic deltas, "
        "not a new zero-shot external-validation claim.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT / "01_class1_metric_comparison.png", dpi=180)
    plt.close(fig)


def gallery_figure(payload: dict, manifest: dict) -> None:
    historical_rows = {
        row["image_id"]: row
        for row in payload["targets"]["historical_class1"]["per_image"]
    }
    guided_rows = {
        row["image_id"]: row
        for row in payload["targets"]["guided_expansion"]["per_image"]
    }
    positives = [
        record
        for record in manifest["images"]
        if guided_rows[record["image_id"]]["truth_positive_fraction"] > 0
    ]
    positives.sort(
        key=lambda record: (
            guided_rows[record["image_id"]]["iou"]
            - historical_rows[record["image_id"]]["iou"]
        )
    )
    selected = [
        positives[-1],
        positives[len(positives) // 2],
        positives[0],
    ]
    normals = [
        record
        for record in manifest["images"]
        if record["category"] == "normal"
    ]
    normals.sort(
        key=lambda record: guided_rows[record["image_id"]][
            "predicted_positive_fraction"
        ]
    )
    selected.append(normals[-1])

    fig, axes = plt.subplots(
        len(selected),
        6,
        figsize=(20, 13.5),
        constrained_layout=True,
    )
    titles = [
        "Input",
        "Class1 truth",
        "Historical",
        "Domain region",
        "Interface / phase / core",
        "Guided expansion",
    ]
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=11)

    with ZipFile(ARCHIVE) as archive:
        for row, record in enumerate(selected):
            image = _read_image(
                archive.read(record["source_relative_path"])
            )
            truth = _read_truth(archive, record)
            with np.load(
                PREDICTIONS / record["prediction_path"],
                allow_pickle=False,
            ) as values:
                historical = np.asarray(
                    values["historical_class1"], dtype=bool
                )
                domain = np.asarray(
                    values["symmetry_domain_region"], dtype=bool
                )
                interface = np.asarray(
                    values["interface_line"], dtype=bool
                )
                phase = np.asarray(
                    values["phase_slip_line"], dtype=bool
                )
                core = np.asarray(
                    values["dislocation_core_point"], dtype=bool
                )
                guided = np.asarray(
                    values["guided_expansion"], dtype=bool
                )
            panels = [
                np.repeat(image[..., None], 3, axis=-1),
                _overlay(image, [(truth, "red")]),
                _overlay(image, [(historical, "yellow")]),
                _overlay(image, [(domain, "cyan")]),
                _overlay(
                    image,
                    [
                        (interface, "green"),
                        (phase, "magenta"),
                        (core, "yellow"),
                    ],
                ),
                _overlay(image, [(guided, "cyan"), (truth, "red")]),
            ]
            for column, panel in enumerate(panels):
                axes[row, column].imshow(panel)
                axes[row, column].axis("off")
            image_id = record["image_id"]
            delta = (
                guided_rows[image_id]["iou"]
                - historical_rows[image_id]["iou"]
            )
            axes[row, 0].set_ylabel(
                f"{record['category']}\n"
                f"guided ΔIoU={delta:+.3f}",
                fontsize=9,
            )
    fig.suptitle(
        "Class1 multi-geometry outputs (red=truth, cyan=domain/guided, "
        "green=interface, magenta=phase-slip, yellow=core/historical)",
        fontsize=14,
    )
    fig.savefig(OUTPUT / "02_class1_multigeometry_gallery.png", dpi=160)
    plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    manifest = json.loads(
        (PREDICTIONS / "prediction_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    metric_figure(payload)
    gallery_figure(payload, manifest)
    print(OUTPUT.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
