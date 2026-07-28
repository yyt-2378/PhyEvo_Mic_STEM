#!/usr/bin/env python
"""Build auditable development figures for the WSe2 two-teacher result."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy import ndimage
from skimage.io import imread

from tfree_detection.site_matching import match_sites_lexicographic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERIODIC_SCORE = (
    PROJECT_ROOT
    / "benchmarks/development_runs/emseek_wse2_periodic_v1_score_v2"
    / "development_metrics.json"
)
LEGACY_SCORE = (
    PROJECT_ROOT
    / "benchmarks/development_runs/emseek_wse2_legacy_atomic_v0_2_score_v2"
    / "development_metrics.json"
)
CONSENSUS_SCORE_DIR = (
    PROJECT_ROOT
    / "benchmarks/development_runs"
    / "emseek_wse2_two_teacher_consensus_v2_score"
)
CONSENSUS_SCORE = CONSENSUS_SCORE_DIR / "development_metrics.json"
CONSENSUS_PREDICTIONS = (
    PROJECT_ROOT
    / "benchmarks/development_runs"
    / "emseek_wse2_two_teacher_consensus_v2_predictions"
    / "predictions"
)
DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets/public/emseek_wse2_atomic_defects"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _truth_xy(pair_id: str) -> np.ndarray:
    label_image = np.asarray(
        imread(DATASET_ROOT / "label" / f"{pair_id}.png")
    )
    mask = np.any(label_image[..., :3] != 0, axis=-1)
    components, count = ndimage.label(mask)
    centers_yx = np.asarray(
        ndimage.center_of_mass(
            mask, components, range(1, count + 1)
        ),
        dtype=float,
    )
    return centers_yx[:, ::-1]


def _save_summary() -> Path:
    periodic = _load(PERIODIC_SCORE)
    legacy = _load(LEGACY_SCORE)
    consensus = _load(CONSENSUS_SCORE)
    names = ["Legacy site residual", "Periodic counterfactual", "2-teacher consensus"]
    payloads = [legacy, periodic, consensus]
    colors = ["#64748B", "#F59E0B", "#0F766E"]
    metrics = ["site_precision", "site_recall", "site_f1"]
    metric_labels = ["Precision", "Recall", "F1"]

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
    x = np.arange(len(metrics))
    width = 0.24
    for index, (name, payload, color) in enumerate(
        zip(names, payloads, colors)
    ):
        values = [
            payload["micro_site_metrics"][metric]
            for metric in metrics
        ]
        axes[0].bar(
            x + (index - 1) * width,
            values,
            width,
            color=color,
            label=name,
        )
        for x_value, value in zip(
            x + (index - 1) * width, values
        ):
            axes[0].text(
                x_value,
                value + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    axes[0].set_xticks(x, metric_labels)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Micro site metric")
    axes[0].set_title("A  Complementary physics teachers")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].grid(axis="y", alpha=0.18)

    distributions = [
        [item["site_f1"] for item in payload["per_image"]]
        for payload in payloads
    ]
    box = axes[1].boxplot(
        distributions,
        labels=["Legacy", "Periodic", "Consensus"],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.6},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)
    random = np.random.default_rng(17)
    for index, (values, color) in enumerate(
        zip(distributions, colors), start=1
    ):
        jitter = random.normal(0, 0.035, len(values))
        axes[1].scatter(
            index + jitter,
            values,
            s=9,
            color=color,
            alpha=0.24,
            edgecolors="none",
        )
    consensus_f1 = consensus["micro_site_metrics"]["site_f1"]
    legacy_f1 = legacy["micro_site_metrics"]["site_f1"]
    periodic_f1 = periodic["micro_site_metrics"]["site_f1"]
    axes[1].text(
        0.03,
        0.97,
        (
            f"Consensus ΔF1 vs legacy: {consensus_f1 - legacy_f1:+.3f}\n"
            f"Consensus ΔF1 vs periodic: {consensus_f1 - periodic_f1:+.3f}\n"
            "107 images; same train-domain labels and matching rule"
        ),
        transform=axes[1].transAxes,
        fontsize=9,
        va="top",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 0.9,
        },
    )
    axes[1].set_ylim(-0.05, 1.02)
    zero_consensus = int(
        np.sum(np.asarray(distributions[2]) == 0)
    )
    axes[1].text(
        3,
        -0.025,
        f"{zero_consensus} zero-consensus images",
        ha="center",
        va="center",
        fontsize=8,
        color="#991B1B",
    )
    axes[1].set_ylabel("Per-image site F1")
    axes[1].set_title("B  Paired image-level distribution")
    axes[1].grid(axis="y", alpha=0.18)
    figure.suptitle(
        "EMSeek WSe₂ development benchmark — not an external generalization test",
        fontsize=13,
        y=1.01,
    )
    figure.tight_layout()
    target = CONSENSUS_SCORE_DIR / "teacher_consensus_summary_final.png"
    figure.savefig(target, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return target


def _densest_crop(
    truth_xy: np.ndarray, shape: tuple[int, int], crop_size: int
) -> tuple[int, int, int, int]:
    density = np.zeros(shape, dtype=np.float32)
    xy = np.rint(truth_xy).astype(int)
    density[
        np.clip(xy[:, 1], 0, shape[0] - 1),
        np.clip(xy[:, 0], 0, shape[1] - 1),
    ] = 1.0
    window = ndimage.uniform_filter(
        density, size=crop_size, mode="constant"
    )
    y, x = np.unravel_index(np.argmax(window), shape)
    half = crop_size // 2
    x1 = int(np.clip(x - half, 0, shape[1] - crop_size))
    y1 = int(np.clip(y - half, 0, shape[0] - crop_size))
    return x1, y1, x1 + crop_size, y1 + crop_size


def _draw_points(
    axis: plt.Axes,
    points: np.ndarray,
    crop: tuple[int, int, int, int],
    *,
    color: str,
    radius: float,
    linewidth: float,
    label: str,
) -> None:
    x1, y1, x2, y2 = crop
    first = True
    for x, y in np.asarray(points, dtype=float).reshape(-1, 2):
        if not (x1 <= x < x2 and y1 <= y < y2):
            continue
        axis.add_patch(
            Circle(
                (x - x1, y - y1),
                radius,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                label=label if first else None,
            )
        )
        first = False


def _save_overlay(pair_id: str = "23") -> Path:
    raw = np.asarray(
        imread(DATASET_ROOT / "image" / f"{pair_id}.png")
    )
    truth = _truth_xy(pair_id)
    record = _load(
        CONSENSUS_PREDICTIONS / f"image-{pair_id}.json"
    )
    consensus = np.asarray(
        record["confirmed_candidate_xy"], dtype=float
    ).reshape(-1, 2)
    periodic_only = np.asarray(
        record["primary_only_uncertain_xy"], dtype=float
    ).reshape(-1, 2)
    legacy_only = np.asarray(
        record["secondary_only_uncertain_xy"], dtype=float
    ).reshape(-1, 2)
    tolerance = float(
        record["consensus_summary"]["tolerance_pixels"]
    )
    crop = _densest_crop(truth, raw.shape[:2], crop_size=360)
    x1, y1, x2, y2 = crop
    crop_image = raw[y1:y2, x1:x2]

    if len(consensus) and len(truth):
        matched = match_sites_lexicographic(
            consensus, truth, tolerance_pixels=tolerance
        )
        true_positive = consensus[matched.predicted_indices]
        predicted_matched = np.zeros(len(consensus), dtype=bool)
        truth_matched = np.zeros(len(truth), dtype=bool)
        predicted_matched[matched.predicted_indices] = True
        truth_matched[matched.truth_indices] = True
        false_positive = consensus[~predicted_matched]
        false_negative = truth[~truth_matched]
    else:
        true_positive = np.empty((0, 2))
        false_positive = consensus
        false_negative = truth

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.35))
    for axis in axes:
        axis.imshow(crop_image, cmap="gray")
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0].set_title("A  Raw HAADF-STEM crop")
    _draw_points(
        axes[1],
        truth,
        crop,
        color="#22D3EE",
        radius=5.0,
        linewidth=1.2,
        label="Author consensus point",
    )
    axes[1].set_title("B  Admitted author labels")
    axes[1].legend(
        loc="lower left", fontsize=7, framealpha=0.75
    )
    _draw_points(
        axes[2],
        consensus,
        crop,
        color="#22C55E",
        radius=5.0,
        linewidth=1.25,
        label="Teacher consensus",
    )
    _draw_points(
        axes[2],
        periodic_only,
        crop,
        color="#F59E0B",
        radius=4.0,
        linewidth=1.0,
        label="Periodic only / uncertain",
    )
    _draw_points(
        axes[2],
        legacy_only,
        crop,
        color="#60A5FA",
        radius=3.0,
        linewidth=0.9,
        label="Legacy only / uncertain",
    )
    axes[2].set_title("C  Physics agreement states")
    axes[2].legend(
        loc="lower left", fontsize=6.6, framealpha=0.78
    )
    _draw_points(
        axes[3],
        true_positive,
        crop,
        color="#22C55E",
        radius=5.0,
        linewidth=1.3,
        label="Matched consensus",
    )
    _draw_points(
        axes[3],
        false_positive,
        crop,
        color="#F43F5E",
        radius=4.5,
        linewidth=1.1,
        label="Consensus false positive",
    )
    _draw_points(
        axes[3],
        false_negative,
        crop,
        color="#22D3EE",
        radius=3.8,
        linewidth=1.0,
        label="Missed author point",
    )
    axes[3].set_title("D  Development scoring only")
    axes[3].legend(
        loc="lower left", fontsize=6.6, framealpha=0.78
    )
    figure.suptitle(
        "Representative image 23 — circles are site centers, not atom-size masks",
        fontsize=12,
    )
    figure.tight_layout()
    target = (
        CONSENSUS_SCORE_DIR
        / "teacher_consensus_overlay_image23_final.png"
    )
    figure.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return target


def main() -> None:
    targets = [_save_summary(), _save_overlay()]
    for target in targets:
        print(
            f"{target.relative_to(PROJECT_ROOT)} "
            f"sha256={__import__('hashlib').sha256(target.read_bytes()).hexdigest()}"
        )


if __name__ == "__main__":
    main()
