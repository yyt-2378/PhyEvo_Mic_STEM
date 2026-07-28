#!/usr/bin/env python
"""Render a compact figure from the immutable locked-score JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORE = (
    PROJECT_ROOT
    / "benchmarks"
    / "locked_runs"
    / "dennler_v1_score"
    / "locked_metrics.json"
)
OUTPUT = SCORE.with_name("locked_primary_summary.png")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {OUTPUT}")
    payload = json.loads(SCORE.read_text(encoding="utf-8"))
    selected = [
        ("Blur/amorphous/\nbeam composite", "blurred_amorphous_or_beam_composite"),
        ("Secondary crystal\nsymmetry", "secondary_crystal_symmetry_composite"),
        ("Any expert-marked\ndefect", "any_expert_marked_defect"),
    ]
    metric_names = ("iou", "precision", "recall")
    values = np.asarray(
        [
            [
                payload["targets"][target]["micro_pixel_metrics"][metric]
                for metric in metric_names
            ]
            for _, target in selected
        ]
    )
    normal_fp = np.asarray(
        [
            payload["targets"][target]["normal_controls"][
                "mean_false_positive_area_fraction"
            ]
            for _, target in selected
        ]
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.4),
        gridspec_kw={"width_ratios": [2.1, 1]},
    )
    x = np.arange(len(selected))
    width = 0.23
    colors = ("#3b82f6", "#f59e0b", "#10b981")
    for index, (metric, color) in enumerate(zip(metric_names, colors)):
        bars = axes[0].bar(
            x + (index - 1) * width,
            values[:, index],
            width,
            label=metric.upper(),
            color=color,
        )
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axes[0].set_xticks(x, [name for name, _ in selected])
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("Pre-registered primary endpoints")
    axes[0].legend(frameon=False, ncol=3, loc="upper center")
    axes[0].grid(axis="y", alpha=0.2)

    bars = axes[1].bar(
        x,
        normal_fp,
        color=("#64748b", "#8b5cf6", "#ef4444"),
        width=0.62,
    )
    axes[1].bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    axes[1].set_xticks(x, ["Blur", "Symmetry", "Any"])
    axes[1].set_ylim(0, max(0.05, float(np.max(normal_fp)) * 1.25))
    axes[1].set_ylabel("Predicted positive area fraction")
    axes[1].set_title("30 normal controls: mean false-positive area")
    axes[1].grid(axis="y", alpha=0.2)

    figure.suptitle(
        "Prospective zero-shot external evaluation — 61 images, "
        "no adaptation, 0 detector failures",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.015,
        "Class 2 is a blur/amorphous/beam-damage composite; "
        "class 1 is secondary crystal symmetry.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
