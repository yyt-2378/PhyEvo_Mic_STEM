#!/usr/bin/env python3
"""Build an audit-first gallery from live SAM2 STEM refinement runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def _load_case(directory: Path) -> dict[str, Any]:
    exchange = directory / "sam3_exchange"
    with (directory / "sam3_refinement_summary.json").open(
        "r", encoding="utf-8"
    ) as stream:
        summary = json.load(stream)
    with (exchange / "sam_request.json").open(
        "r", encoding="utf-8"
    ) as stream:
        request = json.load(stream)
    image = np.load(exchange / "image.npy", allow_pickle=False)
    component_labels = np.load(
        exchange / "component_labels.npy", allow_pickle=False
    )
    with np.load(directory / "sam3_consensus_maps.npz") as archive:
        consensus = {key: archive[key] for key in archive.files}
    return {
        "directory": directory,
        "summary": summary,
        "request": request,
        "image": image,
        "component_labels": component_labels,
        "consensus": consensus,
    }


def _normalized(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    lower, upper = np.percentile(finite, [0.5, 99.5])
    return np.clip(
        (values - lower) / max(float(upper - lower), 1e-8), 0, 1
    )


def _contour(
    axis: plt.Axes,
    mask: np.ndarray,
    color: str,
    *,
    linewidth: float = 1.2,
    linestyle: str = "solid",
) -> None:
    if np.any(mask):
        axis.contour(
            mask.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=linewidth,
            linestyles=linestyle,
        )


def _draw_prompts(axis: plt.Axes, request: dict[str, Any]) -> None:
    for prompt in request["prompts"]:
        if int(prompt["ensemble_index"]) != 0:
            continue
        x1, y1, x2, y2 = prompt["box_xyxy"]
        axis.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="#00e5ff",
                linewidth=1.0,
            )
        )
        positive = np.asarray(prompt["positive_points_xy"], dtype=float)
        negative = np.asarray(prompt["negative_points_xy"], dtype=float)
        if positive.size:
            axis.scatter(
                positive[:, 0],
                positive[:, 1],
                c="#30d158",
                marker="+",
                s=24,
                linewidths=1.4,
            )
        if negative.size:
            axis.scatter(
                negative[:, 0],
                negative[:, 1],
                c="#ff453a",
                marker="x",
                s=18,
                linewidths=1.0,
            )


def _decision_masks(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    summary = case["summary"]
    labels = case["component_labels"]
    final = case["consensus"]["final_mask"].astype(bool)
    fallback = np.zeros(labels.shape, dtype=bool)
    accepted_labels: list[int] = []
    for candidate in summary["candidate_results"]:
        label = int(candidate["component_label"])
        if candidate["accepted"]:
            accepted_labels.append(label)
        else:
            fallback |= labels == label
    accepted = final & ~fallback
    if accepted_labels and not np.any(accepted):
        accepted = np.isin(labels, accepted_labels) & final
    return accepted, fallback


def _metric_lines(case: dict[str, Any]) -> list[str]:
    summary = case["summary"]
    runtime = summary["runtime_and_memory"]
    worker = summary["provenance"]["worker_response"]
    lines = [
        f"status: {summary['status']}",
        (
            f"accepted / fallback: "
            f"{summary['accepted_candidate_count']} / "
            f"{summary['fallback_candidate_count']}"
        ),
        f"prompts: {summary['prompt_count']}",
        (
            f"runtime: {runtime['worker_runtime_seconds']:.2f} s  |  "
            f"GPU peak: "
            f"{runtime['peak_device_memory_bytes'] / 2**30:.2f} GiB"
        ),
        (
            f"{worker['model_variant']}  |  torch "
            f"{worker['torch_version']}  |  CUDA "
            f"{worker['torch_cuda_version']}"
        ),
        "",
    ]
    for candidate in summary["candidate_results"]:
        metrics = candidate["metrics"]
        number = int(candidate["component_label"])
        decision = "REFINED" if candidate["accepted"] else "FALLBACK"
        lines.extend(
            [
                f"C{number}: {decision}",
                (
                    f"  core={metrics['seed_core_recall']:.3f}, "
                    f"area×={metrics['area_ratio']:.3f}, "
                    f"stability={metrics['prompt_stability_mean_iou']:.3f}"
                ),
                (
                    "  reason: "
                    + (
                        ", ".join(candidate["rejection_reasons"])
                        if candidate["rejection_reasons"]
                        else "all gates passed"
                    )
                ),
            ]
        )
    return lines


def _draw_case(
    axes: np.ndarray,
    case: dict[str, Any],
    row_label: str,
) -> None:
    image = _normalized(case["image"])
    consensus = case["consensus"]
    accepted, fallback = _decision_masks(case)
    coarse = consensus["coarse_mask"].astype(bool)

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    _contour(axes[0], coarse, "#ff453a")
    axes[0].set_title(f"{row_label}\nPhysics coarse (red)")

    axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    _draw_prompts(axes[1], case["request"])
    axes[1].set_title("Prompt ensemble 0\n+ foreground, × background")

    axes[2].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(
        np.ma.masked_where(
            consensus["inclusion_probability"] <= 0,
            consensus["inclusion_probability"],
        ),
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.72,
    )
    axes[2].set_title("Live SAM2 proposal\nensemble inclusion")

    axes[3].imshow(image, cmap="gray", vmin=0, vmax=1)
    _contour(
        axes[3], coarse, "#ff453a", linewidth=0.8, linestyle="dashed"
    )
    _contour(axes[3], accepted, "#00e5ff", linewidth=1.4)
    _contour(axes[3], fallback, "#ffd60a", linewidth=1.4)
    axes[3].set_title(
        "Final geometry\ncyan refined · yellow fallback"
    )

    axes[4].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[4].imshow(
        np.ma.masked_where(
            consensus["boundary_uncertainty"] <= 0,
            consensus["boundary_uncertainty"],
        ),
        cmap="magma",
        vmin=0,
        vmax=1,
        alpha=0.78,
    )
    axes[4].set_title("Prompt-instability\nboundary uncertainty")

    axes[5].set_facecolor("#f7f7f7")
    axes[5].text(
        0.02,
        0.98,
        "\n".join(_metric_lines(case)),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        color="#1c1c1e",
        transform=axes[5].transAxes,
    )
    axes[5].set_title("Physics-gate audit")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)


def build_gallery(
    amorphous_directory: Path,
    interface_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=False)
    cases = [
        (
            _load_case(amorphous_directory),
            "Experimental zeolite · amorphous candidates",
        ),
        (
            _load_case(interface_directory),
            "Experimental ZrO₂ · twin/phase interface",
        ),
    ]
    figure, axes = plt.subplots(2, 6, figsize=(24, 8.7))
    for row, (case, label) in enumerate(cases):
        _draw_case(axes[row], case, label)
    figure.suptitle(
        "Live SAM2.1-large on experimental STEM — "
        "frozen proposals, physics-gated candidate-local fallback",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.008,
        "No model training and no experimental Gold boundary: "
        "this figure demonstrates live inference behavior, not accuracy.",
        ha="center",
        fontsize=10,
        color="#3a3a3c",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    figure_path = output_directory / "sam2_real_instance_comparison.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    payload = {
        "validation_kind": "live_sam2_experimental_stem_no_gold",
        "training_or_fitting_used": False,
        "accuracy_claim_permitted": False,
        "figure": figure_path.name,
        "cases": [
            {
                "name": label,
                "source": str(case["directory"].resolve()),
                "refinement": case["summary"],
            }
            for case, label in cases
        ],
    }
    summary_path = output_directory / "gallery_summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return {
        "figure": str(figure_path.resolve()),
        "summary": str(summary_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize live SAM2 refinement on real STEM examples"
    )
    parser.add_argument(
        "--amorphous",
        type=Path,
        default=Path("outputs/reference_amorphous_sam2_live_v1"),
    )
    parser.add_argument(
        "--interface",
        type=Path,
        default=Path("outputs/zro2_phase_boundary_sam2_live_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sam2_real_instance_gallery_v1"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_gallery(args.amorphous, args.interface, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
