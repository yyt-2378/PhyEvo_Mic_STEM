#!/usr/bin/env python3
"""Visual synthetic validation for candidate-local SAM3 fallback.

The proposals are deterministic synthetic masks, not outputs from SAM3.  This
script isolates the acceptance/composition contract and reports boundary gain
against known digital-twin truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tfree_detection.config import SAM3RefinementConfig
from tfree_detection.evaluation import boundary_refinement_comparison
from tfree_detection.stage2.sam3_refine import (
    SAM3Proposal,
    build_physics_prompt_plan,
    evaluate_candidate_sam3_proposals,
)


def _ellipse(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    radii_xy: tuple[float, float],
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    cx, cy = center_xy
    rx, ry = radii_xy
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def _base_image(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    lattice = (
        np.cos(2 * np.pi * xx / 8.0)
        + np.cos(2 * np.pi * yy / 8.0)
    )
    return np.clip(0.50 + 0.17 * lattice, 0, 1)


def _contours(
    axis: plt.Axes,
    image: np.ndarray,
    masks: list[tuple[np.ndarray, str, str]],
) -> None:
    axis.imshow(image, cmap="gray", vmin=0, vmax=1)
    for mask, color, style in masks:
        if np.any(mask):
            axis.contour(
                mask.astype(float),
                levels=[0.5],
                colors=[color],
                linewidths=1.5,
                linestyles=style,
            )
    axis.set_axis_off()


def build(output_directory: Path) -> dict:
    if output_directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite validation output: {output_directory}"
        )
    output_directory.mkdir(parents=True)
    shape = (144, 240)
    truth_left = _ellipse(shape, (68, 72), (35, 28))
    truth_right = _ellipse(shape, (176, 72), (31, 25))
    truth = truth_left | truth_right
    coarse_left = _ellipse(shape, (62, 72), (35, 28))
    coarse_right = truth_right.copy()
    coarse = coarse_left | coarse_right
    score = np.zeros(shape, dtype=np.float32)
    score[truth] = 1.0
    image = _base_image(shape)
    image[truth] = 0.52

    config = SAM3RefinementConfig(
        ensemble_size=8,
        box_padding_pixels=12,
        box_padding_lattice_cells=0,
        minimum_boundary_evidence_alignment=0.0,
    )
    plan = build_physics_prompt_plan(coarse, score, {}, config)
    accepted_geometry = truth_left
    rejected_geometry = truth_right.copy()
    rejected_core = plan.physics_core & (plan.component_labels == 2)
    removed = max(1, int(np.ceil(rejected_core.sum() * 0.06)))
    rejected_geometry.flat[
        np.flatnonzero(rejected_core)[:removed]
    ] = False
    proposals = []
    for candidate_id, component_label, geometry in (
        ("candidate-0001", 1, accepted_geometry),
        ("candidate-0002", 2, rejected_geometry),
    ):
        for ensemble_index in range(config.ensemble_size):
            proposals.append(
                SAM3Proposal(
                    proposal_id=f"{candidate_id}-{ensemble_index:03d}",
                    candidate_id=candidate_id,
                    component_label=component_label,
                    mask=geometry.copy(),
                    score=0.95,
                    ensemble_index=ensemble_index,
                )
            )
    refinement = evaluate_candidate_sam3_proposals(
        coarse, score, plan, proposals, config
    )
    comparison = boundary_refinement_comparison(
        coarse,
        refinement.final_mask,
        truth,
        tolerance_pixels=1.0,
        physics_core=plan.physics_core,
        negative_reference=plan.negative_reference,
    )

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    _contours(
        axes[0, 0],
        image,
        [(truth, "#30d158", "-"), (coarse, "#ff453a", "--")],
    )
    axes[0, 0].set_title("1. Digital-twin truth (green) / coarse (red)")

    axes[0, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].imshow(
        np.ma.masked_where(
            refinement.inclusion_probability <= 0,
            refinement.inclusion_probability,
        ),
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.72,
    )
    axes[0, 1].set_title("2. Candidate-local proposal consensus")
    axes[0, 1].set_axis_off()

    _contours(
        axes[0, 2],
        image,
        [
            (
                refinement.candidate_results[0].final_mask,
                "#00e5ff",
                "-",
            ),
            (
                refinement.candidate_results[1].final_mask,
                "#ffd60a",
                "-",
            ),
        ],
    )
    axes[0, 2].set_title(
        "3. Final: accepted refinement (cyan) / fallback (yellow)"
    )

    coarse_error = np.zeros((*shape, 3), dtype=float)
    coarse_error[coarse & truth] = (0.18, 0.82, 0.38)
    coarse_error[coarse & ~truth] = (1.0, 0.27, 0.23)
    coarse_error[~coarse & truth] = (1.0, 0.84, 0.04)
    axes[1, 0].imshow(coarse_error)
    axes[1, 0].set_title("4. Coarse error: TP green / FP red / FN yellow")
    axes[1, 0].set_axis_off()

    final = refinement.final_mask
    final_error = np.zeros((*shape, 3), dtype=float)
    final_error[final & truth] = (0.18, 0.82, 0.38)
    final_error[final & ~truth] = (1.0, 0.27, 0.23)
    final_error[~final & truth] = (1.0, 0.84, 0.04)
    axes[1, 1].imshow(final_error)
    axes[1, 1].set_title("5. Final error after local composition")
    axes[1, 1].set_axis_off()

    axes[1, 2].axis("off")
    rows = [
        ["Metric", "Coarse", "Final", "Delta"],
        [
            "Boundary F1",
            f"{comparison['coarse']['boundary_f1']:.3f}",
            f"{comparison['final']['boundary_f1']:.3f}",
            f"{comparison['boundary_f1_delta']:+.3f}",
        ],
        [
            "ASSD (px)",
            (
                f"{comparison['coarse']['average_symmetric_surface_distance']:.3f}"
            ),
            (
                f"{comparison['final']['average_symmetric_surface_distance']:.3f}"
            ),
            f"{comparison['assd_delta']:+.3f}",
        ],
        [
            "HD95 (px)",
            f"{comparison['coarse']['hausdorff_distance_95']:.3f}",
            f"{comparison['final']['hausdorff_distance_95']:.3f}",
            f"{comparison['hd95_delta']:+.3f}",
        ],
    ]
    table = axes[1, 2].table(
        cellText=rows,
        cellLoc="center",
        loc="center",
        colWidths=[0.31, 0.20, 0.20, 0.29],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    axes[1, 2].text(
        0.5,
        0.16,
        "Image status: PARTIALLY ACCEPTED\n"
        "candidate-0001 refined · candidate-0002 local fallback",
        ha="center",
        va="center",
        fontsize=9,
        color="#1f2937",
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].set_title("6. Boundary gain and fallback audit")
    figure.suptitle(
        "Synthetic contract validation — proposals are not live SAM3 output",
        fontsize=14,
    )
    figure.tight_layout()

    figure_path = output_directory / "candidate_local_fallback_validation.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report = {
        "validation_kind": "synthetic_contract_not_live_sam3_accuracy",
        "refinement": refinement.to_summary(),
        "boundary_comparison": comparison,
        "figure": figure_path.name,
    }
    report_path = output_directory / "validation_report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
