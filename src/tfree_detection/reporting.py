"""Reproducible artifact and visualization writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from scipy.ndimage import distance_transform_edt
from skimage.io import imsave

from .config import DetectorConfig
from .governance import candidate_records_from_result, write_candidate_jsonl
from .models import DetectionResult


def _draw_overlay(
    axis: plt.Axes, result: DetectionResult, show_labels: bool = True
) -> None:
    axis.imshow(result.normalized_image, cmap="gray", vmin=0, vmax=1)
    geometry = result.final_geometry_mask
    overlay = np.ma.masked_where(~geometry, geometry)
    axis.imshow(overlay, cmap="autumn", alpha=0.42, vmin=0, vmax=1)
    if result.boundary_refinement is not None and np.any(result.mask):
        axis.contour(
            result.mask.astype(float),
            levels=[0.5],
            colors=["#ff2d2d"],
            linewidths=0.9,
            linestyles="dashed",
        )
    if np.any(geometry):
        axis.contour(
            geometry.astype(float),
            levels=[0.5],
            colors=["#00e5ff" if result.boundary_refinement else "#ff2d2d"],
            linewidths=1.1,
        )
    if show_labels:
        maximum_labels = 30
        shown_regions = (
            result.regions
            if len(result.regions) <= maximum_labels
            else sorted(
                result.regions,
                key=lambda region: region.max_score,
                reverse=True,
            )[:maximum_labels]
        )
        for region in shown_regions:
            geometry = result.final_geometry_for_region(region)
            coordinates = np.argwhere(geometry)
            if len(coordinates):
                y1, x1 = coordinates.min(axis=0)
                y2, x2 = coordinates.max(axis=0) + 1
                x1, y1, x2, y2 = (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )
            else:
                x1, y1, x2, y2 = region.bbox_xyxy
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="#00e5ff",
                    linewidth=1.2,
                )
            )
            axis.text(
                x1,
                max(0, y1 - 4),
                f"R{region.region_id}",
                color="#00e5ff",
                fontsize=8,
                weight="bold",
            )
        if len(result.regions) > maximum_labels:
            axis.text(
                8,
                18,
                f"{len(result.regions)} candidates; boxes show top {maximum_labels}",
                color="#00e5ff",
                fontsize=8,
                weight="bold",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
            )
    axis.set_axis_off()


def _save_overlay(
    result: DetectionResult, path: Path, dpi: int
) -> None:
    height, width = result.image_shape
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = figure.add_axes([0, 0, 1, 1])
    _draw_overlay(axis, result)
    figure.savefig(path, dpi=dpi, pad_inches=0)
    plt.close(figure)


def _save_heatmap(
    result: DetectionResult, path: Path, dpi: int
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    shown = axis.imshow(result.score_map, cmap="magma")
    axis.contour(
        result.mask.astype(float),
        levels=[0.5],
        colors=["cyan"],
        linewidths=0.8,
    ) if np.any(result.mask) else None
    titles = {
        "amorphous_region": "Training-free amorphization score",
        "atomic_point_defect": "Atomic-site defect score",
        "stacking_fault": "Lattice phase-slip score",
        "phase_boundary": "Two-phase interface score",
    }
    axis.set_title(titles.get(result.task, "Training-free anomaly score"))
    axis.set_axis_off()
    figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_diagnostics(
    result: DetectionResult, path: Path, dpi: int
) -> None:
    panels = [
        ("Normalized input", result.normalized_image, "gray"),
        ("Fused score", result.score_map, "magma"),
        ("Probability", result.probability_map, "viridis"),
    ]
    for name, values in result.evidence_maps.items():
        if not name.startswith("group_"):
            continue
        panels.append(
            (
                name.replace("group_", "").replace("_", " ").title(),
                values,
                "magma",
            )
        )
    for name in (
        "domain_label",
        "domain_membership_margin",
        "lattice_coherence",
        "strain_magnitude",
        "expected_lattice_sites",
        "observed_atomic_peaks",
    ):
        if name not in result.evidence_maps:
            continue
        cmap = "tab10" if name == "domain_label" else "viridis"
        panels.append(
            (
                name.replace("_", " ").title(),
                result.evidence_maps[name],
                cmap,
            )
        )

    columns = 3
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(14, max(4.5 * rows, 5))
    )
    axes_array = np.atleast_1d(axes).ravel()
    for axis, panel in zip(axes_array, panels):
        title, values, cmap = panel
        shown = axis.imshow(values, cmap=cmap)
        axis.set_title(title)
        axis.set_axis_off()
        if cmap != "gray":
            figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.03)
    for axis in axes_array[len(panels) :]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _robust_limits(values: np.ndarray, signed: bool = False) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    if signed:
        limit = float(np.percentile(np.abs(finite), 98))
        return -limit, limit
    lower, upper = np.percentile(finite, [2, 98])
    return float(lower), float(upper)


def _save_stage2_diagnostics(
    result: DetectionResult, path: Path, dpi: int
) -> None:
    maps = result.evidence_maps
    panels = [
        ("Input and candidates", result.normalized_image, "gray", False),
        ("GPA strain magnitude", maps["gpa_strain_magnitude"], "magma", False),
        ("Lattice coherence", maps["gpa_lattice_coherence"], "viridis", False),
        ("Normal strain exx", maps["gpa_exx"], "coolwarm", True),
        ("Normal strain eyy", maps["gpa_eyy"], "coolwarm", True),
        ("Shear strain exy", maps["gpa_exy"], "coolwarm", True),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 9))
    validity = maps.get("gpa_validity")
    for axis, (title, values, cmap, signed) in zip(axes.flat, panels):
        displayed = (
            np.ma.masked_where(validity < 0.5, values)
            if validity is not None and cmap != "gray"
            else values
        )
        if cmap == "gray":
            axis.imshow(displayed, cmap=cmap, vmin=0, vmax=1)
        else:
            valid_values = (
                values[validity >= 0.5] if validity is not None else values
            )
            lower, upper = _robust_limits(valid_values, signed=signed)
            shown = axis.imshow(
                displayed, cmap=cmap, vmin=lower, vmax=upper
            )
            figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.03)
        if np.any(result.mask):
            axis.contour(
                result.mask.astype(float),
                levels=[0.5],
                colors=["cyan"],
                linewidths=0.7,
            )
        axis.set_title(title)
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_evidence_protocol_diagnostics(
    result: DetectionResult, path: Path, dpi: int
) -> None:
    if (
        result.evidence_bundle is None
        or result.reliability_map is None
        or result.decision_map is None
        or not result.hypothesis_maps
    ):
        raise ValueError("Unified evidence diagnostics require protocol outputs")
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis in axes.flat:
        axis.imshow(
            result.normalized_image,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axis.set_axis_off()

    axes[0, 0].set_title("Input + physics-teacher candidate")
    if np.any(result.mask):
        axes[0, 0].contour(
            result.mask.astype(float),
            levels=[0.5],
            colors=["#00e5ff"],
            linewidths=1.0,
        )

    reliability = result.reliability_map
    shown_reliability = axes[0, 1].imshow(
        np.ma.masked_where(reliability <= 0, reliability),
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.62,
    )
    axes[0, 1].set_title("Measurement reliability q(x)")
    figure.colorbar(
        shown_reliability, ax=axes[0, 1], fraction=0.046, pad=0.03
    )

    decision_colors = [
        "#000000",
        "#ef233c",
        "#ffb703",
        "#c77dff",
        "#6c757d",
    ]
    decision_cmap = ListedColormap(decision_colors)
    axes[1, 0].imshow(
        np.ma.masked_where(result.decision_map == 0, result.decision_map),
        cmap=decision_cmap,
        vmin=0,
        vmax=4,
        alpha=0.70,
        interpolation="nearest",
    )
    axes[1, 0].set_title("Abstention-aware decision state")
    axes[1, 0].legend(
        handles=[
            Patch(color=decision_colors[1], label="confirmed physics candidate"),
            Patch(color=decision_colors[2], label="uncertain candidate"),
            Patch(color=decision_colors[3], label="out of domain"),
            Patch(color=decision_colors[4], label="invalid measurement"),
        ],
        loc="lower left",
        fontsize=7,
        framealpha=0.75,
    )

    hypothesis_name, energy = next(iter(result.hypothesis_maps.items()))
    valid_energy = energy[np.isfinite(energy)]
    upper = (
        max(float(np.percentile(valid_energy, 99)), 1e-8)
        if valid_energy.size
        else 1.0
    )
    normalized = np.clip(energy / upper, 0, 1)
    shown_energy = axes[1, 1].imshow(
        np.ma.masked_where(normalized <= 0, energy),
        cmap="magma",
        vmin=0,
        vmax=upper,
        alpha=0.18 + 0.58 * normalized,
    )
    axes[1, 1].set_title(
        "Unified hypothesis energy\n" + hypothesis_name.replace("_", " ")
    )
    figure.colorbar(
        shown_energy, ax=axes[1, 1], fraction=0.046, pad=0.03
    )
    figure.suptitle(
        "Structural evidence protocol — overlays retain the atomic image",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_sam3_boundary_diagnostics(
    result: DetectionResult, path: Path, dpi: int
) -> None:
    refinement = result.boundary_refinement
    if refinement is None:
        raise ValueError("SAM3 diagnostics require a refinement result")
    backend = str(refinement.provenance.get("backend", "sam3")).lower()
    sam_label = {
        "sam2": "SAM 2",
        "sam3": "SAM 3",
    }.get(backend, backend.upper())
    figure, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    image = result.normalized_image
    coarse = refinement.coarse_mask

    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
    if np.any(coarse):
        axes[0, 0].contour(
            coarse.astype(float),
            levels=[0.5],
            colors=["#ff3b30"],
            linewidths=1.0,
        )
    axes[0, 0].set_title("1. Physics coarse candidate (red)")

    axes[0, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
    shown_prompts = [
        prompt for prompt in refinement.prompts
        if prompt.ensemble_index == 0
    ]
    for prompt in shown_prompts:
        x1, y1, x2, y2 = prompt.box_xyxy
        axes[0, 1].add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="#00e5ff",
                linewidth=1.0,
            )
        )
        if prompt.positive_points_xy:
            positive = np.asarray(prompt.positive_points_xy)
            axes[0, 1].scatter(
                positive[:, 0],
                positive[:, 1],
                c="#35c759",
                marker="+",
                s=28,
            )
        if prompt.negative_points_xy:
            negative = np.asarray(prompt.negative_points_xy)
            axes[0, 1].scatter(
                negative[:, 0],
                negative[:, 1],
                c="#ff453a",
                marker="x",
                s=20,
            )
    axes[0, 1].set_title("2. Physics prompts (ensemble 0)")

    axes[0, 2].imshow(image, cmap="gray", vmin=0, vmax=1)
    shown_probability = axes[0, 2].imshow(
        np.ma.masked_where(
            refinement.inclusion_probability <= 0,
            refinement.inclusion_probability,
        ),
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.72,
    )
    axes[0, 2].set_title("3. Proposal inclusion probability")
    figure.colorbar(
        shown_probability, ax=axes[0, 2], fraction=0.046, pad=0.03
    )

    consensus_state = np.zeros(result.image_shape, dtype=np.uint8)
    consensus_state[refinement.possible_extent] = 1
    consensus_state[refinement.refined_mask] = 2
    consensus_state[refinement.stable_core] = 3
    consensus_cmap = ListedColormap(
        ["#000000", "#ffd60a", "#0a84ff", "#30d158"]
    )
    axes[1, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].imshow(
        np.ma.masked_where(consensus_state == 0, consensus_state),
        cmap=consensus_cmap,
        vmin=0,
        vmax=3,
        alpha=0.60,
        interpolation="nearest",
    )
    axes[1, 0].set_title(
        "4. Possible / selected / stable consensus"
    )

    axes[1, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
    shown_uncertainty = axes[1, 1].imshow(
        np.ma.masked_where(
            refinement.boundary_uncertainty <= 0,
            refinement.boundary_uncertainty,
        ),
        cmap="magma",
        vmin=0,
        vmax=1,
        alpha=0.72,
    )
    axes[1, 1].set_title("5. Boundary uncertainty")
    figure.colorbar(
        shown_uncertainty, ax=axes[1, 1], fraction=0.046, pad=0.03
    )

    axes[1, 2].imshow(image, cmap="gray", vmin=0, vmax=1)
    if np.any(coarse):
        axes[1, 2].contour(
            coarse.astype(float),
            levels=[0.5],
            colors=["#ff453a"],
            linewidths=0.9,
            linestyles="dashed",
        )
    if refinement.candidate_results:
        for candidate in refinement.candidate_results:
            if np.any(candidate.final_mask):
                axes[1, 2].contour(
                    candidate.final_mask.astype(float),
                    levels=[0.5],
                    colors=[
                        "#00e5ff" if candidate.accepted else "#ffd60a"
                    ],
                    linewidths=1.2,
                )
    elif np.any(refinement.final_mask):
        axes[1, 2].contour(
            refinement.final_mask.astype(float),
            levels=[0.5],
            colors=["#00e5ff"],
            linewidths=1.2,
        )
    accepted_count = sum(
        item.accepted for item in refinement.candidate_results
    )
    candidate_count = len(refinement.candidate_results)
    decision = (
        f"{accepted_count}/{candidate_count} CANDIDATES REFINED"
        if candidate_count
        else refinement.status.upper().replace("_", " ")
    )
    axes[1, 2].set_title(f"6. Final geometry — {decision}")

    for axis in axes.flat:
        axis.set_axis_off()
    figure.suptitle(
        f"{sam_label} is a geometry proposer; "
        "physics owns candidate existence",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_strain_overlay(
    result: DetectionResult, path: Path, dpi: int, transition_radius: int
) -> None:
    strain = result.evidence_maps["gpa_strain_magnitude"]
    validity = result.evidence_maps.get(
        "gpa_validity", np.ones_like(strain)
    )
    lower, upper = _robust_limits(strain[validity >= 0.5])
    distance = distance_transform_edt(~result.mask)
    transition_zone = distance <= transition_radius
    visible_values = strain[transition_zone & (validity >= 0.5)]
    visible_threshold = (
        float(np.percentile(visible_values, 35))
        if visible_values.size
        else lower
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(result.normalized_image, cmap="gray", vmin=0, vmax=1)
    shown = axis.imshow(
        np.ma.masked_where(
            (strain < visible_threshold)
            | ~transition_zone
            | (validity < 0.5),
            strain,
        ),
        cmap="magma",
        vmin=lower,
        vmax=upper,
        alpha=0.68,
    )
    if np.any(result.mask):
        axis.contour(
            result.mask.astype(float),
            levels=[0.5],
            colors=["cyan"],
            linewidths=0.9,
        )
        axis.contour(
            distance,
            levels=[transition_radius],
            colors=["#ffe066"],
            linewidths=0.7,
            linestyles="dashed",
        )
    axis.set_title("GPA deformation field around stage-1 candidates")
    axis.set_axis_off()
    figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_result(
    result: DetectionResult,
    config: DetectorConfig,
    output_directory: str | Path,
) -> Dict[str, str]:
    """Write the complete, inspectable output contract for one run."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    candidate_table = output / "candidate_table.jsonl"
    if result.evidence_bundle is not None and candidate_table.exists():
        raise FileExistsError(
            "Refusing to overwrite candidate provenance table; use a new "
            f"versioned output directory: {candidate_table}"
        )
    artifacts = {
        "summary": output / "summary.json",
        "resolved_config": output / "config_resolved.yaml",
        "mask": output / "mask.png",
        "final_geometry_mask": output / "final_geometry_mask.png",
        "overlay": output / "overlay.png",
        "heatmap": output / "score_heatmap.png",
        "diagnostics": output / "diagnostics.png",
    }
    if result.evidence_bundle is not None:
        artifacts["evidence_protocol"] = output / "evidence_protocol.json"
        artifacts["candidate_table"] = candidate_table
        artifacts["evidence_diagnostics"] = (
            output / "evidence_protocol_diagnostics.png"
        )
    with artifacts["summary"].open("w", encoding="utf-8") as stream:
        json.dump(result.summary(), stream, ensure_ascii=False, indent=2)
    with artifacts["resolved_config"].open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.to_dict(), stream, allow_unicode=True, sort_keys=False)
    if result.evidence_bundle is not None:
        with artifacts["evidence_protocol"].open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                result.evidence_bundle.protocol_dict(),
                stream,
                ensure_ascii=False,
                indent=2,
            )
        write_candidate_jsonl(
            candidate_records_from_result(result),
            artifacts["candidate_table"],
        )
        _save_evidence_protocol_diagnostics(
            result,
            artifacts["evidence_diagnostics"],
            config.output.figure_dpi,
        )
    imsave(
        str(artifacts["mask"]),
        (result.mask.astype(np.uint8) * 255),
        check_contrast=False,
    )
    imsave(
        str(artifacts["final_geometry_mask"]),
        (result.final_geometry_mask.astype(np.uint8) * 255),
        check_contrast=False,
    )
    _save_overlay(result, artifacts["overlay"], config.output.figure_dpi)
    _save_heatmap(result, artifacts["heatmap"], config.output.figure_dpi)
    _save_diagnostics(result, artifacts["diagnostics"], config.output.figure_dpi)
    if "gpa_strain_magnitude" in result.evidence_maps:
        stage2_artifacts = {
            "stage2_summary": output / "stage2_summary.json",
            "stage2_diagnostics": output / "stage2_diagnostics.png",
            "strain_overlay": output / "strain_overlay.png",
        }
        with stage2_artifacts["stage2_summary"].open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                result.metadata["stage2"],
                stream,
                ensure_ascii=False,
                indent=2,
            )
        _save_stage2_diagnostics(
            result,
            stage2_artifacts["stage2_diagnostics"],
            config.output.figure_dpi,
        )
        _save_strain_overlay(
            result,
            stage2_artifacts["strain_overlay"],
            config.output.figure_dpi,
            config.stage2.phase.transition_outer_radius,
        )
        artifacts.update(stage2_artifacts)

    if result.boundary_refinement is not None:
        refinement = result.boundary_refinement
        sam3_artifacts = {
            "sam3_summary": output / "sam3_refinement_summary.json",
            "sam3_coarse_mask": output / "sam3_coarse_mask.png",
            "sam3_refined_proposal": output / "sam3_refined_proposal.png",
            "sam3_final_geometry": output / "sam3_final_geometry.png",
            "sam3_stable_core": output / "sam3_stable_core.png",
            "sam3_possible_extent": output / "sam3_possible_extent.png",
            "sam3_boundary_uncertainty": (
                output / "sam3_boundary_uncertainty.png"
            ),
            "sam3_diagnostics": output / "sam3_diagnostics.png",
        }
        existing = [
            path for path in sam3_artifacts.values() if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite SAM3 artifacts: "
                + ", ".join(str(path) for path in existing)
            )
        with sam3_artifacts["sam3_summary"].open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(
                refinement.to_summary(),
                stream,
                ensure_ascii=False,
                indent=2,
            )
        for name, values in (
            ("sam3_coarse_mask", refinement.coarse_mask),
            ("sam3_refined_proposal", refinement.refined_mask),
            ("sam3_final_geometry", refinement.final_mask),
            ("sam3_stable_core", refinement.stable_core),
            ("sam3_possible_extent", refinement.possible_extent),
        ):
            imsave(
                str(sam3_artifacts[name]),
                np.asarray(values, dtype=np.uint8) * 255,
                check_contrast=False,
            )
        imsave(
            str(sam3_artifacts["sam3_boundary_uncertainty"]),
            np.round(
                np.clip(refinement.boundary_uncertainty, 0, 1) * 255
            ).astype(np.uint8),
            check_contrast=False,
        )
        _save_sam3_boundary_diagnostics(
            result,
            sam3_artifacts["sam3_diagnostics"],
            config.output.figure_dpi,
        )
        artifacts.update(sam3_artifacts)

    if config.output.save_numpy_arrays:
        numpy_artifacts = {
            "score_array": output / "score_map.npy",
            "probability_array": output / "probability_map.npy",
            "mask_array": output / "mask.npy",
            "final_geometry_array": output / "final_geometry_mask.npy",
            "feature_archive": output / "feature_maps.npz",
            "evidence_archive": output / "evidence_maps.npz",
        }
        if result.reliability_map is not None:
            numpy_artifacts["reliability_array"] = (
                output / "reliability_map.npy"
            )
        if result.decision_map is not None:
            numpy_artifacts["decision_array"] = output / "decision_map.npy"
        if result.hypothesis_maps:
            numpy_artifacts["hypothesis_archive"] = (
                output / "hypothesis_maps.npz"
            )
        if result.evidence_bundle is not None:
            numpy_artifacts["evidence_reliability_archive"] = (
                output / "evidence_reliability_maps.npz"
            )
        if result.boundary_refinement is not None:
            numpy_artifacts["sam3_consensus_archive"] = (
                output / "sam3_consensus_maps.npz"
            )
            if result.boundary_refinement.proposals:
                numpy_artifacts["sam3_proposal_archive"] = (
                    output / "sam3_proposals.npz"
                )
        np.save(numpy_artifacts["score_array"], result.score_map)
        np.save(numpy_artifacts["probability_array"], result.probability_map)
        np.save(numpy_artifacts["mask_array"], result.mask)
        np.save(
            numpy_artifacts["final_geometry_array"],
            result.final_geometry_mask,
        )
        np.savez_compressed(numpy_artifacts["feature_archive"], **result.feature_maps)
        np.savez_compressed(numpy_artifacts["evidence_archive"], **result.evidence_maps)
        if result.reliability_map is not None:
            np.save(
                numpy_artifacts["reliability_array"],
                result.reliability_map,
            )
        if result.decision_map is not None:
            np.save(numpy_artifacts["decision_array"], result.decision_map)
        if result.hypothesis_maps:
            np.savez_compressed(
                numpy_artifacts["hypothesis_archive"],
                **result.hypothesis_maps,
            )
        if result.evidence_bundle is not None:
            np.savez_compressed(
                numpy_artifacts["evidence_reliability_archive"],
                **{
                    name.replace(".", "__"): evidence.reliability
                    for name, evidence in result.evidence_bundle.fields.items()
                },
            )
        if result.boundary_refinement is not None:
            refinement = result.boundary_refinement
            np.savez_compressed(
                numpy_artifacts["sam3_consensus_archive"],
                coarse_mask=refinement.coarse_mask,
                inclusion_probability=refinement.inclusion_probability,
                refined_mask=refinement.refined_mask,
                stable_core=refinement.stable_core,
                possible_extent=refinement.possible_extent,
                boundary_uncertainty=refinement.boundary_uncertainty,
                final_mask=refinement.final_mask,
                physics_core=refinement.physics_core,
                negative_reference=refinement.negative_reference,
            )
            if refinement.proposals:
                np.savez_compressed(
                    numpy_artifacts["sam3_proposal_archive"],
                    **{
                        proposal.proposal_id.replace("-", "_"): (
                            proposal.mask
                        )
                        for proposal in refinement.proposals
                    },
                )
        artifacts.update(numpy_artifacts)
    return {name: str(path.resolve()) for name, path in artifacts.items()}
