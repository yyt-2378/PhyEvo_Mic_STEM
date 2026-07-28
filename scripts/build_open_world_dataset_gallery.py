#!/usr/bin/env python3
"""Run deterministic cross-dataset smoke tests and build atomic-preserving figures."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy import ndimage
from skimage.io import imread
from skimage.measure import label

from tfree_detection.config import load_config
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.models import DetectionRegion, DetectionResult
from tfree_detection.reporting import save_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_COLORS = {
    "probable": "#00d084",
    "unresolved": "#ffb000",
    "artifact": "#ff3cac",
    "out_of_domain": "#8b5cf6",
    "invalid_measurement": "#9ca3af",
}


@dataclass(frozen=True)
class CaseSpec:
    slug: str
    dataset: str
    task_label: str
    source: str
    config: str
    evidence_title: str
    evidence_keys: tuple[str, ...]
    selection: dict[str, Any]
    selection_rule: str
    reference_label: str | None
    gold_status: str
    interpretation_limit: str


@dataclass
class CaseView:
    spec: CaseSpec
    normalized_image: np.ndarray
    mask: np.ndarray
    reliability: np.ndarray
    primary_evidence: np.ndarray
    regions: list[DetectionRegion]
    hypothesis_maps: list[tuple[str, np.ndarray]]
    summary: dict[str, Any]
    reference_points_xy: np.ndarray | None
    run_directory: Path
    detail_figure: Path


CASES = (
    CaseSpec(
        slug="zeolite_local_amorphization",
        dataset="Stretched zeolite (local)",
        task_label="Amorphous-region branch",
        source="reference_scale_radial_wiener.png",
        config="configs/amorphous.yaml",
        evidence_title="Reciprocal-order loss",
        evidence_keys=("group_spectrum",),
        selection={},
        selection_rule="the only user-provided reference image",
        reference_label=None,
        gold_status="no pixel-wise Gold mask",
        interpretation_limit=(
            "Candidates indicate local order loss; subtype resolution is evidence-based, "
            "not a ground-truth accuracy claim."
        ),
    ),
    CaseSpec(
        slug="emseek_wse2_image_1",
        dataset="EMSeek WSe2 image 1",
        task_label="Atomic-site branch",
        source="datasets/public/emseek_wse2_atomic_defects/image/1.png",
        config="configs/atomic_point_defect.yaml",
        evidence_title="Atomic-site residual",
        evidence_keys=(
            "group_presence",
            "group_intensity",
            "group_displacement",
            "group_extra_peak",
        ),
        selection={},
        selection_rule="lowest numeric image identifier; no outcome-based selection",
        reference_label=(
            "datasets/public/emseek_wse2_atomic_defects/label/1.png"
        ),
        gold_status="author point annotations; development reference, not calibrated Gold",
        interpretation_limit=(
            "The author points are shown for qualitative alignment only; this single "
            "image is not used to estimate platform accuracy."
        ),
    ),
    CaseSpec(
        slug="guo_al_partial_dislocation",
        dataset="Guo Al partial dislocation",
        task_label="Stacking-fault / line branch",
        source=(
            "datasets/public/guo_2021_atomic_defects/"
            "relaxed_single_Al_partial_dislocations_verified.tif"
        ),
        config="configs/public_al_stacking_fault.yaml",
        evidence_title="Lattice phase jump",
        evidence_keys=("group_phase_jump",),
        selection={},
        selection_rule="named public verified example; full 2-D field",
        reference_label=None,
        gold_status="semantic file description; no pixel-wise Gold mask",
        interpretation_limit=(
            "The detector localizes phase-slip evidence; a dislocation identity still "
            "requires topology and/or expert confirmation."
        ),
    ),
    CaseSpec(
        slug="guo_zro2_twin_boundary",
        dataset="Guo monoclinic ZrO2 twin",
        task_label="Phase-interface branch",
        source=(
            "datasets/public/guo_2021_atomic_defects/"
            "ZrO2_monoclinic_twin_boundary.tif"
        ),
        config="configs/phase_boundary.yaml",
        evidence_title="Interface score",
        evidence_keys=(),
        selection={},
        selection_rule="named public twin-boundary example; full 2-D field",
        reference_label=None,
        gold_status="semantic file description; no pixel-wise Gold boundary",
        interpretation_limit=(
            "The branch detects reciprocal-signature interfaces and does not silently "
            "equate every interface with a crystallographic twin."
        ),
    ),
    CaseSpec(
        slug="guo_superscan_frame_0",
        dataset="Guo SuperScan stack",
        task_label="Atomic-site branch",
        source=(
            "datasets/public/guo_2021_atomic_defects/"
            "Recording_of_SuperScan_img_stack.ndata1"
        ),
        config="configs/atomic_point_defect.yaml",
        evidence_title="Atomic-site residual",
        evidence_keys=(
            "group_presence",
            "group_intensity",
            "group_displacement",
            "group_extra_peak",
        ),
        selection={"frame_index": 0},
        selection_rule="frame 0 fixed before detection; no temporal cherry-picking",
        reference_label=None,
        gold_status="no candidate-level Gold labels",
        interpretation_limit=(
            "This is a stack-I/O and candidate-generation test, not an accuracy score."
        ),
    ),
    CaseSpec(
        slug="al_sigma3_ori_channel_0_mean",
        dataset="Al Sigma3 ORI EMD",
        task_label="Phase-interface branch (smoke only)",
        source="datasets/public/al_sigma3_twin_boundaries/Sigma3_ORI.emd",
        config="configs/phase_boundary.yaml",
        evidence_title="Interface score",
        evidence_keys=(),
        selection={
            "dataset_path": (
                "Data/Image/36374cdd2f174c239f19db0eaf17de05/Data"
            ),
            "projection": "mean",
        },
        selection_rule=(
            "lexicographically first EMD image dataset and fixed mean projection"
        ),
        reference_label=None,
        gold_status="channel identity unresolved; no Gold boundary",
        interpretation_limit=(
            "Detector output is smoke-only because the EMD UUID channel has not been "
            "mapped to a physical detector signal."
        ),
    ),
)


def _project_path(relative: str) -> Path:
    candidate = (PROJECT_ROOT / relative).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes project root: {candidate}") from error
    return candidate


def _state_counts(regions: list[DetectionRegion]) -> dict[str, int]:
    return dict(sorted(Counter(region.epistemic_state for region in regions).items()))


def _top_labels(regions: list[DetectionRegion]) -> list[dict[str, Any]]:
    counts = Counter(
        region.resolved_label or region.defect_type or "unresolved"
        for region in regions
    )
    return [
        {"label": label, "count": count}
        for label, count in counts.most_common(4)
    ]


def _reference_points(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    label_image = np.asarray(imread(path))
    if label_image.ndim == 3:
        mask = np.any(label_image[..., :3] != 0, axis=-1)
    else:
        mask = label_image != 0
    components, count = ndimage.label(mask)
    if count == 0:
        return np.empty((0, 2), dtype=float)
    centers_yx = np.asarray(
        ndimage.center_of_mass(mask, components, range(1, count + 1)),
        dtype=float,
    )
    return centers_yx[:, ::-1]


def _combined_evidence(
    result: DetectionResult, keys: tuple[str, ...]
) -> np.ndarray:
    if not keys:
        return np.asarray(result.score_map, dtype=np.float32)
    available = [
        np.asarray(result.evidence_maps[key], dtype=np.float32)
        for key in keys
        if key in result.evidence_maps
    ]
    if not available:
        return np.asarray(result.score_map, dtype=np.float32)
    return np.maximum.reduce(available)


def _rank_hypothesis_maps(
    result: DetectionResult,
) -> list[tuple[str, np.ndarray]]:
    ranked: list[tuple[float, str, np.ndarray]] = []
    support = result.mask
    for name, values in result.hypothesis_maps.items():
        array = np.asarray(values, dtype=np.float32)
        sample = array[support] if np.any(support) else array[np.isfinite(array)]
        score = float(np.mean(sample)) if sample.size else 0.0
        ranked.append((score, name, array))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(name, values) for _, name, values in ranked[:2]]


def _run_case(spec: CaseSpec, output_root: Path) -> CaseView:
    config = load_config(_project_path(spec.config))
    config.output.save_numpy_arrays = False
    config.output.save_feature_maps = False
    config.output.figure_dpi = 130
    framework = TrainingFreeDetectionFramework(config)
    result = framework.detect(
        _project_path(spec.source),
        **spec.selection,
    )
    run_directory = output_root / "runs" / spec.slug
    save_result(result, config, run_directory)
    detail_figure = output_root / "figures" / f"{spec.slug}.png"
    reference_path = (
        _project_path(spec.reference_label) if spec.reference_label else None
    )
    return CaseView(
        spec=spec,
        normalized_image=np.asarray(result.normalized_image, dtype=np.float32),
        mask=np.asarray(result.mask, dtype=bool),
        reliability=(
            np.asarray(result.reliability_map, dtype=np.float32)
            if result.reliability_map is not None
            else np.zeros(result.image_shape, dtype=np.float32)
        ),
        primary_evidence=_combined_evidence(result, spec.evidence_keys),
        regions=list(result.regions),
        hypothesis_maps=_rank_hypothesis_maps(result),
        summary=result.summary(),
        reference_points_xy=_reference_points(reference_path),
        run_directory=run_directory,
        detail_figure=detail_figure,
    )


def _finish_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#64748b")
        spine.set_linewidth(0.6)


def _show_raw(axis: plt.Axes, case: CaseView, title: str) -> None:
    axis.imshow(
        case.normalized_image,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axis.set_title(title, fontsize=10, pad=5)
    _finish_axis(axis)


def _draw_candidate_states(
    axis: plt.Axes,
    case: CaseView,
    *,
    show_legend: bool,
) -> None:
    _show_raw(axis, case, "")
    atomic = case.summary["task"] == "atomic_point_defect"
    if atomic:
        for region in case.regions:
            color = STATE_COLORS.get(region.epistemic_state, "#00e5ff")
            x1, y1, x2, y2 = region.bbox_xyxy
            radius = max((x2 - x1), (y2 - y1), 5) / 2 + 1
            patch = Circle(
                region.centroid_xy,
                radius=radius,
                fill=False,
                edgecolor=color,
                linewidth=0.75,
                alpha=0.92,
            )
            axis.add_patch(patch)
    else:
        component_labels = label(case.mask, connectivity=2)
        for region in case.regions:
            color = STATE_COLORS.get(region.epistemic_state, "#00e5ff")
            component = component_labels == region.region_id
            if not np.any(component):
                continue
            axis.contour(
                component.astype(float),
                levels=[0.5],
                colors=[color],
                linewidths=1.25,
            )
    counts = _state_counts(case.regions)
    state_text = " | ".join(
        f"{name} {count}" for name, count in counts.items()
    ) or "no candidates"
    axis.set_title(
        f"Candidates: {len(case.regions)}\n{state_text}",
        fontsize=9,
        pad=4,
    )
    if show_legend:
        present = [state for state in STATE_COLORS if state in counts]
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=STATE_COLORS[state],
                markersize=6,
                label=state.replace("_", " "),
            )
            for state in present
        ]
        if handles:
            axis.legend(
                handles=handles,
                loc="lower left",
                fontsize=6.3,
                framealpha=0.75,
                borderpad=0.3,
            )


def _field_limits(field: np.ndarray) -> tuple[float, float]:
    finite = field[np.isfinite(field)]
    nonzero = finite[np.abs(finite) > 1e-12]
    values = nonzero if nonzero.size >= 32 else finite
    if not values.size:
        return 0.0, 1.0
    low, high = np.percentile(values, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _show_field(
    figure: plt.Figure,
    axis: plt.Axes,
    case: CaseView,
    field: np.ndarray,
    title: str,
    cmap: str,
    *,
    colorbar: bool,
    reliability: bool = False,
) -> None:
    _show_raw(axis, case, "")
    low, high = (0.0, 1.0) if reliability else _field_limits(field)
    normalized = np.clip((field - low) / max(high - low, 1e-12), 0, 1)
    visible = np.isfinite(field) & (normalized > 0.02)
    rendered = np.ma.masked_where(~visible, field)
    overlay = axis.imshow(
        rendered,
        cmap=cmap,
        vmin=low,
        vmax=high,
        alpha=np.where(visible, 0.12 + 0.52 * np.sqrt(normalized), 0.0),
        interpolation="nearest",
    )
    if np.any(case.mask) and np.any(~case.mask):
        axis.contour(
            case.mask.astype(float),
            levels=[0.5],
            colors=["#00e5ff"],
            linewidths=0.65,
        )
    axis.set_title(title, fontsize=9.5, pad=4)
    if colorbar:
        bar = figure.colorbar(overlay, ax=axis, fraction=0.040, pad=0.018)
        bar.ax.tick_params(labelsize=6, length=2)
        bar.outline.set_linewidth(0.4)


def _show_reference_points(axis: plt.Axes, case: CaseView) -> None:
    _show_raw(axis, case, "")
    points = case.reference_points_xy
    if points is not None and len(points):
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=18,
            facecolors="none",
            edgecolors="#facc15",
            linewidths=0.75,
            label=f"author points ({len(points)})",
        )
        axis.legend(
            loc="lower left",
            fontsize=7,
            framealpha=0.75,
            borderpad=0.3,
        )
    axis.set_title("Author point annotation (reference only)", fontsize=9.5)


def _human_label(name: str) -> str:
    return name.replace("_", " ")


def _write_detail(case: CaseView) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.6, 9.8))
    figure.patch.set_facecolor("#f8fafc")
    _show_raw(axes[0, 0], case, "A  Normalized microscopy input")
    _draw_candidate_states(axes[0, 1], case, show_legend=True)
    _show_field(
        figure,
        axes[0, 2],
        case,
        case.reliability,
        "C  Measurement reliability q(x)",
        "viridis",
        colorbar=True,
        reliability=True,
    )
    _show_field(
        figure,
        axes[1, 0],
        case,
        case.primary_evidence,
        f"D  {case.spec.evidence_title}",
        "magma",
        colorbar=True,
    )
    if case.hypothesis_maps:
        name, field = case.hypothesis_maps[0]
        _show_field(
            figure,
            axes[1, 1],
            case,
            field,
            "E  Top hypothesis\n" + _human_label(name),
            "plasma",
            colorbar=True,
        )
    else:
        _show_raw(axes[1, 1], case, "E  No supported hypothesis map")
    if case.reference_points_xy is not None:
        _show_reference_points(axes[1, 2], case)
    elif len(case.hypothesis_maps) > 1:
        name, field = case.hypothesis_maps[1]
        _show_field(
            figure,
            axes[1, 2],
            case,
            field,
            "F  Competing hypothesis\n" + _human_label(name),
            "cividis",
            colorbar=True,
        )
    else:
        _show_raw(axes[1, 2], case, "F  No second supported hypothesis")
    figure.suptitle(
        f"{case.spec.dataset} — {case.spec.task_label}",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
        y=0.985,
    )
    labels = ", ".join(
        f"{item['label']}×{item['count']}" for item in _top_labels(case.regions)[:2]
    ) or "none"
    figure.text(
        0.5,
        0.023,
        (
            f"Resolved-label summary: {labels}.  "
            f"Reference status: {case.spec.gold_status}."
        ),
        ha="center",
        fontsize=8.6,
        color="#334155",
    )
    figure.text(
        0.5,
        0.007,
        "All scalar maps are translucent overlays on the same normalized input; cyan = detector candidate boundary.",
        ha="center",
        fontsize=7.7,
        color="#64748b",
    )
    figure.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.925,
        bottom=0.065,
        wspace=0.12,
        hspace=0.14,
    )
    figure.savefig(
        case.detail_figure,
        dpi=165,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def _write_overview(cases: list[CaseView], destination: Path) -> None:
    figure, axes = plt.subplots(
        len(cases),
        3,
        figsize=(14.5, 25.5),
        squeeze=False,
    )
    figure.patch.set_facecolor("#f8fafc")
    headers = (
        "Fixed, untouched input",
        "Open-world candidate state",
        "Primary physical evidence",
    )
    for column, header in enumerate(headers):
        figure.text(
            (column + 0.5) / 3,
            0.966,
            header,
            ha="center",
            fontsize=12.5,
            fontweight="bold",
            color="#0f172a",
        )
    for row, case in enumerate(cases):
        _show_raw(axes[row, 0], case, "")
        _draw_candidate_states(axes[row, 1], case, show_legend=True)
        _show_field(
            figure,
            axes[row, 2],
            case,
            case.primary_evidence,
            case.spec.evidence_title,
            "magma",
            colorbar=False,
        )
        axes[row, 0].text(
            -0.035,
            0.5,
            f"{case.spec.dataset}\n{case.spec.task_label}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=9.2,
            fontweight="bold",
            color="#0f172a",
        )
        axes[row, 0].text(
            0.01,
            -0.045,
            case.spec.selection_rule,
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=6.7,
            color="#475569",
        )
        axes[row, 2].text(
            0.99,
            -0.045,
            case.spec.gold_status,
            transform=axes[row, 2].transAxes,
            ha="right",
            va="top",
            fontsize=6.7,
            color="#475569",
        )
    figure.suptitle(
        "Training-Free Open-World STEM Dataset Gallery",
        y=0.992,
        fontsize=19,
        fontweight="bold",
        color="#0f172a",
    )
    figure.text(
        0.5,
        0.009,
        (
            "No model fitting in this run. Candidate count is a pipeline output, not "
            "accuracy when candidate-level Gold is absent. Sigma3 UUID channel is smoke-only."
        ),
        ha="center",
        fontsize=8.2,
        color="#475569",
    )
    figure.subplots_adjust(
        left=0.072,
        right=0.988,
        top=0.945,
        bottom=0.038,
        wspace=0.10,
        hspace=0.18,
    )
    figure.savefig(
        destination,
        dpi=165,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def _manifest_case(case: CaseView) -> dict[str, Any]:
    return {
        "slug": case.spec.slug,
        "dataset": case.spec.dataset,
        "task": case.summary["task"],
        "task_label": case.spec.task_label,
        "source": str(_project_path(case.spec.source)),
        "selection": case.spec.selection,
        "selection_rule": case.spec.selection_rule,
        "image_shape": case.summary["image_shape"],
        "detected_region_count": case.summary["detected_region_count"],
        "detected_area_fraction": case.summary["detected_area_fraction"],
        "resolver_state_counts": _state_counts(case.regions),
        "top_resolved_labels": _top_labels(case.regions),
        "reference_point_count": (
            int(len(case.reference_points_xy))
            if case.reference_points_xy is not None
            else None
        ),
        "gold_status": case.spec.gold_status,
        "interpretation_limit": case.spec.interpretation_limit,
        "run_directory": str(case.run_directory.resolve()),
        "detail_figure": str(case.detail_figure.resolve()),
    }


def build(output: Path) -> list[Path]:
    output_root = output if output.is_absolute() else PROJECT_ROOT / output
    output_root = output_root.resolve()
    try:
        output_root.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("Output must stay inside the project") from error
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_root}"
        )
    (output_root / "runs").mkdir(parents=True)
    (output_root / "figures").mkdir()

    cases: list[CaseView] = []
    for spec in CASES:
        print(f"[run] {spec.slug}", flush=True)
        case = _run_case(spec, output_root)
        _write_detail(case)
        cases.append(case)
        print(
            f"[done] {spec.slug}: {len(case.regions)} candidates, "
            f"states={_state_counts(case.regions)}",
            flush=True,
        )
        gc.collect()

    overview = output_root / "cross_dataset_overview.png"
    _write_overview(cases, overview)
    manifest = output_root / "gallery_summary.json"
    payload = {
        "protocol": "open_world_cross_dataset_gallery_v1",
        "platform_version": "0.4",
        "training_or_fitting_performed": False,
        "sample_selection_policy": (
            "Samples and stack planes were fixed by source name or first-index rules "
            "before detector output inspection."
        ),
        "accuracy_reporting_policy": (
            "No accuracy is reported without candidate-level Gold. Author point labels "
            "are displayed as a development reference only."
        ),
        "visualization_policy": (
            "Every scalar evidence field is drawn translucently over the normalized "
            "microscopy input to preserve the visible atomic phase."
        ),
        "cases": [_manifest_case(case) for case in cases],
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return [overview, manifest, *(case.detail_figure for case in cases)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/open_world_dataset_gallery_v04"),
        help="New project-local output directory",
    )
    return parser.parse_args()


def main() -> None:
    for path in build(parse_args().output):
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
