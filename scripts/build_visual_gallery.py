#!/usr/bin/env python3
"""Build a consistent, training-free microscopy result gallery.

The gallery deliberately draws quantitative fields as translucent overlays on
the normalized microscopy image.  Evidence maps are scalar measurements, not
alternative reconstructions of the specimen, so keeping the atoms visible is
important for interpretation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from skimage.io import imread

from tfree_detection.config import load_config
from tfree_detection.preprocessing import load_grayscale, robust_normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CaseSpec:
    slug: str
    title: str
    output_directory: str
    primary_key: str
    primary_title: str
    primary_cmap: str


@dataclass
class CaseData:
    spec: CaseSpec
    output_directory: Path
    original: np.ndarray
    overlay: np.ndarray
    mask: np.ndarray
    score: np.ndarray
    evidence: dict[str, np.ndarray]
    summary: dict[str, Any]


CASES = (
    CaseSpec(
        slug="amorphous",
        title="Local amorphization",
        output_directory="outputs/reference_amorphous_stage2_final",
        primary_key="group_spectrum",
        primary_title="Reciprocal-order loss",
        primary_cmap="magma",
    ),
    CaseSpec(
        slug="atomic_point_defect",
        title="Atomic point defects",
        output_directory="outputs/public_mowte2_atomic_defects_final_v2",
        primary_key="combined_site_anomaly",
        primary_title="Site-residual evidence",
        primary_cmap="magma",
    ),
    CaseSpec(
        slug="stacking_fault",
        title="Stacking fault / dislocation",
        output_directory="outputs/public_al_stacking_fault_final_v2",
        primary_key="group_phase_jump",
        primary_title="Lattice phase jump",
        primary_cmap="plasma",
    ),
    CaseSpec(
        slug="phase_boundary",
        title="Two-domain boundary",
        output_directory="outputs/public_zro2_phase_boundary_final_v2",
        primary_key="domain_label",
        primary_title="Unsupervised domains",
        primary_cmap="domain",
    ),
)


def _within_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Refusing to read outside project root: {resolved}") from error
    return resolved


def _load_case(spec: CaseSpec) -> CaseData:
    output_directory = _within_project(PROJECT_ROOT / spec.output_directory)
    required = (
        "summary.json",
        "config_resolved.yaml",
        "overlay.png",
        "mask.npy",
        "score_map.npy",
        "evidence_maps.npz",
    )
    missing = [name for name in required if not (output_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{output_directory} is missing required files: {', '.join(missing)}"
        )

    with (output_directory / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    source = summary.get("metadata", {}).get("source")
    if not source:
        raise ValueError(f"{output_directory}/summary.json has no metadata.source")
    source_path = _within_project(Path(source))

    config = load_config(output_directory / "config_resolved.yaml")
    original, _ = robust_normalize(load_grayscale(source_path), config.preprocess)
    overlay = np.asarray(imread(output_directory / "overlay.png"))
    mask = np.asarray(np.load(output_directory / "mask.npy"), dtype=bool)
    score = np.asarray(np.load(output_directory / "score_map.npy"), dtype=np.float32)
    with np.load(output_directory / "evidence_maps.npz") as archive:
        evidence = {key: np.asarray(archive[key]) for key in archive.files}

    shape = original.shape
    if mask.shape != shape or score.shape != shape:
        raise ValueError(f"Image/result shape mismatch in {output_directory}")
    for key, field in evidence.items():
        if field.ndim == 2 and field.shape != shape:
            raise ValueError(f"Evidence map {key!r} has unexpected shape {field.shape}")

    if spec.slug == "atomic_point_defect":
        evidence["combined_site_anomaly"] = np.maximum.reduce(
            [
                evidence["group_presence"],
                evidence["group_intensity"],
                evidence["group_displacement"],
            ]
        )

    return CaseData(
        spec=spec,
        output_directory=output_directory,
        original=original,
        overlay=overlay,
        mask=mask,
        score=score,
        evidence=evidence,
        summary=summary,
    )


def _robust_limits(
    field: np.ndarray,
    validity: np.ndarray | None = None,
    lower: float = 2.0,
    upper: float = 98.0,
) -> tuple[float, float]:
    valid = np.isfinite(field)
    if validity is not None:
        valid &= validity.astype(bool)
    values = field[valid]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [lower, upper])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _finish_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#77808b")


def _show_original(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=5)
    _finish_axis(ax)


def _show_overlay(ax: plt.Axes, overlay: np.ndarray, title: str) -> None:
    ax.imshow(overlay, interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=5)
    _finish_axis(ax)


def _show_continuous_overlay(
    fig: plt.Figure,
    ax: plt.Axes,
    original: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap: str,
    *,
    mask: np.ndarray | None = None,
    validity: np.ndarray | None = None,
    lower: float = 2.0,
    upper: float = 98.0,
) -> None:
    ax.imshow(original, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    valid = np.isfinite(field)
    if validity is not None:
        valid &= validity.astype(bool)
    vmin, vmax = _robust_limits(field, validity, lower, upper)
    normalized = np.clip((field - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    alpha = np.where(valid, 0.12 + 0.58 * np.sqrt(normalized), 0.0)
    rendered = np.ma.masked_where(~valid, field)
    image = ax.imshow(
        rendered,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        interpolation="nearest",
    )
    if mask is not None and np.any(mask) and np.any(~mask):
        ax.contour(
            mask.astype(np.float32),
            levels=[0.5],
            colors=["#00ffff"],
            linewidths=0.75,
        )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.018)
    colorbar.ax.tick_params(labelsize=6, length=2)
    colorbar.outline.set_linewidth(0.5)
    ax.set_title(title, fontsize=10, pad=5)
    _finish_axis(ax)


def _show_domains(
    fig: plt.Figure,
    ax: plt.Axes,
    case: CaseData,
    title: str,
    *,
    include_legend: bool = True,
) -> None:
    domain = case.evidence["domain_label"]
    support = case.evidence["crystalline_support"] > 0.5
    valid = support & np.isfinite(domain) & (domain >= 0)
    rendered = np.ma.masked_where(~valid, domain)
    ax.imshow(case.original, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.imshow(
        rendered,
        cmap=ListedColormap(["#2d7dd2", "#f49d37"]),
        vmin=0,
        vmax=1,
        alpha=np.where(valid, 0.48, 0.0),
        interpolation="nearest",
    )
    if np.any(case.mask) and np.any(~case.mask):
        ax.contour(
            case.mask.astype(np.float32),
            levels=[0.5],
            colors=["#00ffff"],
            linewidths=0.8,
        )
    if include_legend:
        handles = [
            Line2D([0], [0], color="#2d7dd2", lw=5, label="Domain 0"),
            Line2D([0], [0], color="#f49d37", lw=5, label="Domain 1"),
            Line2D([0], [0], color="#00ffff", lw=1.2, label="Boundary"),
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            fontsize=6.5,
            framealpha=0.75,
            borderpad=0.35,
            handlelength=1.6,
        )
    ax.set_title(title, fontsize=10, pad=5)
    _finish_axis(ax)


def _case_caption(case: CaseData) -> str:
    summary = case.summary
    metadata = summary.get("metadata", {})
    count = summary["detected_region_count"]
    fraction = 100.0 * summary["detected_area_fraction"]
    if case.spec.slug == "amorphous":
        return (
            f"{count} candidates | mask area {fraction:.2f}% | "
            "stage 1 structural disorder; stage 2 GPA + frozen DINO verification"
        )
    if case.spec.slug == "atomic_point_defect":
        expected = int(metadata["expected_site_count"])
        sublattices = int(metadata["inferred_sublattice_count"])
        reliability = 100.0 * float(
            np.mean(case.evidence["registration_reliability"])
        )
        return (
            f"{count}/{expected} reliable site candidates ({100 * count / expected:.2f}%) | "
            f"{sublattices} inferred sublattices | reliable registration {reliability:.1f}%"
        )
    if case.spec.slug == "stacking_fault":
        margin = int(metadata["valid_border_margin_pixels"])
        return (
            f"{count} main phase-slip cores | mask area {fraction:.2f}% | "
            f"Fourier validity margin {margin} px"
        )
    domains = metadata["domain_fractions"]
    return (
        f"{count} internal boundary | domain fractions "
        f"{100 * domains[0]:.1f}% / {100 * domains[1]:.1f}% | vacuum masked"
    )


def _detail_fields(case: CaseData) -> list[tuple[str, np.ndarray, str, np.ndarray | None]]:
    if case.spec.slug == "amorphous":
        validity = case.evidence.get("gpa_validity")
        return [
            ("Reciprocal-order loss", case.evidence["group_spectrum"], "magma", None),
            (
                "GPA lattice coherence",
                case.evidence["gpa_lattice_coherence"],
                "viridis",
                validity,
            ),
            (
                "GPA strain magnitude",
                case.evidence["gpa_strain_magnitude"],
                "inferno",
                validity,
            ),
        ]
    if case.spec.slug == "atomic_point_defect":
        return [
            (
                "Presence / intensity residual",
                np.maximum(
                    case.evidence["group_presence"], case.evidence["group_intensity"]
                ),
                "magma",
                None,
            ),
            ("Displacement residual", case.evidence["group_displacement"], "plasma", None),
            (
                "Registration reliability",
                case.evidence["registration_reliability"],
                "cividis",
                None,
            ),
        ]
    if case.spec.slug == "stacking_fault":
        validity = case.evidence.get("phase_slip_validity")
        return [
            ("Lattice phase jump", case.evidence["group_phase_jump"], "plasma", validity),
            (
                "Shear / rotation evidence",
                case.evidence["group_shear_rotation"],
                "magma",
                validity,
            ),
            (
                "Coherence-loss evidence",
                case.evidence["group_coherence_loss"],
                "inferno",
                validity,
            ),
        ]
    return [
        (
            "Reciprocal-signature separation",
            case.evidence["group_reciprocal_signature"],
            "magma",
            case.evidence["boundary_validity"],
        ),
        (
            "Domain membership margin",
            case.evidence["domain_membership_margin"],
            "viridis",
            case.evidence["crystalline_support"],
        ),
        (
            "Crystalline support",
            case.evidence["crystalline_support"],
            "cividis",
            None,
        ),
    ]


def _write_detail(case: CaseData, destination: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.6, 9.3), constrained_layout=False)
    figure.patch.set_facecolor("#f3f5f7")
    _show_original(axes[0, 0], case.original, "A  Normalized microscopy input")
    _show_overlay(axes[0, 1], case.overlay, "B  Final candidate annotation")
    _show_continuous_overlay(
        figure,
        axes[0, 2],
        case.original,
        case.score,
        "C  Fused continuous score",
        "turbo",
        mask=case.mask,
    )

    for index, (title, field, cmap, validity) in enumerate(_detail_fields(case)):
        ax = axes[1, index]
        if case.spec.slug == "phase_boundary" and index == 0:
            _show_domains(figure, ax, case, "D  Unsupervised domain assignment")
        else:
            letter = chr(ord("D") + index)
            _show_continuous_overlay(
                figure,
                ax,
                case.original,
                field,
                f"{letter}  {title}",
                cmap,
                mask=case.mask,
                validity=validity,
                lower=0.0 if "reliability" in title.lower() else 2.0,
                upper=100.0 if "reliability" in title.lower() else 98.0,
            )

    figure.suptitle(
        f"Training-free example: {case.spec.title}",
        x=0.5,
        y=0.978,
        fontsize=17,
        fontweight="bold",
        color="#14213d",
    )
    figure.text(
        0.5,
        0.027,
        _case_caption(case),
        ha="center",
        va="center",
        fontsize=10,
        color="#263238",
    )
    figure.text(
        0.5,
        0.009,
        "Heatmaps are translucent scalar evidence fields, not reconstructed microscopy images. "
        "Cyan contour = final candidate mask.",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#5c6770",
    )
    figure.subplots_adjust(left=0.025, right=0.985, top=0.93, bottom=0.058, wspace=0.12, hspace=0.15)
    figure.savefig(destination, dpi=175, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _write_overview(cases: list[CaseData], destination: Path) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(13.2, 17.2), constrained_layout=False)
    figure.patch.set_facecolor("#f3f5f7")
    column_titles = ("Normalized input", "Final annotation", "Primary physical evidence")
    for column, title in enumerate(column_titles):
        figure.text(
            (column + 0.5) / 3.0,
            0.958,
            title,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#14213d",
        )

    for row, case in enumerate(cases):
        _show_original(axes[row, 0], case.original, "")
        _show_overlay(axes[row, 1], case.overlay, "")
        if case.spec.primary_key == "domain_label":
            _show_domains(
                figure,
                axes[row, 2],
                case,
                case.spec.primary_title,
                include_legend=True,
            )
        else:
            _show_continuous_overlay(
                figure,
                axes[row, 2],
                case.original,
                case.evidence[case.spec.primary_key],
                case.spec.primary_title,
                case.spec.primary_cmap,
                mask=case.mask,
            )
        axes[row, 0].text(
            -0.04,
            0.5,
            case.spec.title,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=12,
            fontweight="bold",
            color="#14213d",
        )
        axes[row, 1].text(
            0.5,
            -0.055,
            _case_caption(case),
            transform=axes[row, 1].transAxes,
            ha="center",
            va="top",
            fontsize=7.6,
            color="#455a64",
            wrap=True,
        )

    figure.suptitle(
        "Training-Free Microscopy Detection Gallery",
        y=0.991,
        fontsize=20,
        fontweight="bold",
        color="#14213d",
    )
    legend = [
        Line2D([0], [0], color="#ff3030", lw=2, label="Final annotation in result overlay"),
        Line2D([0], [0], color="#00ffff", lw=1.5, label="Final mask contour on evidence map"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.006),
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.935,
        bottom=0.055,
        wspace=0.10,
        hspace=0.16,
    )
    figure.savefig(destination, dpi=175, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _write_manifest(cases: list[CaseData], destination: Path) -> None:
    payload = {
        "description": "Unified visual gallery built from existing detector outputs.",
        "training_or_fitting_performed": False,
        "visualization_rule": (
            "Quantitative evidence fields are shown as translucent overlays on the "
            "normalized microscopy input."
        ),
        "cases": [
            {
                "slug": case.spec.slug,
                "title": case.spec.title,
                "source": case.summary["metadata"]["source"],
                "result_directory": str(case.output_directory),
                "detected_region_count": case.summary["detected_region_count"],
                "detected_area_fraction": case.summary["detected_area_fraction"],
                "detail_figure": f"{case.spec.slug}_example.png",
            }
            for case in cases
        ],
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def build_gallery(output_directory: Path) -> list[Path]:
    destination = output_directory
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination = _within_project(destination)
    destination.mkdir(parents=True, exist_ok=True)

    targets = [destination / "multitask_overview.png"]
    targets.extend(destination / f"{spec.slug}_example.png" for spec in CASES)
    targets.append(destination / "gallery_manifest.json")
    existing = [path for path in targets if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing gallery files: {joined}")

    cases = [_load_case(spec) for spec in CASES]
    _write_overview(cases, targets[0])
    for case, target in zip(cases, targets[1:5], strict=True):
        _write_detail(case, target)
    _write_manifest(cases, targets[-1])
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visual_gallery_v1"),
        help="Project-local destination (default: outputs/visual_gallery_v1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = build_gallery(args.output)
    for path in written:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
