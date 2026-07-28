#!/usr/bin/env python
"""Build an auditable visual comparison of data and method iterations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tfree_detection.governance import DataPartition, DataRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "train": "#2A9D8F",
    "development_test": "#E9C46A",
    "locked_external_test": "#E76F51",
    "calibration": "#8E6CBE",
    "blue": "#277DA1",
    "teal": "#43AA8B",
    "orange": "#F8961E",
    "red": "#D1495B",
    "gray": "#6C757D",
    "light": "#E9ECEF",
    "dark": "#243447",
}


def _load_json(relative: str) -> dict[str, Any]:
    return json.loads(
        (PROJECT_ROOT / relative).read_text(encoding="utf-8")
    )


def _registry_summary(relative: str) -> dict[str, Any]:
    registry = DataRegistry.from_yaml(PROJECT_ROOT / relative)
    counts = {
        partition.value: len(registry.assets_for_partition(partition))
        for partition in DataPartition
    }
    return {
        "path": relative,
        "registry_version": registry.registry_version,
        "registry_sha256": registry.content_hash(),
        "asset_count": len(registry.assets),
        "partition_counts": counts,
        "dataset_count": len({asset.dataset_id for asset in registry.assets}),
        "split_group_count": len(
            {asset.split_group_id for asset in registry.assets}
        ),
        "active_learning_asset_count": len(
            registry.active_learning_assets()
        ),
        "datasets": sorted(
            {asset.dataset_id for asset in registry.assets}
        ),
    }


def _atomic_metrics() -> dict[str, dict[str, float]]:
    sources = {
        "Legacy v0.2": (
            "benchmarks/development_runs/"
            "emseek_wse2_legacy_atomic_v0_2_score_v2/"
            "development_metrics.json"
        ),
        "Periodic v1": (
            "benchmarks/development_runs/"
            "emseek_wse2_periodic_v1_score_v2/"
            "development_metrics.json"
        ),
        "Consensus v2": (
            "benchmarks/development_runs/"
            "emseek_wse2_two_teacher_consensus_v2_score/"
            "development_metrics.json"
        ),
    }
    output = {}
    for name, relative in sources.items():
        payload = _load_json(relative)
        metrics = payload["micro_site_metrics"]
        output[name] = {
            "precision": float(metrics["site_precision"]),
            "recall": float(metrics["site_recall"]),
            "f1": float(metrics["site_f1"]),
            "predicted_sites": int(metrics["predicted_sites"]),
            "matched_sites": int(metrics["matched_sites"]),
            "truth_sites": int(metrics["truth_sites"]),
            "mean_localization_error_pixels": float(
                metrics["mean_localization_error_pixels"]
            ),
        }
    return output


def collect_comparison() -> dict[str, Any]:
    registries = [
        _registry_summary(f"datasets/partition_registry{suffix}.yaml")
        for suffix in ("", "_v2", "_v3", "_v4")
    ]
    baseline = _load_json("benchmarks/baseline_v0_2_manifest.json")
    locked = _load_json(
        "benchmarks/locked_runs/dennler_v1_score/locked_metrics.json"
    )
    student = _load_json("benchmarks/latest_student_ablation.json")
    locked_targets = {
        name: {
            "role": values["role"],
            "iou": float(values["micro_pixel_metrics"]["iou"]),
            "precision": float(
                values["micro_pixel_metrics"]["precision"]
            ),
            "recall": float(values["micro_pixel_metrics"]["recall"]),
        }
        for name, values in locked["targets"].items()
        if values["role"] == "primary"
    }
    capabilities = {
        "rows": [
            "四个物理候选生成任务",
            "统一 EvidenceBundle / resolver",
            "sealed registry / 防泄漏治理",
            "周期反事实原子教师",
            "双教师共识正式入口",
            "逐资产 manifest / EMD 解析",
            "线拓扑 / 时序支持",
            "实验 Gold calibration",
        ],
        "columns": ["v0.2 基线", "v0.3 证据化", "v0.4 平台化"],
        "values": [
            [1, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
    }
    return {
        "comparison_format": "tfree-version-iteration-comparison-v1",
        "registries": registries,
        "synthetic_baseline": baseline[
            "synthetic_multitask_benchmark"
        ],
        "atomic_wse2_train_domain": {
            "scope": (
                "107 images in one conservative train group; "
                "not independent generalization"
            ),
            "metrics": _atomic_metrics(),
        },
        "dennler_locked_external": {
            "scope": (
                "61 images; consumed locked zero-shot evaluation; "
                "broad transfer not supported"
            ),
            "targets": locked_targets,
        },
        "student_synthetic_ablation": {
            "scope": student["scope"],
            "evidence_preserving_iou": float(
                student["evidence_preserving_student"]["iou"]
            ),
            "hard_pseudolabel_iou": float(
                student["hard_pseudolabel_student"]["iou"]
            ),
            "iou_difference": float(student["iou_difference"]),
            "release_claim_allowed": bool(
                student["release_claim_allowed"]
            ),
        },
        "capabilities": capabilities,
        "claim_boundary": {
            "calibration_asset_count": 0,
            "nature_level_method_claim_ready": False,
            "data_growth_is_accuracy_growth": False,
        },
    }


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                # Matplotlib indexes the JP face of the shared Noto CJK TTC on
                # this host.  That face contains both Latin and Simplified
                # Chinese glyphs; the SC alias is only visible to fontconfig.
                "Noto Sans CJK JP",
                "Noto Sans CJK SC",
                "Noto Sans CJK TC",
                "DejaVu Sans",
                "Droid Sans Fallback",
            ],
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "#F7F9FC",
            "axes.facecolor": "white",
            "savefig.facecolor": "#F7F9FC",
        }
    )


def _label_bars(axis: plt.Axes, bars: Any, fmt: str = "{:.0f}") -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["dark"],
        )


def build_overview(payload: dict[str, Any], output: Path) -> None:
    _configure_style()
    figure = plt.figure(figsize=(18, 14), constrained_layout=True)
    figure.get_layout_engine().set(rect=(0.02, 0.02, 0.98, 0.90))
    grid = figure.add_gridspec(3, 2, height_ratios=[1.05, 1.05, 1.2])
    registries = payload["registries"]
    labels = [f"Registry v{item['registry_version'][0]}" for item in registries]
    positions = np.arange(len(labels))

    axis = figure.add_subplot(grid[0, 0])
    bottom = np.zeros(len(registries))
    for partition, legend in (
        ("train", "Train"),
        ("development_test", "Development"),
        ("locked_external_test", "Locked external"),
        ("calibration", "Calibration"),
    ):
        values = np.asarray(
            [item["partition_counts"][partition] for item in registries]
        )
        bars = axis.bar(
            positions,
            values,
            bottom=bottom,
            label=legend,
            color=COLORS[partition],
            width=0.66,
        )
        bottom += values
    for x, total in zip(positions, bottom):
        axis.text(
            x,
            total + 5,
            str(int(total)),
            ha="center",
            fontweight="bold",
            color=COLORS["dark"],
        )
    axis.set_title("A  数据资产：5 → 291，但 calibration 始终为 0", loc="left")
    axis.set_ylabel("注册资产数")
    axis.set_xticks(positions, labels)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)

    axis = figure.add_subplot(grid[0, 1])
    width = 0.24
    dataset_counts = [item["dataset_count"] for item in registries]
    group_counts = [item["split_group_count"] for item in registries]
    active_counts = [
        item["active_learning_asset_count"] for item in registries
    ]
    first = axis.bar(
        positions - width,
        dataset_counts,
        width,
        label="数据集数",
        color=COLORS["blue"],
    )
    second = axis.bar(
        positions,
        group_counts,
        width,
        label="独立 split groups",
        color=COLORS["orange"],
    )
    axis.set_title("B  多样性与防泄漏分组持续扩展", loc="left")
    axis.set_ylabel("数据集 / split group 数")
    axis.set_xticks(positions, labels)
    axis.spines[["top"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)
    _label_bars(axis, first)
    _label_bars(axis, second)
    secondary = axis.twinx()
    line = secondary.plot(
        positions,
        active_counts,
        marker="o",
        linewidth=2.5,
        color=COLORS["teal"],
        label="可主动学习图像",
    )
    for x, value in zip(positions, active_counts):
        secondary.text(
            x,
            value + 4,
            str(value),
            ha="center",
            color=COLORS["teal"],
            fontsize=9,
        )
    secondary.set_ylabel("可主动学习资产数")
    handles, legends = axis.get_legend_handles_labels()
    axis.legend(
        handles + line,
        legends + ["可主动学习图像"],
        frameon=False,
        loc="upper left",
    )
    secondary.spines["top"].set_visible(False)

    axis = figure.add_subplot(grid[1, 0])
    atomic = payload["atomic_wse2_train_domain"]["metrics"]
    methods = list(atomic)
    metrics = ("precision", "recall", "f1")
    metric_labels = ("Precision", "Recall", "F1")
    colors = (COLORS["blue"], COLORS["orange"], COLORS["teal"])
    method_positions = np.arange(len(methods))
    width = 0.23
    for offset, metric, metric_label, color in zip(
        (-width, 0, width), metrics, metric_labels, colors
    ):
        values = [atomic[name][metric] for name in methods]
        bars = axis.bar(
            method_positions + offset,
            values,
            width,
            label=metric_label,
            color=color,
        )
        _label_bars(axis, bars, "{:.3f}")
    axis.set_ylim(0, 1.08)
    axis.set_title(
        "C  WSe2 域内原子位点：双教师显著提高 precision/F1",
        loc="left",
    )
    axis.set_ylabel("Site metric")
    axis.set_xticks(method_positions, methods)
    axis.legend(frameon=False, ncol=3, loc="lower center")
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.01,
        0.02,
        "同一保守 train group（107 图）；不是独立跨域泛化结果",
        transform=axis.transAxes,
        fontsize=9,
        color=COLORS["red"],
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF2F2", ec="none"),
    )

    axis = figure.add_subplot(grid[1, 1])
    locked = payload["dennler_locked_external"]["targets"]
    locked_order = [
        "blurred_amorphous_or_beam_composite",
        "secondary_crystal_symmetry_composite",
        "any_expert_marked_defect",
    ]
    locked_labels = ["模糊/非晶/束流", "次晶体对称", "任意专家缺陷"]
    locked_values = [locked[name]["iou"] for name in locked_order]
    bars = axis.bar(
        np.arange(3),
        locked_values,
        color=(COLORS["red"], COLORS["orange"], COLORS["gray"]),
        width=0.58,
    )
    _label_bars(axis, bars, "{:.3f}")
    axis.set_ylim(0, 0.75)
    axis.set_ylabel("IoU")
    axis.set_xticks(np.arange(3), locked_labels)
    axis.set_title(
        "D  外部有效性仍是短板；合成学生增益不能替代真实外测",
        loc="left",
    )
    student = payload["student_synthetic_ablation"]
    student_x = np.asarray([3.7, 4.5])
    student_values = [
        student["hard_pseudolabel_iou"],
        student["evidence_preserving_iou"],
    ]
    student_bars = axis.bar(
        student_x,
        student_values,
        color=(COLORS["light"], COLORS["teal"]),
        edgecolor=(COLORS["gray"], COLORS["teal"]),
        width=0.58,
    )
    _label_bars(axis, student_bars, "{:.3f}")
    axis.set_xticks(
        list(np.arange(3)) + list(student_x),
        locked_labels + ["硬伪标签学生", "证据保持学生"],
    )
    axis.axvline(3.1, color="#C9CED6", linestyle="--", linewidth=1.5)
    axis.text(
        1.0,
        0.65,
        "Dennler locked zero-shot\n广泛迁移不支持",
        ha="center",
        color=COLORS["red"],
        fontweight="bold",
    )
    axis.text(
        4.1,
        0.72,
        "仅 held-out synthetic\nIoU +0.061",
        ha="center",
        color=COLORS["teal"],
        fontweight="bold",
    )
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", rotation=12)

    axis = figure.add_subplot(grid[2, :])
    capabilities = payload["capabilities"]
    matrix = np.asarray(capabilities["values"])
    cmap = mpl.colors.ListedColormap(["#F1F3F5", COLORS["teal"]])
    axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    axis.set_xticks(
        np.arange(len(capabilities["columns"])),
        capabilities["columns"],
        fontsize=12,
        fontweight="bold",
    )
    axis.set_yticks(
        np.arange(len(capabilities["rows"])),
        capabilities["rows"],
        fontsize=11,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                "✓" if matrix[row, column] else "—",
                ha="center",
                va="center",
                fontsize=18,
                color="white" if matrix[row, column] else "#ADB5BD",
                fontweight="bold",
            )
    axis.set_title(
        "E  软件能力迭代：候选生成 → 统一证据 → 可治理、可选择、可审计的平台",
        loc="left",
        pad=14,
    )
    axis.set_xticks(
        np.arange(-0.5, matrix.shape[1], 1), minor=True
    )
    axis.set_yticks(
        np.arange(-0.5, matrix.shape[0], 1), minor=True
    )
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    initial = registries[0]["asset_count"]
    current = registries[-1]["asset_count"]
    legacy = atomic["Legacy v0.2"]
    consensus = atomic["Consensus v2"]
    figure.suptitle(
        "PhyEvo-Mic / Training-Free 平台版本迭代总览",
        fontsize=24,
        fontweight="bold",
        color=COLORS["dark"],
        y=0.992,
    )
    figure.text(
        0.5,
        0.947,
        (
            f"数据资产 {initial}→{current}（{current / initial:.1f}×）  |  "
            f"WSe2 precision +{100 * (consensus['precision'] - legacy['precision']):.1f} pp  |  "
            f"F1 +{100 * (consensus['f1'] - legacy['f1']):.1f} pp  |  "
            "冻结四任务基线保留 100%  |  calibration 仍为 0"
        ),
        ha="center",
        fontsize=13,
        color=COLORS["gray"],
    )
    figure.savefig(output / "version_iteration_overview.png", dpi=220)
    figure.savefig(output / "version_iteration_overview.pdf")
    plt.close(figure)


def build_effect_detail(payload: dict[str, Any], output: Path) -> None:
    _configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.4), constrained_layout=True)
    atomic = payload["atomic_wse2_train_domain"]["metrics"]
    legacy = atomic["Legacy v0.2"]
    periodic = atomic["Periodic v1"]
    consensus = atomic["Consensus v2"]

    axis = axes[0]
    for name, values, color in (
        ("Legacy v0.2", legacy, COLORS["gray"]),
        ("Periodic v1", periodic, COLORS["orange"]),
        ("Consensus v2", consensus, COLORS["teal"]),
    ):
        axis.scatter(
            values["recall"],
            values["precision"],
            s=180,
            color=color,
            edgecolor="white",
            linewidth=1.5,
            label=name,
            zorder=3,
        )
        axis.annotate(
            f"{name}\nF1={values['f1']:.3f}",
            (values["recall"], values["precision"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
        )
    axis.set_xlim(0.60, 0.98)
    axis.set_ylim(0.42, 0.96)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("A  迭代轨迹：更少但更可信的候选", loc="left")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes[1]
    delta = {
        "Precision": consensus["precision"] - legacy["precision"],
        "Recall": consensus["recall"] - legacy["recall"],
        "F1": consensus["f1"] - legacy["f1"],
    }
    bars = axis.bar(
        list(delta),
        [100 * value for value in delta.values()],
        color=[
            COLORS["teal"] if value >= 0 else COLORS["red"]
            for value in delta.values()
        ],
    )
    for bar, value in zip(bars, delta.values()):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            100 * value + (1.2 if value >= 0 else -1.2),
            f"{100 * value:+.1f} pp",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontweight="bold",
        )
    axis.axhline(0, color=COLORS["dark"], linewidth=1)
    axis.set_ylabel("Consensus v2 − Legacy v0.2（百分点）")
    axis.set_title("B  收益不是全面抬升：precision/F1↑，recall↓", loc="left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)

    axis = axes[2]
    baseline = payload["synthetic_baseline"]["tasks"]
    tasks = [
        "amorphous_region",
        "atomic_point_defect",
        "stacking_fault",
        "phase_boundary",
    ]
    task_labels = ["非晶", "原子点", "层错线", "相界面"]
    values = [baseline[task]["iou"] for task in tasks]
    bars = axis.bar(
        np.arange(4),
        values,
        color=[
            COLORS["blue"],
            COLORS["teal"],
            COLORS["orange"],
            COLORS["red"],
        ],
        alpha=0.85,
    )
    _label_bars(axis, bars, "{:.3f}")
    axis.scatter(
        np.arange(4),
        values,
        s=70,
        facecolors="none",
        edgecolors=COLORS["dark"],
        linewidths=1.8,
        label="当前回归结果（与 v0.2 重合）",
        zorder=4,
    )
    axis.set_ylim(0, 1)
    axis.set_xticks(np.arange(4), task_labels)
    axis.set_ylabel("Synthetic IoU")
    axis.set_title("C  冻结数值基线没有因平台扩展而退化", loc="left")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "效果变化的正确读法：域内选择性收益、无回归、但跨域仍未解决",
        fontsize=19,
        fontweight="bold",
        color=COLORS["dark"],
    )
    figure.savefig(output / "version_effect_detail.png", dpi=220)
    figure.savefig(output / "version_effect_detail.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.relative_to(PROJECT_ROOT.resolve())
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite version comparison: {output}"
        )
    output.mkdir(parents=True)
    payload = collect_comparison()
    data_path = output / "version_comparison_data.json"
    data_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    build_overview(payload, output)
    build_effect_detail(payload, output)
    print(
        json.dumps(
            {
                "output": str(output.relative_to(PROJECT_ROOT)),
                "artifacts": sorted(
                    path.name for path in output.iterdir()
                ),
                "registry_asset_counts": [
                    item["asset_count"] for item in payload["registries"]
                ],
                "atomic_metrics": payload[
                    "atomic_wse2_train_domain"
                ]["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
