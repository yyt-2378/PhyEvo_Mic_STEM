#!/usr/bin/env python
"""Visualize the already-unblinded Dennler locked-test data and predictions.

This script is diagnostic only.  It reads the immutable v1 prediction set and
the consumed expert labels, but it does not rerun detectors, tune thresholds,
or write anything into the dataset or locked-run directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
from skimage.io import imread


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "public"
    / "dennler_2021_hrstem_defects"
    / "locked_extracted"
)
DATASET_ROOT = DATA_ROOT / "hrstem_defects_dataset"
PREDICTION_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "locked_runs"
    / "dennler_v1_predictions"
)
MANIFEST_PATH = PREDICTION_ROOT / "prediction_manifest.json"
METRICS_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "locked_runs"
    / "dennler_v1_score"
    / "locked_metrics.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dennler_locked_visualization_v2"

CLASS_COLORS = {
    1: np.asarray([0.96, 0.49, 0.08]),  # orange
    2: np.asarray([0.78, 0.18, 0.58]),  # magenta
}
PREDICTION_COLOR = np.asarray([0.00, 0.67, 0.78])
ERROR_COLORS = {
    "tp": np.asarray([0.08, 0.66, 0.35]),
    "fp": np.asarray([0.86, 0.18, 0.55]),
    "fn": np.asarray([0.98, 0.58, 0.10]),
}


def _configure_fonts() -> None:
    regular = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    bold = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    for path in (regular, bold):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    family = (
        font_manager.FontProperties(fname=str(regular)).get_name()
        if regular.exists()
        else "DejaVu Sans"
    )
    plt.rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "#F8FAFC",
            "text.color": "#0F172A",
            "axes.titlecolor": "#0F172A",
        }
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., :3].astype(np.float64).mean(axis=-1)
    elif array.ndim == 2:
        array = array.astype(np.float64)
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    finite = array[np.isfinite(array)]
    if not finite.size:
        return np.zeros(array.shape, dtype=float)
    low, high = np.percentile(finite, [1.0, 99.5])
    if high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        return np.zeros(array.shape, dtype=float)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _base_rgb(gray: np.ndarray, dim: float = 1.0) -> np.ndarray:
    return np.repeat((gray * dim)[..., None], 3, axis=-1)


def _blend(
    base: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    alpha: float,
) -> np.ndarray:
    output = np.asarray(base, dtype=float).copy()
    boolean = np.asarray(mask, dtype=bool)
    output[boolean] = (
        (1.0 - alpha) * output[boolean] + alpha * np.asarray(color)
    )
    return np.clip(output, 0.0, 1.0)


def _image_record_by_name(manifest: dict, filename: str) -> dict:
    matches = [
        record
        for record in manifest["images"]
        if record["filename"] == filename
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one manifest record for {filename!r}, found {len(matches)}"
        )
    return matches[0]


def _read_source(record: dict) -> np.ndarray:
    return _to_gray(imread(str(DATA_ROOT / record["relative_path"])))


def _read_class_map(record: dict, raw_values: list[float]) -> np.ndarray:
    if record["category"] != "defective":
        source = _read_source(record)
        return np.zeros(source.shape, dtype=np.uint8)
    label_path = (
        DATASET_ROOT / "defective" / "labels" / record["filename"]
    )
    label = np.asarray(imread(str(label_path)))
    if label.ndim == 3:
        label = label[..., :3].astype(float).mean(axis=-1)
    output = np.full(label.shape, 255, dtype=np.uint8)
    for class_id, raw_value in enumerate(raw_values):
        output[np.asarray(label, dtype=float) == float(raw_value)] = class_id
    if np.any(output == 255):
        raise ValueError(f"Could not decode label {label_path}")
    return output


def _prediction_mask(
    record: dict,
    detector_ids: tuple[str, ...],
) -> np.ndarray:
    masks = []
    for detector_id in detector_ids:
        path = (
            PREDICTION_ROOT
            / "predictions"
            / record["image_id"]
            / f"{detector_id}.npz"
        )
        with np.load(path, allow_pickle=False) as payload:
            masks.append(np.asarray(payload["mask"], dtype=bool))
    return np.logical_or.reduce(masks)


def _truth_mask(class_map: np.ndarray, truth_classes: tuple[int, ...]) -> np.ndarray:
    return np.isin(class_map, truth_classes)


def _expert_overlay(gray: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    output = _base_rgb(gray, dim=0.82)
    for class_id, color in CLASS_COLORS.items():
        output = _blend(output, class_map == class_id, color, alpha=0.60)
    return output


def _single_truth_overlay(
    gray: np.ndarray,
    truth: np.ndarray,
    truth_classes: tuple[int, ...],
) -> np.ndarray:
    color = (
        CLASS_COLORS[2]
        if truth_classes == (2,)
        else CLASS_COLORS[1]
        if truth_classes == (1,)
        else np.asarray([0.98, 0.45, 0.08])
    )
    return _blend(_base_rgb(gray, dim=0.80), truth, color, alpha=0.62)


def _prediction_overlay(gray: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return _blend(
        _base_rgb(gray, dim=0.80),
        prediction,
        PREDICTION_COLOR,
        alpha=0.67,
    )


def _error_overlay(
    gray: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    output = _base_rgb(gray, dim=0.32)
    true_positive = truth & prediction
    false_positive = ~truth & prediction
    false_negative = truth & ~prediction
    output[true_positive] = ERROR_COLORS["tp"]
    output[false_positive] = ERROR_COLORS["fp"]
    output[false_negative] = ERROR_COLORS["fn"]
    return output


def _metric_row(metrics: dict, target_id: str, filename: str) -> dict:
    rows = metrics["targets"][target_id]["per_image"]
    matches = [row for row in rows if row["filename"] == filename]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one metric row for {target_id}/{filename}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _style_image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(1.2)


def _save_dataset_overview(
    manifest: dict,
    raw_values: list[float],
) -> Path:
    defective_adf = [
        "ADF  image defective x10M_01.tif",
        "ADF image defective x10M_05.tif",
        "ADF image defective x10M_12.tif",
        "ADF image defective x10M_15.tif",
    ]
    defective_bf = [
        "BF  image defective x10M_17.tif",
        "BF  image defective x10M_19.tif",
        "BF  image defective x8M_28.tif",
        "BF  image defective x6M_31.tif",
    ]
    normal = [
        "ADF image non defective x10M_02.tif",
        "ADF image non defective x10M_04.tif",
        "BF image non defective x10M_15.tif",
        "BF image non defective x8M_22.tif",
    ]
    rows = [
        ("缺陷 ADF｜叠加专家标签", defective_adf),
        ("缺陷 BF｜叠加专家标签", defective_bf),
        ("正常对照｜无专家缺陷标签", normal),
    ]

    figure, axes = plt.subplots(3, 4, figsize=(19.2, 12.0), dpi=180)
    figure.subplots_adjust(
        left=0.055, right=0.985, bottom=0.095, top=0.83, wspace=0.035, hspace=0.19
    )
    figure.suptitle(
        "Dennler III–V HRSTEM 数据概览",
        fontsize=28,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.915,
        "31 张缺陷图 + 30 张正常图｜包含 ADF / BF 与多种放大倍率｜以下均为真实原始图像",
        ha="center",
        fontsize=15,
        color="#334155",
    )
    for row_index, (row_title, filenames) in enumerate(rows):
        axes[row_index, 0].text(
            -0.03,
            1.11,
            row_title,
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color="#0F766E" if row_index < 2 else "#475569",
        )
        for column_index, filename in enumerate(filenames):
            axis = axes[row_index, column_index]
            record = _image_record_by_name(manifest, filename)
            gray = _read_source(record)
            if record["category"] == "defective":
                class_map = _read_class_map(record, raw_values)
                display = _expert_overlay(gray, class_map)
            else:
                display = _base_rgb(gray)
            axis.imshow(display, interpolation="nearest")
            axis.set_title(filename.replace(".tif", ""), fontsize=10.5, pad=6)
            _style_image_axis(axis)

    legend = [
        Patch(facecolor=CLASS_COLORS[1], label="专家 Class 1：次级晶体对称性"),
        Patch(
            facecolor=CLASS_COLORS[2],
            label="专家 Class 2：模糊 / 非晶 / 束损伤复合区",
        ),
        Patch(facecolor="#64748B", label="未标注主晶体 / 正常背景"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.025),
    )
    figure.text(
        0.985,
        0.015,
        "说明：标签类别是数据集的粗粒度语义，不等同于平台四任务的一一对应真值。",
        ha="right",
        fontsize=10.5,
        color="#64748B",
    )
    path = OUTPUT_ROOT / "01_dennler_dataset_overview.png"
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path


def _save_failure_diagnostics(
    manifest: dict,
    metrics: dict,
    raw_values: list[float],
) -> Path:
    examples = [
        {
            "label": "A｜Class 2 大面积漏检",
            "filename": "ADF image defective x10M_05.tif",
            "target": "blurred_amorphous_or_beam_composite",
            "truth_classes": (2,),
            "detectors": ("amorphous_region",),
            "note": "专家区域很大，但非晶教师没有触发",
        },
        {
            "label": "B｜Class 1 局部成功案例",
            "filename": "BF  image defective x10M_19.tif",
            "target": "secondary_crystal_symmetry_composite",
            "truth_classes": (1,),
            "detectors": ("stacking_fault", "phase_boundary"),
            "note": "局部次级对称性与解析教师较好对齐",
        },
        {
            "label": "C｜任意缺陷覆盖不足",
            "filename": "BF  image defective x8M_28.tif",
            "target": "any_expert_marked_defect",
            "truth_classes": (1, 2),
            "detectors": (
                "amorphous_region",
                "stacking_fault",
                "phase_boundary",
            ),
            "note": "专家标注覆盖广，冻结预测只覆盖少量局部",
        },
        {
            "label": "D｜正常图上的误报",
            "filename": "BF image non defective x8M_22.tif",
            "target": "any_expert_marked_defect",
            "truth_classes": (1, 2),
            "detectors": (
                "amorphous_region",
                "stacking_fault",
                "phase_boundary",
            ),
            "note": "无专家缺陷，但结构教师产生少量局部响应",
        },
    ]

    figure, axes = plt.subplots(4, 4, figsize=(19.2, 17.2), dpi=180)
    figure.subplots_adjust(
        left=0.055, right=0.985, bottom=0.09, top=0.825, wspace=0.035, hspace=0.33
    )
    figure.suptitle(
        "Dennler 锁定外测：代表性逐像素诊断",
        fontsize=28,
        fontweight="bold",
        y=0.975,
    )
    any_metrics = metrics["targets"]["any_expert_marked_defect"][
        "micro_pixel_metrics"
    ]
    blur_metrics = metrics["targets"]["blurred_amorphous_or_beam_composite"][
        "micro_pixel_metrics"
    ]
    figure.text(
        0.5,
        0.930,
        (
            "冻结零样本推理｜61 图、183 次推理、0 次运行失败｜"
            f"任意缺陷 IoU {any_metrics['iou']:.3f} / "
            f"Precision {any_metrics['precision']:.3f} / "
            f"Recall {any_metrics['recall']:.3f}"
        ),
        ha="center",
        fontsize=15,
        color="#334155",
    )
    figure.text(
        0.5,
        0.895,
        (
            "核心失败模式：Class 2 precision 很高 "
            f"({blur_metrics['precision']:.3f})，但 recall 只有 "
            f"{blur_metrics['recall']:.3f} —— 不是乱报，而是几乎不报"
        ),
        ha="center",
        fontsize=14,
        color="#B45309",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#FFFBEB",
            "edgecolor": "#F59E0B",
        },
    )

    for row_index, example in enumerate(examples):
        filename = example["filename"]
        record = _image_record_by_name(manifest, filename)
        gray = _read_source(record)
        class_map = _read_class_map(record, raw_values)
        truth = _truth_mask(class_map, example["truth_classes"])
        prediction = _prediction_mask(record, example["detectors"])
        row = _metric_row(metrics, example["target"], filename)

        displays = [
            _base_rgb(gray),
            _single_truth_overlay(gray, truth, example["truth_classes"]),
            _prediction_overlay(gray, prediction),
            _error_overlay(gray, truth, prediction),
        ]
        titles = [
            "原始 STEM 图像",
            f"专家标签｜面积 {100 * row['truth_positive_fraction']:.1f}%",
            f"冻结预测｜面积 {100 * row['predicted_positive_fraction']:.1f}%",
            f"误差分解｜IoU {row['iou']:.3f}",
        ]
        for column_index, (display, title) in enumerate(zip(displays, titles)):
            axis = axes[row_index, column_index]
            axis.imshow(display, interpolation="nearest")
            axis.set_title(title, fontsize=12.5, pad=6)
            _style_image_axis(axis)
        axes[row_index, 0].text(
            -0.03,
            1.18,
            f"{example['label']}｜{filename.replace('.tif', '')}",
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="#0F172A",
        )
        axes[row_index, 3].text(
            1.02,
            0.50,
            example["note"],
            transform=axes[row_index, 3].transAxes,
            ha="left",
            va="center",
            fontsize=10.5,
            color="#475569",
            wrap=True,
        )

    legend = [
        Patch(facecolor=ERROR_COLORS["tp"], label="TP：预测与专家一致"),
        Patch(facecolor=ERROR_COLORS["fp"], label="FP：模型多报"),
        Patch(facecolor=ERROR_COLORS["fn"], label="FN：模型漏报"),
        Patch(facecolor=PREDICTION_COLOR, label="冻结预测区域"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=12.5,
        bbox_to_anchor=(0.5, 0.026),
    )
    figure.text(
        0.5,
        0.009,
        (
            "诊断边界：这是已消费 Dennler v1 的事后可视化，只用于理解失败；"
            "没有重新推理、调阈值、选模型，也不能作为下一版本的新外测。"
        ),
        ha="center",
        fontsize=10.5,
        color="#64748B",
    )
    path = OUTPUT_ROOT / "02_dennler_locked_failure_diagnostics.png"
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    _configure_fonts()
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite output: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    manifest = _read_json(MANIFEST_PATH)
    metrics = _read_json(METRICS_PATH)
    raw_values = [
        float(value) for value in metrics["dataset"]["raw_label_values_sorted"]
    ]
    paths = [
        _save_dataset_overview(manifest, raw_values),
        _save_failure_diagnostics(manifest, metrics, raw_values),
    ]
    for path in paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
