#!/usr/bin/env python
"""Visualize the consumed Dennler v0.3 versus post-hoc v0.4 diagnostic.

The script never changes the dataset or historical locked predictions.  Its
figures are descriptive because Dennler labels were already unblinded before
the v0.4 mechanisms were developed.
"""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile

from tfree_detection.diagnostic_v4 import analyze_posthoc_v4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    PROJECT_ROOT
    / "datasets/public/dennler_2021_hrstem_defects/"
    "hrstem_defects_dataset.zip"
)
HISTORICAL_ROOT = (
    PROJECT_ROOT / "benchmarks/locked_runs/dennler_v1_predictions"
)
HISTORICAL_METRICS = (
    PROJECT_ROOT
    / "benchmarks/locked_runs/dennler_v1_score/locked_metrics.json"
)
V4_ROOT = PROJECT_ROOT / "outputs/dennler_posthoc_v4_diagnostic_v2"
V4_PREDICTIONS = V4_ROOT / "predictions"
V4_METRICS = V4_ROOT / "score/posthoc_metrics.json"
OUTPUT_ROOT = V4_ROOT / "figures"

CLASS_COLORS = {
    1: np.asarray([0.96, 0.49, 0.08]),
    2: np.asarray([0.78, 0.18, 0.58]),
}
OLD_COLOR = np.asarray([0.42, 0.47, 0.55])
NEW_COLOR = np.asarray([0.00, 0.62, 0.72])
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
    values = np.asarray(image)
    if values.ndim == 3:
        values = np.mean(values[..., :3].astype(float), axis=-1)
    if values.ndim != 2:
        raise ValueError(f"Unsupported STEM image shape: {values.shape}")
    values = values.astype(float)
    low, high = np.percentile(values[np.isfinite(values)], [1.0, 99.5])
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def _base_rgb(gray: np.ndarray, dim: float = 1.0) -> np.ndarray:
    return np.repeat((gray * dim)[..., None], 3, axis=-1)


def _blend(
    base: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    alpha: float = 0.66,
) -> np.ndarray:
    output = np.asarray(base, dtype=float).copy()
    active = np.asarray(mask, dtype=bool)
    output[active] = (
        (1.0 - alpha) * output[active] + alpha * np.asarray(color)
    )
    return np.clip(output, 0.0, 1.0)


def _style_image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor("#CBD5E1")
        spine.set_linewidth(1.1)


def _find_record(manifest: dict, filename: str) -> dict:
    matches = [
        record
        for record in manifest["images"]
        if record["filename"] == filename
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one record for {filename!r}")
    return matches[0]


def _read_source(archive: ZipFile, record: dict) -> np.ndarray:
    path = record.get("source_relative_path", record.get("relative_path"))
    return _to_gray(tifffile.imread(BytesIO(archive.read(path))))


def _read_class_map(
    archive: ZipFile,
    record: dict,
    shape: tuple[int, int],
) -> np.ndarray:
    if record["category"] != "defective":
        return np.zeros(shape, dtype=np.uint8)
    source = str(record["source_relative_path"])
    label_path = source.replace("/images/", "/labels/")
    label = np.asarray(tifffile.imread(BytesIO(archive.read(label_path))))
    if label.ndim == 3:
        label = label[..., 0]
    output = np.full(label.shape, 255, dtype=np.uint8)
    for class_id, raw_value in enumerate((0, 127, 255)):
        output[label == raw_value] = class_id
    if np.any(output == 255):
        raise ValueError(f"Could not decode {label_path}")
    return output


def _historical_mask(record: dict, detector_ids: tuple[str, ...]) -> np.ndarray:
    masks = []
    for detector_id in detector_ids:
        path = (
            HISTORICAL_ROOT
            / "predictions"
            / record["image_id"]
            / f"{detector_id}.npz"
        )
        with np.load(path, allow_pickle=False) as values:
            masks.append(np.asarray(values["mask"], dtype=bool))
    return np.logical_or.reduce(masks)


def _v4_mask(record: dict, key: str) -> np.ndarray:
    path = V4_PREDICTIONS / record["prediction_path"]
    with np.load(path, allow_pickle=False) as values:
        return np.asarray(values[key], dtype=bool)


def _iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    union = np.sum(predicted | truth)
    return float(np.sum(predicted & truth) / max(union, 1))


def _truth_overlay(
    gray: np.ndarray,
    class_map: np.ndarray,
    truth_classes: tuple[int, ...],
) -> np.ndarray:
    output = _base_rgb(gray, dim=0.80)
    for class_id in truth_classes:
        output = _blend(
            output,
            class_map == class_id,
            CLASS_COLORS[class_id],
            alpha=0.62,
        )
    return output


def _error_overlay(
    gray: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    output = _base_rgb(gray, dim=0.30)
    output[truth & predicted] = ERROR_COLORS["tp"]
    output[~truth & predicted] = ERROR_COLORS["fp"]
    output[truth & ~predicted] = ERROR_COLORS["fn"]
    return output


def _metric_values(payload: dict, target: str) -> list[float]:
    metrics = payload["targets"][target]["micro_pixel_metrics"]
    return [
        float(metrics[name])
        for name in ("iou", "precision", "recall", "f1")
    ]


def _metric_bars(
    axis: plt.Axes,
    old_values: list[float],
    new_values: list[float],
    title: str,
) -> None:
    locations = np.arange(4)
    width = 0.36
    old_bars = axis.bar(
        locations - width / 2,
        old_values,
        width,
        color=OLD_COLOR,
        label="v0.3 锁定结果",
    )
    new_bars = axis.bar(
        locations + width / 2,
        new_values,
        width,
        color=NEW_COLOR,
        label="v0.4 事后诊断",
    )
    axis.set_xticks(locations, ["IoU", "Precision", "Recall", "F1"])
    axis.set_ylim(0, 1.08)
    axis.set_title(title, fontsize=15, fontweight="bold", loc="left")
    axis.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    for bars in (old_bars, new_bars):
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)


def _save_metric_comparison(old: dict, new: dict) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(16.5, 10.5), dpi=180)
    figure.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=0.09,
        top=0.82,
        wspace=0.23,
        hspace=0.38,
    )
    figure.suptitle(
        "Dennler v0.4 事后诊断重测：有效增益与失败边界",
        fontsize=24,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.91,
        "61/61 图成功｜预测阶段未读标签｜该数据已揭盲，因此不能称为新的 locked / zero-shot 外测",
        ha="center",
        fontsize=12.5,
        color="#475569",
    )
    _metric_bars(
        axes[0, 0],
        _metric_values(old, "blurred_amorphous_or_beam_composite"),
        _metric_values(new, "coherence_loss_region"),
        "A｜Class 2：coherence-loss / 模糊-非晶复合区",
    )
    _metric_bars(
        axes[0, 1],
        _metric_values(old, "any_expert_marked_defect"),
        _metric_values(new, "any_defect_region"),
        "B｜任意专家缺陷：保守组合",
    )
    class1_targets = [
        "secondary_symmetry_region",
        "domain_symmetry_region_ablation",
        "augmented_secondary_symmetry_region_ablation",
    ]
    labels = ["历史结构\n主分支", "局部波矢\n实验分支", "历史 + 实验\n增强分支"]
    ious = [
        new["targets"][target]["micro_pixel_metrics"]["iou"]
        for target in class1_targets
    ]
    normal_fp = [
        new["targets"][target][
            "normal_control_mean_false_positive_area_fraction"
        ]
        for target in class1_targets
    ]
    locations = np.arange(3)
    width = 0.35
    axes[1, 0].bar(
        locations - width / 2,
        ious,
        width,
        color="#2563EB",
        label="Class 1 IoU（越高越好）",
    )
    axes[1, 0].bar(
        locations + width / 2,
        normal_fp,
        width,
        color="#E11D48",
        label="正常图误报面积（越低越好）",
    )
    axes[1, 0].set_xticks(locations, labels)
    axes[1, 0].set_ylim(0, 0.18)
    axes[1, 0].set_title(
        "C｜Class 1 实验分支没有通过正常对照门控",
        fontsize=15,
        fontweight="bold",
        loc="left",
    )
    axes[1, 0].grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.7)
    axes[1, 0].legend(frameon=False, fontsize=9, loc="upper right")
    for location, iou, false_positive in zip(
        locations,
        ious,
        normal_fp,
    ):
        axes[1, 0].text(
            location - width / 2,
            iou + 0.006,
            f"{iou:.3f}",
            ha="center",
            fontsize=9,
        )
        axes[1, 0].text(
            location + width / 2,
            false_positive + 0.006,
            f"{false_positive:.3f}",
            ha="center",
            fontsize=9,
        )

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        "D｜应该怎样解读",
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    statements = [
        ("✓", "#059669", "Class 2 覆盖不足的主要机制已被修复"),
        ("✓", "#059669", "触边候选被完整保留，边缘只降低可靠性"),
        ("✓", "#059669", "保守总掩膜不接纳未验证的晶域提案"),
        ("!", "#D97706", "Precision 下降：仍需新数据集做预注册校准"),
        ("×", "#DC2626", "Class 1 / 结构边界检测尚未取得净增益"),
        ("×", "#DC2626", "这些数值不能作为独立泛化或投稿主证据"),
    ]
    for row, (symbol, color, message) in enumerate(statements):
        y = 0.79 - row * 0.13
        axes[1, 1].text(
            0.03,
            y,
            symbol,
            fontsize=18,
            fontweight="bold",
            color=color,
            va="center",
        )
        axes[1, 1].text(0.10, y, message, fontsize=12, va="center")
    axes[1, 1].add_patch(
        plt.Rectangle(
            (0.00, 0.02),
            0.98,
            0.96,
            transform=axes[1, 1].transAxes,
            fill=False,
            edgecolor="#CBD5E1",
            linewidth=1.2,
        )
    )
    handles = [
        Patch(facecolor=OLD_COLOR, label="v0.3 锁定历史结果"),
        Patch(facecolor=NEW_COLOR, label="v0.4 事后诊断结果"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    path = OUTPUT_ROOT / "01_metric_comparison.png"
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path


def _save_example_gallery(manifest: dict) -> Path:
    examples = [
        {
            "label": "A｜Class 2 典型修复",
            "filename": "ADF image defective x10M_05.tif",
            "truth_classes": (2,),
            "old_detectors": ("amorphous_region",),
            "new_key": "coherence_loss_region",
            "note": "大面积 coherence-loss 从几乎漏空变为完整覆盖",
        },
        {
            "label": "B｜Class 2 仍会过分割",
            "filename": "ADF image defective x10M_02.tif",
            "truth_classes": (2,),
            "old_detectors": ("amorphous_region",),
            "new_key": "coherence_loss_region",
            "note": "真值很小，但新版把更广的低相干区一起检出",
        },
        {
            "label": "C｜Class 1 保持历史分支",
            "filename": "BF  image defective x10M_19.tif",
            "truth_classes": (1,),
            "old_detectors": ("stacking_fault", "phase_boundary"),
            "new_key": "secondary_symmetry_region",
            "note": "实验晶域分支未通过门控，主结果没有冒险替换",
        },
        {
            "label": "D｜正常对照仍有结构误报",
            "filename": "BF image non defective x8M_22.tif",
            "truth_classes": (1, 2),
            "old_detectors": (
                "amorphous_region",
                "stacking_fault",
                "phase_boundary",
            ),
            "new_key": "any_defect_region",
            "note": "coherence-loss 不误报；残留来自历史结构分支",
        },
    ]
    figure, axes = plt.subplots(4, 5, figsize=(20.0, 16.0), dpi=180)
    figure.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.075,
        top=0.86,
        wspace=0.035,
        hspace=0.31,
    )
    figure.suptitle(
        "Dennler 逐例对照：哪里变好了，哪里还没有",
        fontsize=25,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        "灰色为原子像；橙/洋红为专家标签；青色为预测；误差图中绿色 TP、粉色 FP、黄色 FN",
        ha="center",
        fontsize=12.5,
        color="#475569",
    )
    with ZipFile(ARCHIVE) as archive:
        for row_index, example in enumerate(examples):
            record = _find_record(manifest, example["filename"])
            gray = _read_source(archive, record)
            class_map = _read_class_map(archive, record, gray.shape)
            truth = np.isin(class_map, example["truth_classes"])
            old_prediction = _historical_mask(
                record,
                example["old_detectors"],
            )
            new_prediction = _v4_mask(record, example["new_key"])
            displays = [
                _base_rgb(gray),
                _truth_overlay(gray, class_map, example["truth_classes"]),
                _blend(
                    _base_rgb(gray, dim=0.80),
                    old_prediction,
                    OLD_COLOR,
                ),
                _blend(
                    _base_rgb(gray, dim=0.80),
                    new_prediction,
                    NEW_COLOR,
                ),
                _error_overlay(gray, truth, new_prediction),
            ]
            titles = [
                "原始 STEM",
                f"专家真值｜面积 {100 * truth.mean():.1f}%",
                f"v0.3｜IoU {_iou(old_prediction, truth):.3f}",
                f"v0.4｜IoU {_iou(new_prediction, truth):.3f}",
                "v0.4 误差分解",
            ]
            for column_index, (display, title) in enumerate(
                zip(displays, titles)
            ):
                axis = axes[row_index, column_index]
                axis.imshow(display, interpolation="nearest")
                axis.set_title(title, fontsize=11.5, pad=6)
                _style_image_axis(axis)
            axes[row_index, 0].text(
                -0.02,
                1.18,
                (
                    f"{example['label']}｜"
                    f"{example['filename'].replace('.tif', '')}"
                ),
                transform=axes[row_index, 0].transAxes,
                fontsize=14,
                fontweight="bold",
                va="bottom",
            )
            axes[row_index, 4].text(
                1.02,
                0.50,
                example["note"],
                transform=axes[row_index, 4].transAxes,
                fontsize=10,
                color="#475569",
                va="center",
                wrap=True,
            )
    handles = [
        Patch(facecolor=ERROR_COLORS["tp"], label="TP：与专家一致"),
        Patch(facecolor=ERROR_COLORS["fp"], label="FP：多报"),
        Patch(facecolor=ERROR_COLORS["fn"], label="FN：漏报"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    figure.text(
        0.5,
        0.018,
        "事后诊断展示：样本用于解释机制，不是独立外测，也不能据此选择投稿阈值。",
        ha="center",
        fontsize=10.5,
        color="#64748B",
    )
    path = OUTPUT_ROOT / "02_example_gallery.png"
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path


def _synthetic_border_band(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    crystal = (
        np.cos(2 * np.pi * xx / 12.0)
        + 0.9 * np.cos(2 * np.pi * yy / 14.0 + 0.4)
        + 0.25 * np.cos(2 * np.pi * (xx / 12.0 + yy / 14.0))
    )
    truth = (xx >= 92) & (xx <= 164) & (yy <= 150)
    blurred = gaussian_filter(crystal, sigma=5.0, mode="reflect")
    image = np.where(truth, blurred, crystal)
    lower, upper = np.percentile(image, [0.5, 99.5])
    image = np.clip((image - lower) / (upper - lower), 0, 1)
    return np.round(image * 65535).astype(np.uint16), truth


def _save_border_invariant() -> Path:
    image, truth = _synthetic_border_band()
    result = analyze_posthoc_v4(image)
    gray = _to_gray(image)
    figure, axes = plt.subplots(1, 4, figsize=(17.2, 4.7), dpi=180)
    figure.subplots_adjust(
        left=0.035,
        right=0.985,
        bottom=0.12,
        top=0.72,
        wspace=0.08,
    )
    figure.suptitle(
        "边界规则 A：保留整条物理候选，只降低靠边测量可靠性",
        fontsize=21,
        fontweight="bold",
        y=0.96,
    )
    displays = [
        _blend(_base_rgb(gray, dim=0.80), truth, CLASS_COLORS[2]),
        result.evidence_maps["coherence_loss_score"],
        _blend(
            _base_rgb(gray, dim=0.80),
            result.coherence_loss_mask,
            NEW_COLOR,
        ),
        result.reliability_map,
    ]
    titles = [
        "合成贯穿上边缘的模糊带",
        "coherence-loss 证据",
        "新版候选：顶部仍完整保留",
        "可靠性：边缘 0.25 → 内部 1.00",
    ]
    cmaps = [None, "magma", None, "viridis"]
    for axis, display, title, cmap in zip(axes, displays, titles, cmaps):
        image_artist = axis.imshow(
            display,
            interpolation="nearest",
            cmap=cmap,
            vmin=0 if cmap else None,
            vmax=1 if cmap else None,
        )
        axis.set_title(title, fontsize=11.5)
        _style_image_axis(axis)
        if cmap:
            figure.colorbar(
                image_artist,
                ax=axis,
                fraction=0.046,
                pad=0.03,
            )
    overlap = np.sum(result.coherence_loss_mask & truth) / np.sum(truth)
    figure.text(
        0.5,
        0.035,
        (
            f"单元测试观测：真实带覆盖率 {overlap:.1%}；"
            "“触边”不再等于“整块删除”。"
        ),
        ha="center",
        fontsize=11.5,
        color="#334155",
    )
    path = OUTPUT_ROOT / "03_border_reliability_invariant.png"
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    _configure_fonts()
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite output: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    old_metrics = _read_json(HISTORICAL_METRICS)
    new_metrics = _read_json(V4_METRICS)
    manifest = _read_json(V4_PREDICTIONS / "prediction_manifest.json")
    paths = [
        _save_metric_comparison(old_metrics, new_metrics),
        _save_example_gallery(manifest),
        _save_border_invariant(),
    ]
    for path in paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
