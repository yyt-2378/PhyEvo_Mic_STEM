#!/usr/bin/env python3
"""Verify completed cycles and generate a Chinese, data-derived result report."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import textwrap
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from tfree_detection.expert_supervision import contained_path, load_learning_dataset, sha256_file, write_new_json
from tfree_detection.expert_student import point_candidates, matched_points


ROOT = Path(__file__).resolve().parents[1]
GROUP_NAMES = {
    "guo-superscan-single-acquisition": "SuperScan 整段序列",
    "guo_2021_mowte2_acquisition": "MoWTe2 单视野",
    "guo_2021_zro2_twin_acquisition": "ZrO2 单视野",
    "sigma3-orientation-i": "Al Σ3 / ORI",
    "sigma3-orientation-ii": "Al Σ3 / ORII",
}


def verify_cycle(directory: Path, dataset: Path):
    protocol = json.loads((directory / "protocol.json").read_text())
    summary = json.loads((directory / "cycle_summary.json").read_text())
    protocol_hash = sha256_file(directory / "protocol.json")
    if summary["protocol_sha256"] != protocol_hash:
        raise ValueError("Cycle summary/protocol mismatch")
    if protocol["learning_manifest_sha256"] != sha256_file(dataset / "manifest.json"):
        raise ValueError("Cycle used a different expert dataset")
    reports = {}
    for fold in protocol["folds"]:
        for variant in protocol["variants"]:
            run = directory / fold["fold_id"] / variant
            marker = json.loads((run / "worker_completed.json").read_text())
            checkpoint = contained_path(ROOT, marker["checkpoint"])
            if marker["protocol_sha256"] != protocol_hash:
                raise ValueError("Worker/protocol mismatch")
            if sha256_file(checkpoint / "checkpoint.json") != marker["checkpoint_sha256"]:
                raise ValueError("Worker checkpoint metadata changed")
            metadata = json.loads((checkpoint / "checkpoint.json").read_text())
            if sha256_file(checkpoint / "state.npz") != metadata["state_sha256"]:
                raise ValueError("Worker weights changed")
            if metadata["step"] != protocol["steps"] or metadata["train_case_numbers"] != fold["train_case_numbers"]:
                raise ValueError("Worker did not complete the specified training")
            if fold["held_out_group"] in metadata["train_groups"]:
                raise ValueError("Acquisition leakage")
            report = json.loads((run / "evaluation.json").read_text())
            if report["checkpoint_sha256"] != marker["checkpoint_sha256"]:
                raise ValueError("Evaluation checkpoint mismatch")
            if sorted(r["case_number"] for r in report["cases"]) != sorted(fold["validation_case_numbers"]):
                raise ValueError("Evaluation inventory mismatch")
            for case in report["cases"]:
                path = contained_path(run, case["score_path"])
                if sha256_file(path) != case["score_sha256"]:
                    raise ValueError("Evaluation score changed")
                reports[(variant, case["case_number"])] = (case, path)
    checkpoint = contained_path(ROOT, summary["shadow_checkpoint"])
    if sha256_file(checkpoint / "checkpoint.json") != summary["shadow_checkpoint_sha256"]:
        raise ValueError("Final shadow checkpoint changed")
    final_meta = json.loads((checkpoint / "checkpoint.json").read_text())
    if sha256_file(checkpoint / "state.npz") != final_meta["state_sha256"]:
        raise ValueError("Final shadow weights changed")
    lineage = json.loads((directory / "cycle_lineage.json").read_text())
    if lineage["stage"] != "release_rejected" or summary["independent_accuracy_claim_allowed"]:
        raise ValueError("This assisted-label experiment must not claim production/independent accuracy")
    return protocol, summary, reports


def aggregate_group_metrics(summaries):
    values = defaultdict(list)
    for summary in summaries:
        for item in summary["group_metrics"]:
            values[(item["geometry"], item["group"], item["variant"])].append(item)
    aggregate = []
    for (geometry, group, variant), items in sorted(values.items()):
        row = {"geometry": geometry, "group": group, "variant": variant, "seeds": len(items), "cases": items[0]["cases"]}
        keys = ("marked_point_recall_at_20", "marked_point_recall_at_50", "marked_point_recall_at_100",
                "uniform_spatial_null_recall_at_50", "border_only_null_recall_at_50",
                "human_positive_coverage_at_10pct_area", "fraction_above_0_5")
        for key in keys:
            numbers = [item[key] for item in items if item.get(key) is not None]
            if numbers:
                row[key] = {"mean": float(np.mean(numbers)), "std_across_seeds": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else 0.0,
                            "min": min(numbers), "max": max(numbers), "values": numbers}
        aggregate.append(row)
    return aggregate


def evidence_audit(manifest):
    distances, empty = [], Counter()
    points_by_case = {}
    for entry in manifest["entries"]:
        if entry["human_positive_pixels"] and not entry["teacher_positive_pixels"]:
            empty[entry["geometry"]] += 1
        if entry["geometry"] == "point" and entry["action"] == "correct_geometry":
            xy = np.asarray(entry["points_xy"])
            h, w = entry["shape"]
            d = np.minimum.reduce([xy[:, 0], xy[:, 1], w - 1 - xy[:, 0], h - 1 - xy[:, 1]])
            distances.extend(d.tolist())
            points_by_case[entry["case_number"]] = len(xy)
    return {"case_count": len(manifest["entries"]), "explicit_points": len(distances),
            "points_within_24px_border": sum(value < 24 for value in distances),
            "points_within_32px_border": sum(value < 32 for value in distances),
            "empty_teacher_positive_expert": dict(empty),
            "geometry_semantics": dict(Counter(e["semantic_kind"] for e in manifest["entries"])),
            "distance_to_border_px": distances, "points_by_case": points_by_case}


def border_band_sensitivity(manifest):
    """Post-hoc location-only control; no image intensities or defect labels used to rank."""
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in manifest["entries"]:
        if entry["geometry"] != "point" or entry["action"] != "correct_geometry":
            continue
        h, w = entry["shape"]
        yy, xx = np.indices((h, w))
        distance = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy])
        budget = max(1, round(50 * h * w / (512 * 512)))
        for width in (24, 32):
            recalls = []
            for seed in (7, 19, 31):
                rng = np.random.default_rng(seed * 10000 + entry["case_number"])
                # Uniform inside a border strip, unlike monotonic distance
                # whose local maxima mostly collapse onto the outermost edge.
                rank = rng.random((h, w)) + 2 * (distance < width)
                predicted = point_candidates(rank, budget)
                recalls.append(matched_points(predicted, entry["points_xy"]) / len(entry["points_xy"]))
            grouped[entry["group"]][str(width)].append(float(np.mean(recalls)))
    return {group: {width: float(np.mean(values)) for width, values in widths.items()}
            for group, widths in grouped.items()}


def text_page(title, paragraphs):
    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    fig.text(0.07, 0.91, title, fontsize=23, weight="bold", color="#14334c")
    y = 0.83
    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph, width=48, break_long_words=True, replace_whitespace=False):
            fig.text(0.07, y, line, fontsize=13, color="#263b46")
            y -= 0.035
        y -= 0.024
    return fig


def evidence_figure(audit):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = [0, 8, 16, 24, 48, 96, 2048]
    count, _ = np.histogram(audit["distance_to_border_px"], bins)
    axes[0].bar(range(6), count, color="#318b93")
    axes[0].set_xticks(range(6), ["0–8", "8–16", "16–24", "24–48", "48–96", "≥96"])
    axes[0].set_xlabel("标注点距最近图像边缘 / px")
    axes[0].set_ylabel("专家明确标出的点数")
    axes[0].set_title(f"{audit['points_within_24px_border']}/{audit['explicit_points']} 个点位于 24 px 边缘内")
    for index, value in enumerate(count):
        axes[0].text(index, value + 3, str(value), ha="center")
    values = [audit["empty_teacher_positive_expert"].get(key, 0) for key in ["point", "interface"]]
    axes[1].bar(["点位病例", "界面病例"], values, color=["#bf783e", "#527cad"])
    axes[1].set_ylabel("病例数")
    axes[1].set_title("教师没有候选，专家仍标出了缺陷")
    for index, value in enumerate(values):
        axes[1].text(index, value + 0.2, str(value), ha="center", fontsize=16)
    fig.suptitle("现有标注揭示的盲区：教师没有输出，不等于没有缺陷", fontsize=17)
    fig.text(0.5, 0.015, "这是描述性审计；补充式标注、边缘偏好等因素使它不能直接当作模型完整召回率。", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    return fig


def comparison_figure(rows, sensitivity):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    lookup = {(r["geometry"], r["group"], r["variant"]): r for r in rows}
    for ax, geometry, metric in [(axes[0], "point", "marked_point_recall_at_50"),
                                 (axes[1], "interface", "human_positive_coverage_at_10pct_area")]:
        groups = sorted({row["group"] for row in rows if row["geometry"] == geometry})
        xx = np.arange(len(groups))
        for offset, variant, label, color in [(-0.18, "teacher_only", "学生：仅学教师", "#527cad"),
                                               (0.18, "teacher_plus_expert", "学生：教师＋专家", "#bf783e")]:
            entries = [lookup[(geometry, group, variant)][metric] for group in groups]
            means = [100 * e["mean"] for e in entries]
            spread = [100 * e["std_across_seeds"] for e in entries]
            ax.bar(xx + offset, means, width=0.34, yerr=spread, capsize=4, label=label, color=color)
            for x, mean in zip(xx + offset, means):
                ax.text(x, mean + 1, f"{mean:.1f}", ha="center", fontsize=10)
        if geometry == "point":
            for metric2, style, label in [("uniform_spatial_null_recall_at_50", "s:", "空间随机对照"),
                                           ("border_only_null_recall_at_50", "o--", "仅偏好边缘对照")]:
                yy = [100 * lookup[(geometry, group, "teacher_only")][metric2]["mean"] for group in groups]
                ax.plot(xx, yy, style, label=label, color="#477d57" if "border" in metric2 else "#888888")
            ax.plot(xx, [100 * sensitivity[group]["24"] for group in groups], "D-.",
                    label="24px 边缘条带均匀对照（事后）", color="#af4d6c")
            ax.set_title("整采集组留出：已标点找回率\n预算：每 512×512 图像最多 50 个候选")
        else:
            ax.set_title("整采集组留出：专家界面范围覆盖率\n同一候选面积预算：整幅图的 10%")
            ax.axhline(10, color="#888888", linestyle=":", label="随机面积覆盖期望 10%")
        ax.set_xticks(xx, [GROUP_NAMES.get(g, g) for g in groups], rotation=8)
        ax.set_ylim(0, 40)
        ax.set_ylabel("百分比 / %")
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("专家标注带来的变化：按独立采集组报告，不用帧数冒充独立样本数", fontsize=16)
    fig.text(0.5, 0.015, "误差线是 3 个随机种子的标准差，不是总体置信区间。点位不是完整精确率；界面范围不是原子级中心线。", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    return fig


def example_figure(case_numbers, manifest, arrays, reports):
    fig, axes = plt.subplots(len(case_numbers), 3, figsize=(13, 5.0 * len(case_numbers) + 1), squeeze=False)
    entry_by_number = {e["case_number"]: e for e in manifest["entries"]}
    for row, number in enumerate(case_numbers):
        entry = entry_by_number[number]
        plane = arrays[number]
        human_xy = np.asarray(entry["points_xy"])
        for column, variant in enumerate([None, "teacher_only", "teacher_plus_expert"]):
            ax = axes[row, column]
            ax.imshow(plane["image"], cmap="gray", vmin=0, vmax=1)
            if entry["geometry"] == "point":
                ax.scatter(human_xy[:, 0], human_xy[:, 1], facecolors="none", edgecolors="#31c8c0", s=42, linewidths=1.2)
            else:
                ax.plot(human_xy[:, 0], human_xy[:, 1], color="#31c8c0", lw=1.7)
            if variant is None:
                if plane["teacher"].any():
                    ax.contour(plane["teacher"], levels=[0.5], colors=["#e6ad50"], linewidths=0.65)
                label = "原图＋专家（青）＋原教师（琥珀）"
            else:
                case, path = reports[(variant, number)]
                with np.load(path, allow_pickle=False) as archive:
                    score = archive["score"]
                if entry["geometry"] == "point":
                    xy = point_candidates(score, max(1, round(50 * score.size / (512 * 512))))
                    if len(xy):
                        ax.scatter(xy[:, 0], xy[:, 1], marker="+", color="#e8a4e5", s=32, linewidths=0.9)
                    metric = case["point_recall_at_budget"]["50"]
                    suffix = f"已标点命中 {metric['matches']}/{case['annotated_points']}"
                else:
                    selection = np.zeros(score.shape, bool)
                    selection.ravel()[np.argsort(-score.ravel(), kind="stable")[:int(np.ceil(0.1 * score.size))]] = True
                    ax.contour(selection, levels=[0.5], colors=["#e8a4e5"], linewidths=0.75)
                    suffix = f"范围覆盖 {100 * case['human_positive_coverage_at_10pct_area']:.1f}%"
                label = ("仅学教师" if variant == "teacher_only" else "教师＋专家") + f"（紫）\n{suffix}"
            ax.set_title(f"病例 {number} · {label}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("留出病例示例｜随机种子 23｜相应采集组未参与对应模型训练", fontsize=15, y=0.975)
    fig.text(0.5, 0.04, "原始像素坐标。紫色仅为候选；青色之外并非已确认无缺陷。单例表现不代表组平均效果。", ha="center", fontsize=10)
    # Reserve actual space between rows for two-line titles. Tight bounding
    # boxes alone do not prevent a lower title overlapping the image above it.
    fig.subplots_adjust(left=0.035, right=0.98, bottom=0.09, top=0.90, hspace=0.30, wspace=0.14)
    return fig


def build_report(dataset, cycles, output):
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    verified = [verify_cycle(cycle, dataset) for cycle in cycles]
    protocols, summaries, reports = zip(*verified)
    if len({p["seed"] for p in protocols}) != len(protocols):
        raise ValueError("Random seeds must be distinct")
    reference = {k: v for k, v in protocols[0].items() if k != "seed"}
    if any({k: v for k, v in p.items() if k != "seed"} != reference for p in protocols[1:]):
        raise ValueError("Protocols differ beyond random seed; do not pool them")
    if output.exists():
        raise FileExistsError("Refusing to overwrite a result report")
    # Retain actual producer source bytes, not merely hashes of mutable code.
    source_bytes = {}
    for name, expected in protocols[0]["code_hashes"].items():
        path = contained_path(ROOT, name)
        if sha256_file(path) != expected:
            raise ValueError("Training source changed before producer archive was sealed")
        source_bytes[name] = path.read_bytes()
    output.mkdir(parents=True)
    with ZipFile(output / "training_producer_sources.zip", "x", compression=ZIP_DEFLATED) as archive:
        for name, data in source_bytes.items():
            archive.writestr(name, data)
    rows, audit = aggregate_group_metrics(summaries), evidence_audit(manifest)
    sensitivity = border_band_sensitivity(manifest)
    result = {"format": "phyevo-expert-cycle-multiseed-report-v1", "seeds": [p["seed"] for p in protocols],
              "cycles": [str(c.relative_to(ROOT)) for c in cycles], "audit": audit, "group_metrics": rows,
              "independent_accuracy_claim_allowed": False,
              "producer_sources_sha256": sha256_file(output / "training_producer_sources.zip"),
              "report_builder_sha256": sha256_file(Path(__file__)),
              "unvalidated_geometries": manifest["unvalidated_geometries"]}
    result["posthoc_border_band_sensitivity"] = {
        "metric": "marked_point_recall_at_50_candidates_per_512_square",
        "groups": sensitivity, "claim": "posthoc_sensitivity_not_preregistered_independent_validation"}
    write_new_json(output / "aggregate_results.json", result)
    # TTC face zero may have a JP family name despite fontconfig selecting SC.
    # Register the actual file and use its actual family, not a guessed name.
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font_path.is_file():
        raise FileNotFoundError("A CJK font is required for the Chinese report")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": [font_name, "DejaVu Sans"], "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "font.size": 11})
    figures = [
        ("01_annotation_evidence.png", evidence_figure(audit)),
        ("02_group_cv_comparison.png", comparison_figure(rows, sensitivity)),
        ("03_held_out_points.png", example_figure([2, 63], manifest, arrays, reports[0])),
        ("04_held_out_interfaces.png", example_figure([105, 135], manifest, arrays, reports[0])),
    ]
    lookup = {(r["group"], r["variant"]): r for r in rows}
    findings = []
    for group in sorted({r["group"] for r in rows}):
        before, after = lookup[(group, "teacher_only")], lookup[(group, "teacher_plus_expert")]
        metric = "marked_point_recall_at_50" if before["geometry"] == "point" else "human_positive_coverage_at_10pct_area"
        a, b = before[metric]["mean"], after[metric]["mean"]
        findings.append(f"{GROUP_NAMES[group]}：{100*a:.1f}% → {100*b:.1f}%（{100*(b-a):+.1f} 个百分点）。")
    pages = [text_page("PhyEvo-Mic｜专家标注学习闭环：第一轮", [
        "本轮完成：标注封存、监督语义转换、整采集组留出训练、多种子对照、可恢复 checkpoint、可选学生推理及发布判断。专家无需再次操作。",
        f"数据：{audit['case_count']} 个病例；两位专家分工审核，不是每例双专家共识。明确点标注 {audit['explicit_points']} 个，51 例界面标注采用范围监督。",
        "实验：5 个采集组留出折 × 2 个训练分支 × 3 个随机种子；另训练 3 个使用全部可用标签的影子模型。GPU 2–7 并行，未使用已有其他任务的 GPU 0–1。",
        "对照：学生仅学教师 vs 学生学习教师＋专家。数据划分、模型、训练步数与采样方式保持一致。原教师输出也保留用于审计。",
        "发布结果：原训练自由框架保持默认。学生可显式启用作独立候选分支；不把单专家辅助标签冒充 Gold，不自动宣称生产级可靠性。",
        "本报告中的百分比不是完整检测准确率，也不证明原子元素身份或应力定量能力。",
    ]), text_page("结果要点与动机的联系", findings + [
        f"教师空预测但专家标出阳性的病例共有 {sum(audit['empty_teacher_positive_expert'].values())} 例：提示“没有输出”必须区别于“没有缺陷”。",
        f"{audit['points_within_24px_border']}/{audit['explicit_points']} 个标注点处于 24 px 边缘内。因此增加边缘偏好对照，防止将学会标注位置偏好误当成学会缺陷。",
        "结论边界：人工监督改变了输出；是否学到缺陷特征，还要看位置对照和跨组稳定性。下降与种子波动也必须保留。闭环完成不等于每项任务都应替换原模型。",
    ])]
    pdf = output / "PhyEvo_Mic_专家标注学习闭环报告.pdf"
    with pdf.open("xb") as stream, PdfPages(stream) as document:
        for fig in pages:
            document.savefig(fig)
            plt.close(fig)
        for name, fig in figures:
            fig.savefig(output / name, dpi=180, bbox_inches="tight")
            document.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    lines = ["# PhyEvo-Mic 专家学习闭环结果", "", "本报告由真实保存的标注和留出预测生成；未生成、补画或修改专家标注。", "",
             "## 结果", "", *[f"- {line}" for line in findings], "", "## 阅读边界", "",
             "- 点位指标：固定候选预算下已标注点的找回率，不能据此估计完整精确率。",
             "- 界面指标：固定 10% 候选面积下专家范围覆盖率，不是精确中心线定位精度。",
             "- 误差线：不同随机种子的波动，不是独立材料样本的统计置信区间。",
             "- 原单调边缘对照的局部峰偏向最外缘，因此补充 24/32 px 条带均匀位置对照；这是事后敏感性分析，未用于重新训练或调参。",
             "- 线缺陷只有一例拒绝预测；区域只有一个来源组。两者不能宣称已验证泛化。",
             "- 原开发数据用途已在独立学习版本中明确退役，不改写原注册表、不再把训练资产称作学生独立测试集。",
             "- 默认物理框架保持不变；可选学生分支输出未校准候选，不输出虚构的物理证据、OOD 置信度或元素身份。", "",
             "## 主要产物", "", "- `PhyEvo_Mic_专家标注学习闭环报告.pdf`：完整说明及四张必要结果图。",
             "- `aggregate_results.json`：种子级数值、采集组指标和标注分布审计。",
             "- `training_producer_sources.zip`：与训练协议哈希一致的实际代码快照。", "",
             "## 全标签影子模型", "", *[f"- seed {p['seed']}: `{s['shadow_checkpoint']}`" for p, s in zip(protocols, summaries)], "",
             "模型训练不意味着默认替换原框架。每个运行目录的 `cycle_lineage.json` 和 `release_decision.json` 记录完整闭环及生产发布未获准的原因。", ""]
    with (output / "README.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    write_new_json(output / "report_seal.json", {name: sha256_file(output / name) for name in
                   ["aggregate_results.json", "README.md", pdf.name, "training_producer_sources.zip"] + [name for name, _ in figures]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/expert_cycle_001/learning_dataset_v1"))
    parser.add_argument("--cycles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(contained_path(ROOT, args.dataset), [contained_path(ROOT, path) for path in args.cycles],
                          contained_path(ROOT, args.output))
    print(json.dumps({"output": str(args.output), "seeds": result["seeds"], "case_count": result["audit"]["case_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
