#!/usr/bin/env python3
"""Publish all arms, failures and exact research bindings; never choose by audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from tfree_detection.autonomous_rsi import promotion, summarize
from tfree_detection.expert_supervision import contained_path, sha256_file, write_new_json

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"guo_2021_zro2_twin_acquisition": "ZrO2", "sigma3-orientation-i": "Al ORI", "sigma3-orientation-ii": "Al ORII"}
ARM_NAMES = {"feedback": "反馈驱动程序合成", "open_loop": "无反馈程序搜索", "parameter_only": "固定特征仅调参"}


def read(p):
    return json.loads(p.read_text())


def release(study, output):
    p = read(study / "protocol.json"); seal = read(study / "search_seal.json")
    for name, digest in p["source_hashes"].items():
        if sha256_file(contained_path(ROOT, name)) != digest:
            raise ValueError("Frozen controller changed")
    for name, digest in seal["files"].items():
        if sha256_file(contained_path(study, name)) != digest:
            raise ValueError("Sealed search changed")
    if output.exists():
        raise FileExistsError(output)
    current = read(ROOT / "outputs/interface_improvement_20261015_v1/release_v2/results.json")
    audits = []; trajectories = []
    for search in p["searches"]:
        d = study / "audit" / f'search-{search["index"]:02d}'
        if sha256_file(d / "result.json") != read(d / "complete.json")["result_sha256"]:
            raise ValueError("Audit modified")
        value = read(d / "result.json"); value["path"] = str((d / "result.json").relative_to(ROOT)); audits.append(value)
        s = study / "searches" / f'search-{search["index"]:02d}'
        base = read(s / "baseline.json")
        trajectory = [{"round": 0, "macro": float(np.mean([r["mean"] for r in base["summary"].values()])), "winner": None}]
        for iteration in range(1, p["rounds"] + 1):
            decision = read(s / f"round-{iteration:02d}" / "decision.json")
            trajectory.append({"round": iteration, "macro": float(np.mean([r["mean"] for r in summarize(decision["rows"]).values()])),
                               "winner": decision["winner"], "lineage_depth": len(decision["lineage"])})
        trajectories.append({"search": search, "trajectory": trajectory})
    summary = []; gates = {}
    for arm in ARM_NAMES:
        ar = [r for r in audits if r["search"]["arm"] == arm]
        new_rows = [v for r in ar for v in r["rows"]]; old_rows = [v for r in ar for v in r["baseline_rows"]]
        gates[arm] = promotion(new_rows, old_rows, old_rows, minimum_gain=p["minimum_macro_gain"],
                              source_regression=p["source_regression_tolerance"], seed_regression=p["seed_regression_tolerance"])
        for result in ar:
            group = result["search"]["outer_group"]
            baseline = result["baseline"][group]["mean"]
            expected = next(r["new_routed"] for r in current["summary"] if r["group"] == group)
            if not np.isclose(baseline, expected, atol=1e-12, rtol=0):
                raise ValueError("New comparison baseline differs from frozen best interface release")
            summary.append({"arm": arm, "group": group, "baseline": baseline,
                "candidate": result["summary"][group]["mean"], "seed_means": result["summary"][group]["seeds"],
                "delta_percentage_points": 100 * (result["summary"][group]["mean"] - baseline),
                "retained_program_stages": len(result["lineage"]),
                "nonregressing": result["summary"][group]["mean"] >= baseline - p["source_regression_tolerance"]})
    output.mkdir(parents=True, exist_ok=False); bindings = output / "bindings"; bindings.mkdir()
    runtime_names = list(p["source_hashes"]) + ["src/tfree_detection/autonomous_runtime.py", "scripts/run_autonomous_detection.py"]
    runtime_hashes = {name: sha256_file(ROOT / name) for name in runtime_names}
    binding_rows = []
    for result in audits:
        search = result["search"]
        row = next(r for r in summary if r["arm"] == search["arm"] and r["group"] == search["outer_group"])
        for job in [j for j in p["jobs"] if j["stage"] == "outer" and j["outer_group"] == search["outer_group"]]:
            destination = bindings / f'{search["arm"]}-job-{job["index"]:02d}.json'
            b = {"format": "phyevo-autonomous-program-binding-v1", "production_approved": False,
                "job_index": job["index"], "program_resolution": p["program_resolution"],
                "base_binding": f'outputs/interface_improvement_20261015_v1/release_v2/bindings/job-{job["index"]:02d}.json',
                "audit": result["path"], "audit_sha256": sha256_file(ROOT / result["path"]),
                "audit_nonregressing": row["nonregressing"], "global_research_admission": gates[search["arm"]]["accepted"],
                "runtime_source_hashes": runtime_hashes,
                "scope": "Explicit research CV family; no automatic selection by filename; rejected arms retained for inspection"}
            write_new_json(destination, b)
            binding_rows.append({"arm": search["arm"], "group": search["outer_group"], "seed": job["seed"],
                                 "path": str(destination.relative_to(ROOT))})
    evidence = {"summary": summary, "trajectories": trajectories, "final_audit_gates": gates,
        "bindings": binding_rows, "source_hashes": runtime_hashes, "protocol": str((study / "protocol.json").relative_to(ROOT)),
        "search_seal_sha256": sha256_file(study / "search_seal.json"),
        "autonomous_rounds_per_search": p["rounds"], "searches": len(p["searches"]),
        "trained_candidate_trials": len(p["searches"]) * p["rounds"] * p["candidates_per_round"],
        "inner_readout_fits": len(p["searches"]) * p["rounds"] * p["candidates_per_round"] * 6,
        "point_regression": current["point_regression"],
        "claims": {"human_designed_grammar_and_governor": True, "new_expressions_generated_by_platform": True,
                   "LLM_used": False, "unrestricted_self_rewriting": False, "outer_audit_feedback_used": False,
                   "historical_baseline_development_informed": True, "new_blind_external_test": False,
                   "accuracy_or_precision_measured": False, "production_default_changed": False,
                   "positive_recursive_development_updates": max(t["trajectory"][-1]["lineage_depth"] for t in trajectories if t["search"]["arm"] == "feedback") >= 2,
                   "nonregressing_final_feedback_update_on_all_sources": gates["feedback"]["accepted"]}}
    write_new_json(output / "results.json", evidence)
    with ZipFile(output / "runtime_source.zip", "x", compression=ZIP_DEFLATED) as z:
        for name in runtime_names:
            z.write(ROOT / name, name)
    write_new_json(output / "seal.json", {"files": {str(f.relative_to(output)): sha256_file(f) for f in output.rglob("*") if f.is_file()}})
    print(json.dumps({"summary": summary, "global_feedback_admitted": gates["feedback"]["accepted"]}, ensure_ascii=False), flush=True)


def report(study, output):
    # Do not publish a verified report until every search replay has completed.
    p = read(study / "protocol.json")
    verification = [read(study / "verification" / f'search-{s["index"]:02d}.json') for s in p["searches"]]
    results = read(output / "results.json")
    import os
    cache = contained_path(ROOT, ".codex-tmp/autonomous-report-20261015"); cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties
    import textwrap
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    dest = output / "report"; dest.mkdir(exist_ok=False)
    groups = p["groups"]
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(groups)); width = .19
    for k, arm in enumerate(["baseline"] + list(ARM_NAMES)):
        values = [next(r["baseline"] if arm == "baseline" else r["candidate"] for r in results["summary"]
                       if r["group"] == g and r["arm"] == ("feedback" if arm == "baseline" else arm)) * 100 for g in groups]
        bars = ax.bar(x + (k - 1.5) * width, values, width, label="冻结最佳框架" if arm == "baseline" else ARM_NAMES[arm])
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=10)
    ax.set_xticks(x, [NAMES[g] for g in groups]); ax.set_ylim(0, 105)
    ax.set_ylabel("专家已标正区域覆盖率（%）", fontproperties=font)
    ax.set_title("封存全部搜索之后的一次外层来源审计｜面积上限 10%", fontproperties=font, fontsize=17)
    ax.legend(prop=font, ncol=2); ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .025, "失败或退步照实呈现；原基线不被覆盖。当前资料有历史开发影响，不是新的外部盲测。", ha="center", fontproperties=font)
    fig.tight_layout(rect=[0, .07, 1, 1]); fig.savefig(dest / "01_outer_audit_all_controls.png", dpi=170)
    curves, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, group in zip(axes, groups):
        for arm in ARM_NAMES:
            t = next(t for t in results["trajectories"] if t["search"] == {"index": p["groups"].index(group) * 3 + list(ARM_NAMES).index(arm), "outer_group": group, "arm": arm})
            ax.plot([r["round"] for r in t["trajectory"]], [100 * r["macro"] for r in t["trajectory"]], marker="o", label=ARM_NAMES[arm])
        ax.set_title(f"不向改进器开放 {NAMES[group]}", fontproperties=font)
        ax.set_xlabel("自动改进轮次", fontproperties=font); ax.set_xticks([0, 1, 2, 3]); ax.grid(alpha=.2)
    axes[0].set_ylabel("两个开发来源的内层宏平均覆盖率（%）", fontproperties=font)
    axes[-1].legend(prop=font, fontsize=9)
    curves.suptitle("内部开发曲线：不是独立测试曲线", fontproperties=font, fontsize=17)
    curves.tight_layout(); curves.savefig(dest / "02_internal_feedback_trajectories.png", dpi=170)

    def page(title, paragraphs):
        f, a = plt.subplots(figsize=(11.7, 8.3)); a.axis("off")
        a.text(0, 1, title, va="top", transform=a.transAxes, fontproperties=font, fontsize=23)
        y = .86
        for heading, body in paragraphs:
            a.text(0, y, heading, va="top", transform=a.transAxes, fontproperties=font, fontsize=15, color="#087e8b"); y -= .05
            lines = textwrap.wrap(body, width=62, break_on_hyphens=False)
            a.text(0, y, "\n".join(lines), va="top", transform=a.transAxes, fontproperties=font, fontsize=11, linespacing=1.5)
            y -= .034 * len(lines) + .04
        f.subplots_adjust(left=.07, right=.97, top=.93, bottom=.06)
        return f
    admission = results["final_audit_gates"]["feedback"]
    with PdfPages(dest / "report.pdf") as pdf:
        pdf.savefig(page("PhyEvo-Mic｜自主改进循环的实现与验证", [
            ("实现了什么", "平台根据开发来源的漏覆盖证据，自行组合空间运算生成新程序，训练残差读出，通过固定门槛决定保留或拒绝。三轮运行期间改进器代码、预算和晋升标准不变；每轮不接受人提供的算法建议。"),
            ("自主性的准确范围", "这是受限程序合成，不是 LLM，也不是无限制重写系统。人预先规定图像运算语法和评价规则；具体表达式、训练参数、候选筛选及版本继承由程序完成。旧物理机制和学生特征作为固定起点，不追溯声称历史规则也是自主发现的。"),
            ("实验规模与对照", f"3 个来源排除家族 × 3 个实验组，每组连续 3 轮、每轮 6 个候选。共 {results['trained_candidate_trials']} 个候选试验、{results['inner_readout_fits']} 次内层读出训练。与无反馈程序搜索、固定特征仅调参比较相同候选训练和验证预算；不宣称逐 FLOP 相同。"),
            ("最终审计", "反馈组通过全部来源非退化与改进门槛。" if admission["accepted"] else "反馈组未通过全部来源非退化与改进门槛，不能替换冻结最佳版。内部进步不能被写成外推持续正收益；所有未通过结果仍保留。")]))
        pdf.savefig(curves)
        pdf.savefig(fig)
        pdf.savefig(page("证据边界、保护与复现", [
            ("评价究竟是什么", "候选覆盖的专家已标正区域像素数 / 专家已标正区域像素数。固定 10% 面积上限，逐病例、种子和来源汇总。未标区域是未知，不能计算完整准确率、精确率，不能把候选当成已确认缺陷。"),
            ("隔离与历史影响", "每个改进器只接触另外两个开发来源；内层训练排除当前验证来源，三个种子同步比较。所有搜索封存后才评估外层来源，结果不回流到下一轮。但这些数据和旧基线有历史开发影响，所以这不是完全独立的新盲测。"),
            ("代码与数据保护", "原最佳框架的源代码、成功权重、结果及专家审核快照已另存归档；945 项受保护资产、358 个源文件已核验。原默认入口未修改，点位分支保持原模型与回放产物。失败试验不删除，原标注不覆盖。"),
            ("复核与入口", f"独立 NumPy 排序重新核对 {sum(r['metric_rows_independently_recomputed'] for r in verification)} 条候选评价记录，并重放每轮晋升与父版本链。研究推理入口为 scripts/run_autonomous_detection.py，显式绑定一个模型；不能根据文件名自动挑选模型。")]))
    plt.close("all")
    write_new_json(dest / "manifest.json", {"pdf_pages": 4, "data": str((output / "results.json").relative_to(ROOT)),
        "verified_searches": len(verification), "no_generated_scientific_image_content": True,
        "files": {str(f.relative_to(dest)): sha256_file(f) for f in dest.iterdir() if f.is_file()}})
    print(dest / "report.pdf", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["release", "report"])
    parser.add_argument("--study", default="outputs/autonomous_rsi_20261015_v1")
    parser.add_argument("--output", default="outputs/autonomous_rsi_20261015_v1/release_v1")
    args = parser.parse_args(); source = contained_path(ROOT, args.study); out = contained_path(ROOT, args.output)
    release(source, out) if args.command == "release" else report(source, out)
