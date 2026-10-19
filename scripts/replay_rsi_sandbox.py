#!/usr/bin/env python3
"""Re-execute the frozen RSI controller, not a transcript of winning answers.

This is a retrospective mechanism replication, not a new blind benchmark or
an attempt to surpass the incumbent. Preparation copies only protocol/cache
inputs. Old proposals, winners and audit outcomes are unavailable in the new
output until fresh searches and their audit have completed. This is a logical
data-flow sandbox, NOT an OS security boundary against arbitrary Python.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

from tfree_detection.expert_supervision import contained_path, sha256_file, write_new_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = "outputs/autonomous_rsi_20261015_v1"
DEFAULT_OUTPUT = "outputs/rsi_sandbox_replay_20261015_v1"
WRAPPER = "scripts/replay_rsi_sandbox.py"
EXTRA_CODE = [WRAPPER, "scripts/verify_autonomous_rsi.py", "scripts/report_autonomous_rsi.py"]


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def input_inventory(protocol):
    """Exact input whitelist; no historical proposals, results or winners."""
    names = ["protocol.json", "baseline/manifest.json"]
    for job in protocol["jobs"]:
        base = f"cache/job-{job['index']:02d}"
        names.append(base + "/complete.json")
        names.extend(f"{base}/case-{n:03d}.npz" for n in sorted(set(job["train_cases"] + job["test_cases"])))
    return names


def check_hashes(base, hashes):
    for name, expected in hashes.items():
        p = contained_path(base, name)
        require(p.is_file() and sha256_file(p) == expected, f"Changed protected file: {name}")
    return len(hashes)


def prepare(source, output):
    require(not output.exists(), "A new, exclusive sandbox directory is required")
    require(source != output and source not in output.parents and output not in source.parents,
            "Source and replay must be separate sibling studies")
    p = read(source / "protocol.json")
    require(p["format"] == "phyevo-autonomous-program-rsi-v1", "Unsupported frozen protocol")
    seal = read(source / "attempt_seal.json")
    check_hashes(source, seal["files"])
    check_hashes(ROOT, p["source_hashes"])
    baseline = read(source / "baseline/manifest.json")
    check_hashes(ROOT, baseline["protected_files"])
    check_hashes(ROOT, baseline["source_files"])
    check_hashes(source / "baseline", baseline["archives"])
    names = input_inventory(p)
    inventory = {name: sha256_file(contained_path(source, name)) for name in names}
    output.mkdir(parents=True, exist_ok=False)
    for name, digest in inventory.items():
        target = contained_path(output, name); target.parent.mkdir(parents=True, exist_ok=True)
        with contained_path(source, name).open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        require(sha256_file(target) == digest, "Input copy mismatch")
    code = {**p["source_hashes"], **{name: sha256_file(ROOT / name) for name in EXTRA_CODE}}
    manifest = {
        "format": "phyevo-rsi-sandbox-reexecution-v1", "source_study": str(source.relative_to(ROOT)),
        "input_files": inventory, "source_hashes": code,
        "source_attempt_seal_sha256": sha256_file(source / "attempt_seal.json"),
        "historical_outcomes_known_to_designer": True, "new_independent_test": False,
        "old_candidates_supplied_to_search": False, "production_default_modified": False,
        "baseline": "exact V0 of the frozen autonomous study, not the historical 4-to-58 percent repair",
        "sandbox_kind": "separate outputs and frozen field-only candidate interpreter; not OS isolation",
        "purpose": "Can the implemented controller repeat diagnosis/proposal/learning/selection without roundwise advice?",
        "success_conditions": ["fresh multi-round proposals and gradient fits", "all matched controls retained",
            "at least two inherited internal feedback updates", "independently replayed metrics and decisions",
            "rejected outer updates remain rejected", "original best assets and study unchanged"],
        "not_required_or_claimed": ["surpass current best", "independent generalization gain",
            "rediscover historical geometry repair", "LLM authoring", "arbitrary conversation replacement"],
        "prepared_time_ns": time.time_ns(),
    }
    write_new_json(output / "sandbox_manifest.json", manifest)
    with ZipFile(output / "sandbox_source.zip", "x", compression=ZIP_DEFLATED) as z:
        for name in code:
            z.write(ROOT / name, name)
    print("SANDBOX PREPARED", len(inventory), "input-only files", flush=True)


def checked(output):
    m = read(output / "sandbox_manifest.json")
    check_hashes(ROOT, m["source_hashes"])
    check_hashes(output, m["input_files"])
    return m


def stage_commands(output, gpus):
    r = "scripts/run_autonomous_rsi.py"
    return [
        ("search", [r, "search", "--output", str(output), "--gpus", gpus]),
        ("seal", [r, "seal-searches", "--output", str(output)]),
        ("audit", [r, "audit", "--output", str(output), "--gpus", gpus]),
        ("verify", ["scripts/verify_autonomous_rsi.py", "launch", "--output", str(output),
                    "--gpus", gpus, "--attempt", "sandbox"]),
        ("release", ["scripts/report_autonomous_rsi.py", "release", "--study", str(output),
                     "--output", str(output / "release_v1")]),
    ]


def execute(output, gpus):
    checked(output)
    events = output / "execution.jsonl"
    require(not events.exists(), "Replay already started; inspect its recorded processes, do not restart")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
           "TMPDIR": str(ROOT / ".codex-tmp"), "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    with events.open("x") as log:
        for name, args in stage_commands(output, gpus):
            checked(output)
            process = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env)
            log.write(json.dumps({"stage": name, "pid": process.pid, "started_time_ns": time.time_ns(), "argv": args}) + "\n"); log.flush()
            code = process.wait()
            log.write(json.dumps({"stage": name, "exit_code": code, "finished_time_ns": time.time_ns()}) + "\n"); log.flush()
            if code:
                raise RuntimeError(f"Sandbox stage {name} failed ({code}); all evidence retained")


def semantic(value):
    """Only timing and location bookkeeping differ between exact executions."""
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k not in {"elapsed_seconds", "files", "search_seal_sha256"}}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def assess(trajectories, verified, preserved):
    updates = [sum(r.get("winner") is not None for r in t["trajectory"][1:])
               for t in trajectories if t["search"]["arm"] == "feedback"]
    return {"bounded_loop_feasibility_reproduced": bool(verified and preserved and updates and max(updates) >= 2),
            "max_internal_feedback_updates": max(updates, default=0),
            "historical_human_geometry_repair_rediscovered": False,
            "independent_positive_rsi_proven": False, "general_conversation_agent_replaced": False}


def verify(output):
    m = checked(output); source = contained_path(ROOT, m["source_study"])
    require(sha256_file(source / "attempt_seal.json") == m["source_attempt_seal_sha256"], "Original seal changed")
    protected = read(source / "baseline/manifest.json")
    counts = {k: check_hashes(ROOT, protected[k]) for k in ("protected_files", "source_files")}
    counts["original_study"] = check_hashes(source, read(source / "attempt_seal.json")["files"])
    p = read(output / "protocol.json"); results = read(output / "release_v1/results.json")
    require({s["arm"] for s in p["searches"]} == {"feedback", "open_loop", "parameter_only"}, "Missing matched control")
    check_hashes(output, read(output / "search_seal.json")["files"])
    comparisons = []; metric_rows = 0; trials = 0
    # Historical answers are consulted ONLY now, after new search/audit/replay.
    for search in p["searches"]:
        name = f"search-{search['index']:02d}"
        fresh = output / "searches" / name; old = source / "searches" / name
        closed = read(fresh / "closed.json")
        check_hashes(fresh, closed["files"])
        receipt = read(output / "verification" / f"{name}.json")
        require(receipt["closed_sha256"] == sha256_file(fresh / "closed.json"), "Replay receipt mismatch")
        require(all(receipt[k] for k in ("all_masks_match_numpy_stable_topk", "parent_chain_verified",
                                        "source_exclusion_verified", "governor_replayed")), "Independent verification incomplete")
        require(not closed["outer_labels_accessed"] and closed["human_roundwise_interventions"] == 0, "Search intervention detected")
        metric_rows += receipt["metric_rows_independently_recomputed"]
        checks = []
        for iteration in range(1, p["rounds"] + 1):
            rd = f"round-{iteration:02d}"
            names = ["proposals.json", "decision.json"] + [f"candidate-{i:02d}.json" for i in range(p["candidates_per_round"])]
            for f in names:
                checks.append(semantic(read(fresh / rd / f)) == semantic(read(old / rd / f)))
            trials += p["candidates_per_round"]
        comparisons.append({"search": search, "all_proposals_fits_and_decisions_match": all(checks),
                            "closed_state_matches": semantic(closed) == semantic(read(old / "closed.json"))})
    reference = read(source / "release_v1/results.json")
    require(trials == len(p["searches"]) * p["rounds"] * p["candidates_per_round"], "Incomplete candidate accounting")
    same = all(c["all_proposals_fits_and_decisions_match"] and c["closed_state_matches"] for c in comparisons)
    outer_same = results["summary"] == reference["summary"] and results["final_audit_gates"] == reference["final_audit_gates"]
    verdict = assess(results["trajectories"], same and outer_same, True)
    evidence = {"format": "phyevo-rsi-sandbox-evidence-v1", "claims": verdict, "preserved_files": counts,
        "fresh_candidate_trials": trials, "independently_recomputed_metric_rows": metric_rows,
        "exact_controller_reexecution": same, "outer_audit_reproduced": outer_same,
        "comparisons": comparisons, "results": str((output / "release_v1/results.json").relative_to(ROOT)),
        "manifest_sha256": sha256_file(output / "sandbox_manifest.json"),
        "historical_answers_used_only_after_search_sealing": True,
        "interpretation": "Mechanism replication of a bounded automated improvement loop; no new independent evidence of transfer gains."}
    write_new_json(output / "sandbox_verification.json", evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k != "comparisons"}, ensure_ascii=False), flush=True)


def report(output):
    evidence = read(output / "sandbox_verification.json"); result = read(output / "release_v1/results.json")
    cache = contained_path(ROOT, ".codex-tmp/rsi-sandbox-report"); cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyBboxPatch
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    dest = output / "report"; dest.mkdir(exist_ok=False)
    navy = "#153249"; teal = "#087F8C"

    def page(title, subtitle):
        fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
        fig.text(.055, .91, title, fontproperties=font, size=23, color=navy, weight="bold")
        fig.text(.055, .855, subtitle, fontproperties=font, size=12, color="#596B78")
        return fig

    f1 = page("PhyEvo-Mic：把对话式改进转成可执行循环", "沙盘可行性复现｜不要求超过当前最好版，也不追溯声称历史规则由系统发现")
    labels = [("人指出漏检", "自动汇总未覆盖的专家正例"), ("人提出修改思路", "组合空间算子，生成候选表达式"),
              ("人要求训练与比较", "自动拟合修正项＋运行等预算对照"), ("人决定保留或撤回", "固定门槛判定＋继承已接受的父版本")]
    for i, (human, auto) in enumerate(labels):
        y = .72 - i * .14
        f1.text(.07, y, human, fontproperties=font, size=16, color=navy)
        f1.text(.31, y, "→", size=22, color=teal)
        f1.text(.37, y, auto, fontproperties=font, size=16, color=teal)
    f1.text(.07, .12, "人预先规定任务、运算语法、评价与晋升规则；每轮不再提供具体算法修改建议。", fontproperties=font, size=13)
    f1.text(.07, .065, "逻辑隔离沙盘，并非操作系统级隔离；这是受限程序合成，不是 LLM 通用编程代理。", fontproperties=font, size=11, color="#596B78")

    f2 = page("重新执行，而不是播放旧答案", "只复制初始协议与冻结模型输出；旧候选、旧赢家和旧审计不作为搜索输入")
    ax = f2.add_axes([.055, .40, .89, .32]); ax.axis("off")
    stages = ["初始模型\n专家开发数据", "失败诊断\n提出可检验假设", "生成空间程序\n训练修正项", "固定门槛\n保留或拒绝", "继承父版本\n进入下一轮"]
    for i, label in enumerate(stages):
        x = .2 * i
        ax.add_patch(FancyBboxPatch((x + .005, .2), .17, .57, boxstyle="round,pad=0.008", fc="#EAF5F4", ec=teal))
        ax.text(x + .09, .48, label, ha="center", va="center", fontproperties=font, size=12, color=navy)
        if i < 4: ax.annotate("", (x + .202, .48), (x + .18, .48), arrowprops={"arrowstyle": "->", "color": teal})
    f2.text(.06, .31, f"3 个来源排除家族 × 3 个实验分支 × 3 轮；重新执行 {evidence['fresh_candidate_trials']} 个候选试验。", fontproperties=font, size=14)
    f2.text(.06, .235, f"独立重算 {evidence['independently_recomputed_metric_rows']:,} 条内部评价；生成、训练与选择重现：{evidence['exact_controller_reexecution']}。", fontproperties=font, size=14)
    f2.text(.06, .16, "先封存全部搜索，再做外层审核；原最好版本、标注和既有实验不被覆盖。", fontproperties=font, size=13)
    f2.text(.06, .075, "历史 V0 本身已有人工开发影响；本实验不等于重新发现此前 Al ORI 从约 4% 到 58% 的人工几何修复。", fontproperties=font, size=11, color="#596B78")

    f3 = page("实际证据：内部连续改进可以复现", "全部三个来源排除家族均展示；纵轴是开发来源的已标正区域覆盖率，不是测试准确率")
    groups = list(dict.fromkeys(t["search"]["outer_group"] for t in result["trajectories"]))
    names = {"guo_2021_zro2_twin_acquisition": "排除 ZrO₂", "sigma3-orientation-i": "排除 Al ORI", "sigma3-orientation-ii": "排除 Al ORII"}
    arms = {"feedback": ("反馈驱动", teal), "open_loop": ("无反馈搜索", "#C28739"), "parameter_only": ("仅调参", "#7B7C9C")}
    for i, group in enumerate(groups):
        axis = f3.add_axes([.065 + i * .315, .27, .26, .48])
        for arm, (label, color) in arms.items():
            t = next(t for t in result["trajectories"] if t["search"]["outer_group"] == group and t["search"]["arm"] == arm)["trajectory"]
            axis.plot([r["round"] for r in t], [100*r["macro"] for r in t], "o-", color=color, label=label)
        axis.set_title(names[group], fontproperties=font, size=13); axis.set_xticks([0, 1, 2, 3]); axis.grid(alpha=.2)
        axis.set_xlabel("自动改进轮次", fontproperties=font); axis.legend(prop=font, loc="best", fontsize=9)
    zcurve = next(t["trajectory"] for t in result["trajectories"]
                  if t["search"]["arm"] == "feedback" and t["search"]["outer_group"] == "guo_2021_zro2_twin_acquisition")
    curve_text = " → ".join(f"{100*r['macro']:.2f}%" for r in zcurve)
    f3.text(.065, .15, curve_text + "：排除 ZrO₂ 后其余开发来源的内部曲线，不是 ZrO₂ 测试提升。", fontproperties=font, size=11)
    f3.text(.065, .075, "面积上限约为图像的 10%；仅使用专家已标正区域，未标像素不能当成确认负例。", fontproperties=font, size=11, color="#596B78")

    f4 = page("文章可以主张什么，不能主张什么", "机制可行性证据，与独立泛化收益分开呈现")
    outer = {r["group"]: r for r in result["summary"] if r["arm"] == "feedback"}
    def change(group):
        row = outer[group]
        return f"{100*row['baseline']:.2f}%→{100*row['candidate']:.2f}%"
    passed = evidence["claims"]["bounded_loop_feasibility_reproduced"]
    admitted = result["final_audit_gates"]["feedback"]["accepted"]
    paragraphs = [
        ("可以主张", "在人工预先定义的搜索空间和冻结评价规则下，系统可以无需逐轮人工建议，\n自主完成失败反馈、候选程序生成、参数学习、版本选择和多轮继承。"),
        ("本次证据强度", f"机制复现核查通过：{passed}；最多内部反馈更新 {evidence['claims']['max_internal_feedback_updates']} 次；保留全部对照与失败。\n这是回顾性机制复现，不是新增独立样本，也不是系统重新发现了整个历史研发过程。"),
        ("必须同时披露", f"最终外推整体准入通过：{admitted}。ZrO₂ {change('guo_2021_zro2_twin_acquisition')}；\nAl ORI {change('sigma3-orientation-i')}；Al ORII {change('sigma3-orientation-ii')}。此沙盘不替换生产版本。"),
        ("不能据此声称", "持续正收益 RSI、任意场景泛化、完整替代科学家／对话代理、\n自主发现此前人工设计的几何规则，或已经完成生产部署。"),
    ]
    for i, (head, body) in enumerate(paragraphs):
        y = .74 - .17*i
        f4.text(.065, y, head, fontproperties=font, size=15, color=teal)
        f4.text(.22, y, body, fontproperties=font, size=12, color=navy, va="top", linespacing=1.6)
    figures = [f1, f2, f3, f4]
    with PdfPages(dest / "RSI_sandbox_feasibility_zh.pdf") as pdf:
        for i, fig in enumerate(figures, 1):
            fig.text(.94, .025, f"{i}/4", size=9, color="#596B78")
            pdf.savefig(fig)
            fig.savefig(dest / f"page-{i:02d}.png", dpi=120)
    plt.close("all")
    write_new_json(dest / "seal.json", {"files": {p.name: sha256_file(p) for p in dest.iterdir() if p.is_file()},
        "verification_sha256": sha256_file(output / "sandbox_verification.json"), "pages": 4})
    print(dest / "RSI_sandbox_feasibility_zh.pdf", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "execute", "verify", "report"])
    parser.add_argument("--source", default=DEFAULT_SOURCE); parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    args = parser.parse_args(); output = contained_path(ROOT, args.output)
    if args.command == "prepare": prepare(contained_path(ROOT, args.source), output)
    elif args.command == "execute": execute(output, args.gpus)
    elif args.command == "verify": verify(output)
    else: report(output)
