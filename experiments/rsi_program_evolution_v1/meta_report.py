"""Render a checked, non-overwriting report of actual meta-policy experiments."""
import argparse
import json
import sqlite3

from .contracts import file_hash, inside, write_new
from .real_study import ROOT, read


def average(summary):
    return sum(row["mean"] for row in summary.values()) / len(summary)


def collect(directory):
    directory = inside(ROOT, directory)
    config = read(directory / "study.json")
    result = read(directory / "meta_result.json")
    db_path = directory / "archive/archive.sqlite3"
    db = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    try:
        trials = {i: {"candidate": c, "split": s, "outcome": o, "receipt": json.loads(r)}
                  for i, c, s, o, r in db.execute("SELECT * FROM trials")}
        proposals = {i: json.loads(p) for i, p in db.execute("SELECT id,proposal FROM candidates")}
    finally:
        db.close()
    table = []
    for i, pair in enumerate(result["pairs"]):
        base = trials[pair["control"]["baseline_trial"]]
        row = {"start": i, "starting_coverage": average(base["receipt"]["summary"])}
        for side in ("control", "candidate"):
            run = pair[side]
            trial = trials[run["trial"]]
            if trial["candidate"] != run["candidate"] or trial["split"] != "validation" or trial["outcome"] != "ok":
                raise ValueError("Meta summary is not backed by its own validation trial")
            if trial["receipt"]["summary"] != run["summary"]:
                raise ValueError("Meta summary changed after evaluation")
            row[side] = {"coverage": average(run["summary"]), "gain_per_call": run["gain_per_call"],
                         "candidate": run["candidate"], "trial": run["trial"],
                         "mechanism": proposals[run["candidate"]]["hypothesis"],
                         "training_steps": proposals[run["candidate"]]["training_steps"]}
        table.append(row)
    follow_path = directory / "continuation-result.json"
    continuation = read(follow_path) if follow_path.exists() else None
    if continuation and continuation["policy"] != result["selected_policy"]:
        raise ValueError("Next round did not use the admitted policy")
    if continuation:
        trial = trials[continuation["trial"]]
        if (trial["candidate"] != continuation["candidate"] or trial["split"] != "validation"
                or trial["receipt"].get("summary") != continuation.get("summary")):
            raise ValueError("Next-round result is not backed by its actual task execution")
    requests = [key for key in config["allowed_requests"]
                if (directory / "requests" / key / "request.json").exists()]
    total_seconds = sum(ex["seconds"] for t in trials.values()
                        for ex in t["receipt"].get("executions", []))
    return {"study": str(directory.relative_to(ROOT)), "table": table,
            "research_policy": proposals[result["candidate_policy"]],
            "accepted": result["accepted"], "selected_policy": result["selected_policy"],
            "selected_task_program": result["selected_task_program"],
            "paired_gain_per_call": result["policy_selection"]["paired_gain_per_call"],
            "continuation": continuation, "reserved_proposal_sessions": len(requests),
            "design_sessions": 1, "paired_sessions": 4,
            "token_usage": None, "currency_cost": None, "task_execution_seconds": total_seconds,
            "independent_datasets": 1, "paired_starting_states": 2,
            "statistical_significance_claimed": False, "final_audit_used": False,
            "source_result_sha256": file_hash(directory / "meta_result.json")}


def render(directory, destination):
    output = inside(ROOT, destination)
    report = collect(directory)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "report.json", report)
    lines = ["# 研究策略自身改进：真实配对实验", "",
        "这轮评价的是“怎样选择下一次研究”，不是把某个检测分数提高直接算成研究策略提高。",
        "候选程序由独立请求（指定 GPT-5.6-Sol）生成，代码原文封存，在断网沙盒中执行。", "",
        "## 配对结果", "",
        "指标：固定 10% 候选面积下的已知专家正例覆盖率；不是正确率或 IoU。",
        "每对共享同一基础模型、数据、起始任务程序和可用实验记录，每边一次后代提案。", "",
        "| 起始状态 | 起始覆盖 | 原研究策略的后代 | 新研究策略的后代 |", "| --- | ---: | ---: | ---: |"]
    for row in report["table"]:
        lines.append(f"| {row['start'] + 1} | {row['starting_coverage']:.2%} | "
                     f"{row['control']['coverage']:.2%} | {row['candidate']['coverage']:.2%} |")
    lines += ["", "准入结果：" + ("接受新研究策略。" if report["accepted"] else "保留原研究策略；新策略作为研究候选保存。"),
              f"两组平均的非负最佳增益／提案机会差：{100 * report['paired_gain_per_call']:.3f} 个百分点。",
              "准入阈值预先固定为平均差超过 0.2 个百分点，且两组均不得低于原策略。", "",
              "## 模型提出的机制", "", report["research_policy"]["hypothesis"], "",
              report["research_policy"]["intervention"], "", "## 下一轮是否真实继承", ""]
    following = report["continuation"]
    if following:
        value = average(following["summary"]) if following.get("summary") else None
        lines += ["已检查下一轮请求实际调用获选研究策略，并实际执行其生成的任务程序。",
                  f"下一轮结果：{value:.2%}。" if value is not None else "下一轮执行失败，失败回执已保留。",
                  "任务版本决定：" + ("接受更新。" if following["decision"]["accepted"] else "保留先前任务版本。")]
    else:
        lines += ["尚未执行后续轮次；不能仅凭策略准入声称完成后续继承。"]
    lines += ["", "## 成本与解释", "",
        f"本阶段已保留 {report['reserved_proposal_sessions']} 个提案请求：策略设计 1 次、配对后代 4 次，另留 1 次实际继承验证。",
        f"归档的任务沙盒执行时间合计 {report['task_execution_seconds']:.1f} 秒，包含候选重执行；不等于总墙钟时间或模型推理时间。",
        "模型后端未提供 token／费用回执，保持空值。策略设计成本单独列出，不隐藏在配对预算之外。",
        "两组是同一开发来源的两个起始程序，不是两个独立数据集；本轮不作显著性或持续普遍提升结论。",
        "最终审计未参与策略生成或选择。现有生产代码、最佳发布和专家标注不变。", ""]
    with (output / "report.md").open("x") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"report": str(output / "report.md"), "accepted": report["accepted"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("destination")
    args = parser.parse_args()
    render(args.directory, args.destination)
