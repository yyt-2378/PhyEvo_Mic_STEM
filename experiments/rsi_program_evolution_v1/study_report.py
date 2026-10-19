"""Read-only evidence aggregation; transport failure is not algorithm failure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from .contracts import file_hash, inside, write_new


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("full", "generic", "no_memory")


def resolved_rounds(events):
    rows = {e["body"]["round"]: dict(e["body"]) for e in events if e["kind"] == "round_finished"}
    for event in events:
        if event["kind"] == "late_response_executed":
            n = event["body"]["round"]
            if rows.get(n, {}).get("status") != "provider_failed":
                raise ValueError("Late response has no original terminal transport failure")
            rows[n] = {**event["body"], "original_transport_outcome": rows[n]["status"]}
    return [rows[k] for k in sorted(rows)]


def arm_report(study, arm):
    directory = inside(ROOT, study / arm)
    db = sqlite3.connect(f"file:{directory / 'archive.sqlite3'}?mode=ro", uri=True)
    try:
        events = [{"kind": k, "body": json.loads(b)} for k, b in db.execute("SELECT kind,body FROM events ORDER BY seq")]
        trials = {i: {"candidate": c, "split": s, "outcome": o, "receipt": json.loads(r)}
                  for i, c, s, o, r in db.execute("SELECT * FROM trials")}
        proposals = {i: json.loads(p) for i, p in db.execute("SELECT id,proposal FROM candidates")}
    finally:
        db.close()
    baseline = next(e["body"] for e in events if e["kind"] == "baseline_registered")
    incumbent = [e["body"]["incumbent"] for e in events if e["kind"] == "selection"][-1]
    rounds = resolved_rounds(events)
    for row in rounds:
        if "candidate" not in row:
            continue
        p = proposals[row["candidate"]]
        trial = trials[row["trial"]]
        if trial["candidate"] != row["candidate"] or trial["split"] != "validation":
            raise ValueError("Round trial is not its own development execution")
        row.update(action=p["action"], parents=p["parents"], target=p["target"],
                   hypothesis=p["hypothesis"], intervention=p["intervention"],
                   training_steps=p["training_steps"], summary=trial["receipt"].get("summary"),
                   executed_seconds=sum(x["seconds"] for x in trial["receipt"].get("executions", [])))
    calls = [e["body"] for e in events if e["kind"] == "model_call_started"]
    requests = []
    for index, call in enumerate(calls, 1):
        path = directory / "exchange" / f"call-{index:03d}" / "request.json"
        context = call["context"]
        requests.append({"round": call["round"], "request_sha256": file_hash(path),
            "incumbent": context["incumbent"], "parent_ids": context["required_parents"],
            "memory_ids": [m["id"] for m in context["memories"]],
            "memory_count": len(context["memories"]), "archive_candidates_visible": len(context["pool"]),
            "action": context["required_action"], "focus": context["research_plan"]["focus"]})
    selected = [t for t in trials.values() if t["candidate"] == incumbent and t["split"] == "validation" and t["outcome"] == "ok"][-1]
    return {"arm": arm, "baseline": trials[baseline["trial"]]["receipt"]["summary"],
            "incumbent": incumbent, "selected_validation": selected["receipt"]["summary"],
            "rounds": rounds, "requests": requests, "proposal_requests": len(calls),
            "accepted_updates": sum(r["status"] == "accepted" for r in rounds),
            "token_usage": None, "cost_currency": None,
            "transport_timeouts": sum(e["kind"] == "model_call_finished" and
               e["body"]["receipt"]["status"] == "external_model_timeout" for e in events),
            "sealed": any(e["kind"] == "search_sealed" for e in events),
            "audit_trials": [{"candidate": t["candidate"], "summary": t["receipt"].get("summary")}
                             for t in trials.values() if t["split"] == "audit"]}


def render(directory, destination):
    study, output = inside(ROOT, directory), inside(ROOT, destination)
    if output.exists():
        raise FileExistsError("Create a new report snapshot")
    comparison = json.loads((study / "comparison.json").read_text())
    arms = [arm_report(study, arm) for arm in ARMS]
    output.mkdir(parents=True, exist_ok=False)
    result = {"study": str(study.relative_to(ROOT)), "comparison": comparison, "arms": arms,
              "metric": "known_positive_area_coverage_at_fixed_area; not accuracy or IoU",
              "independent_model_repetitions": 1, "significance_claimed": False,
              "baseline_is_source_excluded_frozen_routed_branch": True,
              "comparison_with_historical_best_is_not_same_training_exposure": True}
    write_new(output / "report.json", result)
    lines = ["# 真实电镜数据上的程序级 RSI 实验", "",
        "评价量：固定 10% 候选面积下的已知正例覆盖率，不是总体正确率、IoU 或精确率。",
        "当前为单条模型轨迹、单个模型种子的开发迁移试验，不作显著性推断。", "",
        "| 分支 | 冻结基线 | 当前保留版本 | 接受更新 | 提案请求 |", "| --- | ---: | ---: | ---: | ---: |"]
    labels = {"full": "完整 RSI", "generic": "通用 Agent", "no_memory": "无记忆对照"}
    for arm in arms:
        base = next(iter(arm["baseline"].values()))["mean"]
        current = next(iter(arm["selected_validation"].values()))["mean"]
        lines.append(f"| {labels[arm['arm']]} | {base:.2%} | {current:.2%} | {arm['accepted_updates']} | {arm['proposal_requests']} |")
    lines += ["", "## 每轮提案与真实执行", ""]
    for arm in arms:
        lines += [f"### {labels[arm['arm']]}", ""]
        for row in arm["rounds"]:
            value = next(iter(row["summary"].values()))["mean"] if row.get("summary") else None
            score = f"{value:.2%}" if value is not None else "未产生有效测量"
            lines += [f"- 第 {row['round']} 轮：{row.get('action', '无有效提案')}；{score}；{row['status']}。",
                      f"  - 机制：{row.get('hypothesis', '—')}", f"  - 修改：{row.get('intervention', '—')}"]
        lines += ["", "后续请求实际继承：", ""]
        for request in arm["requests"]:
            lines.append(f"- 第 {request['round']} 轮：{request['memory_count']} 条已测经验，"
                         f"{request['archive_candidates_visible']} 个可见候选；动作 {request['action']}。")
        lines += [""]
    lines += ["## 执行与来源说明", "",
        "原始超时事件保留；迟到回复按原请求哈希接回，没有重新请求模型，也没有重选答案。",
        "第一份提案使用内联请求并由主代理原样转存；其余通过中性命名的封存文件交换。",
        "配额按提案会话数计，未获得后端逐次推理调用数、token 或费用回执，不虚构这些成本。",
        "底座模型曾使用其他来源训练，完整来源见 dataset.json；验证和审计源被两阶段权重训练排除。",
        "现有最佳发布、原始数据、专家标注不变。新候选只在隔离沙盒执行。",
        "最终审计只在所有分支封存后执行；元研究策略需另外做同条件配对实验。", ""]
    with (output / "report.md").open("x") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"report": str(output / "report.md"), "arms": len(arms)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("destination")
    args = parser.parse_args()
    render(args.directory, args.destination)
