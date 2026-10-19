"""Frozen historical-diagnosis regression adapter; NOT a detection execution study.

Execute the selected current research policy, then request the original diagnosis
schema on unchanged old packets. Never supply old answers, research memories or
later task code to solvers. Assess old/new answers with version labels hidden.
All artifacts are new and exclusive-create. No external configuration is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import importlib.util
import json
from pathlib import Path

from .contracts import digest, engine_fingerprint, file_hash, inside, write_new
from .sandbox import Sandbox

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/rsi_history_replay_20261015_v1"
CURRENT = ROOT / "outputs/rsi_program_evolution_20261016_v1"
RUN = ROOT / "outputs/rsi_history_current_20261016_v1"
CASES = [f"R{i:02}" for i in range(1, 17)] + [f"C{i:02}" for i in range(1, 4)]
REPEATS = ["R03", "R07", "R10", "R14"]


def read(path):
    return json.loads(inside(ROOT, path).read_text())


def save(path, value):
    path = inside(ROOT, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new(path, value)


def legacy_validator():
    path = OLD / "_source/replay.py"
    spec = importlib.util.spec_from_file_location("frozen_history_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_response


def check_request(request):
    old_request = read(OLD / "requests" / f"blind-{request['case_id']}.json")
    if request["packet"] != old_request["packet"] or request["system"] != old_request["system"]:
        raise ValueError("Changed historical evidence or diagnosis instruction")
    if request["validated_memory"] != []:
        raise ValueError("Historical answer memories prohibited")
    allowed = {"slot", "case_id", "system", "packet", "validated_memory", "research_plan", "adapter_scope"}
    if set(request) != allowed:
        raise ValueError("Unregistered solver fields")


def prepare():
    target = inside(ROOT, RUN)
    if target.exists():
        raise FileExistsError("Do not overwrite or re-run a registered study")
    protocol = read(OLD / "protocol.json")
    selected = read(CURRENT / "meta_live/meta_result.json")["selected_policy"]
    program = CURRENT / "meta_live/archive/programs" / selected
    sources = {}
    for folder in (OLD, ROOT / "experiments/rsi_program_evolution_v1"):
        for p in folder.rglob("*"):
            if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts:
                sources[str(p.relative_to(ROOT))] = file_hash(p)
    sources[str((program / "policy.py").relative_to(ROOT))] = file_hash(program / "policy.py")
    slots = [{"slot": f"q{i+1:02}", "case_id": cid, "repeat": i >= len(CASES)}
             for i, cid in enumerate(CASES + REPEATS)]
    save(target / "protocol.json", {
        "schema": "current_policy_historical_diagnosis_probe_v1", "slots": slots,
        "requested_model": "gpt-5.6-sol", "engine_fingerprint": engine_fingerprint(),
        "selected_policy": selected, "policy_sha256": file_hash(program / "policy.py"),
        "maximum_solver_sessions": len(slots), "max_hypotheses": 3,
        "one_first_response_per_slot": True, "old_packet_hashes": protocol["packet_hashes"],
        "old_rubric_hashes": protocol["assessor_hashes"], "source_hashes": sources,
        "primary_endpoint": "key_issue_supported AND evidence_grounded on 16 historical cases",
        "secondary": ["full/partial/none", "three retain controls", "four fresh repeats",
                      "same-rubric version-hidden old/new assessment", "existing-assets feasibility"],
        "scope": "Current selected-policy-guided diagnosis adapter; not full Controller.step or historical fix execution",
        "differences_from_old": ["Actual selected policy adds a generic research focus",
            "Fresh model dispatch; old backend model identity not guaranteed matched",
            "No old memory, later task code, rubric or answers supplied to solver"],
        "limitations": ["Curated retrospective text/numeric packets, not raw-image discovery",
            "Current policy was developed after historical events; policy frozen before this regression",
            "No claim of isolated full-framework causal contribution",
            "Agent file-access restriction is instructed, not an OS-isolated solver session",
            "Backend token usage and exact returned model unavailable unless supplied"]})
    for slot in slots:
        cid, sid = slot["case_id"], slot["slot"]
        packet_path = OLD / "packets" / f"{cid}.json"
        rubric = OLD / "assessor_only" / f"{cid}.json"
        if file_hash(packet_path) != protocol["packet_hashes"][cid] or file_hash(rubric) != protocol["assessor_hashes"][cid]:
            raise ValueError("Historical packet/rubric changed")
        packet = read(packet_path)
        context = {"round": 1, "incumbent": digest({"packet": cid}), "pool": [], "memories": [],
                   "diagnostic": packet, "budget": {"remaining_proposal_sessions": 1}}
        folder = target / "policy_runs" / sid
        save(folder / "input/request.json", {"mode": "policy", "context": context})
        receipt = Sandbox(ROOT, ROOT / ".venvs/sam2", timeout=30).run(program, folder / "input", folder / "output")
        if receipt["status"] != "ok":
            raise RuntimeError("Current policy did not execute")
        plan = read(folder / "output/result.json")
        old = read(OLD / "requests" / f"blind-{cid}.json")
        request = {"slot": sid, "case_id": cid, "system": old["system"], "packet": packet,
                   "validated_memory": [], "research_plan": plan,
                   "adapter_scope": "Use the research focus to prioritize the diagnosis. Return exactly the original diagnosis JSON schema. This is a diagnosis-only probe: do not execute a proposed repair or claim an experiment ran. A retain or insufficient_evidence decision remains allowed regardless of the scheduler action."}
        check_request(request)
        save(target / "exchange" / sid / "request.json", request)
    save(target / "requests_seal.json", {s["slot"]: file_hash(target / "exchange" / s["slot"] / "request.json") for s in slots})
    return {"prepared": len(slots), "policy": selected, "destination": str(target)}


def accept(slot, origin):
    protocol = read(RUN / "protocol.json")
    if slot not in {r["slot"] for r in protocol["slots"]}:
        raise ValueError("Unknown reservation")
    folder = RUN / "exchange" / slot
    req = read(folder / "request.json")
    if file_hash(folder / "request.json") != read(RUN / "requests_seal.json")[slot]:
        raise ValueError("Request changed")
    check_request(req)
    result = read(folder / "raw_response.json")
    legacy_validator()(req, result)
    save(folder / "receipt.json", {"request_sha256": file_hash(folder / "request.json"),
        "response_sha256": file_hash(folder / "raw_response.json"), "origin": origin,
        "fork_turns": "none", "requested_model": protocol["requested_model"],
        "returned_model": None, "usage": None,
        "transport_note": "Independent session; read only its sealed request and write only its raw response; no external session log inspected"})
    return {"accepted": slot, "origin": origin}


def review_jobs():
    protocol = read(RUN / "protocol.json")
    mapping = {}
    jobs = [[] for _ in range(4)]
    for i, slot in enumerate(protocol["slots"]):
        sid, cid = slot["slot"], slot["case_id"]
        folder = RUN / "exchange" / sid
        receipt = read(folder / "receipt.json")
        if receipt["response_sha256"] != file_hash(folder / "raw_response.json"):
            raise ValueError("Response changed after acceptance")
        candidates = [("current", read(folder / "raw_response.json"))]
        if not slot["repeat"]:
            candidates.append(("legacy", read(OLD / "responses" / f"blind-{cid}.json")["result"]))
        candidates.sort(key=lambda x: digest({"slot": sid, "condition": x[0], "seed": 23041}))
        answers = []
        for j, (condition, value) in enumerate(candidates):
            label = f"{sid}-{chr(65+j)}"
            mapping[label] = {**slot, "condition": condition, "answer_sha256": digest(value)}
            answers.append({"label": label, "answer": value})
        jobs[i % 4].append({"packet": read(folder / "request.json")["packet"],
            "historical_reference": read(OLD / "assessor_only" / f"{cid}.json"), "answers": answers})
    for i, rows in enumerate(jobs):
        save(RUN / "review" / f"job{i+1}.json", {
            "instruction": "Judge each answer independently using the frozen historical_reference. Versions and old grades are hidden. Accept mechanism-equivalent alternatives, not keyword matches. full requires the essential actionable mechanism, partial means the core issue is found but mechanism incomplete, none means missing/wrong. Do not infer that proposed experiments ran. Distinguish feasibility: new expert annotation or acquisition is NOT available. Score the written experiment, not an imagined replacement. Use no other files. Return a JSON array, one item per answer label, with label, key_issue_supported(bool), evidence_grounded(bool), experiment_discriminates(bool), discovery_level(full|partial|none), feasible_existing_assets(bool), control_decision_correct(bool or null), reason_zh, missing_zh. Cite precise answer content in reasons. Do not force a winner.", "rows": rows})
    save(RUN / "private_review_map.json", mapping)
    return {"jobs": 4, "answers": len(mapping)}


def finish():
    protocol = read(RUN / "protocol.json")
    mapping = read(RUN / "private_review_map.json")
    assessments = []
    for i in range(1, 5):
        job = read(RUN / "review" / f"job{i}.json")
        values = read(RUN / "review" / f"raw_assessment{i}.json")
        expected = {a["label"] for row in job["rows"] for a in row["answers"]}
        if {v["label"] for v in values} != expected or len(values) != len(expected):
            raise ValueError("Incomplete or duplicate evaluation labels")
        for v in values:
            if v["discovery_level"] not in {"full", "partial", "none"}:
                raise ValueError("Invalid grade")
            for k in ("key_issue_supported", "evidence_grounded", "experiment_discriminates", "feasible_existing_assets"):
                if type(v[k]) is not bool:
                    raise ValueError("Invalid binary assessment")
            assessments.append({**mapping[v["label"]], **v})
    changed = [p for p, h in protocol["source_hashes"].items() if file_hash(ROOT / p) != h]
    if changed:
        raise ValueError("Frozen sources changed: " + str(changed))
    summary = {"protocol_sha256": file_hash(RUN / "protocol.json"), "assessment": assessments,
               "protected_files_checked": len(protocol["source_hashes"]), "changed": [],
               "scope": protocol["scope"], "conditions": {}}
    for condition in ("current", "legacy"):
        rows = [r for r in assessments if r["condition"] == condition and not r["repeat"] and r["case_id"].startswith("R")]
        summary["conditions"][condition] = {"n": len(rows),
            "grounded_core_hits": sum(r["key_issue_supported"] and r["evidence_grounded"] for r in rows),
            "grades": dict(Counter(r["discovery_level"] for r in rows)),
            "feasible_existing_assets": sum(r["feasible_existing_assets"] for r in rows)}
    summary["controls"] = [r for r in assessments if r["case_id"].startswith("C")]
    summary["repeats"] = [r for r in assessments if r["repeat"]]
    summary["historical_published_reference"] = {k: read(OLD / "summary_v1.json")[k] for k in ("history_counts", "history_core_supported", "existing_asset_full_experiments")}
    save(RUN / "summary.json", summary)
    e = html.escape
    lines = []
    historical = read(OLD / "summary_v1.json")
    titles = {r["case_id"]: r["title"] for r in historical["history"]}
    for r in [r for r in assessments if r["condition"] == "current"]:
        grade = {"full": "完整", "partial": "部分", "none": "未命中"}[r["discovery_level"]]
        lines.append(f"<tr><td>{e(r['case_id'])}{' 重复' if r['repeat'] else ''}</td><td>{e(titles.get(r['case_id'], '正常对照'))}</td><td>{grade}</td><td>{e(r['reason_zh'])}</td><td>{e(r['missing_zh'])}</td></tr>")
    body = f'''<!doctype html><meta charset="utf-8"><title>新版 RSI 历史诊断复测</title>
<style>body{{font-family:Arial,sans-serif;max-width:1250px;margin:40px auto;padding:20px;background:#fafbf9;color:#18312e;line-height:1.7}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border:1px solid #ccd8d3;text-align:left;vertical-align:top}}pre{{white-space:pre-wrap}}a{{color:#08766c}}</style>
<h1>新版 RSI：历史关键问题诊断复测</h1><p>冻结新版研究策略实际运行；16 个原历史观测包、3 个正常对照、4 个新会话重复。旧答案与评分标准不进入求解输入。独立评审隐藏版本，并重新评审旧答案。</p>
<h2>结果</h2><pre>{e(json.dumps(summary['conditions'], ensure_ascii=False, indent=2))}</pre>
<p>旧报告原成绩为核心问题 16/16、完整 13/16；这里旧答案经同批评审重新评分，不能把评审差异当成算法变化。</p>
<h2>本次到底测了什么</h2><p>这是当前策略引导的研究诊断入口回归，不是完整 Controller.step，不运行历史修复、不重训学生、不代表所有框架机制带来因果收益。每题都是冷启动，无历史记忆或候选库；选定策略在这一输入条件下返回通用的单机制研究重点。现有框架是在历史事件之后开发的，本次不是前瞻未知任务测试。</p>
<h2>逐题结果</h2><table><tr><th>编号</th><th>历史问题</th><th>覆盖</th><th>判定依据</th><th>未覆盖部分</th></tr>{''.join(lines)}</table>
<h2>原始证据</h2><p><a href="summary.json">全部评分与新旧比较</a> · <a href="protocol.json">冻结协议</a> · <a href="requests_seal.json">请求哈希</a></p>
<p>受保护文件核查：{summary['protected_files_checked']} 项无变化。无生产修改、无数据删除、无新增专家标注。</p>'''
    report = inside(ROOT, RUN / "report.html")
    with report.open("x") as f:
        f.write(body)
    save(RUN / "final_seal.json", {str(p.relative_to(RUN)): file_hash(p) for p in sorted(RUN.rglob("*")) if p.is_file()})
    return summary["conditions"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "accept", "review", "finish"])
    parser.add_argument("--slot")
    parser.add_argument("--origin")
    args = parser.parse_args()
    result = prepare() if args.action == "prepare" else accept(args.slot, args.origin) if args.action == "accept" else review_jobs() if args.action == "review" else finish()
    print(json.dumps(result, ensure_ascii=False))
