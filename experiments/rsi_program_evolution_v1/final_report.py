"""Verify sealed evidence and render a self-contained Chinese handoff report.

Reads existing trials and model replies; never launches search, training, API
calls or services. Only an explicitly new report directory is created.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from html import escape
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .contracts import digest, engine_fingerprint, file_hash, inside, verify_snapshot, write_new
from .final_audit import assert_frozen, open_archive
from .meta_report import average, collect as collect_meta
from .real_study import ARMS, DATA, HISTORICAL, PHYSICAL, ROOT, checked, exclusion_audit, load_study, read
from .study_report import arm_report


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def tests_receipt(path):
    tree = ET.parse(path)
    cases = list(tree.iter("testcase"))
    require(bool(cases), "No executed tests in the test report")
    require(not any(c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped")),
            "Completion tests include a failure, error or skip")
    return {"sha256": file_hash(path), "passed": len(cases),
            "tests": [c.attrib["classname"] + "::" + c.attrib["name"] for c in cases]}


def observation_payload(context):
    value = json.loads(json.dumps({k: context[k] for k in ("diagnostic", "budget", "field_contract")}))
    # Separate executions necessarily have different receipt IDs (including
    # paths/timing). All scientific observations, candidate identity and budgets
    # must still match; only this provenance locator is excluded from equality.
    value["diagnostic"].pop("trial")
    return value


def verify_packet(packet):
    transport, published = read(packet / "transport.json"), read(packet / "published.json")
    source = inside(ROOT, transport["source"])
    for request in (packet / "request.json", source / "request.json"):
        require(file_hash(request) == transport["request_sha256"], "Model request changed")
    require(file_hash(packet / "raw_response.json") == published["raw_response_sha256"], "Raw model reply changed")
    require(file_hash(source / "response.json") == published["response_sha256"], "Published reply changed")
    response, request = read(source / "response.json"), read(packet / "request.json")
    require(response["request_sha256"] == transport["request_sha256"], "Reply bound to another request")
    require(response["origin"] == published["origin"], "Model origin changed")
    require(response["model"] == request["model"] == "gpt-5.6-sol", "Requested model changed")
    raw = (packet / "raw_response.json").read_text()
    require(response["raw_text"] == raw and json.loads(raw) == response["proposal"], "Proposal differs from original text")
    return {"packet": packet.name, "origin": published["origin"],
            "request_sha256": transport["request_sha256"], "reply_sha256": published["response_sha256"],
            "requested_model": request["model"], "returned_model": response.get("provider_returned_model"),
            "usage": response.get("usage"), "proposal": response["proposal"]}


def verify_archive(directory, specification):
    archive = open_archive(directory)
    try:
        assert_frozen(archive, specification)
        trials = archive.trials(visible_only=False)
        by_id = {t["id"]: t for t in trials}
        artifacts = 0
        for trial in trials:
            value = {k: trial[k] for k in ("candidate", "split", "outcome", "receipt")}
            require(digest(value) == trial["id"], "Trial record changed")
            for name, sha in trial["receipt"].get("artifacts", {}).items():
                require(file_hash(inside(directory, name)) == sha, "Prediction artifact changed")
                artifacts += 1
            for checkpoint in trial["receipt"].get("checkpoints", {}).values():
                require(file_hash(inside(directory, checkpoint["path"])) == checkpoint["sha256"], "Checkpoint changed")
                artifacts += 1
        for memory in archive.memories():
            trial = by_id[memory["verified_by_trial"]]
            require(trial["split"] != "audit" and trial["candidate"] == memory["candidate"], "Audit/foreign trial in research memory")
            require(digest({k: v for k, v in memory.items() if k != "id"}) == memory["id"], "Research memory changed")
        executions = []
        for path in sorted((directory / "jobs").rglob("execution.json")):
            execution = read(path)
            require(execution["network"] is False and execution["repository_mounted"] is False
                    and execution["host_credentials"] is False, "Execution was not isolated")
            output = inside(ROOT, execution["write_mounts"][0])
            require(output == path.parent.resolve(), "Unexpected sandbox output mount")
            for stream in ("stdout", "stderr"):
                require(file_hash(output / (stream + ".log")) == execution[stream + "_sha256"], "Execution log changed")
            executions.append(execution)
        candidates = {}
        for row in archive.pool():
            candidate = archive.candidate(row["id"])
            require(read(Path(candidate["directory"]) / "proposal.json") ==
                    json.loads(json.dumps(asdict(candidate["proposal"]))), "Stored proposal differs from archive")
            candidates[row["id"]] = {"action": candidate["action"], "parents": candidate["parents"],
                                      "target": candidate["proposal"].target}
        return {"event_tail": archive.verify(), "trials": len(trials), "candidates": candidates,
                "memories": len(archive.memories()), "artifact_hashes_checked": artifacts,
                "audit_trials": sum(t["split"] == "audit" for t in trials),
                "sandbox_processes": len(executions), "sandbox_seconds": sum(x["seconds"] for x in executions),
                "gpu_processes": sum(x["gpu"] is not None for x in executions)}
    finally:
        archive.close()


def verify_sources(directory):
    _, protocol, samples = load_study(directory / "interface_live")
    dataset = read(directory / "interface_live/dataset.json")
    old = read(HISTORICAL / "protocol.json")
    job = old["jobs"][6]  # The preregistered real-study job, not selected by audit score.
    checked(DATA / "manifest.json", old["data_manifest_sha256"])
    entries = {e["case_number"]: e for e in read(DATA / "manifest.json")["entries"]}
    for location, metadata_hash, state_hash in ((job["checkpoint"], job["checkpoint_sha256"], job["state_sha256"]),
                                               (job["base"], job["base_checkpoint_sha256"], job["base_state_sha256"])):
        folder = inside(ROOT, location)
        checked(folder / "checkpoint.json", metadata_hash)
        checked(folder / "state.npz", state_hash)
    source_check = exclusion_audit(job, read(inside(ROOT, job["base"]) / "checkpoint.json"),
                                   read(inside(ROOT, job["checkpoint"]) / "checkpoint.json"), entries)
    require(source_check == dataset["source_audit"], "Training exposure differs from prepared study")
    for entry in dataset["samples"]:
        checked(ROOT / entry["source_plane"], entry["source_sha256"])
        checked(PHYSICAL / "physical" / (entry["key"] + ".npz"), entry["physical_sha256"])
        origin = entry["student_origin"]
        if "path" in origin:
            checked(ROOT / origin["path"], origin["sha256"])
    return {"samples": len(samples), "counts": dict(Counter(protocol.split(s.group) for s in samples)),
            "source_audit": source_check, "data_sha256": protocol.data_version,
            "inherited_weight_training_stages_checked": 2,
            "original_source_planes_checked": len(dataset["samples"]),
            "historical_development_informed": dataset["historical_fields_and_model_selection_development_informed"]}


def collect(directory, tests):
    directory = inside(ROOT, directory)
    protected = verify_snapshot(ROOT, read(directory / "protected_before.json"))
    source = verify_sources(directory)
    plan, audit = read(directory / "final_audit_plan.json"), read(directory / "final_audit_results.json")
    require(audit["plan_sha256"] == file_hash(directory / "final_audit_plan.json"), "Audit plan changed")
    require(not audit["audit_used_for_selection"] and not audit["training_during_audit"], "Audit not frozen")
    integrity = {spec["archive"]: verify_archive(directory / spec["archive"], spec) for spec in plan["archives"]}
    for row in audit["results"]:
        archive = open_archive(directory / row["archive"])
        try:
            trial = next(t for t in archive.trials(visible_only=False) if t["id"] == row["trial"])
            require(trial["split"] == "audit" and trial["candidate"] == row["candidate"]
                    and trial["receipt"]["summary"] == row["summary"], "Final summary differs from executed audit")
        finally:
            archive.close()
    expected = {(s["archive"], cid) for s in plan["archives"] for cid in s["programs"]}
    require(len(audit["results"]) == len(expected) and
            {(r["archive"], r["candidate"]) for r in audit["results"]} == expected, "Missing or duplicate final audits")
    arms = [arm_report(directory / "interface_live", arm) for arm in ARMS]
    comparison = read(directory / "interface_live/comparison.json")
    require(all(a["proposal_requests"] == comparison["calls_per_arm"] for a in arms), "Unmatched proposal budgets")
    first_inputs = []
    for arm in ARMS:
        request = read(directory / "interface_live" / arm / "exchange/call-001/request.json")
        context = request["context"]
        archive = open_archive(directory / "interface_live" / arm)
        try:
            observation_trial = next(t for t in archive.trials() if t["id"] == context["diagnostic"]["trial"])
            require(observation_trial["candidate"] == context["diagnostic"]["candidate"] and
                    observation_trial["split"] == "validation", "Initial diagnostic references a different trial")
        finally:
            archive.close()
        first_inputs.append(observation_payload(context))
    require(all(x == first_inputs[0] for x in first_inputs), "Initial observations or tools differ")
    packets = [verify_packet(directory / "model_packets" / f"q{i:03d}") for i in range(1, 17)]
    require(len({p["origin"] for p in packets}) == 16, "Proposal sessions were not independent")
    meta, learning = collect_meta(directory / "meta_live"), read(directory / "learning_live/result.json")
    require(plan["meta_result_sha256"] == file_hash(directory / "meta_live/meta_result.json") and
            plan["learning_result_sha256"] == file_hash(directory / "learning_live/result.json"), "Post-seal results changed")
    # Recheck stored numeric predictions, not only the integration report's booleans.
    archive = open_archive(directory / "learning_live/archive")
    try:
        trials = {t["id"]: t for t in archive.trials()}
        trained, inherited = trials[learning["trial"]]["receipt"], trials[learning["inference_trial"]]["receipt"]
        require(trained["checkpoints"] == inherited["checkpoints"], "Training state was not inherited")
        require(len(trained["rows"]) == len(inherited["rows"]) == source["counts"]["validation"], "Checkpoint reuse cohort differs")
        import numpy as np
        for first, second in zip(trained["rows"], inherited["rows"]):
            require((first["case"], first["seed"]) == (second["case"], second["seed"]), "Checkpoint replay case mismatch")
            with np.load(inside(archive.root, first["prediction"]), allow_pickle=False) as a, \
                 np.load(inside(archive.root, second["prediction"]), allow_pickle=False) as b:
                require(all(np.array_equal(a[k], b[k]) for k in ("score", "support")), "Checkpoint replay prediction mismatch")
        learning_proposal = asdict(archive.candidate(learning["candidate"])["proposal"])
        require(digest(learning_proposal) == digest(packets[-1]["proposal"]), "Trained code differs from original model proposal")
    finally:
        archive.close()
    tests = tests_receipt(inside(ROOT, tests))
    costs = {"main_proposal_sessions": 9, "meta_design_sessions": 1, "meta_descendant_sessions": 4,
             "meta_continuation_sessions": 1, "learning_integration_sessions": 1, "route_probe_sessions": 1,
             "proposal_sessions_excluding_probe": len(packets), "token_usage": None, "currency_cost": None,
             "sandbox_seconds": sum(a["sandbox_seconds"] for a in integrity.values()),
             "cost_note": "Seconds cover archived sandbox processes, not model latency or total wall time. "
                          "Backend inference counts/token usage unavailable; equal budget is proposal opportunities."}
    return {"schema": "rsi_program_evolution_handoff_v1", "engine_sha256": engine_fingerprint(),
            "protected": protected, "source": source, "comparison": comparison, "arms": arms,
            "meta": meta, "learning": learning, "audit": audit, "integrity": integrity,
            "model_packets": [{k: v for k, v in p.items() if k != "proposal"} for p in packets],
            "tests": tests, "costs": costs,
            "scope": "program/learning/research-policy RSI implementation and bounded real-data verification; "
                     "not a claim of universal or sustained performance superiority",
            "learner_initialization_seed": 1729, "inherited_model_and_evaluation_seed": 23}


def table(headers, rows):
    head = "".join("<th>" + escape(str(x)) + "</th>" for x in headers)
    body = "".join("<tr>" + "".join("<td>" + escape(str(x)) + "</td>" for x in row) + "</tr>" for row in rows)
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table>"


def render_html(report):
    labels = {"full": "完整 RSI", "generic": "通用 Agent", "no_memory": "无记忆对照"}
    rows = []
    for arm in report["arms"]:
        held = next(r for r in report["audit"]["results"] if r["archive"] == "interface_live/" + arm["arm"] and r["incumbent"])
        rows.append([labels[arm["arm"]], f"{average(arm['baseline']):.2%}",
                     f"{average(arm['selected_validation']):.2%}", f"{average(held['summary']):.2%}",
                     f"{arm['accepted_updates']} / {arm['proposal_requests']}"])
    comparison = table(["分支", "开发基线", "开发集保留结果", "留出 ZrO₂", "接受 / 提案"], rows)
    meta_rows = [[r["start"] + 1, f"{r['starting_coverage']:.2%}", f"{r['control']['coverage']:.2%}",
                  f"{r['candidate']['coverage']:.2%}"] for r in report["meta"]["table"]]
    meta_table = table(["起始状态", "起点", "旧研究策略后代", "新研究策略后代"], meta_rows)
    training = next(iter(report["learning"]["training"].values()))
    requirements = table(["要求", "实现与核验"], [
        ["R01 保护既有成果", f"{report['protected']['checked']} 个受保护文件无变化；原始输入及两阶段权重另行逐项核验"],
        ["R02 隔离执行", "真实 bwrap 网络/文件系统隔离；CPU 推理、GPU 6 训练；日志与产物哈希检查"],
        ["R03 可执行进化对象", "完整 Python 分析/路由程序、真实 fit/predict 学习状态、独立 plan 研究策略"],
        ["R04 决策落实为执行", "16 份原始提案请求/回复、机制假设、可证伪条件、实际执行和评价关联"],
        ["R05 四种搜索动作", "generate/tune/evolve/repair 接口；真实任务 tune/evolve，合成故障用例验证 repair"],
        ["R06 持久归档与经验", "父子关系、追加式事件、实测经验；断点续接及条件筛选测试，审计不进入记忆"],
        ["R07 固定评价", "来源分离；5 个归档联合封存后才做 9 次审计；不重训、不回选"],
        ["R08 复用专家标注", "使用已有标注；未申请新标注；未知像素不是专家负例"],
        ["R09 实际继承", "后续请求带入已测经验与父程序；新研究策略实际调用；20 例权重复用逐元素一致"],
        ["R10 研究策略独立评价", "两种起点各一次配对后代、独立策略准入、一次后续实际继承"],
        ["R11 同预算对照", "三组同起点、观测、工具、指定模型，每组 3 次提案机会；token 等价不可测"],
        ["R12 结果、成本与复现", f"真实 30/20/1 平面；{report['tests']['passed']} 项测试通过；本报告、JSON 与运行说明"]])
    costs = report["costs"]
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PhyEvo-Mic 程序级 RSI 验证报告</title>
<style>body{{font:17px/1.7 Arial,"Microsoft YaHei",sans-serif;color:#173447;background:#f3f6f7;margin:0}}
main{{max-width:1100px;margin:auto;padding:36px 30px 70px}}h1{{line-height:1.25}}h2{{margin-top:40px;color:#086b74}}
section,.intro{{background:white;padding:22px 26px;border-radius:12px;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:15px}}
th,td{{border-bottom:1px solid #dce5e8;padding:10px;text-align:left;vertical-align:top}}th{{background:#edf5f5}}
.flow{{padding:18px;background:#edf6f5;font-weight:bold}}.note{{color:#536875}}a{{color:#076b86}}code{{overflow-wrap:anywhere}}
@media print{{body{{background:white}}main{{padding:0}}section{{break-inside:avoid}}}}</style><main>
<p>PhyEvo-Mic · 2026-10-16 · 隔离实验，不替换生产最佳版本</p>
<h1>程序、学生能力与研究策略的 RSI 闭环</h1>
<div class="intro">已完成程序级执行闭环、真实训练状态复用和研究策略的独立配对验证。
本轮实现完成不等于已证明长期持续增益；所有接受、拒绝和留出审计结果均保留。</div>
<div class="flow">实验观测 → 机制假设与可执行程序 → 断网沙盒执行 → 固定协议评价 → 接受或保留备选 → 下一轮继承程序、权重和经验</div>
<section><h2>1. 先分清两类更新</h2><p><b>任务层</b>改变分析、候选生成、学习或物理—学生组合；
<b>研究层</b>改变选父版本、修复优先级、经验筛选与实验安排。任务分数上涨不会自动批准研究策略。
获选策略已在下一轮实际执行，不是事后把历史脚本串成故事。</p></section>
<section><h2>2. 使用哪些数据，怎样评价</h2><p>新增训练使用 Al ORI 的 30 个平面；开发评价使用 Al ORII 的 20 个平面；
最终审计使用 ZrO₂ 的 1 个平面。继承底座还训练过其他来源，完整来源记录在 report.json；
ORII 与 ZrO₂ 均从继承权重的两阶段训练中排除。历史方法设计接触过这些数据，不能称为全新外部盲测。</p>
<p><b>指标是固定 10% 图像面积预算内的已知专家正例覆盖率</b>：被选中且落在专家正例里的像素数 ÷ 专家正例像素总数。
不是 IoU、总体正确率或精确率；未标注区域不当作确认负例。下表起点是源排除的冻结组合分支，不是旧表中的纯物理分支。</p></section>
<section><h2>3. 同预算三组真实执行</h2>{comparison}<p>三组均请求 GPT-5.6 Sol，每组 3 次提案。
完整 RSI 本轮没有提高开发覆盖率；两个对照提高了开发结果，但留出来源低于共同基线 74.44%。
这说明准入保留机制运行正常，也说明本轮不能宣称跨来源普遍提高。</p>
<p class="note">具体每轮假设、修改、拒绝原因与实际继承记录见 report.json 的 arms 字段及
<a href="../progress_after_round3/report.md">逐轮开发报告</a>。</p></section>
<section><h2>4. 研究机制自身是否改进</h2>{meta_table}<p>新策略优先处理失败，并按实测候选表现选择父版本，
不再固定只微调当前版本。每对共享相同起点、证据与预算，每边一次后代提案。
两组“保留起点后的最佳非负增益/提案机会”平均差为 {100 * report['meta']['paired_gain_per_call']:.4f} 个百分点，
达到预设准入条件。第二组双方后代均退步，保留起点后增益均为零，不将退步包装为增益。</p>
<p>下一轮实际使用获选研究策略，得到 38.57%；比当前 38.39% 高 0.1839 个百分点，
小于固定的 0.2 个百分点准入门槛，因此仍保留旧任务版本。
两组是同一来源的两个起点，不是两个独立数据集，也不代表长期显著改善。</p>
<p class="note">准入效用相对各自指定起点计算，不是相对整个候选库的最优值计算。
共同候选库此前已有 38.39% 版本，配对阶段的新策略后代 38.11% 没有超过它。
因此，这一结果支持相对指定起点的研究调度收益，不能当作发现新全局最优算法的证据。
本轮保持原先协议与接受记录，不在看到结果后更改阈值或重新定义成功。</p></section>
<section><h2>5. 学生训练与 checkpoint 是否真实接通</h2><p>另设的学习接口实验由研究者要求启动，
不是新研究策略自主选择训练的证据，也不计入三组同预算比较。
模型生成了 6→8→1 的小型残差网络和正例监督/软先验损失；GPU 上完成 {training['steps']} 步、
更新 {training['parameters']} 个参数，训练损失从 {training['first_loss']:.4f} 到 {training['last_loss']:.4f}。
生成程序固定初始化种子为 1729；继承模型和评价记录种子为 23。</p>
<p>随后不重新 fit，只继承保存的权重，20 个开发病例 score/support 与原结果逐元素一致。
开发覆盖率 37.72%，低于学习起点 38.39%，候选被拒绝；其预先登记的审计覆盖率为 51.39%，
同样未据此改动选择。训练损失下降不等于泛化提升。</p></section>
<section><h2>6. 完成项与证据层次</h2>{requirements}<p>本轮真实数据验证集中于界面迁移；
点缺陷预算评价、修复和条件经验筛选另有合成测试。没有把这些测试当成真实原子位点、CIF 配准或稀土识别性能提高。</p></section>
<section><h2>7. 成本、状态和下一步用法</h2><p>真实提案共 {costs['proposal_sessions_excluding_probe']} 次：
三组任务 9 次、研究策略设计 1 次、配对后代 4 次、策略继承 1 次、学习接口 1 次；
另有 1 次通道冒烟测试。归档沙盒进程累计 {costs['sandbox_seconds']:.1f} 秒，
不含模型延迟及总墙钟时间。后端 token、费用、实际底层推理次数未返回，不填估造数字。</p>
<p>最终 5 个归档均封存，9 次审计仅推理；继续搜索必须另建协议与输出目录，不能把本轮审计反馈喂回此搜索。
当前模型通道由 Codex 会话中转，并非脱离会话的常驻自主服务。</p>
<p>未删除文件、未重启服务、未提交或推送；既有生产分支和最佳发布不变。</p>
<p><a href="report.json">完整可核对数据</a> · <a href="requirements.json">逐项验收索引</a> ·
<a href="../final_audit_plan.json">联合封存计划</a> · <a href="../final_audit_results.json">最终审计结果</a></p></section>
</main></html>'''


def render(directory, destination, tests):
    directory, destination = inside(ROOT, directory), inside(ROOT, destination)
    require(not destination.exists(), "Use a new report destination; evidence cannot be overwritten")
    report = collect(directory, tests)
    requirements = read(directory / "requirements.json")
    evidence = {
        "R01": ["report.json#protected", "report.json#source"],
        "R02": ["report.json#integrity", "../isolation_probe/result.json"],
        "R03": ["report.json#integrity", "../learning_live/result.json", "../meta_live/meta_result.json"],
        "R04": ["report.json#model_packets", "report.json#arms"],
        "R05": ["report.json#arms", "report.json#tests"],
        "R06": ["report.json#integrity", "report.json#arms", "report.json#tests"],
        "R07": ["../final_audit_plan.json", "../final_audit_results.json", "report.json#source"],
        "R08": ["report.json#source", "../learning_live/request/request.json"],
        "R09": ["report.json#arms", "../meta_live/continuation-result.json", "../learning_live/result.json"],
        "R10": ["../meta_progress/report.md", "report.json#meta"],
        "R11": ["../interface_live/comparison.json", "report.json#model_packets", "report.json#arms"],
        "R12": ["report.html", "report.json#tests", "report.json#costs"],
    }
    require(set(evidence) == {r["id"] for r in requirements}, "Requirement register changed")
    # This is an evidence index, not an automatic proof inferred from test count.
    reviewed = [{**r, "status": "implemented_and_verified_with_declared_scope", "evidence": evidence[r["id"]]}
                for r in requirements]
    destination.mkdir(parents=True, exist_ok=False)
    write_new(destination / "report.json", report)
    write_new(destination / "requirements.json", {"requirements": reviewed,
        "scope_notes": "Real microscopy evidence: interface transfer, one trajectory. Repair/point evaluation/"
                       "conditional retrieval use explicitly synthetic tests. No sustained-superiority claim.",
        "original_register_sha256": file_hash(directory / "requirements.json")})
    with (destination / "report.html").open("x", encoding="utf-8") as stream:
        stream.write(render_html(report))
    print(json.dumps({"report": str(destination / "report.html"), "tests": report["tests"]["passed"],
                      "protected": report["protected"], "audit_trials": len(report["audit"]["results"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("destination")
    parser.add_argument("--tests", required=True)
    args = parser.parse_args()
    render(args.directory, args.destination, args.tests)
