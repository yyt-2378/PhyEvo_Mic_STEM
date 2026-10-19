#!/usr/bin/env python3
"""Data-derived sampling audit and Chinese refinement report, exclusive outputs.

Outer best-by-result is explicitly descriptive and is never a routing policy.
The executable selector comes from the already frozen inner-source evidence.
"""
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np

from scripts.analyze_rsi_refinement import load_analysis, verified_metrics
from scripts.run_rsi_refinement import read_protocol
from tfree_detection.expert_supervision import contained_path, load_learning_dataset, sha256_file, write_new_json
from tfree_detection.rsi_refinement import POLICIES, source_probabilities
from tfree_detection.rsi_feedback import feedback_predictions
from tfree_detection.rsi_evaluation import EvaluationProtocol, evaluate_case

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "guo-superscan-single-acquisition": "SuperScan 点位",
    "guo_2021_mowte2_acquisition": "MoWTe2 点位",
    "guo_2021_zro2_twin_acquisition": "ZrO2 界面",
    "sigma3-orientation-i": "Al ORI 界面",
    "sigma3-orientation-ii": "Al ORII 界面",
}
POLICY_LABELS = {"control": "原配置", "source_sqrt": "来源权重", "positive_crops": "增加标注裁块",
                 "photometric": "轻度增强", "weak_bag": "区域弱监督", "weak_roi": "加入占比先验"}


def read(path):
    return json.loads(path.read_text())


def exact_crop_hit_probability(known_mask, patch_size, border_probability=.4):
    """Exact probability of any known pixel under the original crop distribution.

    Uniform crop first, then with probability .4 choose one of four edges and
    reset that coordinate. Corners legitimately receive additional probability.
    Padding never creates a supervision hit. No Monte Carlo or training needed.
    """
    known = np.asarray(known_mask, dtype=bool)
    if known.ndim != 2 or min(known.shape) < 1 or patch_size < 1 or not 0 <= border_probability <= 1:
        raise ValueError("Invalid crop audit input")
    h,w = known.shape
    ys, xs = np.arange(max(1,h-patch_size+1)), np.arange(max(1,w-patch_size+1))
    ye, xe = np.minimum(ys+patch_size,h), np.minimum(xs+patch_size,w)
    integral = np.pad(known.astype(np.int64).cumsum(0).cumsum(1), ((1,0),(1,0)))
    counts = integral[np.ix_(ye,xe)]-integral[np.ix_(ys,xe)]-integral[np.ix_(ye,xs)]+integral[np.ix_(ys,xs)]
    hit = counts > 0
    uniform = float(hit.mean())
    edge = float(np.mean([hit[0].mean(),hit[-1].mean(),hit[:,0].mean(),hit[:,-1].mean()]))
    return {"uniform": uniform, "edge_mixture": edge,
            "combined": (1-border_probability)*uniform+border_probability*edge}


def snapshot_producer(output, names):
    paths = [contained_path(ROOT,name) for name in names]
    with ZipFile(output/"producer_source.zip", "x", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path,str(path.relative_to(ROOT)))
    write_new_json(output/"producer_source.json", {"archive_sha256": sha256_file(output/"producer_source.zip"),
        "entries_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}})


def seal(output, inputs):
    files = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_new_json(output/"result_seal.json", {"format": "phyevo-refinement-report-seal-v1", "files": files, "inputs": inputs})


def verify_seal(directory):
    value = read(directory/"result_seal.json")
    for name,digest in value["files"].items():
        if sha256_file(contained_path(directory,name)) != digest:
            raise ValueError(f"Report/evaluation artifact changed: {name}")
    return value


def sampling_audit(training, output):
    training, output = contained_path(ROOT,training), contained_path(ROOT,output)
    protocol, dataset = read_protocol(training)
    manifest, arrays = load_learning_dataset(ROOT,dataset)
    if output.exists():
        raise FileExistsError("Sampling audit output must be new")
    output.mkdir(parents=True)
    entries = {e["case_number"]: e for e in manifest["entries"]}
    probabilities = {n: exact_crop_hit_probability(p["human_weight"] > 0,protocol["patch_size"])
                     for n,p in arrays.items()}
    rows, actual, inputs = [], [], {}
    for fold in protocol["folds"]:
        if fold["stage"] != "outer":
            continue
        train = [entries[n] for n in fold["train_case_numbers"]]
        geometries = sorted({e["geometry_index"] for e in train})
        for name, policy in POLICIES.items():
            by_source = []
            for geometry in geometries:
                sources, source_p = source_probabilities(train,geometry,policy.source_power)
                for group, conditional in zip(sources,source_p):
                    eligible = [e for e in train if e["geometry_index"] == geometry and e["group"] == group]
                    chance = []
                    for entry in eligible:
                        base = probabilities[entry["case_number"]]["combined"]
                        guided = policy.positive_crop_probability if arrays[entry["case_number"]]["human"].any() else 0
                        chance.append(guided+(1-guided)*base)
                    by_source.append({"group": group, "geometry": eligible[0]["geometry"], "cases": len(eligible),
                        "within_geometry_probability": float(conditional), "global_patch_probability": float(conditional/len(geometries)),
                        "expected_known_patch_fraction": float(np.mean(chance)),
                        "expected_draws_per_case_1200_steps": float(protocol["steps"]*protocol["batch_size"]*conditional/len(geometries)/len(eligible))})
            rows.append({"fold_id": fold["fold_id"], "held_out_group": fold["excluded_groups"][0], "policy": name,
                "expected_known_patch_fraction": sum(s["global_patch_probability"]*s["expected_known_patch_fraction"] for s in by_source),
                "source_distribution": by_source})
            for seed in protocol["seeds"]:
                path = training/name/f"seed-{seed}"/fold["fold_id"]/"step-001200/checkpoint.json"
                if not path.is_file():
                    continue
                meta = read(path)
                if meta["protocol_sha256"] != sha256_file(training/"protocol.json") or meta["fold"] != fold:
                    raise ValueError("Sampling audit checkpoint lineage mismatch")
                counts = meta["sampler_audit"]
                geometry_draws = Counter()
                for number,count in counts["by_case"].items():
                    geometry_draws[entries[int(number)]["geometry"]] += count
                actual.append({"fold_id": fold["fold_id"], "policy": name, "seed": seed,
                    "actual_known_patch_fraction": counts["positive_patches"]/counts["patches"],
                    "geometry_draws": dict(geometry_draws), "audit": counts})
                inputs[str(path.relative_to(ROOT))] = sha256_file(path)
    inputs.update({str((training/"protocol.json").relative_to(ROOT)): sha256_file(training/"protocol.json"),
                   str((dataset/"manifest.json").relative_to(ROOT)): sha256_file(dataset/"manifest.json")})
    for entry in manifest["entries"]:
        inputs[str((dataset/entry["plane"]).relative_to(ROOT))] = entry["plane_sha256"]
    result = {"format": "phyevo-exact-training-crop-audit-v1", "method": "exact integral-image enumeration, not Monte Carlo",
        "predictions_or_validation_outcomes_used": False, "per_case_original_crop_hit": {str(k):v for k,v in probabilities.items()},
        "expectations": rows, "observed_completed_outer_runs_at_audit": actual,
        "known_patch_is_not_negative_completeness": True,
        "limitations": ["case counts do not equal independent experiments", "geometry sampling still equal across available tasks",
                        "single region source retains about one third of joint-task patches", "data weighting is a tested hypothesis not an optimum proof"]}
    write_new_json(output/"sampling_audit.json",result)
    snapshot_producer(output,["scripts/report_rsi_refinement.py","src/tfree_detection/rsi_refinement.py"])
    seal(output,inputs)
    return result


def font_setup():
    # Reuse the project's existing reporting font; never install or change fonts.
    path=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not path.is_file(): raise FileNotFoundError("Existing CJK report font unavailable")
    font_manager.fontManager.addfont(str(path))
    name=font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams.update({"font.family": [name,"DejaVu Sans"], "axes.unicode_minus": False, "font.size": 11})


def text_page(title, paragraphs):
    fig = plt.figure(figsize=(11.69,8.27))
    fig.text(.07,.92,title,fontsize=21,color="#173c53",weight="bold")
    y = .83
    for paragraph in paragraphs:
        lines = []
        for source_line in paragraph.split("\n"):
            lines.extend(textwrap.wrap(source_line,width=56,break_long_words=True,break_on_hyphens=False) or [""])
        wrapped = "\n".join(lines)
        fig.text(.07,y,wrapped,fontsize=12,va="top",linespacing=1.65)
        y -= .037*len(lines)+.04
    if y < .02:
        raise ValueError("Report text page overflow; split paragraphs before rendering")
    return fig


def outer_tables(summary):
    data = summary["outer_comparisons"]
    nested = summary["nested_selected_comparison"]
    result = []
    for group,label in GROUPS.items():
        baseline = next(r for r in data if r["group"] == group and r["policy"] == "control" and r["step"] == 1200 and r["method"] == "student_expert")
        physics = next(r for r in data if r["group"] == group and r["policy"] == "control" and r["step"] == 1200 and r["method"] == "physics_budgeted")
        candidates = [r for r in data if r["group"] == group and r["method"] == "student_expert"]
        best = sorted(candidates,key=lambda r:(-r["mean"],r["step"],r["policy"]))[0]
        control_best = sorted([r for r in candidates if r["policy"]=="control"],key=lambda r:(-r["mean"],r["step"]))[0]
        same_step = next(r for r in data if r["group"] == group and r["policy"] == "control" and r["step"] == best["step"] and r["method"] == "student_expert")
        selected = next(r for r in nested if r["group"] == group and r["method"] == "nested_feedback")
        result.append({"group": group,"label": label,"physics": physics["mean"],"old_student_1200": baseline["mean"],
                       "descriptive_outer_best_student": best,"nested_selected": selected["mean"],
                       "descriptive_outer_best_control":control_best,
                       "gain_over_control_best_checkpoint":best["mean"]-control_best["mean"],
                       "control_at_descriptive_best_step": same_step["mean"],
                       "same_checkpoint_gain_of_descriptive_best": best["mean"]-same_step["mean"],
                       "descriptive_best_is_deployable_policy": False})
    return result


def learning_curves(summary):
    groups = ["guo-superscan-single-acquisition","guo_2021_zro2_twin_acquisition","sigma3-orientation-ii"]
    fig,axes = plt.subplots(1,3,figsize=(13.5,6.5),layout="constrained")
    data = summary["outer_comparisons"]
    for ax,group in zip(axes,groups):
        matrix = np.array([[next(r["mean"]*100 for r in data if r["group"]==group and r["policy"]==p and r["step"]==s and r["method"]=="student_expert") for s in (400,800,1200)] for p in POLICIES])
        ax.imshow(matrix,cmap="YlGnBu",vmin=0,vmax=max(60,float(matrix.max())),aspect="auto")
        for (i,j),v in np.ndenumerate(matrix):
            ax.text(j,i,f"{v:.1f}%",ha="center",va="center",color="white" if v>35 else "#183444",fontsize=12)
        ax.set_xticks(range(3),[400,800,1200]); ax.set_xlabel("训练步数")
        ax.set_yticks(range(len(POLICIES)),list(POLICY_LABELS.values()))
        ax.set_title(GROUPS[group],fontsize=15)
    fig.suptitle("相同预算下的学生覆盖率  三个种子均值\n开发队列逐项对照，不把事后最大值当作独立测试结论",fontsize=15)
    return fig


def sampling_figure(audit):
    rows = [r for r in audit["expectations"] if r["held_out_group"]=="sigma3-orientation-ii" and r["policy"] in ("control","source_sqrt","positive_crops")]
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),layout="constrained")
    labels=[POLICY_LABELS[r["policy"]] for r in rows]
    values=[r["expected_known_patch_fraction"]*100 for r in rows]
    axes[0].bar(labels,values,color=["#7b929f","#328da1","#196b56"])
    for i,v in enumerate(values): axes[0].text(i,v+1,f"{v:.1f}%",ha="center")
    axes[0].set_ylim(0,100); axes[0].set_ylabel("至少包含一个已有监督像素的裁块比例 / %")
    axes[0].set_title("更多训练步不等于更多有效标注裁块")
    sources=rows[0]["source_distribution"]
    display={"local_zeolite_reference_stretch_series":"单张区域图","guo_2021_zro2_twin_acquisition":"单张 ZrO2",
             "sigma3-orientation-i":"30 帧 ORI","guo_2021_mowte2_acquisition":"单张 MoWTe2","guo-superscan-single-acquisition":"100 帧 SuperScan"}
    axes[1].barh([display[s["group"]] for s in sources],[s["global_patch_probability"]*100 for s in sources],color="#658799")
    axes[1].set_xlabel("原采样的总裁块份额 / %"); axes[1].set_title("同任务内等来源  跨任务等几何")
    fig.suptitle("ORII 留出训练的精确采样审计  不使用验证结果",fontsize=15)
    return fig


def comparison_figure(table):
    fig,ax=plt.subplots(figsize=(12,6.5),layout="constrained")
    x=np.arange(len(table)); width=.25
    for index,(key,label,color) in enumerate([("physics","原物理 同预算","#6e8292"),("old_student_1200","原配置学生 1200 步","#d99456"),("nested_selected","内层选择后的反馈系统","#267e6b")]):
        values=[r[key]*100 for r in table]
        ax.bar(x+(index-1)*width,values,width,label=label,color=color)
        for a,b in zip(x+(index-1)*width,values): ax.text(a,b+.4,f"{b:.1f}",ha="center",fontsize=10)
    ax.set_xticks(x,[r["label"] for r in table]); ax.set_ylabel("已标注正例覆盖 / %")
    ax.set_ylim(0,max(35,max(r[k]*100 for r in table for k in ("physics","old_student_1200","nested_selected")))+10)
    ax.legend(frameon=False); ax.set_title("内层选策略  外层报结果\n点位是匹配召回，界面是宽带覆盖；二者不合并成总体准确率")
    return fig


def feedback_audit(analysis):
    values=defaultdict(list); embedding=defaultdict(list)
    for path in sorted((analysis/"run_metrics").glob("*.json")):
        if path.name.endswith(".seal.json"):
            continue
        report=verified_metrics(path)
        if report["fold"]["stage"]!="outer": continue
        for row in report["rows"]:
            if not row["main_comparison_eligible"]: continue
            methods=row["methods"]
            key=row["group"],row["policy"],row["step"],row["seed"]
            values[key].append(methods["student_reranked_physics_plus_rescue"]["positive_coverage"]-methods["hybrid_fixed_budget"]["positive_coverage"])
            embedding[key].append({"veto": row["embedding_context"]["novelty_veto"],
                                   "student_minus_physics": methods["student_expert"]["positive_coverage"]-methods["physics_budgeted"]["positive_coverage"],
                                   "distance": row["embedding_context"]["nearest_distance"]})
    rows=[]
    for key,differences in values.items():
        items=embedding[key]
        rows.append({"group":key[0],"policy":key[1],"step":key[2],"seed":key[3],
                     "rerank_minus_fixed_hybrid_macro":float(np.mean(differences)),
                     "embedding_veto_fraction":float(np.mean([r["veto"] for r in items])),
                     "vetoed_student_delta_mean":float(np.mean([r["student_minus_physics"] for r in items if r["veto"]])) if any(r["veto"] for r in items) else None,
                     "not_vetoed_student_delta_mean":float(np.mean([r["student_minus_physics"] for r in items if not r["veto"]])) if any(not r["veto"] for r in items) else None})
    return rows


def score_behavior(analysis):
    grouped=defaultdict(list)
    for path in sorted((analysis/"run_metrics").glob("*--outer-*.json")):
        if path.name.endswith(".seal.json"): continue
        for row in verified_metrics(path)["rows"]:
            if row["main_comparison_eligible"]:
                grouped[row["group"],row["policy"],row["step"],row["seed"]].append(row["score_diagnostics"])
    seeds=defaultdict(list)
    for (*key,seed),rows in grouped.items():
        seeds[tuple(key)].append({"seed":seed,**{name:float(np.mean([r[name] for r in rows])) for name in rows[0]}})
    return [{"group":group,"policy":policy,"step":step,"by_seed":rows,
             "mean":{name:float(np.mean([r[name] for r in rows])) for name in rows[0] if name!="seed"}}
            for (group,policy,step),rows in sorted(seeds.items())]


def label_budget_bounds(manifest, protocol):
    groups=defaultdict(list)
    for entry in manifest["entries"]:
        if entry["action"]!="correct_geometry" or not entry["human_positive_pixels"]: continue
        if entry["geometry"]=="point":
            positive=len(entry["points_xy"]);budget=protocol.point_budget(entry["shape"])
        else:
            positive=entry["human_positive_pixels"];budget=protocol.area_budget(entry["shape"])
        groups[entry["group"]].append(min(1,budget/positive))
    return {group:{"mean_count_budget_upper_bound":float(np.mean(values)),"cases":len(values),
                   "diagnostic_only_not_routing_or_rescaled_accuracy":True} for group,values in groups.items()}


def point_border_audit(manifest, border=24):
    groups=defaultdict(list)
    for entry in manifest["entries"]:
        if entry["geometry"]!="point" or entry["action"]!="correct_geometry":continue
        points=np.asarray(entry["points_xy"],float)
        if not len(points):continue
        h,w=entry["shape"]
        groups[entry["group"]].extend(np.minimum.reduce([points[:,0],points[:,1],w-1-points[:,0],h-1-points[:,1]]).tolist())
    return {group:{"marked_points":len(values),"within_border":int(sum(x<border for x in values)),
                   "border_width_pixels":border,"fraction":float(np.mean(np.asarray(values)<border)),
                   "not_an_inference_feature":True} for group,values in groups.items()}


def directed_tables(summary):
    """Compare identical policy/checkpoints, not per-test best selections."""
    result=[]
    for target,source in (("sigma3-orientation-ii","ORI"),("sigma3-orientation-i","ORII")):
        for step in (400,800,1200):
            def metric(key):
                matches=[r for r in summary[key] if r["group"]==target and r["policy"]=="control"
                         and r["step"]==step and r["method"]=="student_expert"]
                if len(matches)!=1: raise ValueError("Directed comparison inventory mismatch")
                return matches[0]
            direct,mixed=metric("directed_same_material_comparisons"),metric("outer_comparisons")
            result.append({"training_source":source,"target_group":target,"step":step,
                "directed":direct,"mixed":mixed,"policy":"control",
                "directed_contains_zro2":False,"same_checkpoint_not_posthoc_best":True})
    return result


def route_audit(analysis):
    selectors={}
    for path in sorted((analysis/"selectors").glob("*.json")):
        value=read(path)
        selectors[value["outer_group_excluded"]]=value
    grouped=defaultdict(list)
    for row in read(analysis/"selected_outer_cases.json"):
        if row["main_comparison_eligible"]: grouped[row["group"]].append(row)
    result=[]
    for group,rows in grouped.items():
        result.append({"group":group,"label":GROUPS[group],"scopes":selectors[group]["scopes"],
            "actual_branch_case_seed_counts":dict(Counter(r["selected_branch"] for r in rows)),
            "case_count":len({r["case_number"] for r in rows}),"seeds":sorted({r["seed"] for r in rows}),
            "embedding_vetoed_case_seed_count":sum(bool(r["selected_configuration"] and r["selected_configuration"]["embedding_veto"]
                and r["embedding_context"]["novelty_veto"]) for r in rows)})
    return result


def spatial_control_reference(analysis_protocol, training_protocol):
    directory=ROOT/"outputs/rsi_cycle_001/capability_eval_v1"
    receipt=read(directory/"result_seal.json")
    names=("capabilities.json","evaluation_protocol.json")
    for name in names:
        if sha256_file(directory/name)!=receipt["files"][name]: raise ValueError("Original spatial controls changed")
    protocol=read(directory/"evaluation_protocol.json")
    if (protocol["protocols"]["main"]!=analysis_protocol["evaluation"] or
            protocol["source_input_files_sha256"][training_protocol["learning_dataset"]+"/manifest.json"]!=training_protocol["learning_manifest_sha256"]):
        raise ValueError("Spatial controls use different budget or dataset")
    values=read(directory/"capabilities.json")["spatial_controls_across_draws"]
    return {"source_group_controls":[r for r in values if r["dimension"]=="source_group"],
            "variation_is_spatial_draw_not_training_seed":True,
            "input_files_sha256":{str((directory/name).relative_to(ROOT)):sha256_file(directory/name) for name in names}}


def qualitative_examples(analysis, table, dataset):
    """Fixed examples; best-by-outer config labelled descriptive, never routed."""
    manifest,arrays=load_learning_dataset(ROOT,dataset)
    entries={e["case_number"]:e for e in manifest["entries"]}
    frozen=contained_path(ROOT,manifest["frozen_bundle"])
    records={r["case_number"]:r for r in map(json.loads,(frozen/"cases.jsonl").read_text().splitlines())}
    fig,axes=plt.subplots(3,4,figsize=(14,11),layout="constrained")
    details=[]
    for row_number,number in enumerate((63,103,135)):
        entry,plane=entries[number],arrays[number]
        best=next(r["descriptive_outer_best_student"] for r in table if r["group"]==entry["group"])
        matches=[]
        for path in sorted((analysis/"run_metrics").glob(f"{best['policy']}--seed-23--outer-*.json")):
            if path.name.endswith(".seal.json"):continue
            matches.extend(r for r in verified_metrics(path)["rows"] if r["case_number"]==number and r["step"]==best["step"])
        if len(matches)!=1: raise ValueError("Qualitative score inventory mismatch")
        location=contained_path(ROOT,matches[0]["score_path"])
        if sha256_file(location)!=read(location.with_suffix(".json"))["score_sha256"]: raise ValueError("Qualitative score changed")
        with np.load(location,allow_pickle=False) as archive: score=archive["score"]
        predictions=feedback_predictions(plane["image"],plane["teacher"],records[number]["prediction"]["candidates"],score,entry["geometry"])
        metrics=evaluate_case(entry,plane,predictions)
        names=(None,"physics_budgeted","student_expert","student_reranked_physics_plus_rescue")
        titles=("专家已标正例","原物理 同预算","学生候选 仅研究","学生反馈候选 仅研究")
        for ax,name,title in zip(axes[row_number],names,titles):
            ax.imshow(plane["image"],cmap="gray",vmin=0,vmax=1)
            if entry["geometry"]=="point":
                points=np.asarray(entry["points_xy"])
                ax.scatter(points[:,0],points[:,1],s=35,facecolors="none",edgecolors="#13e3ae",linewidths=1)
            else:
                ax.contour(plane["human"].astype(float),levels=[.5],colors=["#13e3ae"],linewidths=.8)
            suffix=""
            if name is not None:
                pred=predictions[name]
                if pred.points_xy is not None:
                    points=pred.points_xy
                    if len(points):ax.scatter(points[:,0],points[:,1],s=10,c="#ffc857",marker="+")
                else:
                    overlay=np.zeros((*pred.mask.shape,4));overlay[pred.mask]=[1,.66,.12,.5];ax.imshow(overlay)
                suffix=f"\n已标正例覆盖 {metrics['methods'][name]['positive_coverage']*100:.1f}%"
            ax.set_title(title+suffix,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
        axes[row_number,0].set_ylabel(f"{GROUPS[entry['group']]}\n病例 {number}",fontsize=11)
        details.append({"case_number":number,"policy":best["policy"],"step":best["step"],"seed":23,
                        "posthoc_example_not_release_selection":True,"score_sha256":sha256_file(location)})
    fig.suptitle("固定病例举例  青绿为已有专家标注  黄色为候选\n采用事后较好配置说明差异，不是部署策略；未标区域不能直接判为误报",fontsize=14)
    return fig,details


def report(analysis, audit_dir, output):
    analysis, analysis_protocol, _, protocol, dataset=load_analysis(analysis)
    audit_dir,output=contained_path(ROOT,audit_dir),contained_path(ROOT,output)
    verify_seal(analysis); verify_seal(audit_dir)
    summary,audit=read(analysis/"summary.json"),read(audit_dir/"sampling_audit.json")
    if summary["registered_models"]!=len(protocol["policies"])*len(protocol["seeds"])*len(protocol["folds"]):
        raise ValueError("Incomplete comparison")
    if output.exists(): raise FileExistsError("Report output must be new")
    output.mkdir(parents=True)
    table=outer_tables(summary); feedback=feedback_audit(analysis)
    directed=directed_tables(summary); routes=route_audit(analysis)
    behavior=score_behavior(analysis)
    manifest=read(dataset/"manifest.json")
    bounds=label_budget_bounds(manifest,EvaluationProtocol(**analysis_protocol["evaluation"]))
    point_bias=point_border_audit(manifest)
    controls=spatial_control_reference(analysis_protocol,protocol)
    write_new_json(output/"report_data.json",{"main_table":table,"feedback_diagnostics":feedback,
                                             "directed_fixed_control_comparison":directed,"route_audit":routes,
                                             "score_behavior":behavior,"label_budget_bounds":bounds,"point_border_audit":point_bias,
                                             "spatial_controls":controls,"no_production_promotion":True})
    lines=["# PhyEvo Mic 训练改进与能力边界结果","",
        "本轮完成六组训练策略、三个检查点和三个随机种子的来源隔离对照。结果区分事后观察到的潜力，与内层证据能支持的实际路由。现有标注是正例和未知区域，百分数不是总体正确率。","",
        "## 同预算对照","",
        "| 场景 | 原物理 | 原学生 1200 步 | 学生事后最佳 仅描述 | 内层选择后的系统 |","| --- | --- | --- | --- | --- |"]
    for row in table:
        best=row["descriptive_outer_best_student"]
        lines.append(f"| {row['label']} | {row['physics']*100:.2f}% | {row['old_student_1200']*100:.2f}% | {best['mean']*100:.2f}%（{POLICY_LABELS[best['policy']]}，{best['step']} 步） | {row['nested_selected']*100:.2f}% |")
    lines += ["","事后最佳是读过外层结果后的描述上界，不是可直接部署的选择规则。内层选择列才使用排除了外层来源的配置选择过程；这批数据已用于开发，仍不是新的锁定外部测试。","",
        "界面标注是宽带，预算可能不允许覆盖整个宽带。按 min(1, 候选面积预算/标注面积) 逐病例求均值，ORI 与 ORII 的面积预算上界分别为 "
        f"{bounds['sigma3-orientation-i']['mean_count_budget_upper_bound']*100:.2f}% 和 {bounds['sigma3-orientation-ii']['mean_count_budget_upper_bound']*100:.2f}%；ZrO2 为 100%。这是读数背景，不把实测值重新包装为准确率。","",
        "## 应当怎样修正旧结论","",
        "旧版固定 1200 步的成绩不能代表学生模型的能力上限。采样、有效标注裁块和增强的影响随来源变化；某种改动在一个场景改善，也可能在另一个场景退化。训练损失降低和训练更久都不是自动晋升依据。","",
        "尤其不能仅凭旧版 ZrO2 的低覆盖就证明神经网络不能跨材料迁移。需要先控制训练流程与检查点，再讨论物理先验和模型各自的优势；新的覆盖增益也不能自动证明完整误报风险更低。","",
        "与其宣称物理方法全面优于学生，或学生可以取代物理，更准确的结论是二者的收益依赖任务与来源。相同材料的定向迁移和跨材料迁移分别评价；保留原物理优势，并在同预算、来源隔离的证据支持下使用学生补充。","",
        "## 模型反馈与能力边界","",
        "学生重新排序物理候选并补充剩余预算，原物理测量不被改写。重新排序相对固定组合的逐组增益保存在 report_data.json，不能把简单多报区域算作改善。若物理候选本来少于预留额度，重排可能与固定组合完全相同。","",
        "训练特征可以返回相似的已审核参考，但相似度不是材料身份、缺陷真实性或校准 OOD 概率。本轮是瓦片特征加权平均后的整图特征，不直接承担局部缺陷定位。参考距离门控是否有利同样由内层比较决定，不能因为实现了 embedding 就假定它增加可靠性。","",
        "点位留出后往往仅剩一个独立点位来源，无法满足两个来源的自动路由证据要求；MoWTe2 只有一张含七个明确点的图，ZrO2 也仅一张图。线和区域没有完成独立能力验证；未知标注不能支持完整精确率或无缺陷判断。","",
        "## 受控闭环","",
        "专家已有标注进入独立监督 → 来源隔离训练与内层选择 → 同预算能力比较与旧能力保留检查 → 可选候选反馈和参考检索 → 记录收益、退化与拒绝原因。证据不够或可选分支出错时保留原物理；没有要求专家重新标注，也没有自动生产晋升。","",
        "## 训练流程仍有的限制","",
        "六组是逐项累加对照，不是所有超参数的穷举。学习率与任务几何采样在主对照中固定，单张区域图仍占联合训练较大份额。来源帧数不等于独立实验数；当前结果不能证明已经找到全局最优训练方案。","",
        "## 纯铝来源定向迁移","",
        "以下均固定原配置和相同检查点，不按目标测试结果挑最优参数。纯 ORI→ORII 只用 30 帧 ORI，纯 ORII→ORI 只用 20 帧 ORII；两者都不含 ZrO2。多来源对照还包含其他任务和 ZrO2，不能把差异单独归因于材料数量。","",
        "| 训练与测试 | 步数 | 纯单来源训练 | 原多来源训练 |","| --- | --- | --- | --- |"]
    for row in directed:
        lines.append(f"| {row['training_source']} → {GROUPS[row['target_group']]} | {row['step']} | {row['directed']['mean']*100:.2f}% | {row['mixed']['mean']*100:.2f}% |")
    lines += ["","## 实际反馈分工","",
        "下表统计内层选好规则后，在外层实际执行的分支。计数单位是病例与种子的组合，不是独立材料数；空的选择表示当前证据不支持自动引入学生。","",
        "| 场景 | 实际分支及次数 | 特征距离否决次数 |","| --- | --- | --- |"]
    for row in routes:
        lines.append(f"| {row['label']} | {row['actual_branch_case_seed_counts']} | {row['embedding_vetoed_case_seed_count']} |")
    lines += ["","## 高分区域是否偏离关注对象","",
        "ZrO2 上，原配置后期的高分越来越集中在图像最暗部分。下表统计候选最高分的 10% 像素中，有多少落在原图最暗的 10% 像素内。最暗不必然等于伪影，这个无标签诊断不能当误报率，但可以解释排名变化与可视化的关系。","",
        "| ZrO2 配置 | 步数 | 高分候选落入最暗部分 | 全图分数大于 0.5 的比例 |","| --- | --- | --- | --- |"]
    zro_behavior=[r for r in behavior if r['group']=='guo_2021_zro2_twin_acquisition' and r['policy'] in ('control','positive_crops','weak_roi')]
    for row in zro_behavior:
        lines.append(f"| {POLICY_LABELS[row['policy']]} | {row['step']} | {row['mean']['fraction_top10_in_darkest_decile']*100:.2f}% | {row['mean']['fraction_above_0_5']*100:.2f}% |")
    lines += ["","固定病例图中，原物理在 ZrO2 上给出较连贯的细线，学生候选更分散。宽带覆盖提高并不等于细线定位更好。多数分数高于 0.5 也不表示多数像素有缺陷；这些分数未经概率校准。占比先验能降低高分面积，但不能仅凭这一点认定检测更准确。"]
    ss_group="guo-superscan-single-acquisition"
    ss_controls=next(r["methods"] for r in controls["source_group_controls"] if r["value"]==ss_group)
    ss_bias=point_bias[ss_group]
    ss_best=next(r["descriptive_outer_best_student"] for r in table if r["group"]==ss_group)
    ss_range=[r["macro_case_coverage"] for r in ss_best["by_seed"]]
    border_note=(f"SuperScan 的 {ss_bias['marked_points']} 个已修正点中，{ss_bias['within_border']} 个（{ss_bias['fraction']*100:.1f}%）距图边不足 24 像素。"
        f"同预算的均匀随机布点覆盖约 {ss_controls['control_uniform']['macro_case_positive_coverage_mean']*100:.2f}%，边缘 24 像素布点约 {ss_controls['control_border24']['macro_case_positive_coverage_mean']*100:.2f}%。"
        f"学生事后较好配置均值为 {ss_best['mean']*100:.2f}%，三个种子范围 {min(ss_range)*100:.2f}%–{max(ss_range)*100:.2f}%。")
    lines += ["","## 点位标注分布与种子波动","",border_note,"",
        "这些比较说明需要同时控制位置偏好和训练随机性。随机布点对照的三次变化是布点抽样，不是训练种子；99 帧仍来自一个采集来源。观察到高于简单位置基线的覆盖，不等于已经验证了任意材料上的稳定点位检测。"]
    failures=[r for r in table if r["nested_selected"]<r["physics"]-.01]
    lines += ["",("内层保留检查没有阻止所有外层退化，以下来源超过一个百分点下降："+
        "、".join(f"{r['label']}（{(r['nested_selected']-r['physics'])*100:+.2f} 个百分点）" for r in failures)+
        "。这些结果是能力边界反例，不应把对应研究规则自动用于生产。" if failures else
        "本次外层没有出现超过一个百分点的原物理覆盖退化，但这不构成新材料或完整误报风险上的保证。"),"",
        "完整定向对照见 directed_comparisons.json；逐模型指标、采样规则和输入哈希保留在对应分析与训练目录。"]
    lines += ["","## 区分停止时机和训练改动","",
        "若旧配置也允许从三个检查点中事后选最佳，应使用下表比较，不能把所有增益都归因于新采样。两列同样只是开发观察，不用于绕过内层选择。","",
        "| 场景 | 原配置事后最佳 | 新配置事后最佳 | 差值 个百分点 |","| --- | --- | --- | --- |"]
    for row in table:
        old,new=row["descriptive_outer_best_control"],row["descriptive_outer_best_student"]
        lines.append(f"| {row['label']} | {old['mean']*100:.2f}%（{old['step']} 步） | {new['mean']*100:.2f}%（{new['step']} 步） | {row['gain_over_control_best_checkpoint']*100:+.2f} |")
    (output/"results_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_new_json(output/"directed_comparisons.json",summary["directed_same_material_comparisons"])
    font_setup()
    figures=[("01_sampling_audit",sampling_figure(audit)),("02_training_policy_checkpoints",learning_curves(summary)),("03_nested_capability_comparison",comparison_figure(table))]
    examples,example_details=qualitative_examples(analysis,table,dataset)
    figures.append(("04_fixed_case_qualitative_comparison",examples))
    write_new_json(output/"qualitative_examples.json",example_details)
    diagnostic_pages=[]
    with PdfPages(output/"PhyEvo_Mic_training_refinement_results_zh.pdf") as pdf:
        fig=text_page("PhyEvo Mic 训练改进与能力边界",[
            "目标：检验采样、增强和弱监督是否限制了现有学生，并验证模型分数与特征能否反过来帮助原物理框架。",
            f"完成 {summary['registered_models']} 次来源隔离训练。六组策略分别保存 400、800、1200 步，三个种子。五个外层折、十个双来源排除内层折，另做两个同材料定向迁移实验。",
            "读数方式：点位是已标注点召回，界面是专家宽带覆盖。固定预算为每 512×512 像素 50 个点、区域面积 10%。不报告总体准确率或完整精确率。",
            "结论范围：固定最后一步不能代表模型上限；不同场景需要分开验证。事后最佳值是开发观察，不能取代内层选择。默认物理框架和已有标注均保留。"])
        pdf.savefig(fig);plt.close(fig)
        fig=text_page("结果摘要与选择边界",[
            "以下依次为原物理、旧学生 1200 步、内层选择后的系统。均为三个种子平均，种子波动不是独立材料的统计证据。",
            "\n".join(f"{r['label']}：{r['physics']*100:.2f}% → {r['old_student_1200']*100:.2f}% → {r['nested_selected']*100:.2f}%" for r in table),
            "点位来源不足时不自动接管；宽带覆盖不等于精确中心线定位。MoWTe2 与 ZrO2 各一张图不能证明整个材料类别的稳定表现。"])
        pdf.savefig(fig);plt.close(fig)
        fig=text_page("停止时机和训练改动分别贡献多少",[
            "下面同时给旧配置和新配置事后选择检查点的机会，避免把单纯提前停止的收益全部算作训练改进。数字依次为旧配置最佳、新配置最佳及差值。",
            "\n".join(f"{r['label']}：{r['descriptive_outer_best_control']['mean']*100:.2f}% → {r['descriptive_outer_best_student']['mean']*100:.2f}%（{r['gain_over_control_best_checkpoint']*100:+.2f} 个百分点）" for r in table),
            "这仍是同一开发队列上的事后对照，不是部署选参规则。真正的内层选择结果单列；没有把所有场景各自的测试最优模型拼成一个系统。"])
        pdf.savefig(fig);plt.close(fig)
        fig=text_page("点位覆盖中的位置偏好与随机性",[
            border_note,
            "简单边缘布点也有非零覆盖，因此不能把每一个命中都解释成模型学到了缺陷结构。改进学生高于这些位置基线，但三个训练种子仍有明显波动。",
            "99 帧来自同一次 SuperScan 采集，不能当作 99 个独立材料实验。只有两个点位来源时，留出一个后，剩余数据不足以支持两个来源的自动点位路由门槛。",
            "继续保留学生的旁路候选与相似案例检索；不使用留出图的标注分布作为推理特征，也不要求专家重新标注。"])
        diagnostic_pages.append(fig)
        zro_control=[r for r in zro_behavior if r['policy']=='control']
        fig=text_page("覆盖提升和定位质量不能混为一谈",[
            "固定 ZrO2 病例中，物理分支给出较连贯的细线；学生分支增加了宽带内覆盖，也产生更分散的关注区域。现有正例与未知标注不能把所有带外候选计为误报，也不能证明更高覆盖就是更精确的界面。",
            "原配置 ZrO2 高分前 10% 中落入最暗像素的比例："+"、".join(f"{r['step']} 步 {r['mean']['fraction_top10_in_darkest_decile']*100:.1f}%" for r in zro_control)+"。这说明排名随训练改变，不单是显示阈值问题。",
            "增加含标注裁块后，上述暗区偏好明显减少；占比先验也能改变全图高分面积，但排序与覆盖并不一定随之改善。分数不是经校准的缺陷概率。",
            f"相同面积预算下，ORI 和 ORII 的宽带覆盖预算上界约为 {bounds['sigma3-orientation-i']['mean_count_budget_upper_bound']*100:.1f}% 与 {bounds['sigma3-orientation-ii']['mean_count_budget_upper_bound']*100:.1f}%。应同时看候选形状、预算和覆盖，而非只看单个百分数。"])
        diagnostic_pages.append(fig)
        fig=text_page("纯铝来源迁移与混合训练分别报告",[
            "纯 ORI→ORII 只用 30 帧 ORI；纯 ORII→ORI 只用 20 帧 ORII。两者不包含 ZrO2、点位或区域任务。下面固定原配置，依次列纯单来源与多来源训练的覆盖。",
            "\n".join(f"{r['training_source']} → {GROUPS[r['target_group']]}，{r['step']} 步：{r['directed']['mean']*100:.2f}% / {r['mixed']['mean']*100:.2f}%" for r in directed),
            "同材料迁移也受成像条件与标注形式影响。多来源训练同时改变了任务组成、采样次数及来源，不能把两列差异全部解释成某一种材料的作用。"])
        diagnostic_pages.append(fig)
        for name,fig in figures:
            fig.savefig(output/(name+".png"),dpi=180)
            pdf.savefig(fig);plt.close(fig)
        for fig in diagnostic_pages:
            pdf.savefig(fig);plt.close(fig)
        fig=text_page("模型怎样反馈主框架",[
            "候选分数：在相同总预算内重新排序原物理候选，再用学生候选补漏。原始应变、位移、可靠性和几何结果独立保留。若物理支持少于预留份额，重排可能不改变结果。",
            "中间特征：只用训练侧参考库，给出相似的已审核案例。距离阈值仅是实验性门控，相似不是同材料，也不是正确预测的保证。",
            "受控选择：至少两个内层采集来源支持、平均改善至少一个百分点、任一来源退化不超过一个百分点。达不到时保留物理，不用事后最优测试值绕过规则。",
            "仍有边界：本轮未穷举学习率或任务份额；只有正例与未知标签，不能证明完整误报风险。闭环沉淀的是有证据的适用范围与退化记录，不是保证每轮全场景都提升。"])
        pdf.savefig(fig);plt.close(fig)
    snapshot_producer(output,["scripts/report_rsi_refinement.py"])
    seal(output,{**controls["input_files_sha256"],str((analysis/"result_seal.json").relative_to(ROOT)):sha256_file(analysis/"result_seal.json"),
                 str((audit_dir/"result_seal.json").relative_to(ROOT)):sha256_file(audit_dir/"result_seal.json")})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("sampling-audit");p.add_argument("--training",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("report");p.add_argument("--analysis",type=Path,required=True);p.add_argument("--audit",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="sampling-audit": sampling_audit(args.training,args.output)
    else: report(args.analysis,args.audit,args.output)


if __name__=="__main__": main()
