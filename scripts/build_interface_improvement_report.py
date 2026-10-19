#!/usr/bin/env python3
"""Build minimal scientific figures and a Chinese evidence report, no image synthesis."""
from __future__ import annotations

import json
import os
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/".codex-tmp/interface-report-mpl-20261015"
CACHE.resolve().relative_to(ROOT);CACHE.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"]=str(CACHE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np

from tfree_detection.expert_supervision import write_new_json
from tfree_detection.interface_geometry import ranked_support

OUT=ROOT/"outputs/interface_improvement_20261015_v1/release_v2"
BASE=OUT.parent
FONT=FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family":"Noto Sans CJK JP","font.size":12,"axes.unicode_minus":False})
NAMES={"sigma3-orientation-i":"Al ORI（30 帧）","sigma3-orientation-ii":"Al ORII（20 帧）","guo_2021_zro2_twin_acquisition":"ZrO2（1 张）"}


def read(p):return json.loads(p.read_text())


def style_text(ax,text,x,y,size=13,color="#18344a",weight="normal"):
    ax.text(x,y,text,transform=ax.transAxes,fontproperties=FONT,fontsize=size,color=color,
            fontweight=weight,va="top",linespacing=1.4)


def page(title,paragraphs):
    fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis("off")
    style_text(ax,title,0,1,24,weight="bold")
    y=.86
    for heading,body in paragraphs:
        style_text(ax,heading,0,y,16,color="#087e8b",weight="bold");y-=.055
        lines=[]
        for line in body.split("\n"):
            lines.extend(textwrap.wrap(line,width=65,break_long_words=True,break_on_hyphens=False) or [""])
        style_text(ax,"\n".join(lines),0,y,11);y-=.031*len(lines)+.04
    fig.subplots_adjust(left=.07,right=.97,top=.93,bottom=.06)
    return fig


def main():
    destination=OUT/"report_final_v2"
    destination.mkdir(exist_ok=False)
    targets=[destination/n for n in ["report.pdf","01_ORI_geometry_before_after.png","02_interface_regression.png","figure_manifest.json"]]
    if any(p.exists() for p in targets):raise FileExistsError("Never overwrite a report")
    result=read(OUT/"results.json");by={r["group"]:r for r in result["summary"]}
    fig,axes=plt.subplots(2,3,figsize=(13.6,9.2))
    for rr,n in enumerate([105,126]):
        with np.load(ROOT/"outputs/expert_cycle_001/learning_dataset_v1/planes"/f"case-{n:03d}.npz") as a:
            image,human=a["image"],a["human"]
        with np.load(BASE/"physical"/f"case-{n:03d}.npz") as p:
            old=p["legacy_mask"];new=ranked_support(p["score"],int(np.ceil(.1*image.size)),valid=p["valid"])
        old_cov=(old&human).sum()/human.sum();new_cov=(new&human).sum()/human.sum()
        for cc,ax in enumerate(axes[rr]):
            ax.imshow(image,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
            ax.contour(human,levels=[.5],colors=["#7ce8a0"],linewidths=1.2)
            if cc:ax.contour(old if cc==1 else new,levels=[.5],colors=["#ffba55" if cc==1 else "#36d5ff"],linewidths=1.3)
            ax.set_xticks([]);ax.set_yticks([])
            title=[f"病例 {n}：原图＋专家范围",f"旧物理：两侧边缘\n覆盖 {old_cov:.2%}",f"新物理：带内排序，预算 ≤10%\n覆盖 {new_cov:.2%}"][cc]
            ax.set_title(title,fontproperties=FONT,fontsize=13)
    fig.suptitle("ORI：错误发生在‘找带边缘’与‘找异常带内部’的目标解释",fontproperties=FONT,fontsize=19)
    fig.text(.5,.016,"绿线＝原始专家宽带范围；橙线＝旧物理；蓝线＝新预算候选。第二行保留低于 50% 的病例，不挑选全优样例。",ha="center",fontproperties=FONT,fontsize=11)
    fig.subplots_adjust(top=.89,bottom=.055,wspace=.04,hspace=.23)
    fig.savefig(targets[1],dpi=180)
    geometry_fig=fig
    order=["sigma3-orientation-i","guo_2021_zro2_twin_acquisition","sigma3-orientation-ii"]
    plot,ax=plt.subplots(figsize=(12.8,7.0));x=np.arange(3);width=.19
    for i,(key,name,color) in enumerate([
        ("legacy_physics","旧物理二值候选","#b8c5cf"),("legacy_student","历史纯学生","#a9ccd7"),
        ("legacy_hybrid","旧固定比例组合","#e8ae65"),("new_routed","新几何／学生分工","#087e8b")]):
        values=[100*by[g][key] for g in order];bars=ax.bar(x+(i-1.5)*width,values,width,label=name,color=color)
        ax.bar_label(bars,labels=[f"{v:.2f}" for v in values],padding=3,fontsize=11)
    ax.set_xticks(x,[NAMES[g] for g in order],fontproperties=FONT)
    ax.set_ylabel("专家已标范围覆盖率（%）",fontproperties=FONT);ax.set_ylim(0,87)
    ax.legend(prop=FONT,ncol=2,loc="upper left");ax.spines[["top","right"]].set_visible(False)
    ax.set_title("同样的 10% 面积上限；与历史组合比较，三个界面来源的均值均未退步",fontproperties=FONT,fontsize=17,pad=20)
    plot.text(.5,.025,"注意：ORII 新分工 46.49% 仍低于历史纯学生 47.62%；不能宣称所有分支全面提升。",ha="center",fontproperties=FONT,fontsize=12)
    plot.tight_layout(rect=[0,.06,1,1]);plot.savefig(targets[2],dpi=180)
    ori=by["sigma3-orientation-i"];zr=by["guo_2021_zro2_twin_acquisition"];orii=by["sigma3-orientation-ii"]
    with PdfPages(targets[0]) as pdf:
        pdf.savefig(page("PhyEvo-Mic｜ORI 界面改进与回归证据",[
            ("主要结果",f"在同样 10% 面积上限和原有效区约束下，ORI 从旧组合 {ori['legacy_hybrid']:.2%} 提升至 {ori['new_routed']:.2%}。30 帧中 28 帧达到 50%，最低帧约 48.05%。"),
            ("这是什么百分数", "覆盖率＝被预测候选覆盖的专家已标正区域像素数 ÷ 专家已标正区域像素数。先对帧取均值，再对三个种子取均值。未标区域保持未知，因此它不是准确率、IoU、精确率或缺陷鉴定正确率。"),
            ("公平比较与边界", "所有新界面候选最多选择 ceil(0.10×图像像素数)，不扩大预算。57.21% 的 Dennler 作者模型可作同预算迁移参考，但其原训练数据与本项目专家训练数据不同，不能把 58.34% 对 57.21% 写成严格同训练条件的胜出。"),
            ("交付状态", "可复现的研究分支与冻结绑定已建立；旧默认物理配置、历史最佳检查点、原始数据及专家标注均保留。没有自动晋升生产，也没有证明持续多轮正收益 RSI。")]))
        pdf.savefig(geometry_fig)
        pdf.savefig(page("为什么原来只有 4%：三个不同的问题",[
            ("① 目标几何解释反了", "ORI 的两个聚类对应 A｜B｜A：相似投影纹理包围中间异常带。旧算法只描出 A/B 的两条外沿，而专家给的是中间宽带。新规则检查同类纹理的两个连通侧翼、带的连贯性和长宽关系，定位带内；不读取 ORI 文件名。"),
            ("② 正样本稀疏导致整图高分", "历史 ORI 学生在全部像素上都大于 0.5。未知区域不是负样本，单靠正区域损失会鼓励全图抬高。新训练使用原 ROI 内的物理证据排序、远处低纹理变化区的弱参考、连续性和尺度一致性，不要求专家重标。"),
            ("③ 固定各占一半会保留错误候选", "新界面路径不再给物理分支预留一半像素。先验证几何与尺度一致性；有可靠物理支持时采用物理，物理证据不足时采用已准入且当前输出健康的学生。未通过的学生输出会被拒用。"),
            ("实现适用范围", "异常带判据为投影纹理的空间关系，不是化学相鉴定。长界面使用连通性、相对分析窗跨度和尺度佐证；这些阈值属于当前开发资料上的工程规则，不冒充校准置信概率。")]))
        pdf.savefig(page("学生训练：保留失败证据，不把重训等同于改进",[
            ("第一组：36 个匹配对照", "两种监督方式 × 三个种子 × 六个来源排除训练集合，每个保存 200、400、800 步。从头训练的物理排序学生虽然消除了训练侧整图高分，但 ORII 泛化会退步，因此没有用它替换历史优良模型。"),
            ("第二组：27 次保留能力的调整", "从原有最佳学生检查点出发，冻结特征提取网络，只调整几何输出层。训练原 ROI 内的物理排序，并用居中 logit 保留约束限制空间排序漂移。共保存 50、100、200、400 步；特征提取器与原检查点逐参数一致。"),
            ("检查点选择与拒用", "按两个来源同时排除的内层结果选择新步骤，外层标签不参与步骤选择。训练诊断整图高分的检查点不可准入；新输入发生整图高分时拒用该张学生输出，由有依据的物理分支回退。评价的是实际组合系统，不强迫学生覆盖物理已经擅长的全部来源。"),
            ("不能省略的限制", "历史最佳起点本身曾参考开发集成绩；几何规则也依据本次错误分析完善。因此新步骤的内层选择不等于整个方法从未见过开发集。ORII 内层分工由物理主导，不能区分各新步骤时按预先规则选最早步，不声称已找到 ORII 真正最优训练时长。")]))
        pdf.savefig(plot)
        pdf.savefig(page("回归、数据保护与复现入口",[
            ("回归结果", f"ZrO2 新分工 {zr['new_routed']:.2%}，主要来自连续物理分数的预算排序，不能归功于新学生。ORII 新分工 {orii['new_routed']:.2%}，高于旧组合 {orii['legacy_hybrid']:.2%}，但低于历史纯学生 {orii['legacy_student']:.2%}。SuperScan 与 MoWTe2 的模型及点位路径未替换。"),
            ("保护与验证", "63 次 GPU 训练、216 个检查点的产物分别保存；第一组失败和第一版不适当的准入汇总也保留。旧最佳封存资产做了 621 项文件哈希核对；相关测试 155 项通过。三个原始 TIFF/EMD 的运行候选掩膜与离线结果一致；学生分数最大差约 0.000514，不能称分数逐值一致。没有删除或覆盖原数据、ROI、旧权重。"),
            ("运行方式", "原物理自动几何：configs/phase_boundary_geometry_aware.yaml。含学生的新分工入口：scripts/run_interface_detection.py，显式传入 --source、--binding、--output。bindings 中每个文件绑定一个来源排除评估模型，不能根据待测文件名自动选择模型。"),
            ("证据目录", "outputs/interface_improvement_20261015_v1/release_v2/：results.json 包含逐病例、逐种子、分支选择和预算；两张 PNG 为必要结果图；raw_runtime_* 为从原始 TIFF/EMD 重新运行的回执。完整精确步骤见相邻训练 protocol.json 与源代码快照。")]))
    plt.close("all")
    write_new_json(targets[3],{"figures":[{"file":targets[1].name,"purpose":"ORI 两个真实病例：常见改善与未达到50%的边界情况"},
        {"file":targets[2].name,"purpose":"明确区分旧物理、旧纯学生、旧组合与新分工"}],
        "data":str((OUT/"results.json").relative_to(ROOT)),"synthetic_visual_content":False,
        "expert_geometry_modified":False,"pdf_pages":6})
    print(targets[0],flush=True)


if __name__=="__main__":main()
