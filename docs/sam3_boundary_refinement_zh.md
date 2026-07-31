# SAM 3 物理约束边界细化

> 实施日期：2026-07-29
> 当前状态：候选级 v2 协议、逐候选门控/回退、区域与成对域界面适配、下游结果接入和
> 回归测试已完成；真实 SAM 3 权重推理尚未在本机执行。

## 一句话结论

SAM 3 现在不是“第六个缺陷检测器”，而是一个不受信任的几何提案器：

```text
物理分支决定“哪里存在候选”
             ↓
从物理核心/正常外侧生成 point + box + coarse-mask prompts
             ↓
隔离的 SAM 3 worker 返回“候选 × prompt 扰动”的独立 proposals
             ↓
主环境逐候选做共识、物理门控和审计
             ↓
该候选通过：采用 refined geometry
该候选失败：只把该候选回退到自己的 coarse component
             ↓
最后组合整图，不让一个失败候选抹掉已通过的相邻候选
```

`DetectionResult.mask` 始终保留物理候选。组合后的几何单独放在
`DetectionResult.final_geometry_mask` 和 `boundary_refinement` 中，因而不会静默改变
候选的科学语义。整图可能出现 `accepted`、`partially_accepted` 或
`rejected_fallback`；后两者不能简单理解为“整图都没用 SAM”。

## 用户选择的策略 A

本次确定的硬规则是：

> SAM 可以调整边界，但不能删除超过 5% 的高置信物理核心。

代码中对应：

```text
seed_core_recall =
  pixels(refined_mask ∩ high_confidence_physics_core)
  / pixels(high_confidence_physics_core)

某候选 seed_core_recall < 0.95
  → candidate.accepted = false
  → proposal 继续保存用于诊断
  → candidate.final_mask = candidate.coarse_mask
  → 其他已通过候选不受影响
```

该规则不能在启用 SAM 时通过配置削弱；若把
`minimum_seed_core_recall` 调到 0.95 以下，或设置
`stage2.sam3.fallback_to_coarse=false`，配置校验都会直接失败。

## 模块与所有权

```text
主 Python 3.10 环境
├─ task.run
│  └─ coarse mask + score/evidence maps
├─ stage2/sam3_refine.py
│  ├─ 高置信核心和正常外侧参考
│  ├─ 物理 prompts
│  ├─ sam_request.json + 校验数组
│  ├─ 逐候选 proposal 共识
│  ├─ Policy A、泄漏、边缘证据、稳定性和拓扑 guardrails
│  ├─ region / paired-domain interface geometry adapter
│  └─ candidate-local accept / deterministic fallback / composition
└─ reporting.py
   └─ coarse、proposal、final、uncertainty 和六面板

独立 Python 3.12 环境
└─ scripts/run_sam3_boundary_worker.py
   ├─ 读取并校验 request
   ├─ model.predict_inst(points, box, mask_input)
   └─ 每个 prompt 单独写 proposal_candidate_*_ensemble_*.npy
```

这样分离是因为官方 SAM 3 当前要求 Python 3.12、PyTorch 2.7+，与主平台的 Python
3.10 科学环境不应强行混装。worker 不知道缺陷类别、Gold 标签和接受阈值，也不能决定
最终结果。worker 也不再把同一 ensemble 中的多个候选做 OR 合并；这项所有权属于主
框架。

## 物理 prompts

每个 coarse 连通候选默认生成 8 组确定性扰动：

- 3 个 positive points：来自该候选内部的高分、高置信核心，并做空间分散；
- 8 个 negative points：来自候选外侧、低物理异常分数的正常参考环带；
- 1 个 expanded box：默认外扩 24 px；若 GPA 提供直接晶格长度，则按 2 个晶格单元
  换算；
- 1 个 coarse-mask prompt：转成 SAM 交互头要求的低分辨率 mask logits。

不使用 `"amorphous"`、`"defect"` 一类自然语言提示。这样 SAM 不需要理解“非晶”的材料
学含义，只需要回答“在这些前景、背景和粗几何约束下，边界可能在哪里”。

### 两相/孪晶界不是填充区域

`phase_boundary` 不把粗界面带直接交给 SAM 填充。每个界面候选、每次扰动会生成两组
prompt：

```text
domain_a prompt ─┐
                 ├─ SAM region masks ─ shared boundary ─ interface band
domain_b prompt ─┘
```

主环境从 `domain_label` 和 `boundary_validity` 生成两个域的 points、box 和 mask
prompt；拿到两侧 proposal 后，重新计算 shared seam，再按
`interface_half_width_pixels` 形成界面带。缺一侧 proposal 时，只让该界面候选回退。

## Proposal 共识

对每个候选，8 个 proposal 的像素包含频率记为 \(P_c(x)\)：

| 输出 | 默认定义 | 用途 |
|---|---:|---|
| stable core | \(P(x)\ge 0.80\) | 对 prompt 扰动稳定的几何 |
| refined proposal | \(P(x)\ge 0.50\) | 待物理门控的候选边界 |
| possible extent | \(P(x)\ge 0.20\) | 可能外延，不当作确定结果 |
| boundary uncertainty | \(0.20\le P(x)<0.80\) | 专家优先核对的边界带 |

不确定性不是概率校准结果；它只是同图 prompt ensemble 的几何不稳定度。

## 接受门

当前已经可执行的门如下：

| 门 | 默认阈值 | 失败行为 |
|---|---:|---|
| 高置信物理核心召回 | `>= 0.95` | Policy A 回退 |
| negative reference 泄漏 | `<= 0.05` | 回退 |
| proposal/coarse 面积比 | `0.50–2.00` | 回退 |
| proposal 内外物理分数差 | `>= 0` | 回退 |
| prompt 稳定性 mean IoU | `>= 0.50` | 回退 |
| 边界处物理 score 梯度对齐 | `>= 0.05`；无可用梯度时记 `null` | 回退 |
| 单候选拓扑发生无依据改变 | 默认不允许 | 该候选回退 |
| proposal 碰到其他 coarse 候选 | 默认不允许 | 该候选回退 |
| 两个已通过 proposal 在组合时重叠 | 默认不允许 | 涉及的候选分别回退 |

当前 `boundary_evidence_alignment` 是 score-map 梯度的无 Gold 保护门，不等于材料学
法向验证。尚未假装完成的是任务专属的边界法向校准，例如：

- 非晶边界与 GPA coherence/strain 法向的一致性；
- 两相界面与 phase/domain transition ridge 的一致性；
- 不同材料和像素尺度下的阈值 Gold calibration。

这些需要独立边界 Gold，不能根据当前参考图反向调阈值。

## 文件协议与安全性

交换目录默认是新输出目录下的 `sam3_exchange/`：

```text
sam3_exchange/
├─ sam_request.json
├─ image.npy
├─ coarse_mask.npy
├─ score_map.npy
├─ physics_core.npy
├─ negative_reference.npy
├─ component_labels.npy
├─ domain_labels.npy                 # paired-domain 界面任务才有
├─ valid_mask.npy                    # paired-domain 界面任务才有
├─ proposal_candidate_0000_ensemble_000.npy ...
├─ sam_response.json
├─ sam3_worker_stdout.log
└─ sam3_worker_stderr.log
```

v2 request 带协议版本、request ID、geometry kind、candidate ID、prompt role、ensemble
index、shape、dtype 和 SHA-256。response 必须完整覆盖 request 中每个
`(candidate_id, prompt_role, ensemble_index)`，不能再用一张整图 OR mask 冒充多个
候选。response 还必须匹配 request ID；
proposal 还会再次检查相对路径、路径穿越、shape、有限值和文件 SHA-256。所有协议文件
使用排他创建，不覆盖旧运行。

## 三种运行方式

### 1. 不安装 SAM，先验证物理提示与协议

```bash
PYTHONPATH=src python -m tfree_detection detect \
  --input reference_scale_radial_wiener.png \
  --config configs/amorphous_sam3_prepare.yaml \
  --output outputs/reference_amorphous_sam3_candidate_local_v5
```

此模式状态为 `proposal_pending`，第 3–5 个 SAM 面板为空，最终几何按 A 回退到 coarse。
它验证主框架，不代表 SAM 已经运行。

界面 paired-domain 协议可用：

```bash
PYTHONPATH=src python -m tfree_detection detect \
  --input datasets/public/guo_2021_atomic_defects/ZrO2_monoclinic_twin_boundary.tif \
  --config configs/phase_boundary_sam3_prepare.yaml \
  --output outputs/zro2_phase_boundary_sam3_paired_prepare_v3
```

### 2. 建立隔离 SAM 3 环境并一键运行

环境说明见
[`environments/sam3/README.md`](../environments/sam3/README.md)。配置完成后：

```bash
PYTHONPATH=src python -m tfree_detection detect \
  --input reference_scale_radial_wiener.png \
  --config configs/amorphous_sam3.yaml \
  --output outputs/reference_amorphous_sam3_live_v1
```

若环境、checkpoint、response 或 worker 任何环节失败，运行仍会输出可审计失败原因和
coarse fallback，不会把空 mask 当成功。

### 3. 外部 worker 后消费已有 response

将 `inference_mode` 设为 `response_only`，并把 `exchange_directory` 指向原 request
目录。主框架会重新计算物理候选并校验 request ID；图像、coarse mask、score、prompt
策略或模型设置任一变化都会拒绝旧 response。

## 输出怎么读

新输出包括：

- `sam3_coarse_mask.png`：物理候选；
- `sam3_refined_proposal.png`：即使失败也保留的 SAM 共识；
- `sam3_final_geometry.png`：通过时为 refined，失败时为 coarse；
- `sam3_stable_core.png`：高共识部分；
- `sam3_possible_extent.png`：低阈值外延；
- `sam3_boundary_uncertainty.png`：prompt 不稳定边界；
- `sam3_refinement_summary.json`：全部阈值、指标、拒绝原因和 provenance；
- `sam3_diagnostics.png`：六面板总览。
- `final_geometry_mask.png/.npy`：主输出层的组合几何；`mask.*` 继续保留物理粗候选；
- `candidate_table.jsonl`：每个候选新增 `geometry_source`、
  `geometry_bbox_xyxy`、`geometry_area_pixels` 和
  `boundary_refinement_status`；
- expert review overlay：使用该候选的 refined-or-local-fallback geometry。
- live worker summary：记录 worker/subprocess runtime、进程 peak RSS 和 peak device
  memory；prepare-only 时这些字段为 `null`；
- assisted/blinded review response 可用 `evaluation.expert_correction_metrics` 汇总几何修正
  率与标注时间；没有真实专家响应时状态明确为 `not_measured_during_inference`。

参考分子筛 v2 prepare-only 实测生成了 3 个物理候选、24 组独立 prompt，请求 ID 为
`780ae58734669689749d94ce`。采用 HR-STEM 尺度配置后，ZrO₂ 界面协议生成 1 个粗界面
候选、16 组 domain-A/domain-B prompts，请求 ID 为
`3445dd752e7d8b0bbc6186d7`。因为本机尚无
隔离 SAM 3 环境和授权 checkpoint，状态正确地
保持为 `proposal_pending`，最终面积仍是 coarse 的 39,411 px。当前实现图见
[`sam3_diagnostics.png`](../outputs/reference_amorphous_sam3_candidate_local_v5/sam3_diagnostics.png)。
ZrO₂ paired-domain 提示图见
[`sam3_diagnostics.png`](../outputs/zro2_phase_boundary_sam3_paired_prepare_v3/sam3_diagnostics.png)。

逐候选回退的数字孪生验证图见
[`candidate_local_fallback_validation.png`](../outputs/sam3_candidate_local_fallback_validation_v5/candidate_local_fallback_validation.png)。
其中候选 1 通过、候选 2 的核心召回为 0.940 而局部回退；整体 Boundary F1 从
0.584 提升到 1.000，ASSD 从 1.677 px 降至 0，HD95 从 5.831 px 降至 0。该图验证
组合与门控契约，不是 live SAM3 精度声明。

有 Gold mask 时可直接比较 coarse 与 final：

```bash
PYTHONPATH=src python -m tfree_detection evaluate-boundary-refinement \
  --coarse <coarse-mask.png> \
  --final <final-geometry-mask.png> \
  --truth <gold-mask.png> \
  --tolerance-pixels 2 \
  --output <new-boundary-report.json>
```

## 当前验证与真实缺口

已验证：

- Policy A 的 94% 核心召回 proposal 必须拒绝；
- 一个候选失败时只回退自身，不抹掉已通过相邻候选；
- worker/response 必须完整保存 candidate/role/ensemble，不允许整图合并；
- 通过 proposal、stable/possible/uncertainty 共识语义；
- 两相界面必须经 domain A/B 共享边界适配，不能请求一个填充区域；
- prompt 可重复且正负点落在正确物理区域；
- 禁止通过配置关闭 A 型回退；
- `final_geometry_mask`、候选表、复核图和 boundary F1/HD95/ASSD 接口；
- 非晶与 ZrO₂ 界面 prepare-only 的 CLI、报告和 v2 协议端到端。
- 完整自动回归 116 项通过。

未验证：

- 本机真实 SAM 3 checkpoint 推理；
- SAM 3.1 checkpoint 与当前 `predict_inst` 交互头的实测兼容性；
- 实验 STEM Gold 边界上的 boundary F1、HD95、ASSD；
- 专家修边时间是否下降；
- 非晶和两相界面之外的任务收益。

第一轮只允许 `amorphous_region` 和 `phase_boundary`。点缺陷、位错芯和极细相位线不能因为
SAM 能输出平滑轮廓就直接接入。

## 上游依据

- [Meta 官方 SAM 3 仓库](https://github.com/facebookresearch/sam3)
- [官方 SAM 1 交互任务 notebook](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_for_sam1_task_example.ipynb)
- 本项目固定的上游 revision：
  `6dbb02bd38288df755dfa1378000a861e65b84f6`
