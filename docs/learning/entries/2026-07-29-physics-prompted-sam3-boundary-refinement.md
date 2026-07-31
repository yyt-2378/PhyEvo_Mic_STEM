# 物理提示、提案所有权与 SAM 2/3 确定性回退

- Kind: architecture
- First seen: 2026-07-29
- Status: applied
- Confidence: 2
- Related change: `src/tfree_detection/stage2/sam3_refine.py`,
  `scripts/run_sam3_boundary_worker.py`,
  `scripts/run_sam2_boundary_worker.py`,
  `tests/test_sam3_refinement.py`
- Last review: 2026-07-30

## Why this appeared

平台已有物理候选，却缺少通用的像素边界细化。直接让 SAM 覆盖物理 mask 会把自然图像模型
的分割偏好误当成材料学证据，也会在 OOD 情况下静默删除真实核心或合并不同缺陷。

## Mental model

边界细化要区分三种所有权：

- physics teacher 拥有候选存在性、高置信核心和正常外侧参考；
- SAM 2/3 只拥有一个可拒绝的几何 proposal；
- acceptance gate 拥有最终几何选择，并在任何失败时确定性回退。

跨进程协议也是信任边界：Python 3.12 worker 不读取标签、不修改阈值、不写最终 mask，只
返回带 checksum 的 proposal。

## Minimal reproduction

运行：

```bash
PYTHONPATH=src pytest -q \
  tests/test_sam3_refinement.py::test_policy_a_rejects_proposal_that_deletes_more_than_five_percent_core
```

测试构造 100 px 的高置信物理核心，并让 8 个一致的 SAM proposal 都删除其中 6 px。

## Prediction

设计检查点是：当 SAM 的边界看起来更平滑，但只保留 94% 的高置信物理核心时，系统应该
A）拒绝 proposal、保留其诊断文件并回退 coarse mask，还是 B）相信 SAM 的视觉轮廓并
覆盖物理候选。

用户选择：“那就选 A。”

第二个架构检查点是：当同一张图有两个候选，候选 1 的边界细化通过，而候选 2
违反 95% 核心门时，应该让整图全部回退，还是只回退失败候选。

用户选择：“逐候选回退”。

## Observation

测试观察到 `seed_core_recall=0.94`，状态为 `rejected_fallback`，8 个 proposal 仍保留在
结果对象中，拒绝原因为 `seed_core_recall_below_0.950`，且
`final_mask` 与 `coarse_mask` 逐像素相同。

保留测试
`test_candidate_local_failure_does_not_erase_accepted_neighbor` 首次运行时在测试收集阶段
失败：项目不存在 `evaluate_candidate_sam3_proposals`，说明旧状态模型只能表达整图
接受/拒绝。加入候选级结果与组合器后，同一测试观察到候选 1 为 `accepted`、候选 2 为
`rejected_fallback`、整图为 `partially_accepted`；候选 1 使用 refined mask，候选 2
逐像素等于自己的 coarse component。

## Project integration map

```text
task.run
  └─ coarse mask + score map
       ↓
build_physics_prompt_plan
  ├─ physics core
  ├─ negative reference
  └─ point / box / coarse-mask ensembles
       ↓ file protocol
run_sam3_boundary_worker.py
  └─ candidate × role × ensemble 独立 proposal + sam_response.json
       ↓
evaluate_refinement_proposals
  ├─ region proposal，或 domain A/B → shared interface adapter
  └─ evaluate_candidate_sam3_proposals
       ├─ 每候选 inclusion consensus
       ├─ Policy A + leakage/area/contrast/stability/topology gates
       ├─ accepted refined OR candidate-local coarse fallback
       └─ collision-safe composition
       ↓
DetectionResult.final_geometry_mask
  ├─ candidate table geometry fields
  ├─ reporting primary geometry artifacts
  ├─ expert review overlay
  └─ boundary F1 / HD95 / ASSD comparison
```

## Theory and boundaries

核心召回门把“SAM 的视觉确信”置于物理证据之下，但 `0.95` 仍是显式工程先验，不是经过
Gold calibration 的概率。它保护 seed，不保证 proposal 新增的外侧像素正确，所以还要
同时检查 negative leakage、面积比、物理分数差、prompt 稳定性和候选合并。

`stable/possible` 是 prompt ensemble 的频率，不是测量不确定性的完整分解，也不能当作
校准置信区间。

## Alternatives

未采用的 B 方案是让 SAM 直接替换 coarse mask。它简单且轮廓通常更平滑，但 OOD
proposal 可以删除物理核心、跨越正常晶格或合并相邻候选，失败还很难从最终图反查。

另一个方案是把 SAM 的 mask 与 coarse mask 做硬交集。它不会外扩，却无法修复 coarse
边界欠覆盖，并会把 SAM 的缺失直接带进结果，因此也没有采用。

整图原子回退也未采用：它实现简单，但把彼此独立的候选绑定成一个失败域；一个局部
OOD proposal 会丢掉另一处已经通过物理门控的边界收益。

## Failure modes

- coarse seed 本身错误时，95% 核心门会保护错误；需要开放世界 resolver 和专家 Gold；
- 负点环带可能落在另一个未检出的缺陷上；
- mask logits 是从物理粗 mask 构造的先验，过强时 proposal 可能只复制 coarse；
- 自然图像预训练的边缘偏好可能追随扫描纹、对比反转或非物理亮度边；
- 当前任务专属 phase/GPA 法向一致性尚未做 Gold 校准；
- SAM 3.1 与交互 `predict_inst` checkpoint 的实际兼容性仍需本机验证；
- SAM2 单图后端已实测，但视频 predictor 尚未接入平台的时间序列缺陷跟踪。

## User contribution

用户明确选择 A，并把“不可删除超过 5% 的高置信物理核心”确定为架构硬约束；随后又
明确选择“逐候选回退”。两项选择被转成配置不可关闭的 Policy A、候选级结果对象、
`partially_accepted` 状态、独立 `final_geometry_mask`，以及 94% 召回和相邻候选不被
抹除的定向回归测试。

## Validation

```text
TMPDIR="$PWD/.codex-tmp/pytest" pytest -q \
  tests/test_sam3_refinement.py tests/test_evaluation.py
15 passed

TMPDIR="$PWD/.codex-tmp/pytest" pytest -q
116 passed
```

参考分子筛 v2 prepare-only 成功写出 3 × 8 = 24 组 region prompts；采用 HR-STEM
尺度配置后，ZrO₂ paired-domain prepare-only 成功写出 1 × 2 × 8 = 16 组 prompts。
数字孪生验证中，一个
候选通过、一个候选以 `seed_core_recall=0.940` 局部回退，组合结果的 Boundary F1
从 0.584 升至 1.000。真实 SAM 3 环境与授权 checkpoint 当前不存在，因此没有把这些
协议/合成验证写成真实模型效果。

2026-07-30 复用本机现有 CUDA SAM 环境接入 SAM2.1-large。实验分子筛的 3 个候选中
1 个通过、2 个按拓扑门逐候选回退；ZrO₂ 界面 proposal 因核心召回 `0.593` 和拓扑
改变回退。真实 worker 分别运行 `9.92 s` 与 `8.53 s`，GPU 峰值约 `1.31 GiB`。
这验证了“后端可替换但所有权不变”的架构预测；因无独立 Gold，未写成边界精度增益。
加入 SAM2 worker 定向测试后，完整回归为 120 项通过。

## Retrieval questions

1. 为什么“proposal 文件保留”和“最终 mask 回退”必须同时成立？
2. 即使 core recall 为 1.0，哪两类外侧错误仍会让 SAM proposal 被拒绝？

## Learning debt

需要在独立实验 Gold 边界上校准 seed recall、negative leakage 和 task-specific
phase/GPA boundary alignment；还要实测 SAM 3.1 交互 checkpoint，而不是从版本名推断兼容。

## Review history

- 2026-07-29：用户选择 A；94% 核心召回复现通过。
- 2026-07-29：用户选择“逐候选回退”；保留测试先暴露旧整图状态模型缺口，随后候选级
  v2 协议、paired-domain 界面适配、下游几何接入与合成边界增益验证通过。
- 2026-07-30：SAM2.1-large 在两个真实实验 STEM 实例上完成 CUDA 推理；Policy A
  对局部非晶保留 1 个细化并回退 2 个，对 ZrO₂ 界面安全回退。
