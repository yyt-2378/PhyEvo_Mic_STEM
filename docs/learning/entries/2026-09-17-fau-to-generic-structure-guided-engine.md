# 从 FAU site-I 流程拆分通用投影结构位点引擎

- Kind: architecture
- First seen: 2026-09-17
- Status: applied
- Confidence: 2
- Related change: `src/tfree_detection/structure_guided/`、`tasks/structure_guided.py`
- Last review: 2026-09-17

## Why this appeared

FAU v20 已能在 Y/RA/H2 的 `[101]` 投影中比较“两 O 模板”和“两 O + 中心成分”，但源码
同时包含 CIF、FAU 位点、样品命名、配准、GPA、亮度和作图。若继续直接改旧脚本，会把
材料专属语义与可迁移判据绑死，也会失去当前可复现结果。

## Mental model

```text
材料适配器负责：结构来源、区轴、投影、配准、样品角色
通用核心负责：中心/端点探针、剖面模型、ΔBIC、可靠性
主框架负责：DetectionResult、EvidenceBundle、resolver、报告
```

因此冻结旧 FAU 源码，并让通用核心只依赖 `ProjectedSiteProbe` 和 `SampleRole`。FAU 只是
第一个适配器，不是核心数据模型。

## Prediction

用户明确预测并修正了参考样品策略：当一个预先声明“不含目标客体”的参考样品因为噪声
或成像伪影得到 `ΔBIC ≥ 6` 时，不应该保留假阳性审计池，并且解释候选数必须强制为 0。

## Observation

合成复现实验使用相同的两端点和明显中心峰：

- `target_possible` 角色得到 `ΔBIC ≥ 6` 并输出候选；
- `target_free_reference` 角色跳过并丢弃中心模型评分，公开的结构阳性、候选和异常证据图均
  为 0，摘要中不存在 false-positive audit 字段；
- 去掉中心峰后不输出候选；
- 把探针 family 从 `FAU_site_I` 改成无关材料名称，ΔBIC 和候选完全不变。

真实 v20 产物的只读对照进一步显示：RA 34 个、H2 72 个有效探针的逐点 ΔBIC 最大绝对
差分别为 `5.68×10⁻14`、`8.53×10⁻14`，候选数保持 `32/71`；Y 的 173 个旧有效探针
在新规则下不保留模型评分且候选为 0。

## Project integration map

```text
IdealSkeleton(optional node + two endpoints)
  -> projected_probes_from_skeleton
  -> analyze_projected_site_probes
  -> analysis_to_detection_result
  -> StructureGuidedSiteTask
  -> TrainingFreeDetectionFramework
  -> EvidenceBundle / resolver

frozen SkeletonFirstSiteEvidence
  -> adapters/fau.py
  -> the same generic analyzer
```

## Alternatives considered

- 直接重构 `fau_stack_analysis.py`：会破坏已确认的 v20 基线和结果可追溯性，因此拒绝。
- 让核心直接解析任意 CIF：投影区轴和实验配准属于可替换适配器，放入判据层会重新形成
  单体脚本，因此第一步采用已配准二维骨架契约。
- 把参考样品的高 ΔBIC 留作假阳性审计：与用户明确的数据治理要求冲突，因此没有实现。

## Failure modes

- 像素基矢或 `origin_xy` 错误会把探针铺到错误位置；当前任务不会假装自动修正未知配准。
- 两个端点并非真正物理邻居、中心不在线段内部或端点距离过短时，模型无效。
- 中心成分可能来自厚度、拖尾、重建或投影重叠；不能直接宣布元素身份。
- 参考角色必须事前声明；不能看完分数再把样品改成负对照。

## User contribution

用户决定了最关键的解释与数据保留不变量：无目标参考样品既不产生候选，也不保留假阳性
审计池。这一决定被编码为 `StructureGuidedAnalysis` 构造约束和专门的自动测试。

## Validation

- `tests/test_structure_guided.py`：6 passed；
- `tests/test_fau_stack_frozen_baseline.py`：2 passed；
- 本地真实 z-stack 兼容性：RA/H2 候选数 `32/71`，逐点 ΔBIC 达浮点精度等价；
- FAU 冻结源码 SHA-256 保持
  `a8c0b0c15e3aff7a9949801d95f8614f2ab8b7fc8adefefe453975434caa69f4`。

## Retrieval questions

1. 为什么任意材料通用化的核心输入应是“已配准二维投影探针”，而不是 FAU.cif？
2. 为什么 `target_free_reference` 即使内部拟合得到较高 ΔBIC，也不能输出候选或审计池？

## Learning debt

需要用真实 Y/RA/H2 的同一批 probe 坐标对冻结 v20 与通用适配器做逐位点数值对照；完成
等价性报告前，不替换旧 FAU CLI。
