# 边界感知候选与测量可靠性分离

- Kind: concept
- First seen: 2026-07-28
- Status: applied
- Confidence: 2
- Related change: `src/tfree_detection/diagnostic_v4.py`,
  `tests/test_diagnostic_v4.py`
- Last review: 2026-07-28

## Why this appeared

Dennler Class 2 的大面积 coherence-loss 真值经常延伸到图像边缘。旧的
`exclude_border_touching` 后处理会把整个连通候选删除，使一个本来局部正确的物理检测因
视野裁剪而变成零预测。

## Mental model

“候选是否存在”与“当前位置测量是否可靠”是两个不同变量：

- 候选掩膜表达观测到的物理违反，不能因为图像裁剪边界就整块消失；
- 可靠性图表达边缘缺少邻域支撑，允许从边缘低值渐变到内部高值；
- 下游校准、审阅排序和风险控制可以使用可靠性，但可靠性不能在没有校准证据时反向删除
  整个物理候选。

## Minimal reproduction

运行：

```bash
PYTHONPATH=src pytest -q \
  tests/test_diagnostic_v4.py::test_border_touching_coherence_band_is_retained_with_lower_edge_reliability
```

测试构造一个从图像上边缘向内部延伸的合成晶格模糊带。独立诊断中，旧触边删除策略留下
`0` 个候选像素，保留策略留下 `144` 个简化候选像素；新版完整分析的真实带覆盖率为
`88.3%`，边缘可靠性为 `0.25`、内部为 `1.00`。

## Prediction

问题是：当真实 coherence-loss 模糊带从图像上边缘贯穿晶格时，新版应该
A）保留整条候选但把靠边部分标为低可靠性，还是 B）把整个触边连通区删除。

用户预测：“我觉得是 A。”

## Observation

合成复现与用户预测一致：A 保留了真实区域及其连通性；B 会因为一个候选触碰视野边界而
把它全部抹掉。新版测试同时观察到边缘可靠性低于内部可靠性。

## Project integration map

```text
analyze_posthoc_v4
  ├─ _lattice_model → 晶格间距
  ├─ _edge_reliability → 独立 reliability map
  ├─ _coherence_candidates → 保留触边 region mask
  └─ compose_posthoc_v4 → 保守主结果与实验分支隔离
          ↓
run_dennler_posthoc_v4.py
  ├─ coherence_loss_region
  ├─ reliability
  └─ seal + post-hoc score
```

候选与可靠性的所有权在物理诊断模块；评分脚本只消费并封存输出，不重新修改候选。

## Theory and boundaries

该规则适用于可能被有限视野截断的 region、line 和 interface 候选。它不表示边缘永远是
真缺陷：边缘卷积、扫描畸变和 padding 仍可能制造假信号，所以必须降低可靠性并保留正常
对照。可靠性数值 `0.25 → 1.00` 当前是透明的工程先验，尚未经过 Gold calibration，不能
解释为真实概率。

## Alternatives

另一方案是对所有触边连通域做硬删除。它能减少边缘伪影，但会系统性漏掉贯穿视野的真实
非晶带、相界面和线缺陷，因此没有采用。更成熟的替代方案是学习条件化边缘风险，但必须有
独立 Gold calibration，当前证据不足。

## Failure modes

- 低对比但仍有周期性的正常晶区可能被 coherence-loss 过分割；
- padding 方式可能影响最外侧的 carrier amplitude；
- 可靠性若直接乘入二值阈值，会悄悄重新引入边缘截断；
- 局部晶域实验分支可能在正常图产生纹理聚类误报。

对应保护包括边缘连续性单元测试、正常晶格误报测试、合成两相几何测试，以及实验晶域不
进入保守主掩膜的失败隔离测试。

## User contribution

用户明确选择了策略 A。该判断被转化为
`tests/test_diagnostic_v4.py::test_border_touching_coherence_band_is_retained_with_lower_edge_reliability`
的核心不变量，并影响 `PosthocV4Config.edge_reliability_floor` 与独立 reliability map
的实现。

## Validation

```text
PYTHONPATH=src pytest -q tests/test_diagnostic_v4.py
4 passed

PYTHONPATH=src pytest -q
93 passed
```

Dennler v0.4 v2 预测为 61/61 成功；Class 2 IoU `0.6108`、Recall `0.9683`，正常图
coherence-loss 平均误报面积 `0.0001`。

## Retrieval questions

1. 为什么边缘低可靠性不应该直接乘进候选二值阈值？
2. 哪种新数据最能检验策略 A 不是 Dennler 揭盲后的特殊适配？

## Learning debt

需要独立 Gold calibration 把边缘距离、padding 支撑和采集伪影映射为可解释的风险值。

## Review history

- 2026-07-28：用户预测选择 A；合成复现与预测一致；规则已进入实现和回归测试。当前
  confidence 为 2，下一次重点是由用户解释候选与可靠性为何必须分离，以及它在哪些边缘
  伪影条件下会失效。

