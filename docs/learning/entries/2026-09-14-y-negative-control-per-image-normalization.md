# Y 无稀土负对照与逐图亮度归一化

- Kind: algorithm
- First seen: 2026-09-14
- Status: applied
- Confidence: 2
- Related change: `src/tfree_detection/fau_stack_analysis.py`、`tests/test_fau_stack_analysis.py`
- Last review: 2026-09-14

## Why this appeared

Y 被进一步明确为不含稀土的标准结构。Y、RA、H2 的视野、图像尺寸、曝光和局部背景可能
不同；若直接用跨图绝对亮度阈值，可能把整张图的成像差异误当化学差异，也可能因 Y 中某个
伪亮尾点而否决 RA/H2 的真实局部异常。

## Mental model

先在每个平面内对 DoG 响应做 median/MAD 稳健标准化，再在每张图内以附近匹配的 Si 柱
建立局部亮度基线。热点与局部 `>=5σ` 极亮 `bright` 档的交集拥有候选所有权。Y 的同类
交集永远解释为负对照假阳性；Y 热点亮度尾部为 RA/H2 计算经验尾概率，但只审计、不跨图
硬否决。原始 DoG 对比度、图内全局 z 分数和经验尾概率全部保留，供人工回看。

## Minimal reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
  -p no:cacheprovider -q \
  tests/test_fau_stack_analysis.py::test_per_image_normalization_keeps_moving_candidate_below_control_absolute_tail
```

## Prediction

合成边界中，Y 的局部 z 尾点为 `5.4`，H2 的图内候选为 `5.2`。用户预测应先分别归一化，
H2 候选仍应保留；绝对亮度可以参考，但不应成为跨图硬门。

## Observation

测试中 Y 被固定解释为 0 个稀土候选，H2 的 `5.2` 候选仍保留；审计明确记录
`cross_image_absolute_intensity_gate=false`。真实数据中 Y 有 5,816 个热点内审计峰，最大
局部 z 为 `4.045`，交集阳性为 0；H2 的 3 个候选均保留，其 Y 负对照经验尾概率为
`1/5817`。

## Project integration map

```text
每张图独立稳健归一化 -> 本图附近 Si 局部基线 -> 热点 × 极亮候选
                                      |
Y 无稀土负对照尾部 ------------------+-> 经验概率审计（不否决）
原始 DoG / 图内全局 z ----------------+-> CSV/NPZ 人工旁证
```

## Theory and boundaries

逐图归一化使候选判定对整体增益和背景偏移更稳健，但也意味着不能把归一化后数值解释成跨图
绝对散射强度。Y 经验尾概率衡量“相对负对照尾部有多极端”，不是化学后验概率。要恢复绝对
强度比较，仍需同一探测器设置、剂量、厚度、暗场/增益校正和采集元数据。

## Alternatives

未采用“必须比 Y 的最亮点更亮”，因为不同视野下该规则不具备强度可比性。也未丢弃原始
数值；它们保留作审计，只是不拥有最终候选去留权。

## Failure modes

- 每张图内部若存在强烈厚度梯度，局部 Si 基线仍可能偏移；
- 扫描伪影同时制造热点和亮峰时，交集门仍可能误报；
- 负对照样本若被污染或采集条件不同，经验尾概率不可当作正式显著性检验；
- 无 EDS/EELS 或匹配 multislice 时，极亮候选仍不是元素确证。

## User contribution

用户明确选择“逐张图归一化为主，绝对值只作参考”，并预测合成边界中 H2 候选应保留。
这一判断决定了候选所有权、Y 负对照语义和自动化回归测试。

## Validation

- 聚焦边界测试通过：H2 候选未被更高的 Y 尾点跨图否决。
- 真实完整 z-stack：Y `0`、RA `0`、H2 `3` 个最终候选。
- 最终输出：`fau_stack_analysis_101_v7_Y_reference_final`。
- FAU、原子位点清点与 Y 参考形变相关回归测试：`18 passed`。

## Retrieval questions

1. 为什么 Y 中一个更亮的点不能直接否决 H2 的局部异常点？
2. Y 负对照经验尾概率为什么不是“这个点是稀土的概率”？

## Learning debt

需要统一剂量/厚度/探测器校正的采集元数据，以及少量 EDS/EELS 对照，才能判断绝对散射
强度是否值得升级为独立证据教师。

## Review history

- 2026-09-14：用户完成归一化策略检查点；合成边界和真实三组 z-stack 均通过，状态设为
  applied、置信度 2。
