# 畸变区门控的局部极亮稀土候选

- Kind: algorithm
- First seen: 2026-09-14
- Status: applied
- Confidence: 2
- Related change: `src/tfree_detection/fau_stack_analysis.py`、`tests/test_fau_stack_analysis.py`
- Last review: 2026-09-14

## Why this appeared

旧流程把三高斯中最亮总体直接叫作稀土，导致每个投影单胞出现十几个所谓稀土候选。
用户指出稀土应当是畸变/应变异常区域内非常异常明亮的少数点，并补充稀土可能只出现在
某一个重建层，不能把跨层稳定性设为硬门槛。

## Mental model

骨架畸变与元素亮度是两个独立证据源。先用焦面窗口内跨层稳定的峰拟合 Si/O 骨架，
从可靠局部仿射拟合圈出畸变或最大剪应变代理热点；再搜索所有切片，对热点邻域中的峰与
附近匹配 Si 柱做局部稳健亮度比较。只有热点区域与局部 `>=5σ` 极亮 `bright` 档的交集
才是高可信图像候选。跨层次数只进入审计分数，不决定去留。

## Minimal reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
  -p no:cacheprovider -q \
  tests/test_fau_stack_analysis.py::test_distortion_and_extreme_brightness_intersection_keeps_only_b \
  tests/test_fau_stack_analysis.py::test_stack_search_keeps_a_slice_local_bright_peak
```

## Prediction

合成反例包含 A、B、C：A 是热点外全图极亮点；B 是热点核心内、比当地 Si 柱高约
`5σ` 的点；C 在热点内但只高 `0.8σ`。用户预测“选 B”，并指出 B 即使只出现一层也
不应被拒绝。

## Observation

测试仅保留 B；A 被标记为 `rejected_outside_hotspot`，C 被标记为
`rejected_not_extreme_brightness`。独立切片测试确认只在一层出现的极亮峰仍进入全层峰集，
但不会被用作稳定骨架点。

## Project integration map

```text
all planes -> neutral dim/mid/bright peaks --------------------------+
focus-window repeated peaks -> CIF Si/O fit -> local deformation ---+-> intersection
                                                                    +-> candidate CSV/zoom
candidate removed -> fixed-pose framework rematch -> final strain
```

## Theory and boundaries

局部亮度 z 分数以附近匹配 Si 柱的中位数和 MAD 稳健尺度为参照，并使用全局尺度下限避免
局部方差过小。热点种子取可靠局部畸变或最大剪应变代理的第 90 百分位，并按投影单胞尺度
扩张。该交集提高特异性，但会漏掉不产生可见二维骨架畸变的稀土；“候选附近畸变更高”
也是筛选定义的结果，不能作为独立因果证据。

## Alternatives

没有继续使用最亮三高斯总体，因为它会把正常的高投影重数 Si/O 柱当成稀土。没有要求
跨层重复，因为层轴可能代表物理深度；也没有使用全图 `5σ` 作硬门，因为厚度和背景变化
会压掉局部薄区中的真实异常。

## Failure modes

- 扫描伪影同时制造畸变和亮点时仍可能通过交集门；
- 稀土与 Si 柱投影重叠时，局部亮度残差需要 multislice 模拟才能解释；
- 热点阈值和扩张半径尚无 EDS/EELS 真值校准；
- 没有物理 z 步长时只能报告最亮层编号，不能报告原子深度；
- 没有弹性张量时最大剪应变只能叫应力热点代理。

## User contribution

用户提供了候选所有权规则和最小反例的预期答案：先限制在畸变/应力异常区域，再找极端
亮点；A/B/C 中只选择 B；跨层稳定性不得作为必要条件。该判断直接固化在两个聚焦测试中。

## Validation

- FAU 聚焦测试：`11 passed`；包括 A/B/C 交集门、单层峰、Y 负对照归一化和尺度不可识别边界。
- 真实三组完整 TIFF：Y `0`、RA `0`、H2 `3` 个交集候选。
- H2 的 20 个局部极亮点中 17 个因热点外而拒绝；3 个候选局部 z 分数为
  `5.08`、`5.16`、`5.22`。
- 结果图人工检查：连续热点轮廓、候选编号和候选自身最亮层放大图均可读。

## Retrieval questions

1. 为什么跨层次数适合作为置信旁证，却不适合作为稀土候选硬门？
2. 为什么“候选附近畸变更高”不能再作为稀土造成畸变的独立证明？

## Learning debt

需要 EDS/EELS 点标或匹配成像条件的 multislice 模拟来校准 `5σ`、热点分位数和空间半径，
并确认三个 H2 候选不是扫描伪影或投影重叠。

## Review history

- 2026-09-14：用户预测最小反例只保留 B，并补充单层稀土边界；完成合成复现、真实数据
  集成与图像审计，状态更新为 applied、置信度 2。
