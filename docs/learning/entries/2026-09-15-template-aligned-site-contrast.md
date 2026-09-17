# FAU 模板对齐后的逐 O/Si 位点衬度审计

> 2026-09-16 更新：本条保留为旧 O/Si 位点审计记录；最终候选所有权已转移到
> [骨架优先 site-I + 局部等价亮度 + GPA 后验旁证](2026-09-16-skeleton-first-site-i-local-gpa.md)。
> 旧的 O/Si 亮度/热点候选不再作为最终稀土指认。

- Kind: algorithm
- First seen: 2026-09-15
- Status: applied
- Confidence: 2
- Related change: `src/tfree_detection/fau_stack_analysis.py`、`tests/test_fau_stack_analysis.py`
- Result: `real_data_test/4D数据计算晶格变化_0901/fau_stack_analysis_101_v15_H2_crop_regular_region/`
- Last review: 2026-09-15

## Why this appeared

用户提出先把 Y、RA、H2 都配准到 FAU `[101]` 骨架，再统计每个理论 Si/O 位点的衬度，
把异常点圈作稀土候选。这比只在全图找极亮峰多利用了一层信息：同一结晶学投影位点在不同
重复单胞中应具有可比较的背景分布。

## Mental model

骨架拟合与衬度归属必须分开。骨架拟合需要排除疑似稀土峰，避免异常点拉歪晶格；配准固定
以后，衬度归属则要重新纳入所有检测峰，否则真正的替位或投影重叠峰永远不会被审计。

每个理论 O/Si 位点先关联 `4 px` 内的检测峰，在所有堆栈层采样并取最大标准化 DoG 衬度。
它不与全图任意 O/Si 混比，而只与本图中相同 CIF 投影 motif 的重复单胞比较。最终候选还
必须通过畸变热点、亮度类别、`3 px` 严格配准残差和 `0.25` 投影单胞边界可靠性门槛。

## Minimal reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
  -p no:cacheprovider -q \
  tests/test_fau_stack_analysis.py::test_aligned_site_attribution_recovers_peak_excluded_from_framework_fit \
  tests/test_fau_stack_analysis.py::test_same_motif_contrast_normalization_is_affine_invariant \
  tests/test_fau_stack_analysis.py::test_aligned_site_candidate_gate_rejects_cell_scaled_image_boundary \
  tests/test_fau_stack_analysis.py::test_aligned_site_candidate_residual_cannot_exceed_match_tolerance
```

## Prediction

用户预测：三图先按 FAU 模板对齐后，可以统计全部 Si/O 对应位点衬度；相对同类位点异常亮
且落在畸变区的点，应作为稀土原子点位候选圈出。

## Observation

合成测试确认两项不变量：骨架拟合中被排除的亮峰仍能在固定模板上重新归属；对整幅图施加
正增益与背景偏移后，同 motif 稳健 z 分数不变。边界测试进一步确认候选排除距图像边缘少于
`0.25` 投影单胞的点。

真实数据共覆盖 `63,131` 个理论 O/Si 位点，其中 `37,118` 个关联检测峰。Y、RA、H2 的
同 motif 极亮异常分别为 `11 / 9 / 5`；叠加热点、`≤3 px` 配准残差、内区门和 Y 负对照
以后，模板位点候选为 `0 / 2 / 0`。RA 两点均是名义 O 位，坐标约 `(542.90,499.88)` 与
`(401.06,545.69)` px，同 motif z 为 `6.53 / 5.75`。

原有局部 Si 参照的峰级规则仍给出 H2 三点，但其同 motif z 只有 `2.34 / 4.15 / 2.98`；
RA 两个模板位点候选也没有通过原峰规则。两类候选严格交集为 `0`。因此用户预测的前半部分
成立：模板对齐能产生逐位点统计并发现新异常；后半部分需要修正为“异常是优先复核候选，
不能直接等同稀土化学身份”。

后续把用户选择的规整 H2 区域精确映射回 19 层原始堆栈后，H2 的 CIF NCC 从 `0.848`
提高到 `0.880`，骨架匹配率从 `74.9%` 提高到 `88.0%`，畸变中位数从 `0.920%` 降到
`0.688%`。旧全视野的 3 个严格峰级候选全部位于裁掉的非晶/紊乱区；规整区严格候选为 0，
只保留 2 个敏感次级复核区。这说明视野质量选择是检测前提，而不是无关的显示步骤。

## Project integration map

```text
Y / RA / H2 原始 z-stack
  -> 逐图稳健归一化与全层峰检测
  -> 稳定峰拟合 FAU [101] O/Si 骨架
  -> 排除峰级疑似客体并固定最终配准
  -> 所有检测峰重新归属理论 O/Si 位点
  -> 同 CIF motif 的重复单胞稳健 z 分数
  -> 亮度 + 热点 + 残差 + 边界 + Y 负对照
  -> 全位点 CSV + 候选 CSV + 图 13/14
```

## Theory and boundaries

对位点 `i`，令 `c_i` 为所有层的最大标准化 DoG 响应，`k(i)` 是其投影 motif：

```text
r_i = c_i - median(c_j | k(j)=k(i))
z_i = r_i / max(MAD_scale_k, 0.5 * global_residual_MAD_scale)
```

这个 z 分数消除了逐图整体增益/偏置并控制不同结构位点固有亮度，但不能消除厚度、通道效应、
沿束流方向的原子柱重叠或重建伪影。CIF 是纯 SiO2；“名义 O 位候选”只表示异常峰落在
模板投影 O 位置附近，可能是替位、重叠或误配，绝不表示检测到了一个更亮的 O 原子。

## Failure modes

- 候选靠近图像边缘时，DoG 反射边界和不完整局部邻域可制造假异常；
- 关联残差接近 `4 px` 搜索上限时，峰可能属于相邻柱，不能解释成该模板位点；
- 相同 motif 在小视野中重复不足 `12` 次时，流程拒绝形成校准；
- 不同视野不能逐原子跟踪，Y/RA/H2 之间只能比较分布和候选类别；
- 两种亮度参照规则不一致时，不能事后调阈值强行得到交集。
- Y 虽可产生统计衬度尾部用于阈值审计，但作为已知无稀土模板，任何尾部点都不得在结果图上
  用候选或特殊形状标记。
- 专家式敏感复核层会扩大召回率，但边缘异常会随之增多；距边缘不足 `0.75` 个投影单胞的
  区域必须显式保留风险标签，不能和内部区域同等解释。

## User contribution

用户提出以物理尺度 FAU 骨架统一对齐三类样品，并把全部 Si/O 理论位置作为衬度统计索引；
该预测决定了从“峰清单”升级为“完整模板位点表”。用户同时明确 Y 是无稀土参考，这使 Y
阳性只能作为假阳性审计，不能解释成稀土。

用户在查看初版图 13 后进一步指出：Y 既然是模板且没有稀土，就不应出现任何特殊点圈选。
这条反馈被固化为显示层硬不变量：Y 可参与数值校准和直方图统计，但所有候选/异常形状标记
一律关闭，标题明确写出 `final RE sites = 0`。

用户进一步提出按专家阅读局部骨架差异的方式扩大候选区。由此新增独立复核层：在可靠畸变区
内用 Y 的 `99.9%` 尾部与 `z=3.5` 下限筛查同 motif 异常，并与原严格非模板亮峰按
`0.25` 个投影单胞聚类。该层刻意与最终元素指认解耦，避免以扩大审核池偷换最终稀土数量。

## Validation

- FAU 工作流回归：`23 passed`；
- 合成增益/偏置不变性、被骨架排除峰重新归属、晶胞尺度边界门、残差配置边界均通过；
- 当前规整 H2 输出：`45,856` 行全位点数据、`2` 行 RA 模板位点候选、H2 严格候选 0；
- 图 `13_aligned_O_Si_site_contrast_anomalies.png` 与
  `14_aligned_site_candidate_zoom_audit.png` 已逐图目视检查；
- Y 最终模板位点候选固定为 `0`；两种候选规则的严格交集为 `0`，已显式报告而非隐藏。
- 新复核层输出 RA `4` 区、H2 `2` 区、Y `0` 区；RA `4/4` 与 H2 `1/2` 区带
  `EDGE-WARNING`，并在 CSV 与图 15/16 中一致显示。图 16 同时叠加理论 O/Si 骨架和
  观测位移箭头，已逐图目视检查。

## Retrieval questions

1. 为什么骨架拟合排除疑似稀土峰后，衬度归属还要重新纳入所有峰？
2. 为什么同 motif 的 `5σ` 异常仍不能直接称为稀土元素？

## Learning debt

论文级元素归属仍需 EDS/EELS 对候选点的少量标签，或带厚度、探针、探测器角度和多重散射
参数的 multislice 模拟。获得这些监督后，可校准两类候选的融合规则，而不是凭当前无标签
图像选择其中一种。

## Review history

- 2026-09-15：完成合成不变量、真实三图逐位点统计、边界/残差审计与两类候选一致性比较；
  状态更新为 applied、置信度 2。
- 2026-09-15：根据用户纠正，新增“无稀土模板禁止特殊点标记”回归测试并更新图 13；保持
  applied、置信度 2。
- 2026-09-15：新增 Y 尾部校准的专家骨架区域复核层、边界风险标记和局部骨架/位移审计图；
  真实输出 RA 4 区、H2 6 区、Y 0 区，保持 applied、置信度 2。
- 2026-09-15：把用户选择的 H2 规整 PNG 以 NCC `0.988` 映射到原始 19 层的
  `x=[796,1354)` 裁剪；重跑后 H2 严格候选从 3 降为 0，确认旧候选受被裁非晶区驱动，
  当前输出 RA 4 个、H2 2 个专家复核区，保持 applied、置信度 2。
