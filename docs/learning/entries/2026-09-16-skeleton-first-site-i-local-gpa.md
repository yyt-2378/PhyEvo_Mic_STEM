# 骨架优先 site-I 候选与局部等价位点/GPA 闭环

- Kind: algorithm
- First seen: 2026-09-16
- Status: applied
- Confidence: 2
- Related change: `src/tfree_detection/fau_stack_analysis.py`、`src/tfree_detection/reference_deformation.py`、对应测试
- Result: `real_data_test/4D数据计算晶格变化_0901/fau_stack_analysis_101_v17_skeleton_first_local_gpa/`
- Last review: 2026-09-16

## Why this appeared

用户与领域专家指出，旧流程在完成骨架匹配后才从亮度和畸变热点找稀土，因果顺序有问题：
稀土本身可能改变局部骨架，应该先由理想 FAU 骨架找到“本不应存在的额外连接/中心成分”，
再在一个小邻域中与相邻等价位点比较亮度。全图亮度会受焦点、厚度、曝光和重建影响，不适合
直接跨视野作元素判断。候选确定后，GPA 应作为独立空间旁证，而不是再次参与候选门控。

## Mental model

候选所有权分成三步且顺序不可交换：

```text
FAU.cif [101] 骨架
  -> 理论空 16c/site-I 中心与两侧 O 端点
  -> 两 O 模型 vs 两 O + 中心模型（结构先验）
  -> 邻近对称等价 site-I 的局部稳健亮度（成像内参）
  -> B 类候选 / C 类结构复核
  -> 冻结候选后再与 native-coordinate GPA 热点比较（独立旁证）
```

GPA 的角色是回答“候选附近是否也是高畸变/高剪切代理区”，不能回答元素身份，也不能因为
一个点落在 GPA 不可靠区就自动否定结构和亮度证据。

## Minimal reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
  -p no:cacheprovider -q \
  tests/test_fau_stack_analysis.py::test_local_same_species_contrast_is_gain_invariant \
  tests/test_fau_stack_analysis.py::test_template_empty_center_profile_prefers_added_component \
  tests/test_fau_stack_analysis.py::test_skeleton_first_prediction_contract \
  tests/test_fau_stack_analysis.py::test_real_fau_cif_produces_projected_site_i_template \
  tests/test_reference_deformation.py::test_candidate_overlap_is_posthoc_and_uses_native_coordinates
```

合成 A/B/C 反例：A 只改变整体增益但没有中心结构，局部 z 保持 0 并排除；B 增加中心成分
且相对邻近等价位点异常亮，进入候选；C 只有中心结构变化、局部 z 未过门，进入结构复核。

## Prediction

用户/专家预测：A 应排除；B 应作为稀土候选；C 只做结构复核。真实 RA/H2 候选应在独立
GPA 畸变/应力代理热点中得到一定空间支持，但不能要求 100% 命中或用 GPA 反向决定候选。

## Observation

真实数据的最终候选数为 Y/RA/H2=`0/2/3`。RA 两点都同时落在 GPA 相对畸变和最大剪应变
代理前 10% 热点内；H2 三点中两点双热点支持，另一点在代表焦面的 GPA 相干性不可靠区，
距畸变/剪切热点约 `66.89/36.67 px`。这条非全命中结果保留，未事后调阈值或删点。

结构存在门在 RA/H2 的有效模型中通过 `32/34`、`71/72`，显示当前模糊投影会使额外中心
模型接近饱和；它能固定物理 site-I 搜索位置，但特异性有限。最终 2/3 个候选主要由局部
等价位点亮度筛出，这一限制已写入当前报告，没有把“结构先行”夸大为结构单独识别。

RA 两点的局部圆盘畸变 P90 位于可靠像素的 `96.99%/99.74%` 分位；H2 两个受支持点位于
`91.86%/98.79%` 分位。进一步以全部有效 site-I 作空间对照后，RA/H2 普通 site-I 的
双热点背景已达 `22/34` 与 `35/72`；候选富集的单侧超几何 p 值为 `0.412/0.479`，并不
显著。因此 GPA 只支持“多数候选与高畸变共现”，不支持候选相对等价位点显著富集，也不能
证明稀土导致畸变。

## Project integration map

```text
parse_and_expand_cif
  -> project_fau_site_i_template
  -> measure_skeleton_first_site_i_evidence
       -> opposed O endpoints
       -> two-O / two-O-plus-center profile BIC
       -> local equivalent-site robust z
  -> apply_skeleton_first_reference_audit
       -> Y=0 invariant
       -> B candidate / C review
  -> rare_earth_candidates.csv + figures 18/19

current representative TIFF planes
  -> TrainingFreeDetectionFramework.detect
  -> compare_to_reference (existing GPA)
  -> summarize/save_candidate_hotspot_overlap
  -> candidate_gpa_hotspot_overlap.{png,csv,json}
```

## Theory and boundaries

结构证据使用

```text
ΔBIC = BIC(two-O framework) - BIC(two-O framework + site-I center)
```

固定 `ΔBIC >= 6` 作为额外中心成分的结构阳性下限。Y 的 ΔBIC 尾部只作审计，因为不同图
的噪声、焦点和重建质量会改变绝对 BIC。亮度证据只与本图半径 `2.5` 投影单胞内的对称
等价 site-I 比较，最少 `6` 个参照，稳健 `z >= 3.5`。

GPA 热点定义为可靠像素中相对畸变或最大剪应变代理的前 10%，候选中心距热点不超过
`3 px` 记为支持。最大剪应变是无量纲排序，不是 Pa/GPa；RA/H2/Y 不同视野也不允许逐像素
跨图相减或逐原子迁移追踪。

## Failure modes

- site-I 两侧 O 端点的投影分离不足时，结构模型无效并拒绝解释；
- 小视野内局部等价参照不足时，不计算候选；
- H2 裁剪边缘或低 GPA 相干区可造成“候选存在但 GPA 无可判支持”；
- 沿束流原子柱重叠、厚度、通道效应和重建伪影仍可制造额外中心成分；
- 5 个候选不足以建立稀土—畸变的统计因果关系。

## User contribution

用户和老师专家提供了决定性的因果顺序修正：不能先按亮度找点再套骨架，而要从红框所示的
骨架差异位置出发；亮度只应在小范围相邻相同/等价位点之间比较。用户进一步明确 A/B/C
解释：A 排除、B 候选、C 结构复核。这一合同直接成为合成测试和报告术语。

## Validation

- `tests/test_reference_deformation.py tests/test_fau_stack_analysis.py`：`32 passed`；
- 合成 A/B/C、局部增益不变性、中心模型 BIC、真实 CIF site-I 投影均通过；
- 当前 TIFF GPA 复用原框架，输入哈希与代表层记录在 `input_provenance.json`；
- `candidate_gpa_hotspot_control_audit.png`、图 18、图 19 已逐图目视核验；
- Y 图和最终表均为 0 候选；RA/H2 的 2/3 个候选与 CSV/JSON/图中坐标一致；
- 未把 H2 的 GPA 不可靠候选伪装成热点命中，也未用 GPA 改变候选数。

## Retrieval questions

1. 为什么局部亮度必须在结构门之后，而且只能比较邻近等价位点？
2. 为什么候选落在 GPA 不可靠区不能被解释成“GPA 证明它不是稀土”？

## Learning debt

仍需 EDS/EELS 点标、独立像素标定、厚度/探测器参数和 multislice 模拟来校准化学身份；若
要输出真实应力，还需材料在该取向下的弹性张量。扩大样本后才能检验候选—GPA 共现是否
显著高于随机等价 site-I 对照。

## Review history

- 2026-09-16：用户完成 A/B/C 检查点；结构优先、局部等价亮度与 GPA 后验旁证完成真实
  数据闭环，状态设为 applied、置信度 2。
