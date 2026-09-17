# 第 4/10 类 site-I 的结构主判 B1/B2 召回分层

- Kind: algorithm
- First seen: 2026-09-16
- Status: applied
- Confidence: 2
- Related change: `src/tfree_detection/fau_stack_analysis.py`、对应测试与 v18 结果
- Result: `real_data_test/4D数据计算晶格变化_0901/fau_stack_analysis_101_v18_structure_primary_recall/`
- Last review: 2026-09-16

## Why this appeared

v17 只把“结构阳性且局部等价位点中心成分 z≥3.5”的点解释为候选，RA/H2 仅有 2/3 个。
用户指出真正应拥有判读权的是纯 FAU 模板第 4/10 类空 site-I 桥位是否出现额外连接；局部
亮度只应是单胞内支持证据，不应成为必要条件。旧 GPA 黄色轮廓则是广义畸变范围，不能
直接当作稀土标签。

## Mental model

```text
FAU [101] 第 4/10 类理论空桥及周期重复位
  -> 两 O 模板 vs 两 O + 中心成分（结构阳性）
  -> 图内结构差异高尾部 = B2 可能稀土/优先专家复核
  -> 局部等价位点中心成分极端 = B1 高置信复核
  -> 候选冻结后再看 GPA，仅作空间旁证
```

另外两类投影 site-I 与骨架柱重叠，不能用同一个空桥模型；扩大召回不能靠把不可分辨位点
强行加入候选。

## Prediction

用户预测并确认：一个第 4/10 类 site-I 若相对模板存在结构差异，即使局部亮度未过严格门，
也应归为“可能稀土、优先专家复核”；亮度可提高可信度但不是必要条件。

## Observation

将结构阳性池的图内 ΔBIC 最高 20% 设为 B2，并保留 z≥3.5 的 B1 后：Y/RA/H2 的
B1+B2 数量为 `0/9/17`，其中 B1 仍为 `0/2/3`。RA/H2 可测第 4/10 周期位为 34/72。

B1+B2 与 GPA 双热点重合为 RA `6/9`、H2 `7/17`；全部有效 site-I 背景为 `22/34`、
`35/72`，单侧超几何 p=`0.610/0.836`。扩展召回没有得到 GPA 富集确证，因此 B2 必须
保持“专家复核”而非化学身份。

## Project integration map

```text
apply_skeleton_first_reference_audit
  -> raw structural-positive pool
  -> top 20% delta-BIC structural tail -> B2
  -> local center-component z>=3.5 -> B1 (also included in priority review)
  -> rare_earth_priority_review.csv + figures 18/19
  -> post-hoc GPA control audit
```

## Failure modes

- `ΔBIC>=6` 在本批模糊投影中接近饱和，不能单独当作高特异性化学判据；
- 图内最高 20% 是专家工作量/召回折中，不是自然元素分界；
- B2 可包含厚度、焦点、通道效应或重建伪影；
- 旧 PNG GPA 图与当前 TIFF 不是同一视野，不能逐点映射；
- Y 的绝对 ΔBIC 尾部不适合跨图硬阈值，但 Y 解释结果仍必须为 0。

## Validation

- `tests/test_fau_stack_analysis.py tests/test_reference_deformation.py`：`32 passed`；
- v18 从原始 Y/RA/H2 TIFF 重新生成，未覆盖 v17；
- 图 18、全部 26 点的图 20 及 B1+B2 GPA 叠加图已目视核验；
- CSV、summary 与图中数量一致：RA=9、H2=17、Y=0。

## Retrieval questions

1. 为什么 B2 可以不要求亮度，而 B1 仍保留局部中心强度支持？
2. 为什么不能把旧 GPA 黄色畸变轮廓中的每个原子都解释为稀土？

## Learning debt

需要专家对 26 个 B1/B2 点逐点确认“是否确有中心连接”，再用少量 EDS/EELS 点标校准 B1、
B2 的真实阳性率。扩大样本后才能学习数据驱动的结构尾部阈值，替代当前 20% 复核预算。
