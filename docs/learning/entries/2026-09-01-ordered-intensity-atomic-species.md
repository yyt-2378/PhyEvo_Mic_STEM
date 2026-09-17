# 参考晶格对齐后的有序强度元素分型

- Kind: algorithm
- First seen: 2026-09-01
- Status: applied
- Confidence: 2
- Related change: src/tfree_detection/atomic_site_inventory.py、tfree-detect map-atomic-sites
- Last review: 2026-09-01

> 2026-09-14 更正：三档亮度可作为观察分组，但“最高亮度总体＝稀土”的化学命名已被
> 真实 FAU 投影审计否定。当前稀土筛选以[畸变区门控的局部极亮判据](2026-09-14-distortion-gated-local-extreme-rare-earth.md)
> 为准；本条保留为算法迭代记录，不再作为当前候选数量依据。

## Why this appeared

用户希望在不同尺寸、取向和视野的 Y、RA、H2 三张 4D-STEM PNG 上检测全部原子位点，
并区分 Si、O 和稀土原子柱。原框架主要输出异常候选，不能直接给出所有位点的元素清单。

## Mental model

先用全局晶格基矢只统一方向和像素尺度，中心平移仅用于可视化；再在对齐图的真实覆盖
区域内检测局部峰。元素身份不由几何臆测，而由实验者声明的单调亮度约束命名：
最低、中间、最高局部对比度簇分别为 O、Si、稀土。输出必须保留归属概率和簇分离度，
避免把连续强度变化包装成绝对化学鉴定。

## Minimal reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest \
  -p no:cacheprovider -q \
  tests/test_atomic_site_inventory.py::test_ordered_mixture_recovers_imbalanced_species_brightness
```

合成数据包含 600 个暗峰、300 个中等峰和仅 80 个亮峰，检验数量不均衡时能否仍保留三个
强度总体。

## Prediction

用户确认：“这三张 PNG 中原子柱亮度应当按照‘稀土最亮、Si 次之、O 最暗’排序，
就是这样的，因为是 4D-STEM。”

## Observation

朴素 K-means 在合成不均衡数据上把数量最多的暗峰拆成两个簇，并吞掉稀有亮峰；这说明
“知道顺序”还不足以让普通聚类稳定工作。改用多起点的一维高斯混合后，合成中心恢复为
0.1767、0.4794、0.8648，接近设定的 0.18、0.48、0.86。

真实三图的最低相邻簇分离度为 2.58σ、2.63σ、2.75σ，元素比例也相近；用户预测的
亮度顺序与实际拟合顺序一致。

## Project integration map

```text
map-atomic-sites
  ├─ robust normalization + FFT direct basis
  ├─ match_lattice_bases(Y, moving)
  ├─ common Y-scale/Y-axis canvas (center translation is visual only)
  ├─ covered-region local peak detection + subpixel centroid
  ├─ ordered three-component intensity mixture
  └─ CSV/NPZ/JSON + native/aligned/species overlays
```

已有双物理教师的原子缺陷检测不变；新模块负责“全部位点清单”，不把普通原子柱误当成
缺陷候选，也不调用已退役的 DINO 第三观察器。

## Theory and boundaries

强度有序模型只在实验成像模式、处理流程和用户声明使亮度单调关系成立时有效。晶格相似
配准去除的是旋转和等向像素缩放；没有共同地标时，绝对平移与跨图原子身份不可识别。
亮度分型也不等于 EDS/EELS 或多重散射标定后的独立元素鉴定。

## Alternatives

未采用固定全局阈值，因为三图动态范围不同。未采用普通 K-means，因为稀土亮峰数量少时
可能被高丰度暗峰吞并。若补充少量专家点标或谱学信息，可改用受监督阈值校准并独立验证
当前三个强度总体。

## Failure modes

- 图像反相或探测模式改变会把元素名整体映射错误；
- 低对比 O 峰可能接近噪声门，产生漏检或低置信归属；
- 扫描伪影、重叠原子柱和厚度变化会拓宽强度簇；
- 不同视野只能做晶格坐标归一化，不能声称绝对迁移；
- 位点不足时三成分模型必须失败，而不是强行输出三个元素。

## User contribution

用户提供并确认了本批 4D-STEM PNG 的元素亮度硬约束：
稀土最亮、Si 次之、O 最暗。该约束决定了三个有序强度总体的元素名称，并被固化在
tests/test_atomic_site_inventory.py 的不均衡总体测试和真实结果审计字段中。

## Validation

- 相关单元与回归：
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest
  -p no:cacheprovider -q tests/test_atomic_site_inventory.py
  tests/test_reference_deformation.py tests/test_atomic_consensus.py tests/test_config.py
- 观察：13 passed；包含位点不足时拒绝强行三分型的失败边界。
- 真实运行：Y 4,930，RA 5,403，H2 4,179 个位点；三个输入均成功。

## Retrieval questions

1. 为什么三张图中心对齐后仍不能逐点计算“同一个原子”的迁移？
2. 为什么强度簇分得清楚仍不等于完成独立化学鉴定？

## Learning debt

尚缺少专家点标、EDS/EELS 或已标定模拟来独立验证元素名称；也缺少物理像素尺度来把坐标
换成 Å 或 nm。

## Review history

- 2026-09-01：用户确认稀土 > Si > O 的亮度顺序；不均衡合成复现和三图真实运行完成，
  状态更新为 applied、置信度 2。
