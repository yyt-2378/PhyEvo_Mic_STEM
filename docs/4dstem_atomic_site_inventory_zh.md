# 4D-STEM 三图晶格对齐与 O/Si/稀土原子位点清单

> 状态：2026-09-01 历史亮度分群实验，不是当前元素指认结论。输入为
> real_data_test/4D_data/Y.png、RA.png 和 H2.png；Y 是参考结构。
> 后续已明确 Y 不含稀土，因此下文旧表中的“稀土”只表示当时的高亮度簇，
> 不能解释为稀土数量或掺杂比例。当前结构主判、Y 零候选规则请见
> [FAU 冻结流程](fau_zstack_cif_analysis_zh.md)和
> [通用投影结构位点引擎](structure_guided_generalization_zh.md)。

## 做了什么

1. 从每张图的全局倒易峰恢复两组直接晶格基矢。
2. 将 RA、H2 的刚体旋转和等向像素缩放匹配到 Y；三张图放入同一公共画布。
3. 只在各图真实覆盖区域内寻找局部亮峰，并用局部加权质心细化到亚像素坐标。
4. 对去除局部背景后的峰强度拟合有序三成分混合模型。
5. 按实验者确认的 4D-STEM 关系，把暗、中、亮三个峰簇分别记为 O、Si、稀土原子柱。

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m tfree_detection.cli \
  map-atomic-sites \
  --reference real_data_test/4D_data/Y.png \
  --moving real_data_test/4D_data/RA.png real_data_test/4D_data/H2.png \
  --config configs/atomic_point_defect.yaml \
  --output real_data_test/4D_data/atomic_site_inventory_v1
```

## 当前结果

| 图像 | 全部位点 | O | Si | 稀土 | O / Si / 稀土比例 | 最小相邻簇分离度 |
|---|---:|---:|---:|---:|---:|---:|
| Y | 4,930 | 1,146 | 2,581 | 1,203 | 23.2% / 52.4% / 24.4% | 2.58σ |
| RA | 5,403 | 1,132 | 2,905 | 1,366 | 21.0% / 53.8% / 25.3% | 2.63σ |
| H2 | 4,179 | 922 | 2,195 | 1,062 | 22.1% / 52.5% / 25.4% | 2.75σ |

三图的元素比例和强度簇中心顺序相近，是当前分型的内部一致性证据。它不是独立化学
验证；最终元素身份仍建议用少量专家点标、EDS/EELS 或已标定模拟抽查。

RA 到 Y 的等向像素尺度系数为 1.1493、晶格等价旋转为 -56.28°；H2 分别为
1.3070 和 -79.34°。基矢相似拟合残差分别为 0.0786 和 0.0619，说明相似变换可以统一
主要方向和尺度，但仍保留真实晶格形状差异。

## 输出在哪里

- 总览图：real_data_test/4D_data/atomic_site_inventory_v1/atomic_sites_native_and_aligned.png
- 分元素图：real_data_test/4D_data/atomic_site_inventory_v1/atomic_sites_by_species_aligned.png
- 强度分群图：real_data_test/4D_data/atomic_site_inventory_v1/species_intensity_mixtures.png
- 全部逐点坐标：real_data_test/4D_data/atomic_site_inventory_v1/atomic_sites.csv
- 审计摘要：real_data_test/4D_data/atomic_site_inventory_v1/atomic_site_inventory_summary.json
- 每图 NumPy 数组：输出目录下 Y、RA、H2 子目录中的 atomic_sites.npz

CSV 同时保存原图坐标、公共对齐坐标、局部对比度和元素归属概率，后续可以直接用于
配位统计、最近邻距离、缺位检查或专家抽查。

## 不能怎样解释

- 三张图不是已证明的同一视野，中心对齐不能解释成绝对原子迁移。
- 不应逐行把 Y、RA、H2 的点当成同一个原子的一一对应。
- “O、Si、稀土”来自已声明的亮度顺序；没有独立谱学信息时，应称为亮度分型结果。
- 当前坐标单位仍是 Y 参考像素；没有物理像素标定时不能直接写成 Å 或 nm。
