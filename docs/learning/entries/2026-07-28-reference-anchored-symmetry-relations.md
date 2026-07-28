# 参考锚定的晶格群关系与多几何 Class1

- Kind: architecture
- First seen: 2026-07-28
- Status: applied
- Confidence: 2
- Related change: `src/tfree_detection/symmetry/`,
  `tests/test_symmetry_group.py`, `tests/test_symmetry_task.py`
- Last review: 2026-07-28

## Why this appeared

Dennler Class1 不只是窄层错线，还混合平移域、孪晶、晶界和 secondary symmetry。旧局部
波矢/应变分支看不到“频率和取向不变、但晶胞 motif 相位改变”的域，固定分位数又会在
正常图强制产生候选。

## Mental model

局部晶格不是只与参考做像素差，而是问：

> 是否存在一个允许的晶格群作用，使局部 motif 能重新与正常参考对齐？

identity residual 高、group-aligned residual 低表示“不是同一身份，但属于可解释的群关系”。
然而扫描漂移也可能产生平滑平移，所以域还必须由位移不连续边界支持。输出几何必须分成
region、interface、line 和 point，不能再用一个 box 代替。

## Minimal reproduction

```bash
PYTHONPATH=src pytest -q \
  tests/test_symmetry_group.py \
  tests/test_symmetry_task.py
```

合成图左右两侧取向与晶格间距相同，右侧平移 `0.35a`。方向/应变教师在理想域内部接近
零，但 group-phase 教师明显响应。干净晶格允许零候选；相移域输出区域和界面。

## Prediction

设计检查点是：同间距同取向、右侧相移的图像，应该只输出边界，还是同时输出完整
symmetry-domain region 与 interface line？

用户目标明确要求四几何输出，因此预测为后者：**完整域 + 界面线**。

## Observation

群对齐单测正确区分 identity、2-bin translation、90° rotation 和 mirror；多参考选择能
排除相移少数簇。最终合成域区域召回约 `0.828`、正常侧区域误报为 `0`，干净晶格全空。

Dennler 事后诊断中，新域 Recall `0.707`、正常图区域误报 `0.0035`；与历史候选做标签
无关的引导扩展后 IoU/F1 为 `0.1891/0.3181`。

## Project integration map

```text
symmetry/group.py
  └─ motif folding + translation/rotation/mirror/sublattice alignment
symmetry/references.py
  └─ dominant identity cluster + spatially diverse references
symmetry/task.py
  ├─ carrier phase-gradient frequency correction
  ├─ five teachers + consensus gate
  └─ region/interface/line/point geometries
framework.py + evidence.py
  └─ fifth registered task + shared ontology/resolver/reliability
```

## Theory and boundaries

当前离散操作近似二维 D4 式 motif 关系，不是完整空间群索引。多参考仍依赖“正常身份占优”；
异常域占多数时语义可能翻转。单帧 scan teacher 不是时间证据。芯点是 curl/topology proxy，
必须用原子图 Burgers circuit 或专家验证。

## Failure modes

- FFT bin 误差会随坐标累积成假平移，必须用相位梯度校正；
- 只看群平移会把平滑扫描漂移当成域，必须加入位移不连续共识；
- 8 邻域普通线像素有两个邻居，端点必须是 `<=1`，否则芯点掩膜会膨胀；
- 线/点与 Dennler 粗区域标签直接做 IoU 会产生几何失配。

## User contribution

用户提出多正常参考 patch、群作用对齐、五物理教师与四种几何输出，并要求继续把 Class1
真正迭代为可用任务。这一要求直接决定了架构层级和“完整域 + 界面线”的合成不变量。

## Validation

- 群关系与参考选择：2 tests passed；
- 端到端 Class1/no-defect/EvidenceBundle：2 tests passed；
- Dennler v2：61/61 prediction success，prediction/score seal 完整；
- 全项目最终回归结果见本次实施报告。

## Retrieval questions

1. 为什么“平移群作用能解释局部 motif”还不足以证明这里是真实层错？
2. 为什么位错芯的 point 几何不能用 Dennler Class1 粗区域 IoU 单独证明正确？

## Learning debt

需要在未消费数据上加入完整空间群/晶体学索引、原子图 Burgers circuit、正交扫描或时序
证据，并为 region/interface/line/point 分别建立 Gold calibration。

## Review history

- 2026-07-28：完成最小复现、框架集成、端点错误审计和 Dennler 事后诊断；confidence 2。

