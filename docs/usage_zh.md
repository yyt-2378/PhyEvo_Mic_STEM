# 使用说明

## 1. 安装

要求 Python 3.10 或更高版本。建议使用独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

需要冻结 DINO 特征或 evidence-student 时再安装可选依赖：

```bash
python -m pip install -e ".[stage2]"
python -m pip install -e ".[learning]"
python -m pip install -e ".[fourdstem]"
```

默认物理任务不需要训练，也不需要下载神经网络权重。

## 2. 最小示例

仓库自带一张局部非晶分子筛示例：

```bash
tfree-detect detect \
  --input reference_scale_radial_wiener.png \
  --config configs/amorphous.yaml \
  --output outputs/quickstart_amorphous
```

输出目录包含：

```text
summary.json                 区域、证据、可靠性和溯源摘要
mask.png                     二值候选
score_heatmap.png            连续异常分数
overlay.png                  原图上的候选轮廓
diagnostics.png              多证据诊断面板
evidence_protocol.json       统一结构证据协议
candidate_table.jsonl        逐候选证据与来源
```

输出目录必须是新目录；CLI 不会静默覆盖既有结果。

## 3. 六个任务

### 局部非晶或失序区域

```bash
tfree-detect detect \
  --input image.tif \
  --config configs/amorphous.yaml \
  --output outputs/amorphous_run
```

### 原子点缺陷

```bash
tfree-detect detect \
  --input atomic_image.tif \
  --config configs/atomic_point_defect.yaml \
  --output outputs/atomic_run
```

对于已知低对称晶胞，可以额外提供：

```bash
--skeleton schemas/skeleton.example.json
```

### 层错、位错或相位滑移线

```bash
tfree-detect detect \
  --input line_defect.tif \
  --config configs/stacking_fault.yaml \
  --line-topology \
  --output outputs/stacking_run
```

### 两相、孪晶或晶畴界面

```bash
tfree-detect detect \
  --input interface.tif \
  --config configs/phase_boundary.yaml \
  --line-topology \
  --output outputs/interface_run
```

### Class1 群关系与多几何输出

```bash
tfree-detect detect \
  --input class1_image.tif \
  --config configs/symmetry_relation.yaml \
  --output outputs/symmetry_relation_run
```

关键 feature maps：

- `symmetry_domain_region`
- `interface_line`
- `phase_slip_line`
- `dislocation_core_point`

芯点输出是 Burgers-loop/curl 的拓扑代理，不是已确认的 Burgers vector。

### 投影模板可选位点

```bash
tfree-detect detect \
  --input image.tif \
  --config configs/structure_guided_site.yaml \
  --skeleton schemas/structure_guided_skeleton.example.json \
  --output outputs/structure_guided_run
```

骨架必须给出像素基矢、`metadata.origin_xy`，以及由两个非可选端点连接的
`optional: true` 位点。样品若是事先声明的无目标参考，应把
`structure_guided.sample_role` 设为 `target_free_reference`；该角色不会保留中心模型评分，
且候选数固定为 0。完整约定见
[通用投影结构位点引擎](structure_guided_generalization_zh.md)。

### 可选 SAM2 真实边界细化

当前机器已有 CUDA SAM 环境和 SAM2.1-large checkpoint，可直接运行真实模型：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python -m tfree_detection detect \
  --input reference_scale_radial_wiener.png \
  --config configs/amorphous_sam2.yaml \
  --output outputs/reference_amorphous_sam2_live_v2
```

界面任务使用成对晶域提示：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python -m tfree_detection detect \
  --input datasets/public/guo_2021_atomic_defects/ZrO2_monoclinic_twin_boundary.tif \
  --config configs/phase_boundary_sam2.yaml \
  --output outputs/zro2_phase_boundary_sam2_live_v2
```

配置中的 worker 指向 `/home/aiprogram/anaconda3/envs/sam/bin/python`，并要求 CUDA；
若换机器，只需修改 `worker_python` 和 `checkpoint_path`。本次实例、六列图和 DINO
frozen 数据流见
[SAM2 真实实例与 DINO frozen 流程说明](sam2_live_and_dino_frozen_validation_zh.md)。

### SAM3 协议模拟与兼容入口

先在不安装 SAM 的情况下验证 prompts、协议和确定性回退：

```bash
PYTHONPATH=src python -m tfree_detection detect \
  --input reference_scale_radial_wiener.png \
  --config configs/amorphous_sam3_prepare.yaml \
  --output outputs/reference_amorphous_sam3_candidate_local_v5
```

SAM3 真实 worker 必须放在独立 Python 3.12 环境，安装说明见
[`environments/sam3/README.md`](../environments/sam3/README.md)。配置好授权 checkpoint
或 Hugging Face 访问后：

```bash
PYTHONPATH=src python -m tfree_detection detect \
  --input reference_scale_radial_wiener.png \
  --config configs/amorphous_sam3.yaml \
  --output outputs/reference_amorphous_sam3_live_v1
```

无论选择 SAM2 还是 SAM3，`result.mask` 仍是物理 coarse candidate；使用
`result.final_geometry_mask` 读取组合后
的几何。某候选核心召回低于 0.95 或其他门失败时，只回退该候选；整图状态可为
`partially_accepted`。界面任务请使用
`configs/phase_boundary_sam3_prepare.yaml`，它会生成 domain A/domain B 成对提示。完整
协议、六面板和输出字段见
[SAM 3 边界细化说明](sam3_boundary_refinement_zh.md)。

有 Gold 边界时可量化 coarse → final 的收益：

```bash
PYTHONPATH=src python -m tfree_detection evaluate-boundary-refinement \
  --coarse coarse-mask.png \
  --final final-geometry-mask.png \
  --truth gold-mask.png \
  --output boundary-report.json
```

## 4. NData/EMD stack

多帧输入必须显式选择帧或投影：

```bash
tfree-detect detect \
  --input movie.ndata1 \
  --config configs/stacking_fault.yaml \
  --frame-index 50 \
  --output outputs/frame_50
```

或：

```bash
tfree-detect detect \
  --input movie.ndata1 \
  --config configs/stacking_fault.yaml \
  --projection median \
  --output outputs/median_projection
```

多 detector-channel EMD 先检查：

```bash
tfree-detect inspect-emd --input image.emd
```

然后用 `--dataset-path` 指定确切通道。框架不会猜测通道。

## 5. 实验 4D-STEM HDF5

当前支持 Werbrouck fast-mapping archive 的显式 HDF5 schema。先标准化为固定物理证据图，
再运行二维界面分支；框架不会按数组 shape 猜测四个轴。

```bash
python scripts/standardize_fast_4dstem.py \
  --acquisition all \
  --output-root datasets/public/werbrouck_2026_fast_4dstem_phase/standardized_v2 \
  --manifest datasets/standardized/fast_4dstem_phase_v2.json
python scripts/run_fast_4dstem_smoke_test.py \
  --manifest datasets/standardized/fast_4dstem_phase_v2.json \
  --output outputs/public_fast_4dstem_v4 \
  --gallery-copy docs/assets/fast_4dstem_test_gallery_v4.png
```

无参数的历史默认值仍按“两个公开 HDF5 中字节数较小者”选择 TiO₂；上面的当前命令显式
处理 TiO₂ 和 NMC/LGPS 两个 acquisition。原始 HDF5、派生 TIFF 和 parent lineage
分开保存。MIB、DM4 和 WSe₂ ptychography container 已完成校验、封存和安全清单，但不能
直接冒充二维图；其适配状态见
[第二波公开实验 STEM 报告](public_stem_dataset_expansion_2026_07_29_zh.md)。

## 6. Python API

```python
from tfree_detection import TrainingFreeDetectionFramework, load_config

config = load_config("configs/symmetry_relation.yaml")
framework = TrainingFreeDetectionFramework(config)
result = framework.detect("class1_image.tif")

print(result.summary())
domain = result.feature_maps["symmetry_domain_region"]
interface = result.feature_maps["interface_line"]
```

数组输入必须是二维灰度数组：

```python
result = framework.detect(image_array)
```

## 7. 验证

```bash
pytest -q
```

Class1 重点测试：

```bash
pytest -q tests/test_symmetry_group.py tests/test_symmetry_task.py
```

当前完整回归为 151 项通过。第三方库的 NumPy deprecation warning 不影响测试结论。

二维专家审核平台另有资产门禁、SQLite重载、几何和真实Streamlit提交测试。启动、备份及
训练标签导出见[154病例二维STEM在线专家审核平台](expert_review_platform_2d_zh.md)。

## 8. 数据和权重

Git 仓库不携带原始公开数据、完整预测数组或预训练权重。下载/登记脚本位于 `scripts/`，
数据分区协议位于 `datasets/`。不要把新的原始数据、`.env`、访问令牌或审阅私钥提交到
Git；`.gitignore` 已覆盖常见格式。

当前统一数据入口是
[`partition_registry_v6.yaml`](../datasets/partition_registry_v6.yaml) 和
[`sample_manifest_v7.jsonl`](../datasets/standardized/sample_manifest_v7.jsonl)。
每条 manifest 记录都包含 `data_interface`，明确 adapter、必选 channel/frame/map、
规范输出和是否可以直接进入 detector。完整字段和筛选示例见
[当前数据资产总结与统一接口](data_collection_and_interface_standardization_zh.md)。

## 9. 科学解释边界

- training-free 表示没有监督标签拟合或网络反向传播，不表示完全不做每图统计估计；
- Silver candidate 是待复核物理候选，不等于实验 Gold；
- DINO/SAM 只能作为表型或几何辅助，不能代替晶体学因果证据；
- Dennler 已经被揭盲，当前改进只能称为 post-hoc diagnostic；
- 投稿前仍需要全新 locked test、双专家 Gold、跨实验室数据和独立校准。
