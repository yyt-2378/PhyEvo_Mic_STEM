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
```

默认五个 physics teacher 不需要训练，也不需要下载神经网络权重。

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

## 3. 五个任务

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

## 5. Python API

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

## 6. 验证

```bash
pytest -q
```

Class1 重点测试：

```bash
pytest -q tests/test_symmetry_group.py tests/test_symmetry_task.py
```

当前完整回归为 97 项通过。第三方库的 NumPy deprecation warning 不影响测试结论。

## 7. 数据和权重

Git 仓库不携带原始公开数据、完整预测数组或预训练权重。下载/登记脚本位于 `scripts/`，
数据分区协议位于 `datasets/`。不要把新的原始数据、`.env`、访问令牌或审阅私钥提交到
Git；`.gitignore` 已覆盖常见格式。

## 8. 科学解释边界

- training-free 表示没有监督标签拟合或网络反向传播，不表示完全不做每图统计估计；
- Silver candidate 是待复核物理候选，不等于实验 Gold；
- DINO/SAM 只能作为表型或几何辅助，不能代替晶体学因果证据；
- Dennler 已经被揭盲，当前改进只能称为 post-hoc diagnostic；
- 投稿前仍需要全新 locked test、双专家 Gold、跨实验室数据和独立校准。
