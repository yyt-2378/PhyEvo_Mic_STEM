# 当前数据资产总结与统一接口

审计日期：2026-07-29

## 一句话结论

当前数据层已经从“若干文件夹和来源链接”收口成三层可审计资产：

1. 23 个公开来源的候选目录，记录许可、固定来源、规模和准入状态；
2. 12 个已纳管数据集、313 个 checksum 固定资产，共 2,885,299,193 bytes
   （2.687 GiB）；
3. 313 条逐资产统一访问契约，明确格式、角色、选择参数、adapter、规范输出和当前
   detector 可运行性。

当前有 155 个资产的数据读取 adapter 已就绪，其中 141 个连同任务映射一起可以直接进入
现有 detector。另有 3 个需要显式选通道或帧、2 个需要先派生 4D-STEM 物理证据图，
6 个 ZIP 和 1 个 DM4 仍需来源专用 adapter。

这仍然不是“313 个独立带 Gold 标注的图像”。资产中包含图像、作者标注、原始容器、
元数据和派生物理图；`calibration` 分区仍为 0。

## 权威入口

- [partition_registry_v6.yaml](../datasets/partition_registry_v6.yaml)：
  acquisition-safe 分区、任务、材料、父子关系和 SHA-256；
- [sample_manifest_v7.jsonl](../datasets/standardized/sample_manifest_v7.jsonl)：
  当前逐资产数据接口；
- [sample_manifest_v7.summary.json](../datasets/standardized/sample_manifest_v7.summary.json)：
  数据量、材料、任务、格式和接口状态汇总；
- [public_dataset_operational_status_v6.json](../datasets/intake/public_dataset_operational_status_v6.json)：
  23 个公开来源的实际执行状态；
- [public_dataset_candidate_registry_v1.yaml](../datasets/intake/public_dataset_candidate_registry_v1.yaml)：
  来源、许可、优先级、限制和后续动作；
- [public_dataset_wave_v1.json](../datasets/standardized/public_dataset_wave_v1.json)：
  WSe₂、SrTiO₃、辐照 Al 和聚合物大型容器的非解压清单。

`sample_manifest_v5` 仍是旧接口的历史快照。v6 是本次接口迁移时生成的中间审计版本，
其记录内容有效，但 summary 使用了语义偏宽的 `total_source_bytes` 字段；因此当前统一
使用字段已更正为 `total_registered_bytes` 的 v7。旧文件没有删除或覆盖。

## 收集规模

### 公开来源目录

候选目录共 23 个来源：

| 当前状态 | 来源数 | 含义 |
|---|---:|---|
| 已纳管 | 5 | 本地文件和正式 registry 均存在 |
| 完整公开记录已纳管 | 1 | Fast 4D-STEM 完整记录 |
| raw 1657 + 坐标已纳管 | 1 | WSe₂ 3D defects 当前固定 acquisition |
| 远程范围冻结、局部纳管 | 1 | EMSeek 本地均衡子集 |
| 已消费 locked test | 1 | Dennler，只保留历史评估 |
| 普通候选 | 3 | 尚未正式准入 |
| 大型候选 | 7 | 需先冻结抽样或流式策略 |
| 许可隔离 | 4 | 数据许可未确认，不进入正式数据树 |

19/23 个来源已确认数据许可，9 个来源至少有资产进入 registry。这里的“进入 registry”
不代表有实验 Gold，也不代表原始容器已经有可执行 adapter。

### 正式 registry

| 维度 | 当前值 |
|---|---:|
| 数据集 ID | 12 |
| 注册资产 | 313 |
| 注册文件总量 | 2,885,299,193 bytes / 2.687 GiB |
| 材料值 | 29 |
| 任务标签 | 31 |
| `train` | 280 |
| `development_test` | 32 |
| `locked_external_test` | 1 |
| `calibration` | 0 |

资产角色：

| 角色 | 数量 | 解释 |
|---|---:|---|
| image | 161 | 二维图、stack、4D-STEM HDF5 或待适配 DM4 |
| annotation | 139 | 作者标注或标签参考，不自动视为 Gold |
| container | 6 | ZIP 等原始归档，不能直接当二维图 |
| metadata | 7 | README、source manifest、notebook、EBSD 或 calibration |

格式覆盖为 PNG 279、TIFF 14、ZIP 6、NData 2、EMD 2、显式 4D-STEM HDF5 2，
通用 HDF5 1、DM4 1，以及少量 JSON、Markdown、OSC 和 notebook。

任务标签允许重叠，所以不能相加当成样本数。较大的标签计数包括：
`atomic_point_defect=219`、`atomic_column_localization=33`、
`atomic_site_anomaly=32`、`phase_boundary=12`、`amorphous_region=10`、
`stacking_fault=4`；新增 4D 数据还覆盖 reciprocal-space anomaly、strain、
orientation、moire、ptychography 和 irradiation。

## 12 个已纳管数据集

| 数据集 | 资产 / 注册 MiB | 主要内容 | 当前接口状态 |
|---|---:|---|---|
| EMSeek WSe₂ | 215 / 87.94 | 107 图 + 107 作者标注 + provenance | 107 图可直接读；标注为 reference |
| EMSeek 均衡子集 | 65 / 62.77 | 16 个材料/场景组，32 图 + 32 标注 | 32 图可读，18 图已有 detector 任务映射 |
| Guo 静态原子图 | 3 / 11.22 | MoWTe₂、Al 位错/层错、ZrO₂ 界面 | 3/3 可直接运行 |
| Guo SuperScan | 1 / 100.00 | 100 帧 acquisition | 必须显式选帧或 projection |
| 本地分子筛 | 1 / 1.26 | 拉伸后局部非晶参考 | 可直接运行 |
| Al Σ3 | 5 / 265.54 | 两个多通道 EMD、EBSD、simulation、README | EMD 必须选 channel 和 plane |
| Werbrouck Fast 4D-STEM | 14 / 287.85 | TiO₂/SiO₂ 与 NMC/LGPS；2 raw + 12 物理图 | 12 图直接运行；2 raw 先派生 evidence map |
| WSe₂ 3D defects | 3 / 806.90 | 1657 raw、重建和作者坐标 | 已封存；需 EMPAD/MAT/坐标配准 adapter |
| SrTiO₃ 4D-STEM | 2 / 168.93 | 64 个 MIB 分片归档 + notebook | 需流式 MIB virtual-detector adapter |
| 辐照 Al 4D-STEM | 2 / 814.28 | DM4 raw + py4DSTEM calibration H5 | H5 已列树；DM4 adapter 未完成 |
| 聚合物室温 4D-STEM | 1 / 84.27 | ZIP 内 10.75 GB 逻辑 DM4 | 禁止批量解压；需流式 orientation adapter |
| Dennler HRSTEM | 1 / 60.68 | 已揭盲的历史分类归档 | consumed locked test，不得回流开发 |

“注册 MiB”包含原始图、标注、元数据和已登记派生图，不等同于纯 raw 数据量。

## 统一数据接口 v1

接口没有强行把所有文件转成同一种物理格式，而是在每条记录增加
`data_interface`：

```json
{
  "interface_format": "tfree-data-interface-v1",
  "access_mode": "select_channel_and_plane",
  "adapter": "tfree_detection.preprocessing.load_grayscale",
  "adapter_status": "selector_required",
  "required_selectors": [
    {"name": "dataset_path", "constraint": "exactly_one"},
    {
      "name": "plane_selection",
      "parameters": ["frame_index", "projection"],
      "constraint": "exactly_one"
    }
  ],
  "canonical_output": {
    "kind": "image_2d",
    "axes": ["y", "x"],
    "dtype": "float32"
  },
  "source_policy": "read_only_checksum_anchored",
  "ready_for_detector_now": false
}
```

`canonical_output.dtype=float32` 指进入 framework 的规范化 detector 表示；loader 会先保留
原始数值，随后由现有 robust preprocessing 转为这一表示。原始 dtype 和 shape 仍记录在
`inspection`，不会被覆盖。

### 路由规则

| `access_mode` | 数量 | 输入 | 必须提供 | 规范输出 |
|---|---:|---|---|---|
| `direct_2d` | 155 | PNG/TIFF 或二维 NData | 无 | `image_2d[y,x]` |
| `select_plane` | 1 | NData stack | `frame_index` 或 projection 二选一 | `image_2d[y,x]` |
| `select_channel_and_plane` | 2 | 多通道 EMD stack | `dataset_path`，再选 frame/projection | `image_2d[y,x]` |
| `derive_evidence_map` | 2 | 已知 schema 的 4D HDF5 | 六种物理 map 选一种 | `evidence_map_2d[scan_x,scan_y]` |
| `annotation_reference` | 139 | 作者标注 | source-specific label adapter | annotation reference |
| `metadata_only` | 7 | calibration/README/EBSD/notebook | 不进入 detector | metadata |
| `container_adapter_required` | 6 | ZIP | 来源专用、流式 adapter | container members |
| `adapter_required` | 1 | DM4 | DM4 4D-STEM adapter | 尚未定义 |

接口状态汇总：

- `ready=155`：数据读取路径已经明确；
- `selector_required=3`：代码支持，但必须由调用者明确选择，框架不会猜；
- `derivation_required=2`：已有严格 HDF5 adapter，要先产生物理证据图；
- `adapter_required=7`：6 个容器和 1 个 DM4 仍需代码；
- `reference_only=139`：标签只作为来源证据；
- `metadata_only=7`：保留但不作为图像输入。

155 个可读取资产中只有 141 个标为 `ready_for_detector_now=true`。差出的 14 个主要是
现有平台尚无对应 detector 任务映射的形貌/载体界面图，不能仅因 PNG 能读就宣称能检测。

## 如何使用统一清单

重新生成 append-only manifest：

```bash
python scripts/build_sample_manifest.py \
  --registry datasets/partition_registry_v6.yaml \
  --output datasets/standardized/sample_manifest_v8.jsonl \
  --summary datasets/standardized/sample_manifest_v8.summary.json
```

输出文件名必须是新版本，脚本拒绝覆盖旧 manifest。

筛选当前可以直接运行的图像：

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = Path("datasets/standardized/sample_manifest_v7.jsonl")
for line in manifest.open(encoding="utf-8"):
    record = json.loads(line)
    if record["data_interface"]["ready_for_detector_now"]:
        print(record["asset"]["asset_id"], record["asset"]["relative_path"])
PY
```

查找仍需 adapter 的大型数据：

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = Path("datasets/standardized/sample_manifest_v7.jsonl")
for line in manifest.open(encoding="utf-8"):
    record = json.loads(line)
    if record["data_interface"]["adapter_status"] == "adapter_required":
        print(
            record["asset"]["dataset_id"],
            record["inspection"]["format"],
            record["asset"]["relative_path"],
        )
PY
```

## 仍然缺什么

代码侧最明确的 adapter 队列是：

1. 辐照 Al DM4 + calibration 的绑定读取；
2. SrTiO₃ MIB 分片的流式 virtual detector；
3. WSe₂ EMPAD raw、MAT 重建和坐标的方向/原点配准；
4. 聚合物高压缩比 DM4 的受控流式 orientation map；
5. 把当前没有 detector 映射的粒子/载体界面任务接入开放世界几何输出。

不能由数据接口代码自动补齐的核心仍是：

- acquisition-aware 的实验 Gold calibration；
- 两位专家独立标注与共识；
- 一套从未用于开发的新 locked external test；
- 跨实验室采集条件和物理语义复核。

因此当前最准确的表述是：**来源和资产治理已经系统化，二维与已知 HDF5 的统一访问已可用，
大型原始 4D 容器进入可追踪的 adapter 队列；实验 Gold 与真正独立的外测仍未闭环。**
