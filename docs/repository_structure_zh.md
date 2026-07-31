# 仓库结构

## 目录树

```text
PhyEvo_Mic_STEM/
├── src/tfree_detection/       核心 Python 包
│   ├── tasks/                 非晶、点缺陷、相位线、界面任务
│   ├── symmetry/              Class1 群作用、参考选择、多几何任务
│   ├── features/              Fourier、倒易晶格、复载波特征
│   ├── stage2/                GPA、冻结 DINO 与物理约束 SAM 3 proposal 门控
│   ├── evidence.py            EvidenceBundle 和物理证据本体
│   ├── resolver.py            开放世界多假设解析
│   ├── student.py             证据保持学生模型接口
│   ├── active_learning.py     分歧驱动主动学习
│   └── cli.py                 tfree-detect 命令入口
├── configs/                   五任务可复现 YAML 配置
├── tests/                     单元、合成、协议和回归测试
├── scripts/                   数据登记、评估、可视化和审阅脚本
├── environments/sam3/         独立 Python 3.12 SAM 3 环境定义
├── benchmarks/                冻结协议与轻量指标摘要
├── schemas/                   数据、证据、审阅和骨架 schema
├── datasets/                  数据注册表与轻量元数据
├── docs/                      方法、使用、评估和路线图文档
├── reference_*.png            可公开快速示例
├── plan.md                    初始设计
└── pyproject.toml             包、依赖和 CLI 定义
```

## 核心调用关系

```text
CLI / Python API
      ↓
TrainingFreeDetectionFramework
      ↓
task.run(normalized image, config, context)
      ↓
DetectionResult
      ├─ mask / score / probability
      ├─ feature_maps / evidence_maps
      ├─ multi-geometry regions
      └─ metadata + input provenance
      ↓
EvidenceBundle → resolver → reliability / abstention
      ↓
reporting / benchmark / review / student
```

SAM 3 不进入主环境依赖，调用关系是：

```text
DetectionResult.mask + score
      ↓
stage2/sam3_refine.py
      ↓ immutable file protocol
scripts/run_sam3_boundary_worker.py (Python 3.12)
      ↓ candidate × role × ensemble proposals only
candidate-local consensus + physics gates
      ↓
region geometry or domain-A/B shared-interface adapter
      ↓
accepted / partially_accepted / candidate-local coarse fallback
      ↓
final_geometry_mask → report / candidate table / expert review / evaluation
```

## Class1 模块

```text
symmetry/group.py
  ├─ 周期 motif folding
  └─ translation / rotation / mirror / sublattice alignment

symmetry/references.py
  └─ 高质量候选 → dominant identity cluster → 空间分散多参考

symmetry/task.py
  ├─ 载波频率相位梯度校正
  ├─ 五个物理教师
  ├─ 教师共识与伪影竞争
  └─ region / interface / line / point
```

## Git 跟踪边界

公开仓库跟踪：

- `src/`、`configs/`、`tests/`、`scripts/`
- 方法文档、schemas、协议 YAML/JSON
- 不含像素数据的轻量 registry/manifest
- 精选结果图与仓库架构图

本地保留但不跟踪：

- `outputs/` 完整运行结果和 `.npy/.npz`
- `datasets/public/` 原始公开数据
- `vendor/` 下载模型和第三方大文件
- benchmark predictions、private selection key、审阅临时包
- `.env`、credential、secret、缓存和虚拟环境

这条边界避免首次提交超过数 GB，也避免公开第三方数据、凭据或已封存审阅信息。

## 扩展一个新任务

1. 在 `src/tfree_detection/tasks/` 或独立子包实现 `DetectionTask`；
2. 在 `DetectorConfig.validate()` 注册任务名与专属配置；
3. 在 `framework.py` 注册 task class；
4. 在 `evidence.py` 定义 ontology mapping、可靠性和 hypotheses；
5. 添加干净对照、物理阳性和几何不变量测试；
6. 新建 `configs/<task>.yaml`；
7. 用新目录运行 benchmark，禁止覆盖 sealed 输出。
