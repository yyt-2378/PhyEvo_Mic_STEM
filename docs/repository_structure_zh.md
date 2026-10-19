# 仓库结构

## 目录树

```text
PhyEvo_Mic_STEM/
├── src/tfree_detection/       核心 Python 包
│   ├── tasks/                 非晶、点缺陷、相位线、界面和结构引导任务
│   ├── symmetry/              Class1 群作用、参考选择、多几何任务
│   ├── structure_guided/      通用投影探针、双模型比较与材料适配器
│   ├── features/              Fourier、倒易晶格、复载波特征
│   ├── stage2/                GPA、冻结 DINO 与物理约束 SAM 3 proposal 门控
│   ├── evidence.py            EvidenceBundle 和物理证据本体
│   ├── resolver.py            开放世界多假设解析
│   ├── student.py             证据保持学生模型接口
│   ├── active_learning.py     分歧驱动主动学习
│   └── cli.py                 tfree-detect 命令入口
├── configs/                   多任务可复现 YAML 配置
├── tests/                     单元、合成、协议和回归测试
├── scripts/                   数据登记、评估、可视化和审阅脚本
│   └── check_source_release.py 源码专用隔离验证；不挂载 outputs
├── experiments/rsi_program_evolution_v1/
│   ├── controller.py           程序提案、执行、评价与继承
│   ├── composition_runtime.py  组合能力及父节点状态的实际调用
│   ├── composition_study.py    最新组合继承实验编排（含本机路径）
│   └── composition_*audit.py   整体保留检查与封存后评价
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

## 脚本分层与精简原则

2026-10-19 整理只调整脚本导航与源码验证，不修改检测算法、数据、专家标注、权重或 `outputs/`。
包版本 `0.4.0` 与源码快照日期是不同概念；发布用 Git commit 固定，不能用目录名称代替代码版本。

| 层次 | 保留内容 | 使用方式 |
| --- | --- | --- |
| 日常入口 | `tfree-detect`、审核应用、`run_expert_learning_cycle.py`、`run_interface_detection.py` | 根据任务选择，不默认启动训练或服务 |
| 最新 RSI | `controller`、`composition_study`、`composition_runtime`、`composition_*audit`、`retained_portfolio` | 能力库＋学生状态＋采用规则＋经验；接口见 RSI README |
| 当前依赖 | `real_study`、`meta_study`、`learning_study`、`active_study`、`frontier_study` 等 | 最新组合入口仍复用其函数，名称旧不代表可删 |
| 历史研究 | `run_rsi_nested/refinement/continual`、`*_report.py`、早期 `*_study.py` | 复现或解释已有协议；不作为最新统一运行入口 |
| 存储迁移与续跑 | `*_continuation.py`、存储池模块 | 不自动执行；部分用于绑定旧实验，不能因名字相近合并 |
| 源码验收 | `check_source_release.py` | 分为语法、核心、审核、最新 RSI 四个检查范围 |

最新 RSI 主要入口的静态依赖闭包包含 35 个模块（含 `__init__`）。
测试夹具和历史报告还可能使用闭包外模块。因此精简优先减少用户需要选择的入口，
不会将所有闭包外源码判为无用，也不批量删除版本号较旧的文件。
源码删除须逐项确认；被保留的历史脚本不应在后台自动执行。

`check_source_release.py` 仅复制 Git 可见源码及已公开测试夹具到全新的 `.codex-tmp/source-release-*`。
它不复制或挂载 `outputs/`、`paper/`、原始数据、真实审核数据库和第三方下载目录；
源码在 bubblewrap 内只读，写入限于本次临时目录，网络和 GPU 不开放。
这让缺少本地论文材料的 GitHub 克隆也能明确区分代码测试与实验资产核验。

## 扩展一个新任务

1. 在 `src/tfree_detection/tasks/` 或独立子包实现 `DetectionTask`；
2. 在 `DetectorConfig.validate()` 注册任务名与专属配置；
3. 在 `framework.py` 注册 task class；
4. 在 `evidence.py` 定义 ontology mapping、可靠性和 hypotheses；
5. 添加干净对照、物理阳性和几何不变量测试；
6. 新建 `configs/<task>.yaml`；
7. 用新目录运行 benchmark，禁止覆盖 sealed 输出。
