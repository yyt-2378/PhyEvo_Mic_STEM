# 当前工作流与历史代码分离

本页是日常使用入口，不按历史实验版本挑脚本。所有命令从仓库根目录运行，
使用已安装相应依赖的解释器；本页不启动服务、不安装环境，也不执行实验。

## 一个入口，七类操作

`scripts/phyevo.py` 只调度已有实现，不再复制一套算法。原有 CLI 继续兼容。
不带具体操作参数时显示帮助，不自动开始实验或审核服务。

| 目的 | 查看帮助 | 实际实现 |
| --- | --- | --- |
| 检测 | `python scripts/phyevo.py detect --help` | `tfree_detection.cli` 的 detect |
| 专家审核 | `python scripts/phyevo.py review --help` | 原 Streamlit 应用及 SQLite 持久化 |
| 私有项目网页平台 | `python scripts/phyevo.py platform --help` | 个人激活码、项目、分析队列与授权审核 |
| 专家监督学生 | `python scripts/phyevo.py train --help` | prepare / train / worker |
| 程序级 RSI | `python scripts/phyevo.py rsi --help` | 当前组合继承编排 |
| 论文评测 | `python scripts/phyevo.py benchmark --help` | locked / fair / author，见下节 |
| 源码验收 | `python scripts/phyevo.py check --help` | syntax / core / rsi / review / platform |

已准入界面模型另有 `python scripts/phyevo.py interface --help`，要求显式提供 binding。
完整原物理工具（FAU、参考配准等）仍由 `tfree-detect --help` 查阅，不移除这些能力。

```bash
# 只打印将要执行的命令，不打开图片、数据库或服务
python scripts/phyevo.py --dry-run detect --input example.png --output outputs/new_detection
python scripts/phyevo.py --dry-run review --server.address 127.0.0.1 --server.port 8502
```

审核服务端口、访问码及真实数据库仍由原部署配置决定。本轮没有重启服务或改动数据库。
学生训练的参数含义及来源分组规则见 [专家学习流程](expert_learning_cycle_zh.md)；
新入口不把学生训练悄悄换成其他算法。

## 当前 RSI：公共逻辑与历史编排解耦

```text
phyevo.py rsi + 显式运行配置
  └─ composition_study
       ├─ study_io：只加载已准备且绑定哈希的样本
       ├─ system_state → training_trace：检查数值学习记录
       ├─ field_storage_runtime → configured_sandbox：隔离运行配置
       └─ composition_runtime / 准入 / 继承 / retention / final audit
```

当前运行链不再导入 real_study、learning_study、meta_study、active_study、
frontier_study 或 runtime_compat 中的历史编排。旧入口暂以兼容转发复用公共函数，
不是第二份实现；测试会检查这条依赖边界。

运行位置来自 [配置模板](../configs/rsi_composition.example.json)，不再默认打开
某个日期命名的旧目录。配置使用仓库内真实子路径；配置文件本身不授予跨工作区访问权限。

- scope：已声明的新实验范围。
- source：已有保留研究状态的本地输入目录。
- source_diagnostics：与该来源绑定的开发诊断请求。
- physical_source：提供可观测晶畴字段溯源的物理分析目录。
- runtime：已有的项目内 Python 环境。
- source_aliases：训练及两个开发来源的匿名别名；不是推理时按材料挑算法的路由表。

```bash
python scripts/phyevo.py rsi report --run-config configs/my_run.json
python scripts/phyevo.py rsi request --run-config configs/my_run.json --arm full --seed 23
python scripts/phyevo.py rsi execute --run-config configs/my_run.json --arm full --seed 23 --round 1
python scripts/phyevo.py rsi retention --run-config configs/my_run.json
python scripts/phyevo.py rsi seal --run-config configs/my_run.json
python scripts/phyevo.py rsi audit --run-config configs/my_run.json
```

上述是调用形式，不是可无输入运行的演示。模板中的路径是占位符，需先准备相应资产。
prepare 仍要求 scope 中已有父系统回放回执、frozen_parent、protected_before 等输入；
本轮没有生成、迁移或重新验证这些真实实验资产。模型请求仍沿用已有中转机制，
没有新增自动常驻模型服务。

配置与当前工作流代码写入新协议，续跑时核对；不把历史结果重新封存成新版结果。
本轮保持原评价指标、候选预算和准入阈值，不进行新的科学选型。

## 论文复现与历史代码

| 层次 | 保留方式 |
| --- | --- |
| 当前工作流 | 本页入口、核心模块、必要测试 |
| 最终结果评测 | locked 外测、同数据 fair baseline、author 预训练对照及当前组合审计 |
| 旧研究版本和一次性报告 | Git 提交 `56fc212`；不作为默认入口 |

```bash
python scripts/phyevo.py benchmark locked --help
python scripts/phyevo.py benchmark fair --help
python scripts/phyevo.py benchmark author --help
```

fair / author 实际运行必须显式给出 --output，不能默认追加旧轮次。
不同基线的数据条件仍分别遵循各自冻结协议，不把作者预训练与同数据重训混为一种比较。

历史逐字节复现使用
[56fc212 冻结源码](https://github.com/yyt-2378/PhyEvo_Mic_STEM/tree/56fc212771392265c157812b50dd882c15eaf323)，
在独立目录中配合原始协议与资产进行；不能用本轮改过路径的脚本冒充原封存代码。
本轮保留原 results / seal / 数据，不为通过历史哈希检查改写它们。

## 清理状态

本轮已经做了公共函数抽取、运行位置配置化、统一入口和回归测试，不只是修改导航。
[精确删除清单](source_cleanup_plan.json)首批列出 6 个历史报告生成器及 3 个专属测试。
这 9 个文件已经逐项获批并删除，可从清单记录的 Git 提交恢复；已有报告未删除。

其余历史脚本仍在：一部分用于关键论文复现，一部分仍被历史测试或其他历史编排调用。
没有把“当前入口不导入”简单等同于“整库可直接删除”。后续按依赖组处理，不能删一半留下断链。
outputs、数据、专家标注、权重和稿件完全不在这次清理范围。

2026-10-22 经“确认清理删除”授权，已移除 10 个旧脚本及 8 个专属测试，共 18 个文件：
[逐文件记录与提交范围](source_cleanup_20261022.json)。删除前逐项复核路径、哈希与 Git 原件，
均可从提交 `199962c` 恢复。删除后按源码发布范围验证：核心 637 通过、RSI 200 通过，无失败；
原专家审核另有 45 项通过。跳过与显式排除项见发布说明。当前入口没有调用这些旧模块。
`frontier_report.py`、`bank_report.py`、`stable_report.py` 仍有历史测试或封存绑定引用，
本次不单独拆除。私有平台和 FAU 审核分别由 `platform`、`review` 入口进入。
