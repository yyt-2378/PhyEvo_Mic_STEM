# 程序级 RSI：当前入口与历史实验

## 当前源码快照（2026-10-19）

当前框架继承的对象是 **能力库＋学生状态＋采用规则＋研究经验**。
新方案不默认替代旧框架，而是通过固定预算、逐来源保留和后续实际调用检查后纳入系统。
目录名 `v1` 是实现包路径，不能理解成只包含第一轮实验；保留路径可避免破坏旧协议和 import。

| 目的 | 当前模块 |
| --- | --- |
| 通用提案、归档与研究循环 | `contracts.py`、`archive.py`、`controller.py` |
| 模型请求与回复中转 | `provider.py`、`broker.py` |
| 受限执行与固定协议评价 | `executor.py`、`sandbox.py`、`worker.py`、`evaluation.py` |
| 系统能力库和采用规则 | `system_state.py`、`expansion_state.py`、`stability_state.py` |
| 已接纳组合的导出与回放 | `retained_portfolio.py`、`replay_retained_portfolio.py` |
| 新增能力调用其父能力与状态 | `composition_runtime.py` |
| 最新组合研究编排及显式续接 | `composition_study.py`、`composition_continuation.py` |
| 后续调用、非训练来源收益和旧能力保留 | `composition_retention_audit.py` |
| 搜索封存后的固定评价 | `composition_final_audit.py` |
| 有界存储与数值传输 | `field_storage*`、`prediction_storage*`、`storage_budget.py` |

```text
观测与专家反馈 → 系统提出机制、操作和采用条件 → 隔离执行
       ↑                                      ↓
继承能力／状态／规则／经验 ← 整体接受或拒绝 ← 固定协议评价
```

`composition_study.py` 是当前本地实验的编排器，仍含固定实验目录及外部归档来源；
它不是任意新数据的一键运行命令。现有封存目录不能继续追加搜索。
本次源码整理不读取或修改这些结果，不把历史文件组合当作一次新的执行。
新的研究运行应另行授权输入、声明协议并指定新的输出范围；当前默认检测入口不因此改变。

### 无本地结果的源码验证

仓库根目录运行（要求 Linux、bubblewrap 和已准备好的项目内 Python 环境）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venvs/sam2/bin/python scripts/check_source_release.py --profile rsi
```

此范围包含 20 个明确列出的测试文件：实际隔离执行、标签隔离、固定预算、准入、
学生状态继承、组合父节点调用、存储约束和最终封存。全部使用合成输入，
不挂载 `outputs/` 或真实标注。历史编排及本地证据检查不是该范围的一部分。
准确的测试数量、跳过与失败以该次 XML/日志为准，不能沿用旧文档的“52 项通过”。
测试定义见 `scripts/check_source_release.py` 中的 `RSI_TESTS`；不会重新训练科学模型或调用付费模型。

## 历史记录：2026-10-16 首轮程序级实验

本目录是 `RSI_iterate.md` 的新增实验实现，不替换现有 `src/`、生产配置、最佳发布或专家标注。
历史运行目录是 `outputs/rsi_program_evolution_20261016_v1/`；资产是否仍在本机以既有清理记录为准。
以下内容保留用于解释当时的协议与结果，不代表最新源码的验收数量或当前全部实验结论。

## 已接通的循环

```text
真实开发观测 → 可执行研究策略 plan(context)
             → 独立模型生成完整提案与程序
             → bwrap 断网隔离执行 fit / predict
             → 固定源划分、候选预算与评价
             → 任务准入或保留备选 → 下一轮继承程序、状态与经验

研究策略本身 → 同起点、同观测、同模型、同预算的后代实验
             → 独立元策略准入 → 下一轮实际执行获选策略
```

任务准入与研究策略准入不同：提高一次检测覆盖率，不会自动提升该候选附带的研究策略。
旧版本仍留在候选库，可作为后续父版本；程序、模型 checkpoint、经验和实测记录有各自的哈希与关联。

## 模块

| 模块 | 职责 |
| --- | --- |
| `contracts.py` / `archive.py` | 冻结协议、提案、父子关系、追加式事件和实验记忆 |
| `controller.py` | 生成／细化／组合／修复、任务准入、多轮继承 |
| `executor.py` / `sandbox.py` / `worker.py` | 受限数据视图、真实 OS 隔离、训练及推理执行 |
| `evaluation.py` | 固定面积覆盖或固定点数匹配评价；未标注区不当负例 |
| `provider.py` / `broker.py` | 原始模型请求与回复交换；不修改模型生成的算法 |
| `real_study.py` / `study_report.py` | 三组真实界面实验及逐轮报告 |
| `meta_study.py` / `meta_report.py` | 独立元策略配对、可续接请求、实际后续继承及报告 |
| `learning_study.py` | 有界真实训练、数值 checkpoint、零训练步数的精确继承验证 |
| `final_audit.py` / `final_report.py` | 联合封存、纯推理审计、逐项证据核查和中文 HTML 交付 |
| `reconcile.py` | 接回此前已经超时、但后来完整送达的原请求回复 |

## 运行约定

从仓库根目录运行，使用项目内 `.venvs/sam2/bin/python`，不要全局安装依赖。
设置 `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src`；候选默认在 CPU 执行。
需要训练时可显式指定 `--gpu N`，只挂载指定设备，不清理其他 GPU 作业。

已有 `interface_live`、`meta_live` 和 `learning_live` 是证据目录，不能覆盖、重建或删除。
它们现在已经联合封存，不能继续开发搜索；只可核验或复用已缓存的审计。
新实验需要新的精确输出目录、重新声明协议和审计来源，不能继续针对本轮审计结果选型。

任务分支从 `real_study prepare 新目录 --job 6 --rounds 3` 开始，随后对三个 arm 分别执行
`real_study advance 新目录 --arm full|generic|no_memory`。该命令会发布原始请求并等待已授权的
模型中转；不要在模型仍运行时启动重复 advance。请求在 `exchange/call-NNN/`，由下面的 broker
原样发布回复。每轮以归档事件和实际执行结果为准，不把终端断线视为模型失败。

元策略运行顺序：

1. `meta_study prepare 新目录 --source 已完成三组开发实验的目录`：重新执行共同候选，固定两种起点。
2. `meta_study queue 新目录 --key policy`：封存一次研究策略请求。
3. `broker stage 中性包目录 --source 请求目录`：将唯一请求交给不继承历史的指定模型。
4. 模型只读取该请求、写入 `raw_response.json`；`broker publish 中性包目录 --origin 独立会话名` 原样发布。
5. `meta_study evaluate 新目录 --key policy`：校验原始回复，要求任务 pipeline 完全不变，只改 `plan`。
6. 对 `p0-control`、`p0-candidate`、`p1-control`、`p1-candidate` 分别 queue、独立提案、publish、evaluate。
7. `meta_study finish 新目录`：核对配对条件和执行回执，进行元策略准入。
8. 对 `continuation` 再做一次 queue → 提案 → publish → evaluate，实际使用获选策略与任务状态。
9. `meta_report 新目录 新报告目录`：生成逐对结果和后续继承证据。

上述模块名均加前缀 `python -m experiments.rsi_program_evolution_v1.`。
模型生成可以并行；同一实验归档的 CLI 写操作通过项目内锁串行化。
这里的锁指元策略与学习/最终审计 CLI；三组任务的 advance 必须每个 arm 只有一个写入者。
`queue` 不等待模型，终端退出不会丢失请求；重试同一 slot 不增加请求。
没有新的模型回复时，不伪造回复或重新申请另一份答案。

Codex 会话的模型调度工具与兼容 HTTP API 是不同通道。此实验可由 Codex 会话中转模型请求，
不把它宣称为脱离该会话的永久独立模型服务。当前后端未返回 token／费用回执，相关字段为空。

学习和最终审计顺序：

1. `learning_study prepare 新学习目录 --source 任务实验目录 --meta 元策略实验目录`：单独声明一次学习接口验证，不追加到三组比较预算。
2. 使用 broker 发布该目录 `request/` 的一次独立模型提案。
3. `learning_study execute 新学习目录 --gpu N`：运行生成的 fit，检查实际参数更新；生成相同代码、零训练步数的继承版本，逐例核对预测。
4. 全部开发阶段结束后，`final_audit seal 总运行目录` 一次封存三组任务、元策略和学习分支。
5. `final_audit execute 总运行目录` 只推理，不改选择。重复执行只验证内容绑定的缓存。
6. `final_report 总运行目录 全新报告目录 --tests 本轮测试XML` 校验输入、归档、回复、权重和预测，生成 HTML/JSON。

最终审计的当前精确复核命令是：

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venvs/sam2/bin/python -m experiments.rsi_program_evolution_v1.final_audit execute outputs/rsi_program_evolution_20261016_v1
# 已有 final_handoff 不能覆盖；生成另一份核验快照时必须指定全新的报告目录。
```

`final_audit` 采用本次固定目录布局。模型通道未接通时停留在请求状态，不用硬编码算法冒充模型回复。

## 当前证据与结论

- 三组各 3 次任务提案均已真实执行，完整 RSI 未超过这轮两个对照；全部结果保留在 `progress_after_round3/`。
- 元策略已完成两组同预算后代实验、独立准入和一次实际后续调用；详见 `meta_live/` 及 `meta_progress/`。
- 元策略准入效用相对指定起点计算，并非相对整个候选库最优值；配对后代未超过库中已有的 38.39% 版本，因此不视为新的全局最优发现。
- 本轮实测是界面来源迁移，评价量为固定 10% 面积下专家已知正例覆盖率，不是 IoU 或正确率。
- 两组元试验是同一开发来源的两个起始程序，不作独立数据集或统计显著性解读。
- 单独的真实学习接口实验完成 32 步、65 参数的残差网络训练；20 例 checkpoint 复用预测完全一致。开发覆盖率 37.72%，低于保留版本 38.39%，因此拒绝更新。不能把损失下降写成效果提升。
- 5 个归档联合封存后完成 9 次最终审计；三组保留程序在 ZrO₂ 的覆盖率为 74.44%、72.38%、71.81%。只报告结果，没有回流选型；本轮不能声称普遍持续提升。
- 指定模型、工具与提案机会匹配；后端未返回 token/费用或逐次推理回执，因此不声称实际 token 预算完全相等。
- 52 项相关测试通过。修复、点缺陷预算评价及条件经验筛选的部分测试使用明确标记的合成输入；真实实测集中于界面来源迁移。
- 最终入口：[中文完整报告](../../outputs/rsi_program_evolution_20261016_v1/final_review/report.html)、[机器可读证据](../../outputs/rsi_program_evolution_20261016_v1/final_review/report.json)、[逐项验收](../../outputs/rsi_program_evolution_20261016_v1/final_review/requirements.json)。`final_handoff` 是先前汇总快照；`final_review` 补充了元策略效用参照的解释，实验数字没有改动。

相关测试显式选择本目录 `test_*.py`，使用仓库内**全新** `--basetemp` 和 `-p no:cacheprovider`。
不要复用已存在的 pytest basetemp，pytest 可能先删除该目录；不要运行无关稿件测试来代替框架验证。
