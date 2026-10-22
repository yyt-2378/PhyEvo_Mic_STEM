# PhyEvo-Mic-RSI 文档导航

本页按“先使用、再理解、最后复现实验”的顺序组织文档。仓库主入口是
[`README.md`](../README.md)。

## 快速上手

- [当前统一入口：检测、审核、学生、RSI 与评测](current_workflows_zh.md)
- [个人激活码、私有项目网页工作台与专家共享](platform_zh.md)
- [当前 PhyEvo-Mic + RSI 整体框架图](assets/phyevo_mic_rsi_framework.svg)
- [安装、输入格式、多任务命令和 Python API](usage_zh.md)
- [源码、配置、测试、数据和输出目录说明](repository_structure_zh.md)
- [当前效果、精选图、定量指标和科学边界](results_summary_zh.md)
- [GitHub 公开范围、隔离源码检查与主分支发布](publishing_zh.md)

## 当前方法

- [SAM2 真实 GPU 实例、六列图与 DINO frozen 流程](sam2_live_and_dino_frozen_validation_zh.md)
- [SAM 3 物理提示、共识、门控与确定性回退](sam3_boundary_refinement_zh.md)
- [Class1 多参考群关系、五教师与四几何输出](class1_symmetry_relation_final_zh.md)
- [Class1 失败迭代与端点假高分审计](class1_failed_iterations_archive_zh.md)
- [统一 EvidenceBundle 与开放世界 resolver](evidence_protocol_zh.md)
- [周期反事实与双教师共识](periodic_counterfactual_and_consensus_zh.md)
- [4D-STEM 三图晶格对齐与 O/Si/稀土原子位点清单](4dstem_atomic_site_inventory_zh.md)
- [FAU z-stack：CIF 骨架约束下的稀土客体与 Y 参考畸变](fau_zstack_cif_analysis_zh.md)
- [从 FAU v20 冻结流程拆分出的通用投影结构位点引擎](structure_guided_generalization_zh.md)
- [开放世界 v0.4 实现](open_world_v0_4_implementation_zh.md)

## 数据与评估

- [同数据完整模型比较、EMSeek/Dennler 与训练排除材料迁移](fair_baseline_results_zh.md)
- [当前数据资产总结与统一接口](data_collection_and_interface_standardization_zh.md)
- [2026-07-29 公开实验 STEM 第二波：下载、封存、标准化与测试](public_stem_dataset_expansion_2026_07_29_zh.md)
- [2026-07-28 公开实验 STEM 数据扩展、4D 标准化与测试](public_stem_dataset_expansion_2026_07_28_zh.md)
- [公开数据标准化状态](public_dataset_standardization_status_zh.md)
- [缺陷本体和公开数据扩展](public_data_and_defect_ontology_expansion_zh.md)
- [数据治理和 Gold calibration](data_governance_and_calibration_zh.md)
- [Dennler 锁定外测协议](locked_external_test_protocol_zh.md)
- [Dennler 锁定外测结果](locked_external_test_result_zh.md)
- [Dennler v0.4 事后诊断](dennler_posthoc_v4_diagnostic_zh.md)

## 平台与路线图

- [当前程序级 RSI：组合继承、整体准入、最新与历史入口](../experiments/rsi_program_evolution_v1/README.md)
- [脚本分层：当前入口、兼容依赖、历史研究编排](repository_structure_zh.md#脚本分层与精简原则)
- [多轮错误反馈结果、版本选择失效与研究采用决定](rsi_continual_results_zh.md)
- [四轮等步数反馈实验与内部控制协议](rsi_continual_protocol_zh.md)
- [训练改进结果、模型反馈和当前能力边界](rsi_refinement_results_zh.md)
- [六组训练对照、来源隔离和模型反馈实验协议](rsi_refinement_protocol_zh.md)
- [RSI 能力对照、同预算协同、嵌套验证与版本保留](rsi_framework_zh.md)
- [已完成专家标注的真实 GPU 学习闭环、结果与推理入口](expert_learning_cycle_zh.md)
- [当前实施状态](implementation_status_zh.md)
- [154病例二维STEM在线专家审核平台](expert_review_platform_2d_zh.md)
- [二维STEM专家审核一页操作说明](expert_annotation_quick_guide_zh.md)
- [自提升平台路线图](self_improving_platform_roadmap_zh.md)
- [非人工环节完成状态](non_human_loop_completion_status_zh.md)
- [版本迭代对比](version_iteration_comparison_zh.md)
- [专家双盲审阅工作流](expert_review_workflow_zh.md)

## 学习与设计记录

- [Learning index](learning/index.md)
- [边界候选与可靠性分离](learning/entries/2026-07-28-border-aware-candidate-reliability.md)
- [参考锚定群关系与多几何 Class1](learning/entries/2026-07-28-reference-anchored-symmetry-relations.md)
- [物理提示、提案所有权与 SAM 2/3 确定性回退](learning/entries/2026-07-29-physics-prompted-sam3-boundary-refinement.md)

## 推荐阅读顺序

```text
README
  ↓
usage_zh
  ↓
results_summary_zh
  ↓
class1_symmetry_relation_final_zh
  ↓
evidence_protocol_zh
  ↓
implementation_status_zh / roadmap
```
