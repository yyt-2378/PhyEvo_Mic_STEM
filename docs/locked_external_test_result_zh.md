# Dennler III–V HRSTEM 锁定外测结果

> 协议：`dennler-2021-hrstem-zero-shot-v1`  
> 预测集合：
> `f7b126bb10691d81c3e08d677325e2e36dac5d0f1619436bd9e220ef773232de`  
> 评分内容：
> `056b7c3ff5cef80ebb076d028320d04919023d38f8df7df44c2e67af5fe742d0`  
> 结论：冻结解析教师的零样本跨域迁移未达到可用水平

## 结果摘要

61 张图像的 3 个 detector 共完成 183 次冻结推理，失败数为 0。预测在读取标签前完成
SHA-256 封存，揭盲账本随后把该资产永久绑定到唯一 prediction set。

| 预注册主终点 | Pixel IoU | Precision | Recall | Dice/F1 | Normal 平均 FP 面积 |
|---|---:|---:|---:|---:|---:|
| 模糊/非晶/束损伤复合区 | 0.0118 | 0.9319 | 0.0118 | 0.0233 | 0.0000 |
| 次级晶体对称性复合区 | 0.1251 | 0.2291 | 0.2162 | 0.2224 | 0.0351 |
| 任意专家标注缺陷 | 0.1396 | 0.4522 | 0.1680 | 0.2450 | 0.0351 |

![锁定外测主指标](../benchmarks/locked_runs/dennler_v1_score/locked_primary_summary.png)

完整逐图结果保存在
[`locked_metrics.json`](../benchmarks/locked_runs/dennler_v1_score/locked_metrics.json)，
原始预注册见
[`locked_external_test_protocol_zh.md`](locked_external_test_protocol_zh.md)。

## 应该怎样解释

非晶 detector 的 0.932 precision 和 normal 图零假阳性不能单独当作成功。它只召回专家
class 2 的 1.18%，表现为明显的过度保守。该模式与“固定窗口/阈值只接受少量强异常，而
外部标签包含大面积模糊、非晶和束损伤复合区”一致，但这只是结果后的机制假设，不能用
Dennler 标签继续调参验证。

层错与相界 detector 的并集提高了覆盖，但 class 1 的 IoU 仍只有 0.125。数据真值只叫
“次级晶体对称性”，没有区分层错、晶畴和相界，所以该结果既不能证明细分类正确，也不能
把所有错位都归因于同一种缺陷。

三个任务的 reliability 均明显高于实际检测正确率。这并非数值矛盾：当前 \(q(x)\) 表示
窗口是否完整、相位/Fourier 测量是否可用等 measurement support，并未经过 Gold 标签
校准。锁定结果直接否定了“reliability 就是正确概率”的错误解释，也加强了独立 Gold
calibration 的必要性。

## 对平台结论的影响

这个负结果不删除、不替换，也不选取某几张好图重新汇报。当前可以保留的主张是：

- 平台能够严格执行预注册、无标签预测封存和一次性揭盲；
- 当前四个解析教师在开发域和合成任务上可工作；
- 当前冻结教师尚不能直接零样本覆盖外部 III–V HRSTEM 的粗粒度专家缺陷定义；
- 因此“跨材料通用检测”和“真实世界持续增精”仍未得到证据支持。

## 后续治理

1. Dennler 归档已经被消费为 v0.3 的外测，永不进入训练、校准、主动学习或阈值选择；
2. 可以把这次失败总结为未来开发假设，但改进只能使用新的 train/development 数据；
3. 改进后的模型必须另找从未查看标签的新材料/新仪器外测，不能在本归档上重新宣布提升；
4. 下一批最关键资产仍是独立 Gold calibration 和 50–100 个双专家前瞻性 pilot；
5. 在这些证据出现前，发布门禁保持“不支持跨域泛化主张”。
