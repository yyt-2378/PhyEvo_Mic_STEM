# Expert annotation runtime data

`review_store_v1.sqlite3` 是二维 STEM 专家平台的权威审核账本。它保存不可变修订事件和
每位专家、每个病例唯一的当前结果。该数据库、`-wal` 和 `-shm` 文件不得清理、覆盖或
提交到 Git；部署时应纳入服务器数据备份。

可随时从界面下载当前结果与完整历史 JSONL。审核阶段冻结后，用：

```bash
PYTHONPATH=src python scripts/export_expert_reviews.py \
  --reviewer-id expert-01 \
  --output datasets/expert_annotations/exports/expert-01-v1
```

导出目录拒绝覆盖同名版本。接受预测、专家修正和无缺陷结论会生成训练可用记录；拒绝、
artifact 和 unknown 会保留在审核清单中，但不会进入学生模型训练标签。
