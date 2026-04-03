# AGENTS Notes For `feature_feedback`

## Purpose

这个目录存放 feature feedback loop 的辅助逻辑与约定。

## Core Rule

训练集分析和验证集验收必须严格分开：

- `analyze` 只能使用 `train` 错分样本
- `valid` 只能用于接受/拒绝 feature 变体

不要把 `valid` 错分样本、`valid` SHAP 解释或 `valid` 的单样本现象写进 feature 修改建议。

## Variant Policy

- 原始 `codex_generated_code/` 里的特征代码不能直接修改
- 每次 feature 改动都必须创建一个新的版本化副本 backend
- 变体命名固定为 `<base_backend>_feedback_v###`
- 变体目录里必须保留与原模块相同的公开接口

## Acceptance Policy

候选变体只有在下面三个条件同时满足时才算 accepted：

- `mean_train_macro_f1` 提升
- `mean_valid_macro_f1` 提升
- `mean_valid_roc_auc` 不下降

如果不满足，状态应标记为 `rejected`，但代码和报告要保留。

## Runbook Pointer

如果要真正执行一次全量或单任务 feedback loop，不要只看这个 AGENTS 文件。

先读：

- `/data1/tianang/Projects/LLM4SD/feature_feedback/TDC_FEEDBACK_LOOP_RUNBOOK.md`

那个文件是面向 Codex 执行的详细操作手册，包含：

- TDC 任务与 backend 对应关系
- 全量运行顺序
- 中断后如何续跑
- 何时自动复用已有最佳超参数
- 最终汇报时应总结哪些结果
