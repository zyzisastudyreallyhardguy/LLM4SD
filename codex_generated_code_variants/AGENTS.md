# AGENTS Notes For `codex_generated_code_variants`

## Purpose

这个目录存放通过 feedback loop 生成的版本化 feature extractor 副本。

## Non-Negotiable Rule

不要直接修改 `codex_generated_code/` 里的原始实现。

所有新增、删除、修改 feature 的工作，都必须发生在这里的版本化副本中。

## Naming

目录名固定为：

`<base_backend>_feedback_v###`

例如：

- `dili_feedback_v001`
- `dili_feedback_v002`

## Required Files

每个变体目录至少应包含：

- `__init__.py`
- 一个 `*_rule_features.py` 主模块
- `variant_metadata.json`

## Interface Compatibility

每个变体都必须继续暴露与原始特征模块一致的公开接口：

- `featurize_smiles(...)`
- `featurize_smiles_list(...)`
- `get_feature_names()`
- `get_feature_descriptions()`

目标是让 `eval.py`、`tune_tdc_rf.py` 和 `tree_shap_explainer.py` 可以直接通过 backend 名称复用现有流程。
