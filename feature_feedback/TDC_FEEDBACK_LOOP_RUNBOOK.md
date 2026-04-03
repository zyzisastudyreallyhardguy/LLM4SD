# TDC Feedback Loop Runbook

## Purpose

这个 runbook 是给 Codex 执行用的操作手册。

下次用户如果说“读这个 runbook 并把 TDC feedback loop 跑一遍”，应先阅读本文件，再按这里的顺序执行。

## Environment

- 仓库根目录：`/data1/tianang/Projects/LLM4SD`
- 如果在 `node002`，默认使用：
  `/data1/tianang/anaconda3/condabin/conda run -n vllm ...`
- 默认只跑：
  `dataset=TDC`
  `estimator=rf`
  `model=galactica-6.7b`
  `knowledge_type=synthesize`
  `num_samples=50`

## Task To Backend Mapping

全量运行时使用下面这 16 个 TDC 任务和 backend：

- `AMES` -> `ames`
- `BBB_Martins` -> `bbb_martins`
- `Bioavailability_Ma` -> `bioavailability_ma`
- `CYP2C9_Substrate_CarbonMangels` -> `cyp2c9_substrate_carbonmangels`
- `CYP2D6_Substrate_CarbonMangels` -> `cyp2d6_substrate_carbonmangels`
- `CYP3A4_Substrate_CarbonMangels` -> `cyp3a4_substrate_carbonmangels`
- `Carcinogens_Lagunin` -> `carcinogens_lagunin`
- `ClinTox` -> `clintox`
- `DILI` -> `dili`
- `HIA_Hou` -> `hia_hou`
- `PAMPA_NCATS` -> `pampa_ncats`
- `Pgp_Broccatelli` -> `pgp_broccatelli`
- `SARSCoV2_3CLPro_Diamond` -> `sarscov2_3clpro_diamond`
- `SARSCoV2_Vitro_Touret` -> `sarscov2_vitro_touret`
- `Skin_Reaction` -> `skin_reaction`
- `hERG` -> `herg`

## Core Invariants

- `analyze` 只能使用 train 错分样本。
- 不能用 valid 错分样本做 feature 修改建议。
- 不要直接修改 `codex_generated_code/`。
- 每次 feature 修改都必须先创建一个新的版本化副本 backend。
- 候选变体只有在下面三个条件都满足时才接受：
  - `mean_train_macro_f1` 提升
  - `mean_valid_macro_f1` 提升
  - `mean_valid_roc_auc` 不下降

## Hyperparameter Rule

这个 loop 会自动复用之前已经找到的最佳超参数。

具体规则：

1. `eval.py`
- 会优先读取：
  `eval_result/TDC_hyperparameter_search/<Subtask>/<Subtask>_<feature_backend>_best_params.json`
- 如果文件存在，就直接使用该 backend 对应的 tuned RF 参数。
- 如果文件缺失，就回退到 `llm4sd_models.json` 里的 BBBP 默认参数。

2. `feature_feedback_loop.py compare`
- 会先检查候选 backend 和 baseline backend 是否已经有 tuned JSON。
- 如果存在，默认直接复用。
- 只有在文件缺失，或者显式传 `--force_retune` 时，才重新运行 `tune_tdc_rf.py`。

结论：

- 全量循环默认是“尽可能复用已有最佳超参数”
- 不要默认重调参
- 只有缺失或用户明确要求时才 `--force_retune`

## Standard Full-Cycle Order

对每个任务按下面顺序执行。

### Phase 1. Baseline tuning/eval artifacts

目标：确保 baseline backend 有最新的：

- tuned params JSON
- `classification_metrics_per_seed.csv`
- `sample_predictions.csv`

步骤：

1. 检查 tuned params 是否存在
- 路径：
  `eval_result/TDC_hyperparameter_search/<Subtask>/<Subtask>_<backend>_best_params.json`

2. 如果 tuned params 缺失，先跑：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python tune_tdc_rf.py \
  --subtask <Subtask> \
  --feature_backend <backend> \
  --feature_jobs 2 \
  --rf_jobs 1
```

3. 再跑 baseline eval：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python eval.py \
  --dataset TDC \
  --subtask <Subtask> \
  --feature_backend <backend> \
  --model galactica-6.7b \
  --knowledge_type synthesize \
  --num_samples 50 \
  --feature_jobs 2 \
  --rf_jobs 1
```

4. 确认下面两个文件存在：
- `eval_result/galactica-6.7b/TDC/synthesize/galactica-6.7b_TDC_<Subtask>_synthesize_<backend>_classification_metrics_per_seed.csv`
- `eval_result/galactica-6.7b/TDC/synthesize/galactica-6.7b_TDC_<Subtask>_synthesize_<backend>_sample_predictions.csv`

### Phase 2. Train-only analyze

目标：生成 train-only 的修改证据包。

命令：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python feature_feedback_loop.py analyze \
  --subtask <Subtask> \
  --feature_backend <backend> \
  --model_name galactica-6.7b \
  --knowledge_type synthesize \
  --num_samples 50
```

输出目录：

- `feature_feedback_runs/<Subtask>/<run_id>/`

重点文件：

- `persistent_train_errors.csv`
- `train_shap_evidence.jsonl`
- `feature_pattern_summary.csv`
- `dropped_features.csv`
- `codex_edit_brief.md`

### Phase 3. Create candidate variant

目标：建立可编辑的版本化副本 backend。

默认命名：

- 第一轮候选：`<backend>_feedback_v001`
- 如果 `v001` 已存在，下一轮顺延到 `v002`、`v003` ...

命令：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python feature_feedback_loop.py init-variant \
  --base_backend <backend> \
  --variant_backend <backend>_feedback_v001 \
  --subtask <Subtask> \
  --source_run_id <analyze_run_id>
```

### Phase 4. Codex edits candidate code

目标：只修改变体副本，不碰原始 backend。

规则：

- 只读 `feature_feedback_runs/<Subtask>/<run_id>/codex_edit_brief.md`
- 只修改：
  `codex_generated_code_variants/<variant_backend>/`
- 不改：
  `codex_generated_code/`

建议做法：

- 优先处理 recurring harmful features
- 再处理 `dropped_features.csv` 里被 train 预处理整列丢掉但有价值的特征
- 不要用 valid 现象当修改依据

### Phase 5. Compare baseline vs candidate

目标：自动调参、评估并决定是否接受候选变体。

命令：

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python feature_feedback_loop.py compare \
  --subtask <Subtask> \
  --baseline_backend <backend> \
  --candidate_backend <backend>_feedback_v001 \
  --model_name galactica-6.7b \
  --knowledge_type synthesize \
  --num_samples 50 \
  --feature_jobs 2 \
  --rf_jobs 1
```

输出：

- `feature_feedback_runs/<Subtask>/<run_id>/acceptance_report.json`
- `codex_generated_code_variants/<variant_backend>/variant_metadata.json`

接受状态写在：

- `acceptance_report.json`
- `variant_metadata.json`

### Phase 6. Reporting

每个任务最终都要汇报：

- baseline backend 名称
- candidate backend 名称
- 是否 accepted
- baseline `mean_train_macro_f1`
- candidate `mean_train_macro_f1`
- baseline `mean_valid_macro_f1`
- candidate `mean_valid_macro_f1`
- baseline `mean_valid_roc_auc`
- candidate `mean_valid_roc_auc`
- 哪些 recurring harmful features 驱动了这次修改
- 是否有 dropped features 被修复或新增替代特征

## Full Sweep Strategy

如果用户要求“把所有有数据的 TDC 任务都跑一遍”，使用下面顺序：

1. 先对全部 16 个任务补 baseline eval 产物
2. 再对全部 16 个任务跑 `analyze`
3. 再为全部任务创建候选 variant
4. 再逐任务做 Codex feature edits
5. 最后逐任务跑 `compare`
6. 汇总 accepted / rejected / blocked 三类结果

不要混用“有的任务还没 baseline，有的任务已经 compare 完”的状态来做最终汇报。  
最终汇报前，至少确认全部任务都走到了下面三种状态之一：

- `accepted`
- `rejected`
- `blocked`

## Resume Rules

如果上一次运行中断，按下面顺序续跑。

### Baseline resume

如果下面文件已存在，就不要重复 baseline eval：

- `*_classification_metrics_per_seed.csv`
- `*_sample_predictions.csv`

除非：

- 用户明确要求重跑
- 代码发生了会影响 baseline 的改动

### Analyze resume

如果 `feature_feedback_runs/<Subtask>/<run_id>/analysis_metadata.json` 和 `codex_edit_brief.md` 已存在，就可以直接复用该 analyze run。

### Compare resume

如果候选 variant 已经有：

- `variant_metadata.json`
- `acceptance_report.json`

则默认把该轮视为已完成，不重复 compare，除非用户明确要求重跑。

## Parallelization Guidance

如果用户明确允许使用 subagents，可以这样分工：

- 主 agent 负责总调度、状态记录和最终汇总
- subagents 负责分任务读取 `codex_edit_brief.md` 并修改各自的 variant backend

注意：

- 不要让多个 subagent 同时修改同一个 variant 目录
- 一个 subagent 最好只拥有一个或少量任务的 variant 目录
- 主 agent 仍然负责最终读取 `acceptance_report.json` 并给用户汇报

## Final Deliverable Format

面向用户的最终汇报建议包含：

1. 全量统计
- 总任务数
- accepted 数
- rejected 数
- blocked 数

2. 每个任务一行摘要
- `<Subtask> | baseline=<backend> | candidate=<variant_backend> | accepted/rejected/blocked | valid_macro_f1 delta | valid_roc_auc delta`

3. 重点观察
- 哪些任务明显提升
- 哪些任务过拟合到 train 没过 valid gate
- 哪些类型的 feature 修改最常见
- 哪些任务反复因为 dropped features 或 NaN survival 问题受限

## Short Prompt For Reuse

下次如果用户想直接触发这套流程，可以使用类似下面的请求：

“读取 `feature_feedback/TDC_FEEDBACK_LOOP_RUNBOOK.md`，按里面的流程把所有有数据的 TDC 任务跑一遍，并汇报 accepted / rejected / blocked 结果。”
