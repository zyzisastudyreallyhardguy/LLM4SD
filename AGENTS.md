# AGENTS Notes

## Env Instructions

If you are on `node002`, default to the `vllm` conda environment when you need RDKit or the local project dependencies. conda is at: /data1/tianang/anaconda3/condabin/conda

## Purpose

This file records current maintenance notes for the evaluation and feature-backend stack in this repository.

## Current Pipeline

1. `synthesize.py`
Generates prior-knowledge rule text into `synthesize_model_response/`.

2. `inference.py`
Generates data-inferred rule text into `inference_model_response/`.

3. `summarize_rules.py`
Summarizes and deduplicates inference rules into `summarized_inference_rules/`.

4. `code_gen_and_eval.py`
Turns rule text into Python functions, validates them, trains/evaluates RandomForest models, and exports rule-importance CSVs.

5. `eval.py`
Runs evaluation either from generated rule functions or from a registered feature backend.

6. `tune_tdc_rf.py`
Tunes RandomForest hyperparameters for TDC classification tasks using registered feature backends and writes task/backend-specific best-parameter JSON files.

## Current Architecture

- `codex_generated_code/`
  Stores hand-built feature extractors derived from DeepResearch rule responses, such as `BBB_Martins` and `DILI`.

- `feature_backends/`
  Wraps those feature extractors behind a shared interface used by training and evaluation code.

- `eval.py`
  Loads either `generated_rules` or a registered backend from `feature_backends.registry`.

- `tune_tdc_rf.py`
  Uses the same backend names as `eval.py` and writes tuned files under:
  `eval_result/TDC_hyperparameter_search/<Subtask>/<Subtask>_<feature_backend>_best_params.json`

## Current Rule-Importance Behavior

The repository currently:

- reads `clf.feature_importances_`
- maps importances to surviving post-preprocessing feature names
- writes per-seed and aggregated CSV outputs

The repository does not yet:

- compute permutation importance
- map feature names back to original textual rule chunks
- map feature names back to original code snippets
- write a dedicated human-readable top-k report

## Output Files

Current rule-importance outputs are:

- `*_rule_importance_per_seed.csv`
- `*_rule_importance_summary.csv`

For non-generated backends such as `bbb_martins` and `dili`, the exported `source` column defaults to the backend name.

## Important Maintenance Rule

Whenever a new task is added under `codex_generated_code/`, do not stop at adding the feature module itself.

You must also check whether the new task needs updates in:

- `feature_backends/` adapters and `feature_backends/registry.py`
- `eval.py`
- `tune_tdc_rf.py`
- relevant README or AGENTS notes

In particular:

- if the new feature extractor should participate in TDC evaluation, make sure `eval.py` can load it as a backend
- if the new feature extractor should support RF hyperparameter tuning, make sure `tune_tdc_rf.py` can precompute/cache its features and write backend-specific best-parameter files

Do not assume `codex_generated_code/` changes are complete until the tuning path has been checked too.

## Current Pitfalls

### 1. Feature names are the real unit of importance

Importance is aligned to the final feature columns that survive preprocessing, not to raw rule text lines.

### 2. Surviving columns come from training data only

`prepare_feature_matrices(...)` decides the final column set from training features, then reindexes valid/test to that order.

### 3. `knowledge_type=all` preserves source provenance

When synthesize and inference code are concatenated, downstream exports preserve source labels for the surviving generated features.

### 4. TDC tuned parameters are backend-specific

For TDC tasks, `eval.py` first looks for:

- `eval_result/TDC_hyperparameter_search/<Subtask>/<Subtask>_<feature_backend>_best_params.json`

If that file is missing, evaluation falls back to the BBBP RandomForest defaults from `llm4sd_models.json`.

### 5. TDC tasks may not have a dedicated test split

Some TDC subtasks only have `train` and `valid`. `eval.py` may fall back to the validation split for test-time reporting.

## Main Places To Modify

- `/data1/tianang/Projects/LLM4SD/eval.py`
- `/data1/tianang/Projects/LLM4SD/tune_tdc_rf.py`
- `/data1/tianang/Projects/LLM4SD/feature_backends/`
- `/data1/tianang/Projects/LLM4SD/codex_generated_code/`
- `/data1/tianang/Projects/LLM4SD/rule_importance.py`

## Feedback Loop Notes

The repository now also has a train-only feature feedback loop for TDC classification:

- `feature_feedback_loop.py`
- `feature_feedback/`
- `codex_generated_code_variants/`
- `feature_feedback/TDC_FEEDBACK_LOOP_RUNBOOK.md`

Important rules:

- Analyze only `train` errors when proposing feature edits
- Use `valid` only as the acceptance gate
- Never edit `codex_generated_code/` in place when testing feedback-driven feature changes
- Store all accepted or rejected feature edits as versioned copies under `codex_generated_code_variants/`
