#!/usr/bin/env bash

set -euo pipefail

############################################################################################################
# Evaluate saved TDC RandomForest checkpoints on the train split without retraining.
#
# Default behavior:
# - scans checkpoints/forest/TDC__*
# - evaluates the train split with eval_wo_training.py
# - uses the best checkpoint per task by saved validation metric
# - writes per-task reports under OUTPUT_DIR
# - writes a combined TSV summary under OUTPUT_DIR
#
# Optional environment variables:
#   OUTPUT_DIR=eval_result_wo_training_tdc_train
#   CHECKPOINT_SELECTION=best            # or all
#   TASK_PATTERN='TDC__*'                # glob under checkpoints/forest
#   MAX_TASKS=0                          # 0 means no limit
#   PYTHON_BIN='python'                  # override interpreter command
############################################################################################################

OUTPUT_DIR="${OUTPUT_DIR:-eval_result_wo_training_tdc_train}"
CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-best}"
TASK_PATTERN="${TASK_PATTERN:-TDC__*}"
MAX_TASKS="${MAX_TASKS:-0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  HOSTNAME_VALUE="$(hostname)"
  if [[ "${HOSTNAME_VALUE}" == "node002" ]]; then
    PYTHON_CMD=("/data1/tianang/anaconda3/condabin/conda" "run" "-n" "vllm" "python")
  else
    PYTHON_CMD=("python")
  fi
fi

SUMMARY_PATH="${OUTPUT_DIR}/tdc_train_summary.tsv"
mkdir -p "${OUTPUT_DIR}"
printf "task\tsubtask\tmodel\tknowledge_type\tfeature_backend\tcheckpoint_selection\tnum_bundles\tavg_train_roc_auc\tstd_train_roc_auc\tavg_train_macro_f1\tstd_train_macro_f1\tavg_train_accuracy\tstd_train_accuracy\treport_path\n" > "${SUMMARY_PATH}"

task_count=0
for task_dir in checkpoints/forest/${TASK_PATTERN}; do
  if [[ ! -d "${task_dir}" ]]; then
    continue
  fi

  if [[ "${MAX_TASKS}" != "0" && "${task_count}" -ge "${MAX_TASKS}" ]]; then
    break
  fi

  task_name="$(basename "${task_dir}")"
  task_count=$((task_count + 1))

  echo "================================================="
  echo "Evaluating ${task_name} on train split"

  run_output="$("${PYTHON_CMD[@]}" eval_wo_training.py \
    --bundle_path "${task_dir}" \
    --eval_split train \
    --checkpoint_selection "${CHECKPOINT_SELECTION}" \
    --output_dir "${OUTPUT_DIR}" 2>&1)"

  printf '%s\n' "${run_output}"

  report_path="$(printf '%s\n' "${run_output}" | awk -F'to ' '/Saved evaluation report to /{print $2}' | tail -n 1)"
  if [[ -z "${report_path}" || ! -f "${report_path}" ]]; then
    echo "Failed to locate report for ${task_name}" >&2
    exit 1
  fi

  subtask="$(awk -F': ' '/^subtask: /{print $2}' "${report_path}")"
  model="$(awk -F': ' '/^model: /{print $2}' "${report_path}")"
  knowledge_type="$(awk -F': ' '/^knowledge_type: /{print $2}' "${report_path}")"
  feature_backend="$(awk -F': ' '/^feature_backend: /{print $2}' "${report_path}")"
  num_bundles="$(awk -F': ' '/^num_bundles: /{print $2}' "${report_path}")"
  avg_train_roc_auc="$(awk -F': ' '/^average_train_roc_auc: /{print $2}' "${report_path}")"
  std_train_roc_auc="$(awk -F': ' '/^std_train_roc_auc: /{print $2}' "${report_path}")"
  avg_train_macro_f1="$(awk -F': ' '/^average_train_macro_f1: /{print $2}' "${report_path}")"
  std_train_macro_f1="$(awk -F': ' '/^std_train_macro_f1: /{print $2}' "${report_path}")"
  avg_train_accuracy="$(awk -F': ' '/^average_train_accuracy: /{print $2}' "${report_path}")"
  std_train_accuracy="$(awk -F': ' '/^std_train_accuracy: /{print $2}' "${report_path}")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task_name}" \
    "${subtask}" \
    "${model}" \
    "${knowledge_type}" \
    "${feature_backend}" \
    "${CHECKPOINT_SELECTION}" \
    "${num_bundles}" \
    "${avg_train_roc_auc}" \
    "${std_train_roc_auc}" \
    "${avg_train_macro_f1}" \
    "${std_train_macro_f1}" \
    "${avg_train_accuracy}" \
    "${std_train_accuracy}" \
    "${report_path}" >> "${SUMMARY_PATH}"
done

echo "================================================="
echo "Finished evaluating ${task_count} TDC task(s)."
echo "Combined summary saved to ${SUMMARY_PATH}"
