#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from feature_feedback import (
    is_feedback_backend_name,
    resolve_backend_source,
    resolve_variant_metadata_path,
    resolve_variant_package_dir,
)
from model_bundle import get_checkpoint_dir, list_bundle_metadata, load_model_bundle
from tree_shap_explainer import explain_smiles_list_with_tree_shap


REPO_ROOT = Path(__file__).resolve().parent
DATASET_NAME = "TDC"
ESTIMATOR_NAME = "rf"
FLOAT_TOLERANCE = 1e-6


def build_parser():
    parser = argparse.ArgumentParser(
        description="Feedback loop utilities for versioned TDC feature backends.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--subtask", required=True, help="TDC subtask name, for example DILI")
    shared.add_argument("--model_name", default="galactica-6.7b", help="Saved bundle/eval model_name")
    shared.add_argument(
        "--knowledge_type",
        default="synthesize",
        help="Saved bundle/eval knowledge_type",
    )
    shared.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Saved bundle/eval num_samples",
    )
    shared.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Checkpoint root used by eval.py",
    )
    shared.add_argument(
        "--eval_output_dir",
        default="eval_result",
        help="Evaluation output root used by eval.py",
    )
    shared.add_argument(
        "--feedback_run_root",
        default="feature_feedback_runs",
        help="Root directory for analyze/compare reports",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        parents=[shared],
        help="Analyze persistent train-set errors for one backend.",
    )
    analyze_parser.add_argument(
        "--feature_backend",
        required=True,
        help="Registered or feedback backend to analyze",
    )
    analyze_parser.add_argument(
        "--run_id",
        help="Optional explicit run id; defaults to a timestamped id",
    )
    analyze_parser.add_argument(
        "--persistent_error_threshold",
        type=int,
        default=3,
        help="Minimum number of wrong seeds required to keep a train sample",
    )
    analyze_parser.add_argument(
        "--shap_top_k",
        type=int,
        default=20,
        help="Number of top SHAP features stored per persistent error",
    )

    init_variant_parser = subparsers.add_parser(
        "init-variant",
        help="Create a versioned feature-code copy without touching codex_generated_code.",
    )
    init_variant_parser.add_argument(
        "--base_backend",
        required=True,
        help="Source backend to copy from; may be a static backend or an accepted feedback variant",
    )
    init_variant_parser.add_argument(
        "--variant_backend",
        required=True,
        help="New feedback backend name, for example dili_feedback_v001",
    )
    init_variant_parser.add_argument(
        "--subtask",
        default="",
        help="Optional TDC subtask stored in variant metadata",
    )
    init_variant_parser.add_argument(
        "--source_run_id",
        default="",
        help="Optional analyze run id recorded in variant metadata",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Tune, evaluate, and compare a candidate feedback variant against a baseline backend.",
    )
    compare_parser.add_argument("--subtask", required=True, help="TDC subtask name")
    compare_parser.add_argument("--baseline_backend", required=True, help="Baseline backend name")
    compare_parser.add_argument("--candidate_backend", required=True, help="Candidate feedback backend name")
    compare_parser.add_argument("--model_name", default="galactica-6.7b", help="eval.py --model value")
    compare_parser.add_argument("--knowledge_type", default="synthesize", help="eval.py --knowledge_type value")
    compare_parser.add_argument("--num_samples", type=int, default=50, help="eval.py --num_samples value")
    compare_parser.add_argument("--checkpoint_dir", default="checkpoints", help="eval.py --checkpoint_dir value")
    compare_parser.add_argument("--eval_output_dir", default="eval_result", help="eval.py --output_dir value")
    compare_parser.add_argument(
        "--feedback_run_root",
        default="feature_feedback_runs",
        help="Root directory for acceptance reports",
    )
    compare_parser.add_argument(
        "--tune_output_dir",
        default="eval_result/TDC_hyperparameter_search",
        help="tune_tdc_rf.py --output_dir value",
    )
    compare_parser.add_argument(
        "--precomputed_feature_dir",
        default="scaffold_datasets/TDC_precomputed_features",
        help="Feature cache directory shared by eval.py and tune_tdc_rf.py",
    )
    compare_parser.add_argument("--feature_jobs", type=int, default=4, help="Feature worker count")
    compare_parser.add_argument("--rf_jobs", type=int, default=1, help="RandomForest n_jobs")
    compare_parser.add_argument(
        "--run_id",
        help="Optional explicit compare run id; defaults to a timestamped id",
    )
    compare_parser.add_argument(
        "--force_retune",
        action="store_true",
        help="Retune RandomForest hyperparameters even if backend-specific tuned params already exist",
    )
    compare_parser.add_argument(
        "--force_recompute_features",
        action="store_true",
        help="Force recomputation of cached feature CSVs during tuning/evaluation",
    )

    return parser


def build_timestamped_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_feedback_run_dir(feedback_run_root: str, subtask: str, run_id: str) -> Path:
    run_dir = REPO_ROOT / feedback_run_root / subtask / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_eval_result_dir(eval_output_dir: str, model_name: str, knowledge_type: str, num_samples: int) -> Path:
    result_dir = REPO_ROOT / eval_output_dir / model_name / DATASET_NAME / knowledge_type
    if knowledge_type in {"inference", "all"}:
        result_dir = result_dir / f"sample_{num_samples}"
    return result_dir


def build_eval_result_prefix(
    *,
    model_name: str,
    knowledge_type: str,
    subtask: str,
    feature_backend: str,
) -> str:
    prefix = f"{model_name}_{DATASET_NAME}_{subtask}_{knowledge_type}"
    if feature_backend != "generated_rules":
        prefix += f"_{feature_backend}"
    return prefix


def get_eval_output_paths(
    *,
    eval_output_dir: str,
    model_name: str,
    knowledge_type: str,
    num_samples: int,
    subtask: str,
    feature_backend: str,
) -> tuple[Path, Path]:
    result_dir = build_eval_result_dir(eval_output_dir, model_name, knowledge_type, num_samples)
    result_prefix = build_eval_result_prefix(
        model_name=model_name,
        knowledge_type=knowledge_type,
        subtask=subtask,
        feature_backend=feature_backend,
    )
    metrics_path = result_dir / f"{result_prefix}_classification_metrics_per_seed.csv"
    predictions_path = result_dir / f"{result_prefix}_sample_predictions.csv"
    return metrics_path, predictions_path


def select_filtered_bundle_record(
    *,
    subtask: str,
    checkpoint_dir: str,
    model_name: str,
    feature_backend: str,
    knowledge_type: str,
    num_samples: int,
) -> dict[str, object]:
    bundle_dir = get_checkpoint_dir(
        checkpoint_root=checkpoint_dir,
        estimator=ESTIMATOR_NAME,
        dataset=DATASET_NAME,
        subtask=subtask,
    )
    records = list_bundle_metadata(bundle_dir)
    filtered_records = []
    for record in records:
        if record.get("model_name") != model_name:
            continue
        if record.get("feature_backend_name") != feature_backend:
            continue
        if record.get("knowledge_type") != knowledge_type:
            continue
        if int(record.get("num_samples", -1)) != num_samples:
            continue
        filtered_records.append(record)

    if not filtered_records:
        raise FileNotFoundError(
            "No checkpoint metadata matched the requested backend selection. "
            f"subtask={subtask}, model_name={model_name}, feature_backend={feature_backend}, "
            f"knowledge_type={knowledge_type}, num_samples={num_samples}"
        )

    def selection_key(record: dict[str, object]):
        metrics = dict(record.get("metrics") or {})
        metric_value = metrics.get("selection_metric_value")
        higher_is_better = bool(metrics.get("selection_higher_is_better", True))
        seed = int(record.get("seed", 0))
        if metric_value is None:
            return (0, float("-inf"), -seed)
        score = float(metric_value)
        if not higher_is_better:
            score = -score
        return (1, score, -seed)

    return max(filtered_records, key=selection_key)


def read_eval_csvs(
    *,
    eval_output_dir: str,
    model_name: str,
    knowledge_type: str,
    num_samples: int,
    subtask: str,
    feature_backend: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    metrics_path, predictions_path = get_eval_output_paths(
        eval_output_dir=eval_output_dir,
        model_name=model_name,
        knowledge_type=knowledge_type,
        num_samples=num_samples,
        subtask=subtask,
        feature_backend=feature_backend,
    )
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing classification metrics CSV: {metrics_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing sample predictions CSV: {predictions_path}")
    return (
        pd.read_csv(metrics_path),
        pd.read_csv(predictions_path),
        metrics_path,
        predictions_path,
    )


def ensure_feedback_candidate_name(candidate_backend: str):
    if not is_feedback_backend_name(candidate_backend):
        raise ValueError(
            f"Candidate backend must match '<base_backend>_feedback_v###', got {candidate_backend!r}"
        )


def build_subprocess_command(script_name: str, cli_args: list[str]) -> list[str]:
    return [sys.executable, str(REPO_ROOT / script_name), *cli_args]


def run_command(command: list[str]):
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def ensure_tuned_params(
    *,
    subtask: str,
    feature_backend: str,
    tune_output_dir: str,
    precomputed_feature_dir: str,
    feature_jobs: int,
    rf_jobs: int,
    force_retune: bool,
    force_recompute_features: bool,
) -> Path:
    tuned_path = (
        REPO_ROOT
        / tune_output_dir
        / subtask
        / f"{subtask}_{feature_backend}_best_params.json"
    )
    if tuned_path.exists() and not force_retune:
        return tuned_path

    command = build_subprocess_command(
        "tune_tdc_rf.py",
        [
            "--subtask",
            subtask,
            "--feature_backend",
            feature_backend,
            "--precomputed_feature_dir",
            precomputed_feature_dir,
            "--feature_jobs",
            str(feature_jobs),
            "--rf_jobs",
            str(rf_jobs),
            "--output_dir",
            tune_output_dir,
        ],
    )
    if force_recompute_features:
        command.append("--force_recompute_features")
    run_command(command)

    if not tuned_path.exists():
        raise FileNotFoundError(f"Expected tuned-parameter file was not created: {tuned_path}")
    return tuned_path


def run_eval_for_backend(
    *,
    subtask: str,
    feature_backend: str,
    model_name: str,
    knowledge_type: str,
    num_samples: int,
    checkpoint_dir: str,
    eval_output_dir: str,
    precomputed_feature_dir: str,
    feature_jobs: int,
    rf_jobs: int,
    force_recompute_features: bool,
):
    command = build_subprocess_command(
        "eval.py",
        [
            "--dataset",
            DATASET_NAME,
            "--subtask",
            subtask,
            "--feature_backend",
            feature_backend,
            "--model",
            model_name,
            "--knowledge_type",
            knowledge_type,
            "--num_samples",
            str(num_samples),
            "--checkpoint_dir",
            checkpoint_dir,
            "--output_dir",
            eval_output_dir,
            "--precomputed_feature_dir",
            precomputed_feature_dir,
            "--feature_jobs",
            str(feature_jobs),
            "--rf_jobs",
            str(rf_jobs),
            "--estimator",
            ESTIMATOR_NAME,
        ],
    )
    if force_recompute_features:
        command.append("--force_recompute_features")
    run_command(command)


def summarize_metric_means(metric_df: pd.DataFrame) -> dict[str, float]:
    summary = {}
    for column in [
        "train_macro_f1",
        "train_roc_auc",
        "valid_macro_f1",
        "valid_roc_auc",
        "test_macro_f1",
        "test_roc_auc",
    ]:
        summary[f"mean_{column}"] = float(metric_df[column].mean())
    return summary


def serialize_record(record: dict[str, object]) -> dict[str, object]:
    serialized = {}
    for key, value in record.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, pd.Timestamp):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized


def write_json(path: Path, payload: dict[str, object]):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def join_unique_values(values) -> str:
    unique_values = sorted({str(value) for value in values})
    return ";".join(unique_values)


def build_persistent_train_errors(
    prediction_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    persistent_error_threshold: int,
) -> pd.DataFrame:
    train_df = prediction_df[prediction_df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("No train rows were found in sample_predictions.csv")

    grouped_rows = []
    for (sample_index, smiles, y_true), group in train_df.groupby(
        ["sample_index", "smiles", "y_true"],
        dropna=False,
    ):
        total_seed_count = int(group["seed"].nunique())
        error_mask = ~group["correct"].astype(bool)
        error_count = int(error_mask.sum())
        grouped_rows.append(
            {
                "sample_index": int(sample_index),
                "smiles": str(smiles),
                "y_true": int(y_true),
                "total_seed_count": total_seed_count,
                "error_count": error_count,
                "error_rate": float(error_count / total_seed_count) if total_seed_count else 0.0,
                "wrong_seed_list": join_unique_values(group.loc[error_mask, "seed"].tolist()),
                "mean_prob_1_across_seeds": float(group["prob_1"].mean()),
                "error_type": "false_positive" if int(y_true) == 0 else "false_negative",
            }
        )

    persistent_df = pd.DataFrame(grouped_rows)
    persistent_df = persistent_df[persistent_df["error_count"] >= persistent_error_threshold].copy()
    if persistent_df.empty:
        return persistent_df

    wrong_rows = train_df[~train_df["correct"].astype(bool)].copy()
    wrong_rows = wrong_rows.merge(
        metric_df[["seed", "bundle_path", "valid_macro_f1", "valid_roc_auc"]],
        on=["seed", "bundle_path"],
        how="left",
    )
    wrong_rows = wrong_rows.sort_values(
        by=["sample_index", "valid_macro_f1", "valid_roc_auc", "seed"],
        ascending=[True, False, False, True],
    )
    selected_rows = wrong_rows.drop_duplicates(subset=["sample_index"], keep="first").rename(
        columns={
            "seed": "explanation_seed",
            "bundle_path": "explanation_bundle_path",
            "valid_macro_f1": "explanation_bundle_valid_macro_f1",
            "valid_roc_auc": "explanation_bundle_valid_roc_auc",
            "y_pred": "selected_wrong_prediction",
            "prob_1": "selected_wrong_prob_1",
        }
    )
    persistent_df = persistent_df.merge(
        selected_rows[
            [
                "sample_index",
                "explanation_seed",
                "explanation_bundle_path",
                "explanation_bundle_valid_macro_f1",
                "explanation_bundle_valid_roc_auc",
                "selected_wrong_prediction",
                "selected_wrong_prob_1",
            ]
        ],
        on="sample_index",
        how="left",
    )
    if persistent_df["explanation_bundle_path"].isna().any():
        raise ValueError("Some persistent train errors could not be matched back to a wrong-seed bundle path.")
    persistent_df = persistent_df.sort_values(
        by=["error_count", "error_rate", "sample_index"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return persistent_df


def build_dropped_features_df(bundle: dict[str, object]) -> pd.DataFrame:
    input_feature_names = list(bundle["input_feature_names"])
    surviving_feature_names = set(bundle["surviving_feature_names"])
    feature_descriptions = dict(bundle.get("feature_descriptions") or {})

    rows = []
    for feature_name in input_feature_names:
        if feature_name in surviving_feature_names:
            continue
        rows.append(
            {
                "feature_name": feature_name,
                "feature_description": feature_descriptions.get(feature_name, feature_name),
                "dropped_reason": "dropped_by_training_preprocessor_due_to_train_nan_column",
                "feature_backend": bundle["feature_backend_name"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "feature_name",
            "feature_description",
            "dropped_reason",
            "feature_backend",
        ],
    )


def build_shap_evidence_records(
    persistent_df: pd.DataFrame,
    shap_top_k: int,
) -> list[dict[str, object]]:
    evidence_records = []
    if persistent_df.empty:
        return evidence_records

    for bundle_path, group in persistent_df.groupby("explanation_bundle_path", dropna=False):
        smiles_list = group["smiles"].astype(str).tolist()
        explanations = explain_smiles_list_with_tree_shap(
            smiles_list=smiles_list,
            bundle_path=bundle_path,
            top_k=shap_top_k,
        )
        for row, explanation in zip(group.to_dict("records"), explanations):
            evidence_records.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "smiles": str(row["smiles"]),
                    "y_true": int(row["y_true"]),
                    "error_type": str(row["error_type"]),
                    "error_count": int(row["error_count"]),
                    "total_seed_count": int(row["total_seed_count"]),
                    "error_rate": float(row["error_rate"]),
                    "wrong_seed_list": str(row["wrong_seed_list"]),
                    "explanation_seed": int(row["explanation_seed"]),
                    "explanation_bundle_path": str(row["explanation_bundle_path"]),
                    "explanation_bundle_valid_macro_f1": float(row["explanation_bundle_valid_macro_f1"]),
                    "explanation_bundle_valid_roc_auc": float(row["explanation_bundle_valid_roc_auc"]),
                    "selected_wrong_prediction": int(row["selected_wrong_prediction"]),
                    "selected_wrong_prob_1": float(row["selected_wrong_prob_1"]),
                    "shap_explanation": explanation,
                }
            )
    return evidence_records


def build_feature_pattern_summary(
    evidence_records: list[dict[str, object]],
    persistent_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_columns = [
        "error_type",
        "feature_name",
        "feature_description",
        "sample_count",
        "affected_fraction",
        "mean_harmful_shap_value",
        "mean_abs_harmful_shap_value",
        "mean_raw_value",
        "mean_model_input_value",
        "example_sample_indices",
    ]
    summary_rows = []
    if not evidence_records:
        return pd.DataFrame(columns=summary_columns)

    total_counts_by_error_type = (
        persistent_df.groupby("error_type")["sample_index"].nunique().to_dict()
        if not persistent_df.empty else {}
    )
    accumulator = defaultdict(list)

    for record in evidence_records:
        error_type = record["error_type"]
        sample_index = int(record["sample_index"])
        for feature in record["shap_explanation"]["features"]:
            if float(feature["shap_value"]) <= 0.0:
                continue
            accumulator[
                (
                    error_type,
                    str(feature["feature_name"]),
                    str(feature["feature_description"]),
                )
            ].append(
                {
                    "sample_index": sample_index,
                    "shap_value": float(feature["shap_value"]),
                    "abs_shap_value": float(feature["abs_shap_value"]),
                    "raw_value": feature["raw_value"],
                    "model_input_value": float(feature["model_input_value"]),
                }
            )

    for (error_type, feature_name, feature_description), rows in accumulator.items():
        sample_indices = sorted({row["sample_index"] for row in rows})
        raw_values = [float(row["raw_value"]) for row in rows if row["raw_value"] is not None]
        summary_rows.append(
            {
                "error_type": error_type,
                "feature_name": feature_name,
                "feature_description": feature_description,
                "sample_count": len(sample_indices),
                "affected_fraction": float(
                    len(sample_indices) / max(total_counts_by_error_type.get(error_type, 1), 1)
                ),
                "mean_harmful_shap_value": float(sum(row["shap_value"] for row in rows) / len(rows)),
                "mean_abs_harmful_shap_value": float(sum(row["abs_shap_value"] for row in rows) / len(rows)),
                "mean_raw_value": float(sum(raw_values) / len(raw_values)) if raw_values else None,
                "mean_model_input_value": float(sum(row["model_input_value"] for row in rows) / len(rows)),
                "example_sample_indices": ";".join(str(value) for value in sample_indices[:10]),
            }
        )

    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(
        by=["error_type", "sample_count", "mean_harmful_shap_value", "feature_name"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def write_jsonl(path: Path, rows: list[dict[str, object]]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_codex_edit_brief(
    *,
    subtask: str,
    feature_backend: str,
    run_id: str,
    persistent_df: pd.DataFrame,
    feature_summary_df: pd.DataFrame,
    dropped_features_df: pd.DataFrame,
    reference_bundle_path: str,
    shap_evidence_path: Path,
) -> str:
    lines = [
        f"# Codex Edit Brief: {subtask} / {feature_backend}",
        "",
        f"- Analyze run id: `{run_id}`",
        f"- Reference bundle: `{reference_bundle_path}`",
        f"- SHAP evidence file: `{shap_evidence_path}`",
        "- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.",
        "- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.",
        "- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.",
        "",
        "## Persistent Train Errors",
    ]

    if persistent_df.empty:
        lines.extend(
            [
                "",
                "No persistent train errors met the configured threshold. Avoid inventing changes; rerun with a lower threshold only if you explicitly want a more exploratory loop.",
            ]
        )
    else:
        error_counts = persistent_df["error_type"].value_counts().to_dict()
        lines.extend(
            [
                "",
                f"- Persistent train errors: {len(persistent_df)}",
                f"- False positives: {int(error_counts.get('false_positive', 0))}",
                f"- False negatives: {int(error_counts.get('false_negative', 0))}",
            ]
        )

    lines.extend(["", "## Recurring Harmful Features"])
    if feature_summary_df.empty:
        lines.extend(["", "No positive-SHAP harmful feature patterns were extracted from the selected wrong-seed explanations."])
    else:
        for error_type in ["false_positive", "false_negative"]:
            subset = feature_summary_df[feature_summary_df["error_type"] == error_type].head(10)
            lines.extend(["", f"### {error_type}"])
            if subset.empty:
                lines.append("")
                lines.append("No recurring harmful features found.")
                continue
            lines.append("")
            for row in subset.to_dict("records"):
                lines.append(
                    "- "
                    f"{row['feature_name']} | samples={int(row['sample_count'])} | "
                    f"affected_fraction={row['affected_fraction']:.3f} | "
                    f"mean_harmful_shap={row['mean_harmful_shap_value']:.4f} | "
                    f"description={row['feature_description']}"
                )

    lines.extend(["", "## Dropped Features"])
    if dropped_features_df.empty:
        lines.extend(["", "No features were dropped by the train-time preprocessing step."])
    else:
        lines.append("")
        lines.append(
            f"{len(dropped_features_df)} features never survived the train preprocessor because their training columns contained NaN."
        )
        for row in dropped_features_df.head(10).to_dict("records"):
            lines.append(f"- {row['feature_name']} | {row['feature_description']}")

    lines.extend(
        [
            "",
            "## Editing Guidance",
            "",
            "- Prefer changes that address recurring train-side patterns across multiple persistent errors.",
            "- Do not justify an edit using any valid-set sample behavior.",
            "- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.",
        ]
    )
    return "\n".join(lines) + "\n"


def command_analyze(args):
    if args.feature_backend == "generated_rules":
        raise ValueError("V1 feedback-loop analyze only supports registered/static or feedback backends, not generated_rules.")

    metric_df, prediction_df, metrics_path, predictions_path = read_eval_csvs(
        eval_output_dir=args.eval_output_dir,
        model_name=args.model_name,
        knowledge_type=args.knowledge_type,
        num_samples=args.num_samples,
        subtask=args.subtask,
        feature_backend=args.feature_backend,
    )
    run_id = args.run_id or build_timestamped_run_id(f"{args.feature_backend}_analyze")
    run_dir = build_feedback_run_dir(args.feedback_run_root, args.subtask, run_id)

    persistent_df = build_persistent_train_errors(
        prediction_df=prediction_df,
        metric_df=metric_df,
        persistent_error_threshold=args.persistent_error_threshold,
    )

    selected_bundle_record = select_filtered_bundle_record(
        subtask=args.subtask,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        feature_backend=args.feature_backend,
        knowledge_type=args.knowledge_type,
        num_samples=args.num_samples,
    )
    reference_bundle = load_model_bundle(selected_bundle_record["bundle_path"])
    dropped_features_df = build_dropped_features_df(reference_bundle)
    evidence_records = build_shap_evidence_records(
        persistent_df=persistent_df,
        shap_top_k=args.shap_top_k,
    )
    feature_summary_df = build_feature_pattern_summary(evidence_records, persistent_df)

    persistent_path = run_dir / "persistent_train_errors.csv"
    shap_path = run_dir / "train_shap_evidence.jsonl"
    feature_summary_path = run_dir / "feature_pattern_summary.csv"
    dropped_features_path = run_dir / "dropped_features.csv"
    brief_path = run_dir / "codex_edit_brief.md"
    metadata_path = run_dir / "analysis_metadata.json"

    persistent_df.to_csv(persistent_path, index=False)
    write_jsonl(shap_path, evidence_records)
    feature_summary_df.to_csv(feature_summary_path, index=False)
    dropped_features_df.to_csv(dropped_features_path, index=False)
    brief_path.write_text(
        build_codex_edit_brief(
            subtask=args.subtask,
            feature_backend=args.feature_backend,
            run_id=run_id,
            persistent_df=persistent_df,
            feature_summary_df=feature_summary_df,
            dropped_features_df=dropped_features_df,
            reference_bundle_path=str(selected_bundle_record["bundle_path"]),
            shap_evidence_path=shap_path,
        ),
        encoding="utf-8",
    )
    write_json(
        metadata_path,
        {
            "dataset": DATASET_NAME,
            "subtask": args.subtask,
            "feature_backend": args.feature_backend,
            "model_name": args.model_name,
            "knowledge_type": args.knowledge_type,
            "num_samples": args.num_samples,
            "metrics_csv": str(metrics_path),
            "sample_predictions_csv": str(predictions_path),
            "selected_reference_bundle_path": str(selected_bundle_record["bundle_path"]),
            "persistent_error_threshold": args.persistent_error_threshold,
            "shap_top_k": args.shap_top_k,
            "persistent_train_error_count": int(len(persistent_df)),
        },
    )

    print(f"Saved persistent train errors to {persistent_path}")
    print(f"Saved train-only SHAP evidence to {shap_path}")
    print(f"Saved feature pattern summary to {feature_summary_path}")
    print(f"Saved dropped-feature report to {dropped_features_path}")
    print(f"Saved Codex edit brief to {brief_path}")


def command_init_variant(args):
    if args.base_backend == "generated_rules":
        raise ValueError("V1 feedback variants do not support generated_rules as a source backend.")

    ensure_feedback_candidate_name(args.variant_backend)
    source_spec = resolve_backend_source(args.base_backend)
    expected_base = args.variant_backend.split("_feedback_v", 1)[0]
    if expected_base != source_spec.base_backend:
        raise ValueError(
            f"Variant backend {args.variant_backend!r} must use base backend prefix "
            f"{source_spec.base_backend!r}, not {expected_base!r}."
        )

    variant_dir = resolve_variant_package_dir(args.variant_backend)
    if variant_dir.exists():
        raise FileExistsError(f"Variant backend directory already exists: {variant_dir}")
    variant_dir.mkdir(parents=True, exist_ok=False)

    for source_path in sorted(source_spec.package_dir.iterdir()):
        if source_path.name == "__pycache__":
            continue
        if source_path.name == "variant_metadata.json":
            continue
        target_path = variant_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)

    metadata = {
        "variant_backend": args.variant_backend,
        "parent_backend": args.base_backend,
        "base_backend": source_spec.base_backend,
        "subtask": args.subtask,
        "source_run_id": args.source_run_id,
        "status": "draft",
        "acceptance_summary": None,
        "feature_module": source_spec.feature_module_stem,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path = resolve_variant_metadata_path(args.variant_backend)
    write_json(metadata_path, metadata)

    print(f"Created variant backend directory {variant_dir}")
    print(f"Saved variant metadata to {metadata_path}")


def update_variant_metadata(
    candidate_backend: str,
    *,
    status: str,
    acceptance_summary: dict[str, object],
):
    metadata_path = resolve_variant_metadata_path(candidate_backend)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = status
    metadata["acceptance_summary"] = acceptance_summary
    metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(metadata_path, metadata)


def command_compare(args):
    if args.baseline_backend == "generated_rules" or args.candidate_backend == "generated_rules":
        raise ValueError("V1 feedback-loop compare does not support generated_rules.")

    ensure_feedback_candidate_name(args.candidate_backend)
    run_id = args.run_id or build_timestamped_run_id(f"{args.candidate_backend}_compare")
    run_dir = build_feedback_run_dir(args.feedback_run_root, args.subtask, run_id)

    baseline_tuned_path = ensure_tuned_params(
        subtask=args.subtask,
        feature_backend=args.baseline_backend,
        tune_output_dir=args.tune_output_dir,
        precomputed_feature_dir=args.precomputed_feature_dir,
        feature_jobs=args.feature_jobs,
        rf_jobs=args.rf_jobs,
        force_retune=args.force_retune,
        force_recompute_features=args.force_recompute_features,
    )
    candidate_tuned_path = ensure_tuned_params(
        subtask=args.subtask,
        feature_backend=args.candidate_backend,
        tune_output_dir=args.tune_output_dir,
        precomputed_feature_dir=args.precomputed_feature_dir,
        feature_jobs=args.feature_jobs,
        rf_jobs=args.rf_jobs,
        force_retune=args.force_retune,
        force_recompute_features=args.force_recompute_features,
    )

    run_eval_for_backend(
        subtask=args.subtask,
        feature_backend=args.baseline_backend,
        model_name=args.model_name,
        knowledge_type=args.knowledge_type,
        num_samples=args.num_samples,
        checkpoint_dir=args.checkpoint_dir,
        eval_output_dir=args.eval_output_dir,
        precomputed_feature_dir=args.precomputed_feature_dir,
        feature_jobs=args.feature_jobs,
        rf_jobs=args.rf_jobs,
        force_recompute_features=args.force_recompute_features,
    )
    run_eval_for_backend(
        subtask=args.subtask,
        feature_backend=args.candidate_backend,
        model_name=args.model_name,
        knowledge_type=args.knowledge_type,
        num_samples=args.num_samples,
        checkpoint_dir=args.checkpoint_dir,
        eval_output_dir=args.eval_output_dir,
        precomputed_feature_dir=args.precomputed_feature_dir,
        feature_jobs=args.feature_jobs,
        rf_jobs=args.rf_jobs,
        force_recompute_features=args.force_recompute_features,
    )

    baseline_metric_df, _, baseline_metrics_path, _ = read_eval_csvs(
        eval_output_dir=args.eval_output_dir,
        model_name=args.model_name,
        knowledge_type=args.knowledge_type,
        num_samples=args.num_samples,
        subtask=args.subtask,
        feature_backend=args.baseline_backend,
    )
    candidate_metric_df, _, candidate_metrics_path, _ = read_eval_csvs(
        eval_output_dir=args.eval_output_dir,
        model_name=args.model_name,
        knowledge_type=args.knowledge_type,
        num_samples=args.num_samples,
        subtask=args.subtask,
        feature_backend=args.candidate_backend,
    )

    baseline_summary = summarize_metric_means(baseline_metric_df)
    candidate_summary = summarize_metric_means(candidate_metric_df)
    acceptance_checks = {
        "train_macro_f1_improved": (
            candidate_summary["mean_train_macro_f1"] > baseline_summary["mean_train_macro_f1"] + FLOAT_TOLERANCE
        ),
        "valid_macro_f1_improved": (
            candidate_summary["mean_valid_macro_f1"] > baseline_summary["mean_valid_macro_f1"] + FLOAT_TOLERANCE
        ),
        "valid_roc_auc_not_decreased": (
            candidate_summary["mean_valid_roc_auc"] + FLOAT_TOLERANCE >= baseline_summary["mean_valid_roc_auc"]
        ),
    }
    accepted = all(acceptance_checks.values())

    report = {
        "dataset": DATASET_NAME,
        "subtask": args.subtask,
        "run_id": run_id,
        "baseline_backend": args.baseline_backend,
        "candidate_backend": args.candidate_backend,
        "model_name": args.model_name,
        "knowledge_type": args.knowledge_type,
        "num_samples": args.num_samples,
        "baseline_tuned_params_path": str(baseline_tuned_path),
        "candidate_tuned_params_path": str(candidate_tuned_path),
        "baseline_metrics_csv": str(baseline_metrics_path),
        "candidate_metrics_csv": str(candidate_metrics_path),
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "acceptance_checks": acceptance_checks,
        "accepted": accepted,
        "float_tolerance": FLOAT_TOLERANCE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    report_path = run_dir / "acceptance_report.json"
    write_json(report_path, report)
    update_variant_metadata(
        args.candidate_backend,
        status="accepted" if accepted else "rejected",
        acceptance_summary={
            "accepted": accepted,
            "report_path": str(report_path),
            "baseline_backend": args.baseline_backend,
            "baseline_summary": baseline_summary,
            "candidate_summary": candidate_summary,
        },
    )

    print(f"Saved acceptance report to {report_path}")
    print(f"Accepted: {accepted}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        command_analyze(args)
        return 0
    if args.command == "init-variant":
        command_init_variant(args)
        return 0
    if args.command == "compare":
        command_compare(args)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
