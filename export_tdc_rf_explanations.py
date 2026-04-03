#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model_bundle import get_checkpoint_dir, list_bundle_metadata, select_best_bundle_metadata
from tree_shap_explainer import (
    explain_smiles_list_with_tree_shap,
    render_tree_shap_concise_text,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="scaffold_datasets/TDC",
        help="Root directory containing TDC scaffold split folders.",
    )
    parser.add_argument(
        "--checkpoint_root",
        default="checkpoints",
        help="Root directory containing saved model bundles.",
    )
    parser.add_argument(
        "--output_root",
        default="RF_explanations",
        help="Directory where task-level JSONL outputs will be written.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Number of top TreeSHAP features to keep per sample.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=128,
        help="Number of samples to explain per batch within each split.",
    )
    return parser.parse_args()


def list_tdc_tasks(dataset_root: Path) -> list[str]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def validate_task_checkpoints(task_names: list[str], checkpoint_root: str) -> None:
    missing_tasks = []
    for task_name in task_names:
        bundle_dir = get_checkpoint_dir(
            checkpoint_root=checkpoint_root,
            estimator="rf",
            dataset="TDC",
            subtask=task_name,
        )
        try:
            list_bundle_metadata(bundle_dir)
        except FileNotFoundError:
            missing_tasks.append(task_name)

    if missing_tasks:
        missing_text = ", ".join(missing_tasks)
        raise FileNotFoundError(
            "Missing RF checkpoints for TDC tasks: "
            f"{missing_text}. No outputs were written."
        )


def load_and_clean_split(split_path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    if not split_path.exists():
        raise FileNotFoundError(f"Missing TDC split file: {split_path}")

    frame = pd.read_csv(split_path)
    if "smiles" not in frame.columns:
        raise ValueError(f"Split file has no 'smiles' column: {split_path}")

    label_columns = [column for column in frame.columns if column != "smiles"]
    if len(label_columns) != 1:
        raise ValueError(
            f"Expected exactly one label column besides 'smiles' in {split_path}, "
            f"but found {label_columns}."
        )

    label_column = label_columns[0]
    frame = frame.copy()
    frame["smiles"] = frame["smiles"].astype(str)

    conflicting_smiles = (
        frame.groupby("smiles")[label_column]
        .nunique()
        .loc[lambda series: series > 1]
        .index.tolist()
    )
    if conflicting_smiles:
        frame = frame.loc[~frame["smiles"].isin(conflicting_smiles)].copy()

    return frame, label_column, sorted(conflicting_smiles)


def _coerce_json_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def export_task_split(
    *,
    dataset_root: Path,
    checkpoint_root: str,
    output_root: Path,
    task_name: str,
    split_name: str,
    top_k: int,
    chunk_size: int,
) -> Path:
    split_path = dataset_root / task_name / f"{task_name}_{split_name}.csv"
    frame, label_column, conflicting_smiles = load_and_clean_split(split_path)

    smiles_list = frame["smiles"].tolist()
    bundle_dir = get_checkpoint_dir(
        checkpoint_root=checkpoint_root,
        estimator="rf",
        dataset="TDC",
        subtask=task_name,
    )
    best_bundle = select_best_bundle_metadata(bundle_dir)
    bundle_path = str(best_bundle["bundle_path"])

    explanations = []
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(total=len(smiles_list), desc=f"TDC/{task_name} {split_name}", unit="sample")
    try:
        for start_index in range(0, len(smiles_list), chunk_size):
            smiles_chunk = smiles_list[start_index:start_index + chunk_size]
            chunk_explanations = explain_smiles_list_with_tree_shap(
                smiles_list=smiles_chunk,
                bundle_path=bundle_path,
                checkpoint_root=checkpoint_root,
                dataset="TDC",
                subtask=task_name,
                top_k=top_k,
            )
            explanations.extend(chunk_explanations)
            if progress_bar is not None:
                progress_bar.update(len(smiles_chunk))
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if len(explanations) != len(frame):
        raise ValueError(
            f"Explanation count mismatch for {task_name} {split_name}: "
            f"{len(explanations)} explanations for {len(frame)} rows."
        )

    task_output_dir = output_root / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = task_output_dir / f"{split_name}.jsonl"

    with output_path.open("w", encoding="utf-8") as output_file:
        for (_, row), explanation in zip(frame.iterrows(), explanations):
            record = {
                "drug": str(row["smiles"]),
                "Y": _coerce_json_scalar(row[label_column]),
                "pseudo_label": _coerce_json_scalar(explanation["predicted_class"]),
                "explanation": render_tree_shap_concise_text(explanation),
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Saved {len(frame)} rows to {output_path} "
        f"(dropped {len(conflicting_smiles)} conflicting SMILES)"
    )
    return output_path


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    if args.top_k <= 0:
        raise ValueError("--top_k must be a positive integer")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be a positive integer")

    task_names = list_tdc_tasks(dataset_root)
    validate_task_checkpoints(task_names, args.checkpoint_root)

    for task_name in task_names:
        for split_name in ("train", "valid"):
            export_task_split(
                dataset_root=dataset_root,
                checkpoint_root=args.checkpoint_root,
                output_root=output_root,
                task_name=task_name,
                split_name=split_name,
                top_k=args.top_k,
                chunk_size=args.chunk_size,
            )


if __name__ == "__main__":
    raise SystemExit(main())
