from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def extract_function_names(generated_code: str) -> list[str]:
    return [
        line.split()[1].split('(')[0]
        for line in generated_code.splitlines()
        if line.startswith('def ')
    ]


def collect_feature_importance_rows(
    *,
    clf,
    feature_names: list[str],
    feature_descriptions: dict[str, str] | None,
    dataset: str,
    subtask: str,
    knowledge_type: str,
    seed: int,
    default_source: str,
    source_map: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        coefficients = getattr(clf, "coef_", None)
        if coefficients is None:
            raise ValueError(
                f"Model {type(clf).__name__} does not expose feature_importances_ or coef_."
            )
        coefficient_array = np.asarray(coefficients, dtype=float)
        if coefficient_array.ndim == 1:
            importances = np.abs(coefficient_array)
        else:
            importances = np.mean(np.abs(coefficient_array), axis=0)
    if len(importances) != len(feature_names):
        raise ValueError(
            "Feature importance length does not match feature name length: "
            f"{len(importances)} vs {len(feature_names)}"
        )

    rows = []
    source_map = source_map or {}
    feature_descriptions = feature_descriptions or {}
    for rule_name, importance in zip(feature_names, importances):
        rows.append(
            {
                "rule_name": rule_name,
                "rule_description": feature_descriptions.get(rule_name, rule_name),
                "source": source_map.get(rule_name, default_source),
                "dataset": dataset,
                "subtask": subtask,
                "knowledge_type": knowledge_type,
                "seed": seed,
                "importance": float(importance),
            }
        )
    return rows


def write_rule_importance_outputs(
    *,
    records: list[dict[str, object]],
    output_dir: str,
    result_prefix: str,
):
    if not records:
        raise ValueError("No rule importance records were provided.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    per_seed_df = pd.DataFrame(records)
    per_seed_df = per_seed_df.sort_values(
        by=["seed", "importance", "rule_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    per_seed_path = output_path / f"{result_prefix}_rule_importance_per_seed.csv"
    per_seed_df.to_csv(per_seed_path, index=False)

    summary_df = (
        per_seed_df
        .groupby(
            ["rule_name", "rule_description", "source", "dataset", "subtask", "knowledge_type"],
            dropna=False,
            as_index=False,
        )["importance"]
        .agg(mean_importance="mean", std_importance="std")
    )
    summary_df["std_importance"] = summary_df["std_importance"].fillna(0.0)
    summary_df = summary_df.sort_values(
        by=["mean_importance", "rule_name"],
        ascending=[False, True],
    ).reset_index(drop=True)
    summary_df["rank"] = summary_df.index + 1

    summary_path = output_path / f"{result_prefix}_rule_importance_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return per_seed_path, summary_path, summary_df
