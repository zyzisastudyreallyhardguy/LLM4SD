from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from feature_backends import load_feature_backend, transform_feature_frame
from model_bundle import (
    get_checkpoint_dir,
    list_bundle_metadata,
    load_model_bundle,
)
from rule_importance import extract_function_names


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "NA"
    numeric_value = float(value)
    if np.isclose(numeric_value, round(numeric_value)):
        return str(int(round(numeric_value)))
    return f"{numeric_value:.3f}"


def _resolve_dataset_split_path(dataset: str, subtask: str, split: str) -> Path:
    if dataset == "TDC":
        if not subtask:
            raise ValueError("TDC dataset requires a subtask such as DILI.")
        split_path = Path("scaffold_datasets") / "TDC" / subtask / f"{subtask}_{split}.csv"
        if split == "test" and not split_path.exists():
            fallback_path = Path("scaffold_datasets") / "TDC" / subtask / f"{subtask}_valid.csv"
            if fallback_path.exists():
                return fallback_path
        return split_path

    dataset_dir = Path("scaffold_datasets") / dataset
    file_stem = dataset if not subtask else subtask
    return dataset_dir / f"{file_stem}_{split}.csv"


def load_dataset_split_frame(
    *,
    dataset: str,
    subtask: str = "",
    split: str,
) -> pd.DataFrame:
    split_path = _resolve_dataset_split_path(dataset, subtask, split)
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find dataset split file: {split_path}")
    return pd.read_csv(split_path)


def get_dataset_sample(
    *,
    dataset: str,
    subtask: str = "",
    split: str,
    sample_index: int,
) -> dict[str, object]:
    split_df = load_dataset_split_frame(dataset=dataset, subtask=subtask, split=split)
    if sample_index < 0 or sample_index >= len(split_df):
        raise IndexError(
            f"sample_index {sample_index} is out of range for {dataset}/{subtask or dataset} "
            f"{split} split with {len(split_df)} rows."
        )

    smiles_column = "mol" if dataset == "bace" else "smiles"
    if smiles_column not in split_df.columns:
        raise ValueError(f"Dataset split {dataset}/{subtask or dataset}/{split} has no {smiles_column!r} column.")

    row = split_df.iloc[sample_index]
    return {
        "dataset": dataset,
        "subtask": subtask,
        "split": split,
        "sample_index": sample_index,
        "smiles": str(row[smiles_column]),
        "row": row.to_dict(),
    }


def _resolve_bundle_path(
    *,
    bundle_path: str | Path | None = None,
    checkpoint_root: str = "checkpoints",
    estimator: str = "rf",
    dataset: str | None = None,
    subtask: str = "",
    model_name: str | None = None,
    feature_backend: str | None = None,
    knowledge_type: str | None = None,
    num_samples: int | None = None,
) -> Path:
    if bundle_path is not None:
        bundle_path = Path(bundle_path)
        if bundle_path.is_file():
            return bundle_path
        if bundle_path.is_dir():
            metadata_records = list_bundle_metadata(bundle_path)
        else:
            raise FileNotFoundError(f"Bundle path does not exist: {bundle_path}")
    else:
        if dataset is None:
            raise ValueError("dataset is required when bundle_path is not provided.")
        bundle_dir = get_checkpoint_dir(
            checkpoint_root=checkpoint_root,
            estimator=estimator,
            dataset=dataset,
            subtask=subtask,
        )
        metadata_records = list_bundle_metadata(bundle_dir)

    filtered_records = []
    for record in metadata_records:
        if model_name is not None and record.get("model_name") != model_name:
            continue
        if feature_backend is not None and record.get("feature_backend_name") != feature_backend:
            continue
        if knowledge_type is not None and record.get("knowledge_type") != knowledge_type:
            continue
        if num_samples is not None and int(record.get("num_samples", -1)) != num_samples:
            continue
        filtered_records.append(record)

    if not filtered_records:
        raise FileNotFoundError(
            "No checkpoint metadata matched the requested filters. "
            f"model_name={model_name!r}, feature_backend={feature_backend!r}, "
            f"knowledge_type={knowledge_type!r}, num_samples={num_samples!r}"
        )

    def _selection_key(record: dict[str, object]):
        metrics = dict(record.get("metrics") or {})
        metric_name = metrics.get("selection_metric_name")
        metric_value = metrics.get("selection_metric_value")
        higher_is_better = bool(metrics.get("selection_higher_is_better", True))
        seed = int(record.get("seed", 0))

        if metric_name is None or metric_value is None:
            return (0, float("-inf"), -seed)

        score = float(metric_value)
        if not higher_is_better:
            score = -score
        return (1, score, -seed)

    best_record = max(filtered_records, key=_selection_key)
    return Path(str(best_record["bundle_path"]))


def _load_shap_module():
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "The shap package is required for TreeSHAP explanations. "
            "Please install shap in the active environment."
        ) from exc
    return shap


def _build_backend_from_bundle(bundle: dict[str, object]):
    backend_name = str(bundle["feature_backend_name"])
    if backend_name != "generated_rules":
        return load_feature_backend(backend_name)

    backend_payload = dict(bundle["backend_payload"])
    generated_code = str(backend_payload["generated_code"])
    namespace: dict[str, object] = {}
    exec(generated_code, namespace)
    function_names = list(backend_payload.get("function_names") or extract_function_names(generated_code))
    return load_feature_backend(
        backend_name,
        function_names=function_names,
        namespace=namespace,
    )


def _prepare_single_smiles_inputs(
    smiles: str,
    bundle: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_surviving_df, transformed_df = _prepare_smiles_list_inputs([smiles], bundle)
    return raw_surviving_df, transformed_df


def _prepare_smiles_list_inputs(
    smiles_list: list[str],
    bundle: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    backend = _build_backend_from_bundle(bundle)
    if hasattr(backend, "featurize_smiles_list"):
        raw_feature_df = backend.featurize_smiles_list(smiles_list, on_error="raise")
    else:
        raw_feature_df = pd.DataFrame([backend.featurize_smiles(smiles) for smiles in smiles_list])

    input_feature_names = list(bundle["input_feature_names"])
    raw_feature_df = raw_feature_df.reindex(columns=input_feature_names)
    raw_surviving_df, transformed_df = transform_feature_frame(
        raw_feature_df,
        bundle["preprocessor"],
    )
    return raw_surviving_df, transformed_df


def _select_output_vector(values: np.ndarray, class_index: int) -> np.ndarray:
    return _select_output_matrix(values, [class_index])[0]


def _select_output_matrix(values: np.ndarray, class_indices: list[int]) -> np.ndarray:
    num_rows = len(class_indices)
    if values.ndim == 1:
        if num_rows != 1:
            raise ValueError(f"Expected one class index for 1D SHAP values, got {num_rows}.")
        return values.reshape(1, -1)
    if values.ndim == 2:
        if values.shape[0] != num_rows:
            if num_rows == 1:
                return values[[0], :]
            raise ValueError(
                f"2D SHAP values row count {values.shape[0]} does not match class index count {num_rows}."
            )
        return values
    if values.ndim == 3:
        if values.shape[0] == num_rows:
            return np.stack(
                [values[row_index, :, class_indices[row_index]] for row_index in range(num_rows)],
                axis=0,
            )
        if values.shape[1] == num_rows:
            return np.stack(
                [values[class_indices[row_index], row_index, :] for row_index in range(num_rows)],
                axis=0,
            )
    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


def _select_base_value(base_values: np.ndarray, class_index: int) -> float:
    return float(_select_base_values(base_values, [class_index])[0])


def _select_base_values(base_values: np.ndarray, class_indices: list[int]) -> np.ndarray:
    num_rows = len(class_indices)
    if base_values.ndim == 0:
        return np.full(num_rows, float(base_values))
    if base_values.ndim == 1:
        if len(base_values) == 1:
            return np.full(num_rows, float(base_values[0]))
        if num_rows == 1:
            return np.asarray([float(base_values[class_indices[0]])])
        if len(base_values) == num_rows:
            return np.asarray(base_values, dtype=float)
        raise ValueError(
            f"Unsupported 1D SHAP base value length {len(base_values)} for {num_rows} rows."
        )
    if base_values.ndim == 2:
        if base_values.shape[0] == num_rows:
            return np.asarray(
                [float(base_values[row_index, class_indices[row_index]]) for row_index in range(num_rows)],
                dtype=float,
            )
        if base_values.shape[1] == num_rows:
            return np.asarray(
                [float(base_values[class_indices[row_index], row_index]) for row_index in range(num_rows)],
                dtype=float,
            )
    raise ValueError(f"Unsupported SHAP base value shape: {base_values.shape}")


def _coerce_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_json_scalar(value):
    value = _coerce_python_scalar(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _extract_model_outputs(
    model,
    transformed_array: np.ndarray,
) -> tuple[np.ndarray | None, list[object] | None, list[list[float]] | None, list[object] | None]:
    predicted_raw = None
    predicted_classes = None
    predicted_probabilities = None
    model_classes = None

    if hasattr(model, "predict"):
        predicted_raw = np.asarray(model.predict(transformed_array))
        predicted_classes = [_coerce_python_scalar(value) for value in predicted_raw.tolist()]
    if hasattr(model, "predict_proba"):
        probability_rows = np.asarray(model.predict_proba(transformed_array))
        predicted_probabilities = [
            [float(probability) for probability in probability_row]
            for probability_row in probability_rows.tolist()
        ]
    if hasattr(model, "classes_"):
        model_classes = [_coerce_python_scalar(value) for value in model.classes_.tolist()]

    return predicted_raw, predicted_classes, predicted_probabilities, model_classes


def _resolve_class_indices(
    *,
    class_index: int | None,
    predicted_classes: list[object] | None,
    model_classes: list[object] | None,
    num_rows: int,
) -> list[int]:
    if class_index is not None:
        return [class_index] * num_rows

    if predicted_classes is not None and model_classes is not None:
        class_indices = []
        for predicted_class in predicted_classes:
            if predicted_class in model_classes:
                class_indices.append(model_classes.index(predicted_class))
            else:
                class_indices.append(1)
        return class_indices

    return [1] * num_rows


def _build_feature_rows(
    *,
    raw_row: pd.Series,
    transformed_row: pd.Series,
    shap_values: np.ndarray,
    feature_descriptions: dict[str, str],
    top_k: int | None,
) -> list[dict[str, object]]:
    features = []
    for feature_name, shap_value in zip(transformed_row.index.tolist(), shap_values.tolist()):
        features.append(
            {
                "feature_name": feature_name,
                "feature_description": feature_descriptions.get(feature_name, feature_name),
                "raw_value": None if pd.isna(raw_row[feature_name]) else float(raw_row[feature_name]),
                "model_input_value": float(transformed_row[feature_name]),
                "shap_value": float(shap_value),
                "abs_shap_value": abs(float(shap_value)),
            }
        )
    features.sort(key=lambda row: row["abs_shap_value"], reverse=True)
    if top_k is not None:
        features = features[:top_k]
    return features


def explain_smiles_list_with_tree_shap(
    *,
    smiles_list: list[str],
    bundle_path: str | Path | None = None,
    checkpoint_root: str = "checkpoints",
    estimator: str = "rf",
    dataset: str | None = None,
    subtask: str = "",
    model_name: str | None = None,
    feature_backend: str | None = None,
    knowledge_type: str | None = None,
    num_samples: int | None = None,
    class_index: int | None = None,
    top_k: int | None = None,
) -> list[dict[str, object]]:
    if not smiles_list:
        return []

    bundle_path = _resolve_bundle_path(
        bundle_path=bundle_path,
        checkpoint_root=checkpoint_root,
        estimator=estimator,
        dataset=dataset,
        subtask=subtask,
        model_name=model_name,
        feature_backend=feature_backend,
        knowledge_type=knowledge_type,
        num_samples=num_samples,
    )
    bundle = load_model_bundle(bundle_path)
    if str(bundle["estimator"]) != "rf":
        raise ValueError("TreeSHAP explanations are only supported for RandomForest bundles.")

    shap = _load_shap_module()
    model = bundle["model"]
    raw_surviving_df, transformed_df = _prepare_smiles_list_inputs(smiles_list, bundle)
    transformed_array = transformed_df.to_numpy()

    predicted_raw, predicted_classes, predicted_probabilities, model_classes = _extract_model_outputs(
        model,
        transformed_array,
    )
    resolved_class_indices = _resolve_class_indices(
        class_index=class_index,
        predicted_classes=predicted_classes,
        model_classes=model_classes,
        num_rows=len(smiles_list),
    )

    explainer = shap.TreeExplainer(model)
    explanation = explainer(transformed_df)

    shap_matrix = _select_output_matrix(np.asarray(explanation.values), resolved_class_indices)
    base_values = _select_base_values(np.asarray(explanation.base_values), resolved_class_indices)
    feature_descriptions = dict(bundle["feature_descriptions"])

    results = []
    for row_index, smiles in enumerate(smiles_list):
        raw_row = raw_surviving_df.iloc[row_index]
        transformed_row = transformed_df.iloc[row_index]
        features = _build_feature_rows(
            raw_row=raw_row,
            transformed_row=transformed_row,
            shap_values=shap_matrix[row_index],
            feature_descriptions=feature_descriptions,
            top_k=top_k,
        )

        result = {
            "smiles": smiles,
            "bundle_path": str(bundle_path),
            "task_name": bundle["task_name"],
            "dataset": bundle["dataset"],
            "subtask": bundle["subtask"],
            "feature_backend_name": bundle["feature_backend_name"],
            "model_class": bundle["model_class"],
            "class_index": resolved_class_indices[row_index],
            "base_value": float(base_values[row_index]),
            "features": features,
        }

        if predicted_classes is not None:
            result["predicted_class"] = _coerce_json_scalar(predicted_classes[row_index])
        if predicted_probabilities is not None:
            result["predicted_probabilities"] = predicted_probabilities[row_index]
        if model_classes is not None:
            result["model_classes"] = [_coerce_json_scalar(value) for value in model_classes]
            result["explained_class"] = _coerce_json_scalar(model_classes[resolved_class_indices[row_index]])
        elif predicted_raw is not None:
            result["predicted_value"] = float(predicted_raw[row_index])

        results.append(result)

    return results


def explain_smiles_with_tree_shap(
    *,
    smiles: str,
    bundle_path: str | Path | None = None,
    checkpoint_root: str = "checkpoints",
    estimator: str = "rf",
    dataset: str | None = None,
    subtask: str = "",
    model_name: str | None = None,
    feature_backend: str | None = None,
    knowledge_type: str | None = None,
    num_samples: int | None = None,
    class_index: int | None = None,
    top_k: int | None = None,
) -> dict[str, object]:
    return explain_smiles_list_with_tree_shap(
        smiles_list=[smiles],
        bundle_path=bundle_path,
        checkpoint_root=checkpoint_root,
        estimator=estimator,
        dataset=dataset,
        subtask=subtask,
        model_name=model_name,
        feature_backend=feature_backend,
        knowledge_type=knowledge_type,
        num_samples=num_samples,
        class_index=class_index,
        top_k=top_k,
    )[0]


def explain_dataset_sample_with_tree_shap(
    *,
    bundle_path: str | Path | None = None,
    checkpoint_root: str = "checkpoints",
    estimator: str = "rf",
    dataset: str,
    subtask: str = "",
    split: str,
    sample_index: int,
    model_name: str | None = None,
    feature_backend: str | None = None,
    knowledge_type: str | None = None,
    num_samples: int | None = None,
    class_index: int | None = None,
    top_k: int | None = None,
) -> dict[str, object]:
    sample = get_dataset_sample(
        dataset=dataset,
        subtask=subtask,
        split=split,
        sample_index=sample_index,
    )
    result = explain_smiles_with_tree_shap(
        smiles=sample["smiles"],
        bundle_path=bundle_path,
        checkpoint_root=checkpoint_root,
        estimator=estimator,
        dataset=dataset,
        subtask=subtask,
        model_name=model_name,
        feature_backend=feature_backend,
        knowledge_type=knowledge_type,
        num_samples=num_samples,
        class_index=class_index,
        top_k=top_k,
    )
    result["sample"] = sample
    return result


def build_tree_shap_summary_dict(explanation: dict[str, object]) -> OrderedDict[str, object]:
    summary = OrderedDict()
    for key in [
        "bundle_path",
        "task_name",
        "dataset",
        "subtask",
        "feature_backend_name",
        "model_class",
        "class_index",
        "explained_class",
        "base_value",
        "model_classes",
        "predicted_probabilities",
    ]:
        if key in explanation:
            summary[key] = explanation[key]

    for feature_index, feature in enumerate(explanation.get("features", []), start=1):
        summary[f"feature_{feature_index}"] = (
            f"{feature['feature_name']} | {feature['feature_description']} | "
            f"raw_value={feature['raw_value']} | model_input_value={feature['model_input_value']} | "
            f"shap_value={feature['shap_value']} | abs_shap_value={feature['abs_shap_value']}"
        )

    return summary


def format_tree_shap_summary_string(summary: OrderedDict[str, object] | dict[str, object]) -> str:
    return "\n".join(f"{key}:{value}" for key, value in summary.items())


def render_tree_shap_concise_text(explanation: dict[str, object]) -> str:
    lines = []
    if "sample" in explanation:
        sample = explanation["sample"]
        lines.append(f"SMILES: {sample['smiles']}")
    else:
        lines.append(f"SMILES: {explanation['smiles']}")

    explained_class = explanation.get("explained_class")
    if explanation.get("model_classes") == [0, 1] and explained_class in [0, 1]:
        explained_label = "B" if explained_class == 1 else "A"
    else:
        explained_label = explained_class

    if "predicted_probabilities" in explanation:
        model_classes = explanation.get("model_classes")
        if model_classes == [0, 1] and len(explanation["predicted_probabilities"]) == 2:
            prob_a = _format_number(explanation["predicted_probabilities"][0])
            prob_b = _format_number(explanation["predicted_probabilities"][1])
            lines.append(f"Predicted probabilities: A={prob_a}, B={prob_b}")
        else:
            formatted_probabilities = [
                _format_number(probability) for probability in explanation["predicted_probabilities"]
            ]
            lines.append(f"Predicted probabilities: {formatted_probabilities}")
        if explanation.get("model_classes") == [0, 1]:
            predicted_label = "B" if explanation["predicted_class"] == 1 else "A"
            lines.append(f"Predicted class: {predicted_label} ({explanation['predicted_class']})")
        else:
            lines.append(f"Predicted class: {explanation['predicted_class']}")
    else:
        lines.append(f"Predicted value: {_format_number(explanation['predicted_value'])}")

    lines.append(f"Base probability of choosing {explained_label}: {_format_number(explanation['base_value'])}")
    lines.append("Top features:")
    for row in explanation["features"]:
        concise_description = row["feature_description"].rstrip(".")
        lines.append(
            f"- {concise_description}={_format_number(row['raw_value'])}, "
            f"adds probability = {_format_number(row['shap_value'])} "
            f"to choosing ({explained_label})."
        )
    return "\n".join(lines)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_path", help="Path to a saved model bundle, or a task checkpoint directory.")
    parser.add_argument("--checkpoint_root", default="checkpoints", help="Root directory for saved model bundles.")
    parser.add_argument("--estimator", default="rf", choices=["rf", "linear"], help="Estimator family to search under checkpoints.")
    parser.add_argument("--smiles", help="SMILES string to explain.")
    parser.add_argument("--dataset", help="Dataset name used to pick a sample from scaffold_datasets.")
    parser.add_argument("--subtask", default="", help="Optional dataset subtask, for example DILI.")
    parser.add_argument("--split", choices=["train", "valid", "test"], help="Dataset split used to pick the sample.")
    parser.add_argument("--sample_index", type=int, help="0-based sample index within the selected dataset split.")
    parser.add_argument("--model_name", help="Optional model name filter used when auto-selecting the best seed.")
    parser.add_argument("--feature_backend", help="Optional feature backend filter used when auto-selecting the best seed.")
    parser.add_argument("--knowledge_type", help="Optional knowledge type filter used when auto-selecting the best seed.")
    parser.add_argument("--num_samples", type=int, help="Optional num_samples filter used when auto-selecting the best seed.")
    parser.add_argument("--class_index", type=int, help="Optional target class index for classification bundles. Defaults to the predicted class.")
    parser.add_argument("--mode", choices=["concise", "detailed"], default="concise", help="Output verbosity mode.")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top features to print.")
    cli_args = parser.parse_args()

    if cli_args.smiles:
        explanation = explain_smiles_with_tree_shap(
            smiles=cli_args.smiles,
            bundle_path=cli_args.bundle_path,
            checkpoint_root=cli_args.checkpoint_root,
            estimator=cli_args.estimator,
            dataset=cli_args.dataset,
            subtask=cli_args.subtask,
            model_name=cli_args.model_name,
            feature_backend=cli_args.feature_backend,
            knowledge_type=cli_args.knowledge_type,
            num_samples=cli_args.num_samples,
            class_index=cli_args.class_index,
            top_k=cli_args.top_k,
        )
    else:
        if cli_args.dataset is None or cli_args.split is None or cli_args.sample_index is None:
            raise ValueError(
                "Provide either --smiles or the trio --dataset/--split/--sample_index "
                "(plus --subtask when needed)."
            )
        explanation = explain_dataset_sample_with_tree_shap(
            bundle_path=cli_args.bundle_path,
            checkpoint_root=cli_args.checkpoint_root,
            estimator=cli_args.estimator,
            dataset=cli_args.dataset,
            subtask=cli_args.subtask,
            split=cli_args.split,
            sample_index=cli_args.sample_index,
            model_name=cli_args.model_name,
            feature_backend=cli_args.feature_backend,
            knowledge_type=cli_args.knowledge_type,
            num_samples=cli_args.num_samples,
            class_index=cli_args.class_index,
            top_k=cli_args.top_k,
        )

    if cli_args.mode == "detailed":
        print(f"Task: {explanation['task_name']}")
        print(f"Bundle: {explanation['bundle_path']}")
        if "sample" in explanation:
            sample = explanation["sample"]
            print(
                f"Dataset sample: {sample['dataset']}/{sample['subtask'] or sample['dataset']} "
                f"{sample['split']}[{sample['sample_index']}]"
            )

    if "sample" in explanation:
        sample = explanation["sample"]
        print(f"SMILES: {sample['smiles']}")
    else:
        print(f"SMILES: {explanation['smiles']}")

    explained_class = explanation.get("explained_class")
    if cli_args.mode == "concise" and explanation.get("model_classes") == [0, 1] and explained_class in [0, 1]:
        explained_label = "B" if explained_class == 1 else "A"
    else:
        explained_label = explained_class

    if cli_args.mode == "concise":
        print("\n".join(render_tree_shap_concise_text(explanation).splitlines()[1:]))
        return

    if "predicted_probabilities" in explanation:
        print(f"Predicted probabilities: {explanation['predicted_probabilities']}")
        if "model_classes" in explanation:
            print(f"Model classes: {explanation['model_classes']}")
        print(f"Predicted class: {explanation['predicted_class']}")
        if explained_class is not None:
            print(f"Explaining class: {explained_class}")
    else:
        print(f"Base value: {explanation['base_value']}")
        print(f"Predicted value: {explanation['predicted_value']}")
    print("Top features:")
    for row in explanation["features"]:
        print(
            f"- {row['feature_name']}: shap={row['shap_value']:.6f}, "
            f"raw={row['raw_value']}, model_input={row['model_input_value']:.6f}, "
            f"description={row['feature_description']}"
        )


if __name__ == "__main__":
    _main()
