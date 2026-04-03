from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score

from feature_backends import load_feature_backend, transform_feature_frame
from model_bundle import get_checkpoint_dir, list_bundle_metadata, load_model_bundle
from rule_importance import extract_function_names

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


def configure_single_thread_runtime():
    thread_limits = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in thread_limits.items():
        os.environ.setdefault(key, value)


configure_single_thread_runtime()

_FEATURE_BACKEND = None
_FEATURE_TEMPLATE = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved RandomForest checkpoints without retraining.",
    )
    parser.add_argument("--dataset", type=str, help="dataset name, e.g. bbbp or TDC")
    parser.add_argument("--subtask", type=str, default="", help="subtask for tox21/sider/qm9/TDC")
    parser.add_argument("--model", type=str, help="saved bundle model_name filter")
    parser.add_argument("--knowledge_type", type=str, help="saved bundle knowledge_type filter")
    parser.add_argument("--feature_backend", type=str, help="saved bundle feature_backend_name filter")
    parser.add_argument("--num_samples", type=int, help="saved bundle num_samples filter")
    parser.add_argument("--seed", type=int, help="evaluate only one saved seed")
    parser.add_argument(
        "--checkpoint_selection",
        type=str,
        choices=["all", "best"],
        default="all",
        help="whether to evaluate all matched checkpoints or only the best one by saved validation metric",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["train", "valid", "test"],
        default="test",
        help="dataset split to evaluate on",
    )
    parser.add_argument(
        "--bundle_path",
        type=str,
        help="optional path to a saved .joblib bundle or a task checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="checkpoint root; bundles are loaded from checkpoints/forest/<task>",
    )
    parser.add_argument(
        "--precomputed_feature_dir",
        type=str,
        default="scaffold_datasets/TDC_precomputed_features",
        help="directory for cached precomputed TDC features",
    )
    parser.add_argument(
        "--feature_jobs",
        type=int,
        default=4,
        help="number of worker processes used for TDC feature precomputation",
    )
    parser.add_argument(
        "--force_recompute_features",
        action="store_true",
        help="recompute cached TDC features even if they already exist",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_result_wo_training",
        help="directory for evaluation reports",
    )
    return parser.parse_args()


def _get_tdc_file_path(args, which: str = "train") -> tuple[Path, str]:
    if not args.subtask:
        raise ValueError("TDC dataset requires --subtask, for example --subtask BBB_Martins")

    dataset_folder = Path("scaffold_datasets") / "TDC" / args.subtask
    requested_path = dataset_folder / f"{args.subtask}_{which}.csv"
    resolved_split = which

    if which == "test" and not requested_path.exists():
        resolved_split = "valid"
        requested_path = dataset_folder / f"{args.subtask}_{resolved_split}.csv"
        warnings.warn(
            f"TDC task {args.subtask} does not provide a test split. "
            f"Falling back to {requested_path.name} for test-time evaluation."
        )

    if not requested_path.exists():
        raise FileNotFoundError(f"Could not find TDC split file: {requested_path}")

    return requested_path, resolved_split


def load_data(args, which: str = "train") -> tuple[list[str], list[float], str]:
    resolved_split = which
    if args.dataset == "TDC":
        file_path, resolved_split = _get_tdc_file_path(args, which)
    else:
        dataset_folder = Path("scaffold_datasets") / args.dataset
        file_stem = args.dataset if args.subtask == "" else args.subtask
        file_path = dataset_folder / f"{file_stem}_{which}.csv"

    df = pd.read_csv(file_path)

    if args.dataset == "bbbp":
        y = df["p_np"].tolist()
    elif args.dataset == "clintox":
        y = df["CT_TOX"].tolist()
    elif args.dataset == "hiv":
        y = df["HIV_active"].tolist()
    elif args.dataset == "bace":
        y = df["Class"].tolist()
    elif args.dataset == "lipophilicity":
        y = df["exp"].tolist()
    elif args.dataset == "esol":
        y = df["ESOL predicted log solubility in mols per litre"].tolist()
    elif args.dataset == "freesolv":
        y = df["calc"].tolist()
    elif args.dataset in ["tox21", "sider", "qm9", "TDC"]:
        y = df[args.subtask].tolist()
    else:
        raise NotImplementedError(f"Unsupported dataset: {args.dataset}")

    smiles_column = "mol" if args.dataset == "bace" else "smiles"
    smiles_list = df[smiles_column].tolist()
    return smiles_list, y, resolved_split


def _init_feature_backend_worker(feature_backend_name: str):
    global _FEATURE_BACKEND, _FEATURE_TEMPLATE
    backend = load_feature_backend(feature_backend_name)
    _FEATURE_BACKEND = backend
    _FEATURE_TEMPLATE = {name: math.nan for name in backend.get_feature_names()}


def _featurize_backend_smiles(smiles: str):
    row = dict(_FEATURE_TEMPLATE)
    try:
        row.update(_FEATURE_BACKEND.featurize_smiles(smiles))
    except Exception as exc:
        print(f"Feature precompute failed for SMILES {smiles!r}: {str(exc)}")
    row["smiles"] = smiles
    return row


def _get_precomputed_feature_path(args, feature_backend_name: str, split_name: str) -> Path:
    if not args.subtask:
        raise ValueError("Precomputed TDC features require --subtask")

    cache_dir = Path(args.precomputed_feature_dir) / args.subtask
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{args.subtask}_{split_name}_{feature_backend_name}_features.csv"


def _load_precomputed_feature_frame(feature_path: Path, smiles_list: list[str]) -> pd.DataFrame:
    df = pd.read_csv(feature_path)
    if "smiles" not in df.columns:
        raise ValueError(f"Precomputed feature file is missing 'smiles' column: {feature_path}")

    cached_smiles = df["smiles"].astype(str).tolist()
    expected_smiles = [str(smiles) for smiles in smiles_list]
    if cached_smiles != expected_smiles:
        raise ValueError(
            f"Cached features in {feature_path} do not align with the requested dataset split order."
        )
    return df.drop(columns=["smiles"])


def _compute_and_cache_backend_features(
    args,
    smiles_list: list[str],
    split_name: str,
    feature_backend_name: str,
) -> pd.DataFrame:
    feature_path = _get_precomputed_feature_path(args, feature_backend_name, split_name)
    worker_count = max(1, args.feature_jobs)
    print(
        f"Precomputing {len(smiles_list)} molecules for TDC/{args.subtask} {split_name} "
        f"with backend {feature_backend_name} using {worker_count} workers"
    )
    progress_desc = f"TDC/{args.subtask} {split_name}"

    if worker_count == 1:
        _init_feature_backend_worker(feature_backend_name)
        rows = [
            _featurize_backend_smiles(smiles)
            for smiles in tqdm(smiles_list, total=len(smiles_list), desc=progress_desc)
        ]
    else:
        chunksize = max(1, len(smiles_list) // (worker_count * 4))
        with multiprocessing.Pool(
            worker_count,
            initializer=_init_feature_backend_worker,
            initargs=(feature_backend_name,),
        ) as pool:
            iterator = pool.imap(_featurize_backend_smiles, smiles_list, chunksize=chunksize)
            rows = list(tqdm(iterator, total=len(smiles_list), desc=progress_desc))

    feature_df = pd.DataFrame(rows)
    if "smiles" in feature_df.columns:
        ordered_columns = ["smiles"] + [column for column in feature_df.columns if column != "smiles"]
        feature_df = feature_df[ordered_columns]
    feature_df.to_csv(feature_path, index=False)
    print(f"Saved precomputed features to {feature_path}")
    return feature_df.drop(columns=["smiles"])


def _get_tdc_feature_frame(args, smiles_list: list[str], split_name: str, feature_backend_name: str) -> pd.DataFrame:
    feature_path = _get_precomputed_feature_path(args, feature_backend_name, split_name)
    if feature_path.exists() and not args.force_recompute_features:
        print(f"Loading cached features from {feature_path}")
        return _load_precomputed_feature_frame(feature_path, smiles_list)
    return _compute_and_cache_backend_features(args, smiles_list, split_name, feature_backend_name)


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


def get_feature_frame(args, bundle: dict[str, object], backend, split_name: str, smiles_list: list[str]) -> pd.DataFrame:
    if args.dataset == "TDC" and backend.name != "generated_rules":
        return _get_tdc_feature_frame(args, smiles_list, split_name, backend.name)

    raw_feature_df = backend.featurize_smiles_list(smiles_list, on_error="nan")
    input_feature_names = list(bundle["input_feature_names"])
    return raw_feature_df.reindex(columns=input_feature_names)


def is_classification_task(dataset: str) -> bool:
    return dataset in ["bbbp", "clintox", "hiv", "bace", "tox21", "sider", "TDC"]


def compute_macro_f1(y_true, y_pred):
    if len(set(y_true)) <= 1:
        warnings.warn(
            "Cannot calculate macro F1 because the ground-truth labels contain only one class. Returning 0.0."
        )
        return 0.0
    return f1_score(y_true, y_pred, average="macro")


def build_result_folder(args) -> Path:
    subfolder = ""
    if args.knowledge_type in ["inference", "all"] and args.num_samples is not None:
        subfolder = f"sample_{args.num_samples}"

    result_folder = Path(args.output_dir) / args.model / args.dataset / args.knowledge_type
    if subfolder:
        result_folder = result_folder / subfolder
    result_folder.mkdir(parents=True, exist_ok=True)
    return result_folder


def build_result_prefix(args, split_name: str) -> str:
    subtask_part = f"_{args.subtask}" if args.subtask else ""
    prefix = f"{args.model}_{args.dataset}{subtask_part}_{args.knowledge_type}_{split_name}"
    if args.feature_backend:
        prefix += f"_{args.feature_backend}"
    prefix += "_loaded_checkpoint"
    return prefix


def _require_single_metadata_value(records, field_name: str, cli_flag: str):
    values = {record.get(field_name) for record in records}
    if len(values) != 1:
        raise ValueError(
            f"Matched checkpoints contain multiple {field_name} values: {sorted(values)!r}. "
            f"Please narrow them with {cli_flag}."
        )
    return next(iter(values))


def resolve_bundle_records(args) -> list[dict[str, object]]:
    if args.bundle_path:
        bundle_path = Path(args.bundle_path)
        if bundle_path.is_file():
            return [{"bundle_path": str(bundle_path)}]
        if not bundle_path.is_dir():
            raise FileNotFoundError(f"Bundle path does not exist: {bundle_path}")
        metadata_records = list_bundle_metadata(bundle_path)
    else:
        if not args.dataset:
            raise ValueError("--dataset is required when --bundle_path is not provided")
        bundle_dir = get_checkpoint_dir(
            checkpoint_root=args.checkpoint_dir,
            estimator="rf",
            dataset=args.dataset,
            subtask=args.subtask,
        )
        metadata_records = list_bundle_metadata(bundle_dir)

    filtered_records = []
    for record in metadata_records:
        if args.model is not None and record.get("model_name") != args.model:
            continue
        if args.feature_backend is not None and record.get("feature_backend_name") != args.feature_backend:
            continue
        if args.knowledge_type is not None and record.get("knowledge_type") != args.knowledge_type:
            continue
        if args.num_samples is not None and int(record.get("num_samples", -1)) != args.num_samples:
            continue
        if args.seed is not None and int(record.get("seed", -1)) != args.seed:
            continue
        filtered_records.append(record)

    if not filtered_records:
        raise FileNotFoundError(
            "No checkpoint metadata matched the requested filters. "
            f"model={args.model!r}, feature_backend={args.feature_backend!r}, "
            f"knowledge_type={args.knowledge_type!r}, num_samples={args.num_samples!r}, seed={args.seed!r}"
        )

    if args.checkpoint_selection == "best":
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
        return [best_record]

    filtered_records.sort(key=lambda record: int(record.get("seed", 0)))
    return filtered_records


def infer_missing_cli_args(args, reference_bundle: dict[str, object]):
    if args.dataset is None:
        args.dataset = str(reference_bundle["dataset"])
    if args.subtask == "":
        args.subtask = str(reference_bundle.get("subtask", ""))
    if args.model is None:
        args.model = str(reference_bundle["model_name"])
    if args.knowledge_type is None:
        args.knowledge_type = str(reference_bundle["knowledge_type"])
    if args.feature_backend is None:
        args.feature_backend = str(reference_bundle["feature_backend_name"])
    if args.num_samples is None:
        args.num_samples = int(reference_bundle["num_samples"])


def evaluate_loaded_bundles(args):
    bundle_records = resolve_bundle_records(args)
    bundle_paths = [Path(str(record["bundle_path"])) for record in bundle_records]
    first_bundle = load_model_bundle(bundle_paths[0])

    if args.model is None and len(bundle_records) > 1:
        args.model = str(_require_single_metadata_value(bundle_records, "model_name", "--model"))
    if args.knowledge_type is None and len(bundle_records) > 1:
        args.knowledge_type = str(
            _require_single_metadata_value(bundle_records, "knowledge_type", "--knowledge_type")
        )
    if args.feature_backend is None and len(bundle_records) > 1:
        args.feature_backend = str(
            _require_single_metadata_value(bundle_records, "feature_backend_name", "--feature_backend")
        )
    if args.num_samples is None and len(bundle_records) > 1:
        args.num_samples = int(_require_single_metadata_value(bundle_records, "num_samples", "--num_samples"))

    infer_missing_cli_args(args, first_bundle)

    smiles_list, y_true, resolved_split = load_data(args, args.eval_split)
    result_folder = build_result_folder(args)
    result_prefix = build_result_prefix(args, resolved_split)
    report_path = result_folder / f"{result_prefix}.txt"

    task_is_classification = is_classification_task(args.dataset)
    per_bundle_records = []

    print(f"Evaluating {len(bundle_paths)} bundle(s) on split: {resolved_split}")
    print(f"Checkpoint selection mode: {args.checkpoint_selection}")
    print(f"Result report will be written to {report_path}")

    for bundle_path in bundle_paths:
        bundle = load_model_bundle(bundle_path)
        backend = _build_backend_from_bundle(bundle)
        raw_feature_df = get_feature_frame(args, bundle, backend, resolved_split, smiles_list)
        _, transformed_df = transform_feature_frame(raw_feature_df, bundle["preprocessor"])

        model = bundle["model"]
        seed = int(bundle["seed"])
        record = {
            "seed": seed,
            "bundle_path": str(bundle_path),
        }

        if task_is_classification:
            y_pred = model.predict(transformed_df)
            y_proba = model.predict_proba(transformed_df)[:, 1]
            roc_auc = roc_auc_score(y_true, y_proba)
            macro_f1 = compute_macro_f1(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            record["roc_auc"] = float(roc_auc)
            record["macro_f1"] = float(macro_f1)
            record["accuracy"] = float(accuracy)
            print(
                f"seed={seed} split={resolved_split} ROC-AUC={roc_auc:.6f} "
                f"macro_F1={macro_f1:.6f} accuracy={accuracy:.6f} bundle={bundle_path}"
            )
        elif args.dataset in ["esol", "lipophilicity", "freesolv"]:
            y_pred = model.predict(transformed_df)
            rmse = sqrt(mean_squared_error(y_true, y_pred))
            record["rmse"] = float(rmse)
            print(f"seed={seed} split={resolved_split} RMSE={rmse:.6f} bundle={bundle_path}")
        elif args.dataset == "qm9":
            y_pred = model.predict(transformed_df)
            mae = mean_absolute_error(y_true, y_pred)
            record["mae"] = float(mae)
            print(f"seed={seed} split={resolved_split} MAE={mae:.6f} bundle={bundle_path}")
        else:
            raise NotImplementedError(f"Unsupported dataset for evaluation: {args.dataset}")

        per_bundle_records.append(record)

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"subtask: {args.subtask}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"knowledge_type: {args.knowledge_type}\n")
        f.write(f"feature_backend: {args.feature_backend}\n")
        f.write(f"eval_split: {resolved_split}\n")
        f.write(f"num_bundles: {len(per_bundle_records)}\n\n")

        if task_is_classification:
            roc_auc_values = [record["roc_auc"] for record in per_bundle_records]
            macro_f1_values = [record["macro_f1"] for record in per_bundle_records]
            accuracy_values = [record["accuracy"] for record in per_bundle_records]
            for record in per_bundle_records:
                f.write(
                    f"seed={record['seed']}\troc_auc={record['roc_auc']}\t"
                    f"macro_f1={record['macro_f1']}\taccuracy={record['accuracy']}\t"
                    f"bundle={record['bundle_path']}\n"
                )
            f.write("\n")
            f.write(f"average_{resolved_split}_roc_auc: {np.mean(roc_auc_values)}\n")
            f.write(f"std_{resolved_split}_roc_auc: {np.std(roc_auc_values)}\n")
            f.write(f"average_{resolved_split}_macro_f1: {np.mean(macro_f1_values)}\n")
            f.write(f"std_{resolved_split}_macro_f1: {np.std(macro_f1_values)}\n")
            f.write(f"average_{resolved_split}_accuracy: {np.mean(accuracy_values)}\n")
            f.write(f"std_{resolved_split}_accuracy: {np.std(accuracy_values)}\n")

            print("=================================================")
            print(f"Average {resolved_split} ROC-AUC: {np.mean(roc_auc_values)}")
            print(f"Std {resolved_split} ROC-AUC: {np.std(roc_auc_values)}")
            print(f"Average {resolved_split} macro F1: {np.mean(macro_f1_values)}")
            print(f"Std {resolved_split} macro F1: {np.std(macro_f1_values)}")
            print(f"Average {resolved_split} accuracy: {np.mean(accuracy_values)}")
            print(f"Std {resolved_split} accuracy: {np.std(accuracy_values)}")
            print("=================================================")
        elif args.dataset in ["esol", "lipophilicity", "freesolv"]:
            rmse_values = [record["rmse"] for record in per_bundle_records]
            for record in per_bundle_records:
                f.write(f"seed={record['seed']}\trmse={record['rmse']}\tbundle={record['bundle_path']}\n")
            f.write("\n")
            f.write(f"average_{resolved_split}_rmse: {np.mean(rmse_values)}\n")
            f.write(f"std_{resolved_split}_rmse: {np.std(rmse_values)}\n")

            print("=================================================")
            print(f"Average {resolved_split} RMSE: {np.mean(rmse_values)}")
            print(f"Std {resolved_split} RMSE: {np.std(rmse_values)}")
            print("=================================================")
        else:
            mae_values = [record["mae"] for record in per_bundle_records]
            for record in per_bundle_records:
                f.write(f"seed={record['seed']}\tmae={record['mae']}\tbundle={record['bundle_path']}\n")
            f.write("\n")
            f.write(f"average_{resolved_split}_mae: {np.mean(mae_values)}\n")
            f.write(f"std_{resolved_split}_mae: {np.std(mae_values)}\n")

            print("=================================================")
            print(f"Average {resolved_split} MAE: {np.mean(mae_values)}")
            print(f"Std {resolved_split} MAE: {np.std(mae_values)}")
            print("=================================================")

    print(f"Saved evaluation report to {report_path}")


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    evaluate_loaded_bundles(args)


if __name__ == "__main__":
    main()
