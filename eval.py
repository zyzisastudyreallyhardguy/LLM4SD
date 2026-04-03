import os
import numpy as np
import argparse
import math
import multiprocessing
from math import sqrt
import pandas as pd
import json
import warnings
from pathlib import Path

from rdkit import Chem

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, mean_absolute_error, f1_score
from feature_backends import load_feature_backend, prepare_feature_matrices
from model_bundle import get_task_name, save_model_bundle
from rule_importance import (
    collect_feature_importance_rows,
    extract_function_names,
    write_rule_importance_outputs,
)

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


def configure_worker_runtime():
    configure_single_thread_runtime()
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
    except Exception:
        pass


configure_single_thread_runtime()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bbbp', help='dataset name in lower case')
parser.add_argument('--subtask', type=str, default='', help='subtask of tox21/sider/qm9 dataset')
parser.add_argument('--model', type=str, default='galactica-6.7b', help='LLM model')
parser.add_argument('--knowledge_type', type=str, default='synthesize', help='synthesize/inference/all')
parser.add_argument(
    '--feature_backend',
    type=str,
    default='generated_rules',
    help='feature backend name, including generated_rules, registered static backends, or feedback variants such as dili_feedback_v001',
)
parser.add_argument('--num_samples', type=int, default=50, help='number of sample lists (30/50) for inference')
parser.add_argument('--output_dir', type=str, default='eval_result', help='output folder')
parser.add_argument('--code_gen_folder', type=str, default='eval_code_generation_repo', help='Loading code from this folder')
parser.add_argument('--precomputed_feature_dir', type=str, default='scaffold_datasets/TDC_precomputed_features', help='directory for cached precomputed TDC features')
parser.add_argument('--feature_jobs', type=int, default=4, help='number of worker processes used for TDC feature precomputation')
parser.add_argument('--rf_jobs', type=int, default=1, help='number of CPU workers used by RandomForest itself')
parser.add_argument('--estimator', type=str, choices=['rf', 'linear'], default='rf', help='model family used after feature extraction: rf or linear')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory for saved trained-model bundles')
parser.add_argument('--force_recompute_features', action='store_true', help='recompute cached TDC features even if they already exist')
args = parser.parse_args()


_FEATURE_BACKEND = None
_FEATURE_TEMPLATE = None


def _get_tdc_file_path(which='train'):
    if not args.subtask:
        raise ValueError("TDC dataset requires --subtask, for example --subtask BBB_Martins")

    dataset_folder = Path('scaffold_datasets') / 'TDC' / args.subtask
    requested_path = dataset_folder / f'{args.subtask}_{which}.csv'
    resolved_split = which

    if which == 'test' and not requested_path.exists():
        resolved_split = 'valid'
        requested_path = dataset_folder / f'{args.subtask}_{resolved_split}.csv'
        warnings.warn(
            f"TDC task {args.subtask} does not provide a test split. "
            f"Falling back to {requested_path.name} for test-time evaluation."
        )

    if not requested_path.exists():
        raise FileNotFoundError(f"Could not find TDC split file: {requested_path}")

    return requested_path, resolved_split


def _init_feature_backend_worker(feature_backend_name: str):
    global _FEATURE_BACKEND, _FEATURE_TEMPLATE
    configure_worker_runtime()
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


def _get_precomputed_feature_path(split_name: str) -> Path:
    if not args.subtask:
        raise ValueError("Precomputed TDC features require --subtask")

    cache_dir = Path(args.precomputed_feature_dir) / args.subtask
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{args.subtask}_{split_name}_{args.feature_backend}_features.csv"


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


def _compute_and_cache_backend_features(smiles_list: list[str], split_name: str, feature_backend_name: str) -> pd.DataFrame:
    feature_path = _get_precomputed_feature_path(split_name)
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


def _get_tdc_feature_frame(split_name: str, smiles_list: list[str], feature_backend_name: str) -> pd.DataFrame:
    feature_path = _get_precomputed_feature_path(split_name)
    if feature_path.exists() and not args.force_recompute_features:
        print(f"Loading cached features from {feature_path}")
        return _load_precomputed_feature_frame(feature_path, smiles_list)
    return _compute_and_cache_backend_features(smiles_list, split_name, feature_backend_name)


def ensure_precomputed_feature_frames(feature_backend, split_payloads):
    if args.dataset != 'TDC' or feature_backend.name == 'generated_rules':
        return

    seen_splits = set()
    for split_name, smiles_list in split_payloads:
        if split_name in seen_splits:
            continue
        seen_splits.add(split_name)
        _get_tdc_feature_frame(split_name, smiles_list, feature_backend.name)


def get_feature_frame(feature_backend, split_name: str, smiles_list: list[str]) -> pd.DataFrame:
    if args.dataset == 'TDC' and feature_backend.name != 'generated_rules':
        return _get_tdc_feature_frame(split_name, smiles_list, feature_backend.name)
    return feature_backend.featurize_smiles_list(smiles_list, on_error="nan")


# load csv datasets from a directory
def load_data(which='train'):
    resolved_split = which
    if args.dataset == 'TDC':
        file_path, resolved_split = _get_tdc_file_path(which)
    else:
        dataset_folder = os.path.join('scaffold_datasets', args.dataset)
        if args.subtask == "":
            file_name = args.dataset + '_' + which + '.csv'
        else:
            file_name = args.subtask + '_' + which + '.csv'
        file_path = os.path.join(dataset_folder, file_name)
    df = pd.read_csv(file_path)

    if args.dataset == 'bbbp':
        y = df['p_np'].tolist()
    elif args.dataset == 'clintox':
        y = df['CT_TOX'].tolist()
    elif args.dataset == 'hiv':
        y = df['HIV_active'].tolist()
    elif args.dataset == 'bace':
        y = df['Class'].tolist()
    elif args.dataset == 'lipophilicity':
        y = df['exp'].tolist()
    elif args.dataset == 'esol':
        y = df['ESOL predicted log solubility in mols per litre'].tolist()
    elif args.dataset == 'freesolv':
        y = df['calc'].tolist()
    elif args.dataset in ['tox21', 'sider', 'qm9', 'TDC']:
        y = df[args.subtask].tolist()
    else:
        raise NotImplementedError(f"Load Dataset Error")

    if args.dataset != 'bace':
        smiles_list = df['smiles'].tolist()
    else:
        smiles_list = df['mol'].tolist()
    return smiles_list, y, resolved_split


def get_function_code(generated_code, function_name):
    lines = generated_code.split('\n')
    start = -1
    for i in range(len(lines)):
        if lines[i].strip().startswith('def ' + function_name):
            start = i
            break
    if start == -1:
        return None  # Function not found
    end = start + 1
    while end < len(lines) and lines[end].startswith(' '):  # Continue until we reach a line not indented
        end += 1
    return '\n'.join(lines[start:end])


def exec_code(generated_code, smiles_list, valid_function_names):
    smiles_feat = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        features = []
        for function_name in valid_function_names:  # Loop over the valid function names
            try:
                feature = globals()[function_name](mol)
                if feature is not None and isinstance(feature, (int, float)):
                    features.append(feature)
                else:
                    features.append(np.nan)
            except Exception as e:
                print(f"Unexpected error in function {function_name}: {str(e)}")
                features.append(np.nan)
        smiles_feat.append(features)

    return smiles_feat


def dropna(X):
    X = pd.DataFrame(X)
    X = X.dropna(axis=1)
    X = X.values.tolist()
    return X


def build_feature_backend(valid_function_names=None):
    if args.feature_backend == 'generated_rules':
        if valid_function_names is None:
            raise ValueError("generated_rules backend requires valid_function_names")
        return load_feature_backend(
            args.feature_backend,
            function_names=valid_function_names,
            namespace=globals(),
        )
    return load_feature_backend(args.feature_backend)


def get_result_prefix(subtask_name):
    prefix = f'{args.model}_{args.dataset}{subtask_name}_{args.knowledge_type}'
    if args.feature_backend != 'generated_rules':
        prefix += f'_{args.feature_backend}'
    if args.estimator != 'rf':
        prefix += f'_{args.estimator}'
    return prefix


def get_default_importance_source(feature_backend_name):
    if feature_backend_name == 'generated_rules':
        return args.knowledge_type
    return feature_backend_name


def get_feature_description_map(feature_backend, surviving_feature_names):
    if hasattr(feature_backend, 'get_feature_descriptions'):
        description_map = dict(feature_backend.get_feature_descriptions())
        return {name: description_map.get(name, name) for name in surviving_feature_names}
    return {name: name for name in surviving_feature_names}


def build_backend_payload(*, feature_backend, function_names=None, generated_code=None):
    payload = {"feature_backend_name": feature_backend.name}
    if feature_backend.name == 'generated_rules':
        if generated_code is None:
            raise ValueError("generated_rules bundle payload requires generated_code")
        payload.update(
            {
                "function_names": list(function_names or []),
                "generated_code": generated_code,
            }
        )
    return payload


def save_trained_model_artifact(
    *,
    clf,
    seed: int,
    is_classification: bool,
    feature_backend,
    preprocessor,
    input_feature_names,
    surviving_feature_names,
    feature_description_map,
    backend_payload,
    metrics=None,
):
    bundle = {
        "dataset": args.dataset,
        "subtask": args.subtask,
        "task_name": get_task_name(args.dataset, args.subtask),
        "model_name": args.model,
        "knowledge_type": args.knowledge_type,
        "num_samples": args.num_samples,
        "estimator": args.estimator,
        "seed": seed,
        "is_classification": is_classification,
        "model_class": type(clf).__name__,
        "feature_backend_name": feature_backend.name,
        "backend_payload": backend_payload,
        "input_feature_names": list(input_feature_names),
        "surviving_feature_names": list(surviving_feature_names),
        "feature_descriptions": dict(feature_description_map),
        "preprocessor": preprocessor,
        "metrics": dict(metrics or {}),
        "model": clf,
    }
    return save_model_bundle(bundle=bundle, checkpoint_root=args.checkpoint_dir)


def get_best_rf_model_params():
    with open('llm4sd_models.json', 'r') as model_parms:
        best_models = json.load(model_parms)

    if args.dataset in ['sider', 'tox21']:
        return dict(best_models[args.dataset][args.subtask])
    if args.dataset == 'TDC':
        tuned_param_path = (
            Path('eval_result')
            / 'TDC_hyperparameter_search'
            / args.subtask
            / f'{args.subtask}_{args.feature_backend}_best_params.json'
        )
        if tuned_param_path.exists():
            with tuned_param_path.open('r', encoding='utf-8') as f:
                tuned_payload = json.load(f)
            warnings.warn(f"Loading tuned TDC RF hyperparameters from {tuned_param_path}")
            return dict(tuned_payload['best_params'])
        warnings.warn(
            "TDC tasks do not have dedicated RF hyperparameters in llm4sd_models.json yet. "
            "Falling back to the BBBP RandomForest setting."
        )
        return dict(best_models['bbbp'])
    dataset_params = best_models.get(args.dataset)
    if dataset_params is None:
        warnings.warn(
            f"No dedicated RandomForest hyperparameters found for dataset {args.dataset}. "
            "Falling back to sklearn RandomForest defaults."
        )
        return {}
    return dict(dataset_params)


def build_estimator(*, seed: int, is_classification: bool):
    if args.estimator == 'rf':
        best_params = get_best_rf_model_params()
        best_params['random_state'] = seed
        best_params['n_jobs'] = args.rf_jobs
        if is_classification:
            return RandomForestClassifier(**best_params)
        return RandomForestRegressor(**best_params)

    if is_classification:
        return LogisticRegression(
            max_iter=1000,
            solver='liblinear',
            random_state=seed,
        )
    return LinearRegression()


def get_eval_seeds():
    if args.estimator == 'rf':
        return [0, 1, 2, 3, 4]
    return [0]


def get_estimator_label(*, is_classification: bool) -> str:
    if args.estimator == 'rf':
        return 'RandomForestClassifier' if is_classification else 'RandomForestRegressor'
    return 'LogisticRegression' if is_classification else 'LinearRegression'


def compute_macro_f1(y_true, y_pred):
    if len(set(y_true)) <= 1:
        warnings.warn("Cannot calculate macro F1 because the ground-truth labels contain only one class. Returning 0.0.")
        return 0.0
    return f1_score(y_true, y_pred, average='macro')


def build_selection_metrics(
    *,
    valid_metric_name: str,
    valid_metric_value: float,
    test_metrics: dict[str, float] | None = None,
    higher_is_better: bool = True,
):
    metrics = {
        "selection_metric_name": valid_metric_name,
        "selection_metric_value": float(valid_metric_value),
        "selection_higher_is_better": higher_is_better,
        valid_metric_name: float(valid_metric_value),
    }
    for metric_name, metric_value in (test_metrics or {}).items():
        metrics[metric_name] = float(metric_value)
    return metrics


def build_classification_metric_row(
    *,
    seed: int,
    bundle_path,
    train_macro_f1: float,
    train_roc_auc: float,
    valid_macro_f1: float,
    valid_roc_auc: float,
    test_macro_f1: float,
    test_roc_auc: float,
    test_resolved_split: str,
):
    return {
        "dataset": args.dataset,
        "subtask": args.subtask,
        "model_name": args.model,
        "knowledge_type": args.knowledge_type,
        "feature_backend": args.feature_backend,
        "num_samples": args.num_samples,
        "seed": seed,
        "bundle_path": str(bundle_path),
        "train_macro_f1": float(train_macro_f1),
        "train_roc_auc": float(train_roc_auc),
        "valid_macro_f1": float(valid_macro_f1),
        "valid_roc_auc": float(valid_roc_auc),
        "test_macro_f1": float(test_macro_f1),
        "test_roc_auc": float(test_roc_auc),
        "test_resolved_split": test_resolved_split,
    }


def build_classification_prediction_rows(
    *,
    seed: int,
    bundle_path,
    split_label: str,
    resolved_split: str,
    smiles_list: list[str],
    y_true: list[float],
    y_pred,
    y_proba,
    model_classes,
):
    probability_index_by_class = {}
    for index, class_label in enumerate(model_classes):
        probability_index_by_class[int(class_label)] = index

    prob_0_index = probability_index_by_class.get(0)
    prob_1_index = probability_index_by_class.get(1)

    rows = []
    for sample_index, (smiles, true_label, predicted_label, probability_row) in enumerate(
        zip(smiles_list, y_true, y_pred, y_proba)
    ):
        prob_0 = float(probability_row[prob_0_index]) if prob_0_index is not None else math.nan
        prob_1 = float(probability_row[prob_1_index]) if prob_1_index is not None else math.nan
        rows.append(
            {
                "dataset": args.dataset,
                "subtask": args.subtask,
                "model_name": args.model,
                "knowledge_type": args.knowledge_type,
                "feature_backend": args.feature_backend,
                "num_samples": args.num_samples,
                "seed": seed,
                "bundle_path": str(bundle_path),
                "split": split_label,
                "resolved_split": resolved_split,
                "sample_index": sample_index,
                "smiles": str(smiles),
                "y_true": int(true_label),
                "y_pred": int(predicted_label),
                "prob_0": prob_0,
                "prob_1": prob_1,
                "correct": bool(int(predicted_label) == int(true_label)),
            }
        )
    return rows


def write_classification_feedback_outputs(
    *,
    metric_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    output_dir: str,
    result_prefix: str,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metric_df = pd.DataFrame(metric_rows).sort_values(by=["seed"]).reset_index(drop=True)
    metric_path = output_path / f"{result_prefix}_classification_metrics_per_seed.csv"
    metric_df.to_csv(metric_path, index=False)

    split_order = {"train": 0, "valid": 1, "test": 2}
    prediction_df = pd.DataFrame(prediction_rows)
    prediction_df["_split_order"] = prediction_df["split"].map(split_order).fillna(99)
    prediction_df = prediction_df.sort_values(
        by=["seed", "_split_order", "sample_index"],
    ).drop(columns=["_split_order"]).reset_index(drop=True)
    prediction_path = output_path / f"{result_prefix}_sample_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    return metric_path, prediction_path


def evaluation(feature_backend, feature_source_map=None, generated_code=None, function_names=None):
    train_smiles, y_train, train_split = load_data('train')
    valid_smiles, y_valid, valid_split = load_data('valid')
    test_smiles, y_test, test_split = load_data('test')

    ensure_precomputed_feature_frames(
        feature_backend,
        [
            (train_split, train_smiles),
            (valid_split, valid_smiles),
            (test_split, test_smiles),
        ],
    )

    X_train_df = get_feature_frame(feature_backend, train_split, train_smiles)
    X_valid_df = get_feature_frame(feature_backend, valid_split, valid_smiles)
    X_test_df = get_feature_frame(feature_backend, test_split, test_smiles)
    is_classification = args.dataset in ['bbbp', 'clintox', 'hiv', 'bace', 'tox21', 'sider', 'TDC']
    seeds = get_eval_seeds()

    if args.subtask != '':
        subtask_name = '_' + args.subtask
    else:
        subtask_name = ''

    if args.knowledge_type in ['inference', 'all']:
        subfolder = f"sample_{args.num_samples}"
    else:
        subfolder = ''
    result_folder = os.path.join(args.output_dir, args.model, args.dataset, args.knowledge_type, subfolder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    scale_features = args.dataset in ['bbbp', 'clintox', 'hiv', 'bace', 'tox21', 'sider', 'qm9', 'TDC']
    X_train, X_valid, X_test, surviving_feature_names, preprocessor = prepare_feature_matrices(
        X_train_df,
        X_valid_df,
        X_test_df,
        scale_features=scale_features,
        return_preprocessor=True,
    )
    print(f"Using feature backend: {feature_backend.name}")
    print(f"Number of surviving features: {len(surviving_feature_names)}")
    importance_records = []
    default_importance_source = get_default_importance_source(feature_backend.name)
    feature_description_map = get_feature_description_map(feature_backend, surviving_feature_names)
    backend_payload = build_backend_payload(
        feature_backend=feature_backend,
        function_names=function_names,
        generated_code=generated_code,
    )
    input_feature_names = list(preprocessor["input_columns"])

    if is_classification:

        classification_metric_rows = []
        sample_prediction_rows = []
        average_roc_auc_test = []
        average_roc_auc_valid = []
        average_macro_f1_test = []
        average_macro_f1_valid = []

        for seed in seeds:
            clf = build_estimator(seed=seed, is_classification=True)
            clf.fit(X_train, y_train)
            y_train_proba = clf.predict_proba(X_train)
            y_valid_proba = clf.predict_proba(X_valid)[:, 1]
            y_test_proba = clf.predict_proba(X_test)[:, 1]
            y_train_pred = clf.predict(X_train)
            y_valid_pred = clf.predict(X_valid)
            y_test_pred = clf.predict(X_test)

            train_roc_auc = roc_auc_score(y_train, y_train_proba[:, 1])
            valid_roc_auc = roc_auc_score(y_valid, y_valid_proba)
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
            train_macro_f1 = compute_macro_f1(y_train, y_train_pred)
            valid_macro_f1 = compute_macro_f1(y_valid, y_valid_pred)
            test_macro_f1 = compute_macro_f1(y_test, y_test_pred)

            bundle_path, metadata_path = save_trained_model_artifact(
                clf=clf,
                seed=seed,
                is_classification=True,
                feature_backend=feature_backend,
                preprocessor=preprocessor,
                input_feature_names=input_feature_names,
                surviving_feature_names=surviving_feature_names,
                feature_description_map=feature_description_map,
                backend_payload=backend_payload,
                metrics=build_selection_metrics(
                    valid_metric_name='valid_macro_f1',
                    valid_metric_value=valid_macro_f1,
                    test_metrics={
                        'train_roc_auc': train_roc_auc,
                        'train_macro_f1': train_macro_f1,
                        'valid_roc_auc': valid_roc_auc,
                        'test_roc_auc': test_roc_auc,
                        'test_macro_f1': test_macro_f1,
                    },
                    higher_is_better=True,
                ),
            )
            print(f"Saved trained model bundle to {bundle_path}")
            print(f"Saved trained model metadata to {metadata_path}")
            importance_records.extend(
                collect_feature_importance_rows(
                    clf=clf,
                    feature_names=surviving_feature_names,
                    feature_descriptions=feature_description_map,
                    dataset=args.dataset,
                    subtask=args.subtask,
                    knowledge_type=args.knowledge_type,
                    seed=seed,
                    default_source=default_importance_source,
                    source_map=feature_source_map,
                )
            )
            classification_metric_rows.append(
                build_classification_metric_row(
                    seed=seed,
                    bundle_path=bundle_path,
                    train_macro_f1=train_macro_f1,
                    train_roc_auc=train_roc_auc,
                    valid_macro_f1=valid_macro_f1,
                    valid_roc_auc=valid_roc_auc,
                    test_macro_f1=test_macro_f1,
                    test_roc_auc=test_roc_auc,
                    test_resolved_split=test_split,
                )
            )
            sample_prediction_rows.extend(
                build_classification_prediction_rows(
                    seed=seed,
                    bundle_path=bundle_path,
                    split_label='train',
                    resolved_split=train_split,
                    smiles_list=train_smiles,
                    y_true=y_train,
                    y_pred=y_train_pred,
                    y_proba=y_train_proba,
                    model_classes=clf.classes_,
                )
            )
            sample_prediction_rows.extend(
                build_classification_prediction_rows(
                    seed=seed,
                    bundle_path=bundle_path,
                    split_label='valid',
                    resolved_split=valid_split,
                    smiles_list=valid_smiles,
                    y_true=y_valid,
                    y_pred=y_valid_pred,
                    y_proba=clf.predict_proba(X_valid),
                    model_classes=clf.classes_,
                )
            )
            sample_prediction_rows.extend(
                build_classification_prediction_rows(
                    seed=seed,
                    bundle_path=bundle_path,
                    split_label='test',
                    resolved_split=test_split,
                    smiles_list=test_smiles,
                    y_true=y_test,
                    y_pred=y_test_pred,
                    y_proba=clf.predict_proba(X_test),
                    model_classes=clf.classes_,
                )
            )

            average_roc_auc_test.append(test_roc_auc)
            average_roc_auc_valid.append(valid_roc_auc)
            average_macro_f1_test.append(test_macro_f1)
            average_macro_f1_valid.append(valid_macro_f1)
        print('=================================================')
        print(f"Dataset: {args.dataset}, Sub Task: {args.subtask}, Knowledge Type: {args.knowledge_type}, Sample_number: {args.num_samples}")
        print(f"Estimator: {get_estimator_label(is_classification=True)}")
        print(f"Average test ROC-AUC: {np.mean(average_roc_auc_test)}")
        print(f"Average valid ROC-AUC: {np.mean(average_roc_auc_valid)}")
        print(f"Average test macro F1: {np.mean(average_macro_f1_test)}")
        print(f"Average valid macro F1: {np.mean(average_macro_f1_valid)}")

        # standard deviation
        print(f"Standard deviation of test ROC-AUC: {np.std(average_roc_auc_test)}")
        print(f"Standard deviation of valid ROC-AUC: {np.std(average_roc_auc_valid)}")
        print(f"Standard deviation of test macro F1: {np.std(average_macro_f1_test)}")
        print(f"Standard deviation of valid macro F1: {np.std(average_macro_f1_valid)}")
        print('===================================================')

        # store the results
        file_name = f'{get_result_prefix(subtask_name)}_rules_test_roc_auc.txt'
        file_path = os.path.join(result_folder, file_name)
        with open(file_path, 'w') as f:
            for item in average_roc_auc_test:
                f.write("%s\n" % item)
            f.write(f"\n\nAverage test ROC-AUC: {np.mean(average_roc_auc_test)} \n")
            f.write(f"Standard deviation of test ROC-AUC: {np.std(average_roc_auc_test)}")

        file_name = f'{get_result_prefix(subtask_name)}_rules_test_macro_f1.txt'
        file_path = os.path.join(result_folder, file_name)
        with open(file_path, 'w') as f:
            for item in average_macro_f1_test:
                f.write("%s\n" % item)
            f.write(f"\n\nAverage test macro F1: {np.mean(average_macro_f1_test)} \n")
            f.write(f"Standard deviation of test macro F1: {np.std(average_macro_f1_test)}")

        classification_metrics_path, sample_predictions_path = write_classification_feedback_outputs(
            metric_rows=classification_metric_rows,
            prediction_rows=sample_prediction_rows,
            output_dir=result_folder,
            result_prefix=get_result_prefix(subtask_name),
        )
        print(f"Saved per-seed classification metrics to {classification_metrics_path}")
        print(f"Saved per-sample predictions to {sample_predictions_path}")
    elif args.dataset in ['esol', 'lipophilicity', 'freesolv']:
        rmse_test_list = []
        rmse_valid_list = []
        for seed in seeds:
            clf = build_estimator(seed=seed, is_classification=False)

            clf.fit(X_train, y_train)
            y_valid_pred = clf.predict(X_valid)
            y_test_pred = clf.predict(X_test)
            # Compute the RMSE
            rmse_test = sqrt(mean_squared_error(y_test, y_test_pred))
            rmse_valid = sqrt(mean_squared_error(y_valid, y_valid_pred))

            bundle_path, metadata_path = save_trained_model_artifact(
                clf=clf,
                seed=seed,
                is_classification=False,
                feature_backend=feature_backend,
                preprocessor=preprocessor,
                input_feature_names=input_feature_names,
                surviving_feature_names=surviving_feature_names,
                feature_description_map=feature_description_map,
                backend_payload=backend_payload,
                metrics=build_selection_metrics(
                    valid_metric_name='valid_rmse',
                    valid_metric_value=rmse_valid,
                    test_metrics={'test_rmse': rmse_test},
                    higher_is_better=False,
                ),
            )
            print(f"Saved trained model bundle to {bundle_path}")
            print(f"Saved trained model metadata to {metadata_path}")
            importance_records.extend(
                collect_feature_importance_rows(
                    clf=clf,
                    feature_names=surviving_feature_names,
                    feature_descriptions=feature_description_map,
                    dataset=args.dataset,
                    subtask=args.subtask,
                    knowledge_type=args.knowledge_type,
                    seed=seed,
                    default_source=default_importance_source,
                    source_map=feature_source_map,
                )
            )
            rmse_test_list.append(rmse_test)
            rmse_valid_list.append(rmse_valid)

        print(f"Estimator: {get_estimator_label(is_classification=False)}")
        print(f"Average test RMSE: {np.mean(rmse_test_list)}")
        print(f"Average valid RMSE: {np.mean(rmse_valid_list)}")

        # standard deviation
        print(f"Standard deviation of test RMSE: {np.std(rmse_test_list)}")
        print(f"Standard deviation of valid RMSE: {np.std(rmse_valid_list)}")

        # store the results
        file_name = f'{get_result_prefix(subtask_name)}_rules_test_rmse.txt'
        file_path = os.path.join(result_folder, file_name)
        with open(file_path, 'w') as f:
            for item in rmse_test_list:
                f.write("%s\n" % item)
            f.write(f"\n\nAverage test RMSE: {np.mean(rmse_test_list)} \n")
            f.write(f"Standard deviation of test RMSE: {np.std(rmse_test_list)}")
    elif args.dataset == 'qm9':
        mae_test_list = []
        mae_valid_list = []

        for seed in seeds:
            clf = build_estimator(seed=seed, is_classification=False)

            clf.fit(X_train, y_train)
            y_valid_pred = clf.predict(X_valid)
            y_test_pred = clf.predict(X_test)

            # Compute the MAE
            mae_test = mean_absolute_error(y_test, y_test_pred)
            mae_valid = mean_absolute_error(y_valid, y_valid_pred)

            bundle_path, metadata_path = save_trained_model_artifact(
                clf=clf,
                seed=seed,
                is_classification=False,
                feature_backend=feature_backend,
                preprocessor=preprocessor,
                input_feature_names=input_feature_names,
                surviving_feature_names=surviving_feature_names,
                feature_description_map=feature_description_map,
                backend_payload=backend_payload,
                metrics=build_selection_metrics(
                    valid_metric_name='valid_mae',
                    valid_metric_value=mae_valid,
                    test_metrics={'test_mae': mae_test},
                    higher_is_better=False,
                ),
            )
            print(f"Saved trained model bundle to {bundle_path}")
            print(f"Saved trained model metadata to {metadata_path}")
            importance_records.extend(
                collect_feature_importance_rows(
                    clf=clf,
                    feature_names=surviving_feature_names,
                    feature_descriptions=feature_description_map,
                    dataset=args.dataset,
                    subtask=args.subtask,
                    knowledge_type=args.knowledge_type,
                    seed=seed,
                    default_source=default_importance_source,
                    source_map=feature_source_map,
                )
            )
            mae_test_list.append(mae_test)
            mae_valid_list.append(mae_valid)
        print(f"Estimator: {get_estimator_label(is_classification=False)}")
        print(f"Average test MAE: {np.mean(mae_test_list)}")
        print(f"Average valid MAE: {np.mean(mae_valid_list)}")

        # standard deviation
        print(f"Standard deviation of test MAE: {np.std(mae_test_list)}")
        print(f"Standard deviation of valid MAE: {np.std(mae_valid_list)}")
        # store the results
        file_name = f'{get_result_prefix(subtask_name)}_rules_test_mae.txt'
        file_path = os.path.join(result_folder, file_name)
        with open(file_path, 'w') as f:
            for item in mae_test_list:
                f.write("%s\n" % item)
            f.write(f"\n\nAverage test MAE: {np.mean(mae_test_list)} \n")
            f.write(f"Standard deviation of test MAE: {np.std(mae_test_list)}")

    per_seed_path, summary_path, summary_df = write_rule_importance_outputs(
        records=importance_records,
        output_dir=result_folder,
        result_prefix=get_result_prefix(subtask_name),
    )
    print(f"Saved per-seed rule importance to {per_seed_path}")
    print(f"Saved aggregated rule importance to {summary_path}")
    print("Top 10 features by mean importance:")
    print(summary_df.head(10).to_string(index=False))


def split_string_into_parts(s, max_lines_per_part=10):
    lines = s.splitlines()
    num_parts = math.ceil(len(lines) / max_lines_per_part)
    split_points = [len(lines) * i // num_parts for i in range(num_parts + 1)]
    return ['\n'.join(lines[split_points[i]:split_points[i + 1]]) for i in range(num_parts)]


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    dataset = args.dataset
    model = args.model

    if args.feature_backend != 'generated_rules':
        feature_backend = build_feature_backend()
        evaluation(feature_backend)
    else:
        feature_source_map = None
        if args.subtask != '':
            if args.dataset in ['tox21', 'sider']:
                subtask_name = f'{args.dataset}_{args.subtask}'
            elif args.dataset == 'qm9' and args.num_samples == 50:
                subtask_name = args.subtask
            else:
                raise NotImplementedError(f"Folder Name error")
        else:
            subtask_name = args.dataset

        file_folder = os.path.join(args.code_gen_folder, args.model, args.dataset)
        synthesize_folder = os.path.join(file_folder, 'synthesize')
        synthesize_file_name = f'{args.model}_{subtask_name}_pk_rules.txt'
        synthesize_file_path = os.path.join(synthesize_folder, synthesize_file_name)

        if args.knowledge_type != 'synthesize' and args.num_samples not in [30, 50]:
            raise NotImplementedError(f"num_samples should be 30 or 50")

        inference_folder = os.path.join(file_folder, 'inference', f"sample_{args.num_samples}")
        inference_file_name = f'{args.model}_{subtask_name}_dk_rules.txt'
        inference_file_path = os.path.join(inference_folder, inference_file_name)

        if args.knowledge_type == 'synthesize':
            with open(synthesize_file_path, 'r') as f:
                generated_code = f.read()
            feature_source_map = {name: 'synthesize' for name in extract_function_names(generated_code)}
        elif args.knowledge_type == 'inference':
            with open(inference_file_path, 'r') as f:
                generated_code = f.read()
            feature_source_map = {name: 'inference' for name in extract_function_names(generated_code)}
        elif args.knowledge_type == 'all':
            with open(synthesize_file_path, 'r') as f:
                synthesize_code = f.read()
            with open(inference_file_path, 'r') as f:
                inference_code = f.read()
            generated_code = synthesize_code + '\n' + inference_code  # combine synthesize_code and inference_code
            feature_source_map = {name: 'synthesize' for name in extract_function_names(synthesize_code)}
            feature_source_map.update({name: 'inference' for name in extract_function_names(inference_code)})
        else:
            raise NotImplementedError(f"Knowledge_type is wrong.(synthesize/inference/all)")

        exec(generated_code, globals())
        function_names = extract_function_names(generated_code)
        feature_backend = build_feature_backend(function_names)
        evaluation(
            feature_backend,
            feature_source_map=feature_source_map,
            generated_code=generated_code,
            function_names=function_names,
        )
