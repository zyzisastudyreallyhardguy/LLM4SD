from __future__ import annotations

import json
import re
from pathlib import Path

import joblib


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^\w.-]+", "_", str(value).strip())
    return sanitized.strip("._") or "unknown"


def get_task_name(dataset: str, subtask: str) -> str:
    if subtask:
        return f"{dataset}__{subtask}"
    return dataset


def get_checkpoint_family(estimator: str) -> str:
    if estimator == "rf":
        return "forest"
    if estimator == "linear":
        return "linear"
    raise ValueError(f"Unsupported estimator: {estimator}")


def get_checkpoint_dir(
    *,
    checkpoint_root: str,
    estimator: str,
    dataset: str,
    subtask: str,
) -> Path:
    task_name = get_task_name(dataset, subtask)
    return (
        Path(checkpoint_root)
        / get_checkpoint_family(estimator)
        / _sanitize_path_component(task_name)
    )


def build_bundle_stem(
    *,
    model_name: str,
    feature_backend: str,
    knowledge_type: str,
    num_samples: int,
) -> str:
    parts = [
        _sanitize_path_component(model_name),
        _sanitize_path_component(feature_backend),
        _sanitize_path_component(knowledge_type),
    ]
    if knowledge_type in {"inference", "all"}:
        parts.append(f"sample_{num_samples}")
    return "__".join(parts)


def save_model_bundle(
    *,
    bundle: dict[str, object],
    checkpoint_root: str,
) -> tuple[Path, Path]:
    checkpoint_dir = get_checkpoint_dir(
        checkpoint_root=checkpoint_root,
        estimator=str(bundle["estimator"]),
        dataset=str(bundle["dataset"]),
        subtask=str(bundle["subtask"]),
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bundle_stem = build_bundle_stem(
        model_name=str(bundle["model_name"]),
        feature_backend=str(bundle["feature_backend_name"]),
        knowledge_type=str(bundle["knowledge_type"]),
        num_samples=int(bundle["num_samples"]),
    )
    seed = int(bundle["seed"])
    bundle_path = checkpoint_dir / f"{bundle_stem}__seed_{seed}.joblib"
    metadata_path = checkpoint_dir / f"{bundle_stem}__seed_{seed}.json"

    joblib.dump(bundle, bundle_path)

    metadata = {
        "bundle_path": str(bundle_path),
        "dataset": bundle["dataset"],
        "subtask": bundle["subtask"],
        "task_name": bundle["task_name"],
        "estimator": bundle["estimator"],
        "model_class": bundle["model_class"],
        "model_name": bundle["model_name"],
        "seed": bundle["seed"],
        "feature_backend_name": bundle["feature_backend_name"],
        "knowledge_type": bundle["knowledge_type"],
        "num_samples": bundle["num_samples"],
        "surviving_feature_count": len(bundle["surviving_feature_names"]),
        "surviving_feature_names": bundle["surviving_feature_names"],
        "metrics": bundle.get("metrics", {}),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return bundle_path, metadata_path


def load_model_bundle(bundle_path: str | Path) -> dict[str, object]:
    return joblib.load(Path(bundle_path))


def load_bundle_metadata(metadata_path: str | Path) -> dict[str, object]:
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def list_bundle_metadata(bundle_dir: str | Path) -> list[dict[str, object]]:
    directory = Path(bundle_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {directory}")

    metadata_records = []
    for metadata_path in sorted(directory.glob("*.json")):
        metadata_records.append(load_bundle_metadata(metadata_path))
    if not metadata_records:
        raise FileNotFoundError(f"No bundle metadata files were found in {directory}")
    return metadata_records


def select_best_bundle_metadata(bundle_dir: str | Path) -> dict[str, object]:
    metadata_records = list_bundle_metadata(bundle_dir)

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

    return max(metadata_records, key=_selection_key)
