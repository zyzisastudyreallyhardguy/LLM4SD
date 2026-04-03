from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


FEEDBACK_BACKEND_PATTERN = re.compile(r"^(?P<base_backend>[a-z0-9_]+)_feedback_v(?P<version>\d{3})$")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_VARIANTS_ROOT = _REPO_ROOT / "codex_generated_code_variants"


@dataclass(frozen=True)
class BackendSource:
    backend_name: str
    package_import: str
    package_dir: Path
    feature_module_stem: str
    feature_module_filename: str
    base_backend: str
    parent_backend: str
    metadata: dict[str, object] | None = None


def is_feedback_backend_name(name: str) -> bool:
    return FEEDBACK_BACKEND_PATTERN.fullmatch(name) is not None


def resolve_variant_package_dir(backend_name: str) -> Path:
    return _VARIANTS_ROOT / backend_name


def resolve_variant_metadata_path(backend_name: str) -> Path:
    return resolve_variant_package_dir(backend_name) / "variant_metadata.json"


def load_variant_metadata(backend_name: str) -> dict[str, object]:
    metadata_path = resolve_variant_metadata_path(backend_name)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing variant metadata file: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _discover_feature_module_stem(package_dir: Path) -> str:
    feature_modules = sorted(
        path.stem
        for path in package_dir.glob("*_rule_features.py")
        if path.is_file()
    )
    if len(feature_modules) != 1:
        raise ValueError(
            f"Expected exactly one '*_rule_features.py' file in {package_dir}, found {len(feature_modules)}."
        )
    return feature_modules[0]


def _resolve_static_backend_source(backend_name: str) -> BackendSource:
    adapter_path = _REPO_ROOT / "feature_backends" / f"{backend_name}.py"
    if not adapter_path.exists():
        raise ValueError(f"Unknown static backend: {backend_name}")

    adapter_text = adapter_path.read_text(encoding="utf-8")
    match = re.search(
        r"from\s+(codex_generated_code(?:\.[A-Za-z0-9_]+)+)\s+import\s*\(?\s*([A-Za-z0-9_]+)",
        adapter_text,
    )
    if match is None:
        raise ValueError(
            f"Could not infer codex_generated_code source module from backend adapter: {adapter_path}"
        )

    package_import = match.group(1)
    feature_module_stem = match.group(2)
    package_dir = _REPO_ROOT / package_import.replace(".", "/")
    feature_module_filename = f"{feature_module_stem}.py"

    if not package_dir.exists():
        raise FileNotFoundError(f"Resolved package directory does not exist: {package_dir}")
    if not (package_dir / feature_module_filename).exists():
        raise FileNotFoundError(
            f"Resolved feature module does not exist: {package_dir / feature_module_filename}"
        )

    return BackendSource(
        backend_name=backend_name,
        package_import=package_import,
        package_dir=package_dir,
        feature_module_stem=feature_module_stem,
        feature_module_filename=feature_module_filename,
        base_backend=backend_name,
        parent_backend=backend_name,
    )


def _resolve_feedback_backend_source(backend_name: str) -> BackendSource:
    metadata = load_variant_metadata(backend_name)
    package_dir = resolve_variant_package_dir(backend_name)
    feature_module_stem = str(metadata.get("feature_module") or _discover_feature_module_stem(package_dir))
    feature_module_filename = f"{feature_module_stem}.py"
    return BackendSource(
        backend_name=backend_name,
        package_import=f"codex_generated_code_variants.{backend_name}",
        package_dir=package_dir,
        feature_module_stem=feature_module_stem,
        feature_module_filename=feature_module_filename,
        base_backend=str(metadata.get("base_backend") or backend_name),
        parent_backend=str(metadata.get("parent_backend") or backend_name),
        metadata=metadata,
    )


def resolve_backend_source(backend_name: str) -> BackendSource:
    if is_feedback_backend_name(backend_name):
        return _resolve_feedback_backend_source(backend_name)
    return _resolve_static_backend_source(backend_name)
