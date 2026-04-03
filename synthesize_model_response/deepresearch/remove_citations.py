#!/usr/bin/env python3
"""Remove DeepResearch citation markers from generated response files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_GLOB = "**/deepresearch_pk_response.txt"
CITATION_PATTERNS = (
    re.compile(r"\ue200cite\ue202.*?\ue201", re.DOTALL),
    re.compile(r"【[^】]+†[^】]+】"),
)


def clean_text(text: str) -> tuple[str, int]:
    cleaned = text
    replacements = 0
    for pattern in CITATION_PATTERNS:
        cleaned, count = pattern.subn("", cleaned)
        replacements += count
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"(?<=\S) {2,}(?=\S)", " ", cleaned)
    return cleaned, replacements


def iter_target_files(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def process_file(path: Path) -> int:
    original = path.read_text(encoding="utf-8")
    cleaned, replacements = clean_text(original)
    if replacements and cleaned != original:
        path.write_text(cleaned, encoding="utf-8")
    return replacements


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove DeepResearch citation markers from response files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Root directory to search from. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_GLOB,
        help=f"Glob pattern relative to root. Defaults to {DEFAULT_GLOB!r}.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    files = iter_target_files(root, args.pattern)
    if not files:
        print(f"No files matched pattern {args.pattern!r} under {root}.")
        return 0

    total_replacements = 0
    changed_files = 0

    for path in files:
        replacements = process_file(path)
        total_replacements += replacements
        if replacements:
            changed_files += 1
            print(f"Cleaned {replacements} citation(s): {path}")

    print(
        f"Finished. Updated {changed_files} file(s) with "
        f"{total_replacements} citation marker(s) removed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
