#!/usr/bin/env python3
"""Convert TDC KNN_3_per_label JSONL files into scaffold_datasets-style CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


SOURCE_ROOT = Path("/data1/tianang/Projects/Intern-S1/DataPrepare/TDC_prepended/KNN_3_per_label")
OUTPUT_ROOT = Path("/data1/tianang/Projects/LLM4SD/scaffold_datasets/TDC")
SPLITS = ("train", "valid")


def convert_task_split(task_name: str, split: str) -> int:
    input_path = SOURCE_ROOT / split / f"{task_name}.jsonl"
    task_output_dir = OUTPUT_ROOT / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = task_output_dir / f"{task_name}_{split}.csv"

    row_count = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        writer = csv.DictWriter(dst, fieldnames=["smiles", task_name])
        writer.writeheader()

        for line_number, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if "drug" not in record:
                raise KeyError(f"{input_path}:{line_number} is missing required field 'drug'")
            if "Y" not in record:
                raise KeyError(f"{input_path}:{line_number} is missing required field 'Y'")

            writer.writerow(
                {
                    "smiles": record["drug"],
                    task_name: record["Y"],
                }
            )
            row_count += 1

    return row_count


def discover_tasks(split: str) -> list[str]:
    split_dir = SOURCE_ROOT / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    return sorted(path.stem for path in split_dir.glob("*.jsonl"))


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    train_tasks = discover_tasks("train")
    valid_tasks = discover_tasks("valid")
    if train_tasks != valid_tasks:
        raise ValueError(
            "Train/valid task lists do not match.\n"
            f"train={train_tasks}\n"
            f"valid={valid_tasks}"
        )

    summary_rows: list[dict[str, str | int]] = []
    for task_name in train_tasks:
        for split in SPLITS:
            row_count = convert_task_split(task_name, split)
            summary_rows.append(
                {
                    "task": task_name,
                    "split": split,
                    "num_rows": row_count,
                    "source_file": str(SOURCE_ROOT / split / f"{task_name}.jsonl"),
                    "output_file": str(OUTPUT_ROOT / task_name / f"{task_name}_{split}.csv"),
                }
            )
            print(f"Wrote {task_name}_{split}.csv with {row_count} rows")

    summary_path = OUTPUT_ROOT / "conversion_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=["task", "split", "num_rows", "source_file", "output_file"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary file: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
