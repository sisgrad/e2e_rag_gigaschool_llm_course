#!/usr/bin/env python3
"""Run and persist the reusable v0 label-wise TF-IDF evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gigaschool_rag.versions.v0 import (
    dataset_root,
    evaluation_output_root,
)
from gigaschool_rag.versions.v0.evaluation import (
    evaluate_label_retrieval,
)


COURSE_ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=dataset_root(COURSE_ROOT),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=evaluation_output_root(COURSE_ROOT),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = evaluate_label_retrieval(args.dataset_root)
    pd.DataFrame(result["metrics"]).to_csv(
        args.output / "fact_retrieval_metrics.csv", index=False
    )
    write_jsonl(
        args.output / "fact_retrieval_details.jsonl",
        result["details"],
    )
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"metrics", "details"}
    }
    (args.output / "fact_retrieval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
