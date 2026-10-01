#!/usr/bin/env python3
"""Run and persist the reusable v0 EDA analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from gigaschool_rag.versions.v0 import (
    dataset_root,
    eda_output_root,
)
from gigaschool_rag.versions.v0.eda import (
    analyze_dataset,
    write_analysis,
)


COURSE_ROOT = Path(__file__).resolve().parents[2]


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
        default=eda_output_root(COURSE_ROOT),
    )
    args = parser.parse_args()
    result = analyze_dataset(args.dataset_root)
    write_analysis(result, args.output)
    print(
        {
            "documents": len(result["documents"]),
            "raw_tokens": result["lengths"]["raw_json"][
                "tokens_total"
            ],
            "semantic_tokens": result["lengths"][
                "summary_facts_evidence"
            ]["tokens_total"],
            "exact_groups": len(
                result["exact_duplicate_groups"]
            ),
            "near_pairs": len(result["near_duplicate_pairs"]),
            "clusters": result["clusters"]["selected_k"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
