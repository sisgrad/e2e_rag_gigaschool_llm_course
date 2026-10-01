#!/usr/bin/env python3
"""Derive predictable retrieval chunks from v0 structured facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gigaschool_rag.versions.v0 import (
    FACT_CHUNK_SCHEMA_VERSION,
)


DEFAULT_ROOT = Path(__file__).resolve().parent / "output"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    facts = read_jsonl(
        args.dataset_root / "annotations/vlm_inferred/facts_all.jsonl"
    )
    chunks = []
    for fact in facts:
        location = fact["locations"][0]
        evidence = (fact.get("evidence") or "").strip()
        sentence = (fact.get("sentence") or "").strip()
        text = sentence if not evidence else f"{sentence} Evidence: {evidence}"
        chunks.append(
            {
                "schema_version": FACT_CHUNK_SCHEMA_VERSION,
                "chunk_id": fact["fact_id"],
                "parent_document_id": fact["image_id"],
                "image_id": fact["image_id"],
                "image_rel_path": fact["image_rel_path"],
                "text": text,
                "label_id": fact["label_id"],
                "layer": location["layer"],
                "entity": location["entity"],
                "trusted_fact": fact["trusted_fact"],
                "agreement_state": fact["agreement_state"],
                "derivation": (
                    "Deterministic one-fact-per-chunk projection from the structured "
                    "producer response; no semantic boundary detection or overlap."
                ),
            }
        )
    write_jsonl(args.dataset_root / "offline_chunking/fact_chunks_all.jsonl", chunks)
    write_jsonl(
        args.dataset_root / "offline_chunking/fact_chunks_trusted.jsonl",
        [x for x in chunks if x["trusted_fact"]],
    )
    print({"all": len(chunks), "trusted": sum(x["trusted_fact"] for x in chunks)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
