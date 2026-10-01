#!/usr/bin/env python3
"""Validate a generated v0 release without requiring image files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent / "output"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.dataset_root
    descriptions = read_jsonl(root / "annotations/vlm_inferred/descriptions.jsonl")
    verifications = read_jsonl(root / "annotations/vlm_inferred/verifications.jsonl")
    facts = read_jsonl(root / "annotations/vlm_inferred/facts_all.jsonl")
    trusted = read_jsonl(root / "annotations/vlm_inferred/facts_trusted.jsonl")
    untrusted = read_jsonl(root / "annotations/vlm_inferred/facts_untrusted.jsonl")
    chunks = read_jsonl(root / "offline_chunking/fact_chunks_all.jsonl")
    trusted_chunks = read_jsonl(
        root / "offline_chunking/fact_chunks_trusted.jsonl"
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    counts = manifest["counts"]

    assert len(descriptions) == len(verifications) == counts["images"]
    assert len(facts) == len(trusted) + len(untrusted)
    assert len(facts) == counts["unique_facts"]
    assert len(trusted) == counts["trusted_facts"]
    assert len(untrusted) == counts["untrusted_facts"]
    assert len({x["fact_id"] for x in facts}) == len(facts)
    assert all(x["trusted_fact"] for x in trusted)
    assert all(not x["trusted_fact"] for x in untrusted)
    image_ids = {x["image_id"] for x in descriptions}
    assert {x["image_id"] for x in descriptions} == image_ids
    assert {x["image_id"] for x in verifications} == image_ids
    assert {x["image_id"] for x in facts} <= image_ids
    assert all(
        x["image_rel_path"].startswith("../images/") for x in facts
    )
    assert len(chunks) == len(facts)
    assert len(trusted_chunks) == len(trusted)
    assert {x["chunk_id"] for x in chunks} == {
        x["fact_id"] for x in facts
    }
    print(
        {
            "ok": True,
            "images": len(image_ids),
            "facts": len(facts),
            "trusted": len(trusted),
            "untrusted": len(untrusted),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
