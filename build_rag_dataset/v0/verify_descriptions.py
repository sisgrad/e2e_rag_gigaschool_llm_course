#!/usr/bin/env python3
"""Verify generated v0 descriptions with the configured verifier model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from cursor_vlm import load_config, run_cursor_json


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "input"
DEFAULT_DESCRIPTIONS = (
    HERE / "output/annotations/vlm_inferred/descriptions.jsonl"
)
DEFAULT_OUTPUT = (
    HERE / "output/annotations/vlm_inferred/verifications.jsonl"
)
DEFAULT_CACHE = HERE / ".cache"
COURSE_ROOT = HERE.parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_output(
    data: dict[str, Any] | None,
    image_id: str,
    expected: list[str],
) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "schema_valid": False,
            "missing_labels": expected,
            "extra_labels": [],
            "duplicate_labels": [],
            "errors": ["missing_or_non_object_output"],
        }
    errors = []
    if data.get("image_id") != image_id:
        errors.append("image_id_mismatch")
    verdicts = data.get("label_verdicts")
    if not isinstance(verdicts, list):
        verdicts = []
        errors.append("label_verdicts_not_list")
    emitted = [
        row.get("label_id")
        for row in verdicts
        if isinstance(row, dict) and row.get("label_id")
    ]
    missing = sorted(set(expected) - set(emitted))
    extra = sorted(set(emitted) - set(expected))
    duplicates = sorted(
        {label for label in emitted if emitted.count(label) > 1}
    )
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            errors.append("non_object_verdict")
            continue
        if verdict.get("verdict") not in {
            "supported",
            "unsupported",
            "uncertain",
        }:
            errors.append(f"bad_verdict:{verdict.get('label_id')}")
        for key in ("sentence_faithful", "evidence_faithful"):
            if not isinstance(verdict.get(key), bool):
                errors.append(
                    f"bad_{key}:{verdict.get('label_id')}"
                )
    return {
        "schema_valid": (
            not errors and not missing and not extra and not duplicates
        ),
        "missing_labels": missing,
        "extra_labels": extra,
        "duplicate_labels": duplicates,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--descriptions", type=Path, default=DEFAULT_DESCRIPTIONS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--producer-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(args.input_root / "config/models.yaml")
    ontology = yaml.safe_load(
        (args.input_root / "ontology/labels.yaml").read_text(
            encoding="utf-8"
        )
    )
    ontology_prompt = [
        {
            "label_id": row["label_id"],
            "display_name": row["display_name"],
            "definition": row["definition"],
            "confusable_with": row.get("confusable_with", []),
            "mutually_exclusive_with": row.get(
                "mutually_exclusive_with", []
            ),
        }
        for row in ontology["labels"]
    ]
    image_rows = read_jsonl(
        args.input_root / "provenance/images.jsonl"
    )
    images = {row["image_id"]: row for row in image_rows}
    descriptions = [
        row
        for row in read_jsonl(args.descriptions)
        if row["run_id"] == args.producer_run_id
    ]
    descriptions = list(
        {row["image_id"]: row for row in descriptions}.values()
    )
    producer_hashes = {
        row["image_id"]: row.get("provenance_hash")
        for row in descriptions
    }
    prompt_template = (
        args.input_root / "prompts/verifier.md"
    ).read_text(encoding="utf-8")

    completed = set()
    if args.output.exists():
        completed = {
            row["image_id"]
            for row in read_jsonl(args.output)
            if row.get("run_id") == args.run_id
            and row.get("producer_run_id") == args.producer_run_id
            and row.get("producer_provenance_hash")
            == producer_hashes.get(row["image_id"])
            and row.get("validation", {}).get("schema_valid")
            and row.get("verification") is not None
            and not row.get("verifier", {}).get("error")
        }

    for index, description in enumerate(descriptions, 1):
        image_id = description["image_id"]
        if image_id in completed:
            continue
        expected = [
            row["label_id"] for row in description["allowed_labels"]
        ]
        prompt = (
            prompt_template
            + "\n\nIMAGE_ID:\n"
            + image_id
            + "\n\nFULL_ONTOLOGY:\n"
            + json.dumps(
                ontology_prompt,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n\nALLOWED_LABELS:\n"
            + json.dumps(
                description["allowed_labels"], ensure_ascii=False
            )
            + "\n\nPRODUCER_OUTPUT:\n"
            + json.dumps(
                description.get("description"), ensure_ascii=False
            )
            + "\n\nPRODUCER_VALIDATION:\n"
            + json.dumps(
                description.get("validation"), ensure_ascii=False
            )
        )
        image_relative = images[image_id]["source_image_rel_path"]
        result = run_cursor_json(
            role="verifier",
            image_path=args.images_root / image_relative,
            prompt=prompt,
            config=config,
            cache_root=args.cache_root,
            cache_namespace=(
                f"{args.run_id}:{args.producer_run_id}:"
                f"{description.get('provenance_hash')}:{image_id}"
            ),
            workspace=COURSE_ROOT,
        )
        validation = validate_output(
            result.get("data"), image_id, expected
        )
        row = {
            "schema_version": "rag-verification-1",
            "run_id": args.run_id,
            "producer_run_id": args.producer_run_id,
            "producer_provenance_hash": description.get(
                "provenance_hash"
            ),
            "image_id": image_id,
            "verification": result.get("data"),
            "validation": validation,
            "verifier": {
                key: result.get(key)
                for key in (
                    "model",
                    "sdk_version",
                    "created_at",
                    "cache_key",
                    "cache_hit",
                    "attempts",
                    "error",
                )
                if key in result
            },
        }
        append_jsonl(args.output, row)
        error = row["verifier"].get("error")
        print(
            f"[{index}/{len(descriptions)}] {image_id}: "
            f"valid={validation['schema_valid']}"
            + (f" error={error}" if error else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
