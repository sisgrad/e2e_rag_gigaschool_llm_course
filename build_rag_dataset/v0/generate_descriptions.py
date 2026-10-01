#!/usr/bin/env python3
"""Generate v0 producer descriptions from contracts and supplied images."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from cursor_vlm import (
    load_config,
    run_cursor_json,
    stable_hash,
)


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "input"
DEFAULT_OUTPUT = (
    HERE / "output/annotations/vlm_inferred/descriptions.jsonl"
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


def compact_ontology(ontology: dict[str, Any]) -> list[dict[str, Any]]:
    return [
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


def validate_output(
    data: dict[str, Any] | None,
    image_id: str,
    allowed: list[dict[str, str]],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    expected = [row["label_id"] for row in allowed]
    if not isinstance(data, dict):
        return {
            "schema_valid": False,
            "expected_labels": expected,
            "emitted_labels": [],
            "missing_labels": expected,
            "extra_labels": [],
            "duplicate_labels": [],
            "lexical_leaks": [],
            "errors": ["missing_or_non_object_output"],
        }
    errors = []
    if data.get("image_id") != image_id:
        errors.append("image_id_mismatch")
    facts = data.get("facts")
    if not isinstance(facts, list):
        facts = []
        errors.append("facts_not_list")
    emitted = [
        row.get("label_id")
        for row in facts
        if isinstance(row, dict) and row.get("label_id")
    ]
    duplicates = sorted(
        {label for label in emitted if emitted.count(label) > 1}
    )
    missing = sorted(set(expected) - set(emitted))
    extra = sorted(set(emitted) - set(expected))
    for fact in facts:
        if not isinstance(fact, dict):
            errors.append("non_object_fact")
            continue
        if fact.get("visibility") not in {
            "clear",
            "partial",
            "uncertain",
        }:
            errors.append(f"bad_visibility:{fact.get('label_id')}")
        for key in ("evidence", "sentence"):
            if (
                not isinstance(fact.get(key), str)
                or not fact[key].strip()
            ):
                errors.append(f"missing_{key}:{fact.get('label_id')}")
    description = data.get("description", "")
    if not isinstance(description, str) or not description.strip():
        errors.append("missing_description")

    allowed_set = set(expected)
    combined = " ".join(
        [description]
        + [
            f"{row.get('evidence', '')} {row.get('sentence', '')}"
            for row in facts
            if isinstance(row, dict)
        ]
    ).casefold()
    leaks = []
    for label in ontology["labels"]:
        if label["label_id"] in allowed_set:
            continue
        phrase = label["display_name"].casefold()
        if len(phrase) >= 4 and re.search(
            rf"(?<!\w){re.escape(phrase)}(?!\w)", combined
        ):
            leaks.append(label["label_id"])
    return {
        "schema_valid": (
            not errors and not missing and not extra and not duplicates
        ),
        "expected_labels": expected,
        "emitted_labels": emitted,
        "missing_labels": missing,
        "extra_labels": extra,
        "duplicate_labels": duplicates,
        "lexical_leaks": sorted(set(leaks)),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-labels-per-image", type=int)
    args = parser.parse_args()

    config = load_config(args.input_root / "config/models.yaml")
    ontology = yaml.safe_load(
        (args.input_root / "ontology/labels.yaml").read_text(
            encoding="utf-8"
        )
    )
    contracts = read_jsonl(
        args.input_root / "annotations/image_label_contracts.jsonl"
    )
    image_rows = read_jsonl(
        args.input_root / "provenance/images.jsonl"
    )
    images = {row["image_id"]: row for row in image_rows}
    contracts = sorted(contracts, key=lambda row: row["image_id"])
    if args.max_images is not None:
        contracts = contracts[: args.max_images]
    prompt_template = (
        args.input_root / "prompts/producer.md"
    ).read_text(encoding="utf-8")
    ontology_prompt = compact_ontology(ontology)

    completed = set()
    if args.output.exists():
        completed = {
            row["image_id"]
            for row in read_jsonl(args.output)
            if row.get("run_id") == args.run_id
            and row.get("validation", {}).get("schema_valid")
            and row.get("description") is not None
            and not row.get("producer", {}).get("error")
        }

    for index, contract in enumerate(contracts, 1):
        image_id = contract["image_id"]
        if image_id in completed:
            continue
        allowed = contract["allowed_labels"]
        if args.max_labels_per_image is not None:
            selected_ids = set(
                list(
                    dict.fromkeys(
                        row["label_id"] for row in allowed
                    )
                )[: args.max_labels_per_image]
            )
            allowed = [
                row for row in allowed if row["label_id"] in selected_ids
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
            + json.dumps(allowed, ensure_ascii=False)
        )
        image_relative = images[image_id]["source_image_rel_path"]
        result = run_cursor_json(
            role="producer",
            image_path=args.images_root / image_relative,
            prompt=prompt,
            config=config,
            cache_root=args.cache_root,
            cache_namespace=f"{args.run_id}:{image_id}",
            workspace=COURSE_ROOT,
        )
        validation = validate_output(
            result.get("data"), image_id, allowed, ontology
        )
        row = {
            "schema_version": "rag-description-1",
            "run_id": args.run_id,
            "image_id": image_id,
            "image_rel_path": image_relative,
            "allowed_labels": allowed,
            "description": result.get("data"),
            "validation": validation,
            "producer": {
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
            "provenance_hash": stable_hash(
                {
                    "ontology": ontology.get("ontology_hash"),
                    "contract": contract,
                    "prompt": prompt_template,
                }
            ),
        }
        append_jsonl(args.output, row)
        error = row["producer"].get("error")
        print(
            f"[{index}/{len(contracts)}] {image_id}: "
            f"valid={validation['schema_valid']}"
            + (f" error={error}" if error else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
