#!/usr/bin/env python3
"""Build the v0 release from immutable annotation inputs, without images."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from gigaschool_rag.versions.v0 import FACT_SCHEMA_VERSION


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "input"
DEFAULT_OUTPUT = HERE / "output"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wilson(success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    probability = success / total
    denominator = 1 + z * z / total
    centre = (probability + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            (probability * (1 - probability) + z * z / (4 * total))
            / total
        )
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def subset_rows(
    descriptions: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    images: list[dict[str, Any]],
    max_images: int | None,
    max_labels_per_image: int | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    descriptions = sorted(descriptions, key=lambda row: row["image_id"])
    if max_images is not None:
        descriptions = descriptions[:max_images]
    image_ids = {row["image_id"] for row in descriptions}
    verification_by_image = {
        row["image_id"]: row
        for row in verifications
        if row["image_id"] in image_ids
    }
    contract_by_image = {
        row["image_id"]: row for row in contracts if row["image_id"] in image_ids
    }
    image_by_id = {
        row["image_id"]: row for row in images if row["image_id"] in image_ids
    }

    selected_descriptions = []
    selected_verifications = []
    selected_contracts = []
    for source_description in descriptions:
        image_id = source_description["image_id"]
        description = deepcopy(source_description)
        verification = deepcopy(verification_by_image[image_id])
        contract = deepcopy(contract_by_image[image_id])
        if max_labels_per_image is not None:
            selected_ids = list(
                dict.fromkeys(
                    item["label_id"] for item in description["allowed_labels"]
                )
            )[:max_labels_per_image]
            selected = set(selected_ids)
            description["allowed_labels"] = [
                item
                for item in description["allowed_labels"]
                if item["label_id"] in selected
            ]
            payload = description.get("description") or {}
            payload["facts"] = [
                fact
                for fact in payload.get("facts", [])
                if fact["label_id"] in selected
            ]
            validation = description.get("validation") or {}
            for key in (
                "expected_labels",
                "emitted_labels",
                "missing_labels",
                "extra_labels",
                "duplicate_labels",
            ):
                if key in validation:
                    validation[key] = [
                        label for label in validation[key] if label in selected
                    ]
            verification_payload = verification.get("verification") or {}
            verification_payload["label_verdicts"] = [
                verdict
                for verdict in verification_payload.get("label_verdicts", [])
                if verdict["label_id"] in selected
            ]
            for key in ("allowed_labels", "effective_labels"):
                contract[key] = [
                    item
                    for item in contract.get(key, [])
                    if item["label_id"] in selected
                ]
        selected_descriptions.append(description)
        selected_verifications.append(verification)
        selected_contracts.append(contract)

    return (
        selected_descriptions,
        selected_verifications,
        selected_contracts,
        [image_by_id[image_id] for image_id in sorted(image_ids)],
    )


def compute_agreement(
    descriptions: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
    ontology: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = {row["label_id"]: row for row in ontology["labels"]}
    verifier_by_image = {row["image_id"]: row for row in verifications}
    ledger: list[dict[str, Any]] = []
    trusted_images = 0

    for image_index, description in enumerate(descriptions):
        image_id = description["image_id"]
        producer_facts = {
            fact["label_id"]: fact
            for fact in (description.get("description") or {}).get("facts", [])
        }
        verification_row = verifier_by_image.get(image_id)
        verification = (verification_row or {}).get("verification") or {}
        verdicts = {
            verdict["label_id"]: verdict
            for verdict in verification.get("label_verdicts", [])
        }
        reasons: list[str] = []
        producer_valid = bool(
            description.get("validation", {}).get("schema_valid")
        )
        verifier_valid = bool(
            (verification_row or {}).get("validation", {}).get("schema_valid")
        )
        if not producer_valid:
            reasons.append("producer_schema_invalid")
        if not verification_row:
            reasons.append("missing_verification")
        elif not verifier_valid:
            reasons.append("verifier_schema_invalid")
        if description.get("validation", {}).get("lexical_leaks"):
            reasons.append("out_of_whitelist_lexical_leak")
        if verification.get("out_of_whitelist_mention"):
            reasons.append("verifier_found_out_of_whitelist_mention")
        if verification.get("invented_detail"):
            reasons.append("verifier_found_invented_detail")
        if verification and not verification.get("all_labels_covered", False):
            reasons.append("verifier_found_incomplete_coverage")

        image_rows = []
        for allowed in description["allowed_labels"]:
            label_id = allowed["label_id"]
            fact = producer_facts.get(label_id)
            verdict = verdicts.get(label_id)
            verdict_name = verdict.get("verdict") if verdict else "missing"
            faithful = bool(
                verdict
                and verdict.get("sentence_faithful")
                and verdict.get("evidence_faithful")
            )
            supported = verdict_name == "supported" and faithful
            if fact is None:
                state = "producer_omitted"
                reasons.append(f"producer_omitted:{label_id}")
            elif verdict_name == "unsupported":
                state = "agreed_bad"
                reasons.append(f"unsupported:{label_id}")
            elif verdict_name == "uncertain":
                state = "verifier_uncertain"
                reasons.append(f"uncertain:{label_id}")
            elif verdict_name == "supported" and faithful:
                state = "agreed_good"
            elif verdict_name == "supported":
                state = "producer_only"
                reasons.append(f"unfaithful_text:{label_id}")
            else:
                state = "failed"
                reasons.append(f"missing_verdict:{label_id}")
            label_metadata = metadata[label_id]
            image_rows.append(
                {
                    "run_id": config["agreement_run_id"],
                    "producer_run_id": config["producer_run_id"],
                    "verifier_run_id": config["verifier_run_id"],
                    "image_id": image_id,
                    "label_id": label_id,
                    "layer": allowed["layer"],
                    "entity": allowed["entity"],
                    "frequency": label_metadata["frequency"],
                    "label_status": label_metadata["status"],
                    "merge_target": label_metadata.get("merge_target"),
                    "batch_index": image_index
                    // config["agreement_batch_size"],
                    "producer_emitted": fact is not None,
                    "producer_visibility": (
                        fact.get("visibility") if fact else None
                    ),
                    "producer_schema_valid": producer_valid,
                    "lexical_leak": bool(
                        description.get("validation", {}).get(
                            "lexical_leaks"
                        )
                    ),
                    "verifier_verdict": verdict_name,
                    "verifier_supported": supported,
                    "sentence_faithful": bool(
                        verdict and verdict.get("sentence_faithful")
                    ),
                    "evidence_faithful": bool(
                        verdict and verdict.get("evidence_faithful")
                    ),
                    "agreement_state": state,
                    "is_disagreement": state != "agreed_good",
                    "producer_model": description.get("producer", {}).get(
                        "model"
                    ),
                    "verifier_model": (verification_row or {})
                    .get("verifier", {})
                    .get("model"),
                }
            )
        image_trusted = not reasons and all(
            row["agreement_state"] == "agreed_good" for row in image_rows
        )
        trusted_images += int(image_trusted)
        for row in image_rows:
            row["image_trusted"] = image_trusted
            row["trust_reasons"] = "|".join(sorted(set(reasons)))
        ledger.extend(image_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        grouped[row["label_id"]].append(row)
    category_summary = []
    for label_id, rows in sorted(grouped.items()):
        supported = sum(row["verifier_supported"] for row in rows)
        total = len(rows)
        low, high = wilson(supported, total)
        rate = supported / total
        category_summary.append(
            {
                "label_id": label_id,
                "entity": rows[0]["entity"],
                "facts": total,
                "supported": supported,
                "support_rate": rate,
                "support_wilson_low": low,
                "support_wilson_high": high,
                "disagreement_rate": sum(
                    row["is_disagreement"] for row in rows
                )
                / total,
                "omission_rate": 1
                - sum(row["producer_emitted"] for row in rows) / total,
                "untrusted_image_contribution": sum(
                    not row["image_trusted"] for row in rows
                ),
                "recommended_action": (
                    "insufficient_evidence"
                    if total < 5
                    else "retain"
                    if rate >= 0.8
                    else "review"
                    if rate >= 0.5
                    else "quarantine"
                ),
            }
        )
    state_counts = Counter(row["agreement_state"] for row in ledger)
    summary = {
        "schema_version": 1,
        "run_id": config["agreement_run_id"],
        "generated_at": utc_now(),
        "producer_run_id": config["producer_run_id"],
        "verifier_run_id": config["verifier_run_id"],
        "images": len(descriptions),
        "verified_images": len(verifications),
        "trusted_images": trusted_images,
        "untrusted_images": len(descriptions) - trusted_images,
        "facts": len(ledger),
        "trusted_facts": state_counts["agreed_good"],
        "untrusted_facts": len(ledger) - state_counts["agreed_good"],
        "agreement_states": dict(state_counts),
        "category_summary": category_summary,
    }
    return ledger, summary


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_root
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    config = load_yaml(input_root / "config/build.yaml")
    for key, value in (
        ("producer_run_id", args.producer_run_id),
        ("verifier_run_id", args.verifier_run_id),
        ("agreement_run_id", args.agreement_run_id),
    ):
        if value is not None:
            config[key] = value
    if config["dataset_version"] != "v0":
        raise ValueError("The v0 builder only accepts dataset_version: v0")
    descriptions = read_jsonl(args.descriptions)
    verifications = read_jsonl(args.verifications)
    contracts = read_jsonl(
        input_root / "annotations/image_label_contracts.jsonl"
    )
    images = read_jsonl(input_root / "provenance/images.jsonl")
    descriptions = [
        row
        for row in descriptions
        if row["run_id"] == config["producer_run_id"]
    ]
    descriptions = list(
        {row["image_id"]: row for row in descriptions}.values()
    )
    verifications = [
        row
        for row in verifications
        if row["run_id"] == config["verifier_run_id"]
        and row["producer_run_id"] == config["producer_run_id"]
    ]
    verifications = list(
        {row["image_id"]: row for row in verifications}.values()
    )
    descriptions, verifications, contracts, images = subset_rows(
        descriptions,
        verifications,
        contracts,
        images,
        args.max_images,
        args.max_labels_per_image,
    )
    image_ids = {row["image_id"] for row in descriptions}
    if not image_ids:
        raise ValueError("No descriptions selected")
    if (
        {row["image_id"] for row in verifications} != image_ids
        or {row["image_id"] for row in contracts} != image_ids
        or {row["image_id"] for row in images} != image_ids
    ):
        raise ValueError("Descriptions, verifications, contracts and images differ")

    ontology = load_yaml(input_root / "ontology/labels.yaml")
    ledger, agreement_summary = compute_agreement(
        descriptions, verifications, ontology, config
    )
    ledger_by_key = {
        (row["image_id"], row["label_id"]): row for row in ledger
    }
    ledger_occurrences = Counter(
        (row["image_id"], row["label_id"]) for row in ledger
    )
    verifier_by_image = {row["image_id"]: row for row in verifications}
    images_by_id = {row["image_id"]: row for row in images}

    facts = []
    for description in descriptions:
        image_id = description["image_id"]
        producer_facts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for fact in (description.get("description") or {}).get("facts", []):
            producer_facts[fact["label_id"]].append(fact)
        verdicts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        verification = (
            verifier_by_image[image_id].get("verification") or {}
        )
        for verdict in verification.get("label_verdicts", []):
            verdicts[verdict["label_id"]].append(verdict)
        allowed: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in description["allowed_labels"]:
            allowed[item["label_id"]].append(item)

        for label_id in sorted(allowed):
            producer_fact = (
                producer_facts[label_id][-1]
                if producer_facts[label_id]
                else {}
            )
            verdict = verdicts[label_id][-1] if verdicts[label_id] else {}
            agreement = ledger_by_key[(image_id, label_id)]
            locations = sorted(
                {
                    (item["layer"], item["entity"])
                    for item in allowed[label_id]
                }
            )
            fact_id = f"{image_id}:{locations[0][1]}:{label_id}"
            facts.append(
                {
                    "schema_version": FACT_SCHEMA_VERSION,
                    "fact_id": fact_id,
                    "image_id": image_id,
                    "image_rel_path": (
                        "../images/"
                        f"{config['image_dataset']}/"
                        f"{images_by_id[image_id]['source_image_rel_path']}"
                    ),
                    "label_id": label_id,
                    "locations": [
                        {"layer": layer, "entity": entity}
                        for layer, entity in locations
                    ],
                    "sentence": producer_fact.get("sentence"),
                    "evidence": producer_fact.get("evidence"),
                    "visibility": producer_fact.get("visibility"),
                    "producer_source_occurrences": len(
                        producer_facts[label_id]
                    ),
                    "annotation_source_occurrences": len(allowed[label_id]),
                    "verifier": {
                        "verdict": verdict.get("verdict", "missing"),
                        "sentence_faithful": verdict.get(
                            "sentence_faithful"
                        ),
                        "evidence_faithful": verdict.get(
                            "evidence_faithful"
                        ),
                        "rationale": verdict.get("rationale"),
                    },
                    "agreement_state": agreement["agreement_state"],
                    "trusted_fact": (
                        agreement["agreement_state"] == "agreed_good"
                    ),
                    "trust_reasons": [
                        reason
                        for reason in agreement.get(
                            "trust_reasons", ""
                        ).split("|")
                        if reason
                    ],
                    "source_ledger_occurrences": ledger_occurrences[
                        (image_id, label_id)
                    ],
                    "provenance": {
                        "producer_run_id": config["producer_run_id"],
                        "verifier_run_id": config["verifier_run_id"],
                        "agreement_run_id": config["agreement_run_id"],
                    },
                }
            )
    facts.sort(key=lambda row: row["fact_id"])
    trusted = [row for row in facts if row["trusted_fact"]]
    untrusted = [row for row in facts if not row["trusted_fact"]]

    write_jsonl(
        output / "annotations/vlm_inferred/descriptions.jsonl",
        descriptions,
    )
    write_jsonl(
        output / "annotations/vlm_inferred/verifications.jsonl",
        verifications,
    )
    write_jsonl(
        output / "annotations/gt/image_labels.jsonl",
        contracts,
    )
    write_jsonl(
        output / "annotations/vlm_inferred/facts_all.jsonl", facts
    )
    write_jsonl(
        output / "annotations/vlm_inferred/facts_trusted.jsonl", trusted
    )
    write_jsonl(
        output / "annotations/vlm_inferred/facts_untrusted.jsonl", untrusted
    )

    ontology_by_id = {
        row["label_id"]: row for row in ontology["labels"]
    }
    observed_locations: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for fact in facts:
        for location in fact["locations"]:
            observed_locations[fact["label_id"]].add(
                (location["layer"], location["entity"])
            )
    label_catalog = []
    for label_id, metadata in sorted(ontology_by_id.items()):
        locations = observed_locations.get(label_id, set())
        layers = sorted({layer for layer, _ in locations})
        label_catalog.append(
            {
                "label_id": label_id,
                "display_name": metadata["display_name"],
                "definition": metadata["definition"],
                "layers": layers,
                "label_level": (
                    "base_and_detailed"
                    if len(layers) > 1
                    else "base"
                    if layers == ["base_static"]
                    else "detailed"
                    if layers == ["detailed_static"]
                    else "not_observed"
                ),
                "entities": sorted(
                    {entity for _, entity in locations}
                ),
                "observed_facts": sum(
                    fact["label_id"] == label_id for fact in facts
                ),
                "trusted_facts": sum(
                    fact["label_id"] == label_id
                    and fact["trusted_fact"]
                    for fact in facts
                ),
            }
        )
    copy_file(
        input_root / "ontology/labels.yaml",
        output / "annotations/gt/ontology/labels.yaml",
    )
    dump_json(
        output / "annotations/gt/ontology/label_catalog.json",
        label_catalog,
    )
    copy_file(
        input_root / "prompts/producer.md",
        output / "prompts/producer.md",
    )
    copy_file(
        input_root / "prompts/verifier.md",
        output / "prompts/verifier.md",
    )
    copy_file(
        input_root / "config/models.yaml",
        output / "provenance/models.yaml",
    )
    dump_json(
        output / "provenance/agreement_summary.json",
        agreement_summary,
    )
    copy_file(
        input_root / "dataset_README.md",
        output / "README.md",
    )

    level_counts = Counter(row["label_level"] for row in label_catalog)
    manifest = {
        "schema_version": "v0",
        "description": (
            f"{len(descriptions)} camera images with raw structured producer "
            "output, independent verifier output and normalized fact-level "
            "trust annotations."
        ),
        "counts": {
            "images": len(descriptions),
            "raw_descriptions": len(descriptions),
            "raw_verifications": len(verifications),
            "unique_facts": len(facts),
            "trusted_facts": len(trusted),
            "untrusted_facts": len(untrusted),
            "ontology_labels": len(label_catalog),
            "observed_labels": sum(
                row["label_level"] != "not_observed"
                for row in label_catalog
            ),
            "label_levels": dict(level_counts),
        },
        "semantics": {
            "raw_unit": (
                "one complete structured producer response per image"
            ),
            "fact_unit": "one normalized image-label assertion",
            "missing_label": "unjudged",
            "raw_layer_prechunked": False,
            "derived_fact_chunks_included": True,
        },
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=DEFAULT_INPUT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=DEFAULT_INPUT / "annotations/descriptions.jsonl",
    )
    parser.add_argument(
        "--verifications",
        type=Path,
        default=DEFAULT_INPUT / "annotations/verifications.jsonl",
    )
    parser.add_argument("--producer-run-id")
    parser.add_argument("--verifier-run-id")
    parser.add_argument("--agreement-run-id")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-labels-per-image", type=int)
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    if (
        args.max_labels_per_image is not None
        and args.max_labels_per_image < 1
    ):
        parser.error("--max-labels-per-image must be positive")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
