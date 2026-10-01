"""Reusable retrieval evaluation functions and canonical v0 resources."""

from __future__ import annotations

import json
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import DATASET_VERSION


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_resource_jsonl(name: str) -> list[dict[str, Any]]:
    resource = files(__package__).joinpath(
        "resources", "evaluation", name
    )
    return [
        json.loads(line)
        for line in resource.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_gold() -> list[dict[str, Any]]:
    """Load the immutable 50-question v0 validation/test benchmark."""
    return _load_resource_jsonl("gold.jsonl")


def load_rejected_candidates() -> list[dict[str, Any]]:
    """Load questions rejected while curating the canonical benchmark."""
    return _load_resource_jsonl("rejected_candidates.jsonl")


def evaluate_queries(
    *,
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    """Rank parent images for arbitrary queries and calculate IR metrics."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=16000,
    )
    matrix = vectorizer.fit_transform([row["text"] for row in chunks])
    image_ids = sorted({row["image_id"] for row in chunks})
    image_position = {
        image_id: index for index, image_id in enumerate(image_ids)
    }
    chunk_image_positions = np.asarray(
        [image_position[row["image_id"]] for row in chunks]
    )
    similarities = cosine_similarity(
        vectorizer.transform([row["question"] for row in queries]),
        matrix,
    )

    metric_rows = []
    details = []
    for query_index, query in enumerate(queries):
        scores = np.full(len(image_ids), -np.inf)
        np.maximum.at(
            scores, chunk_image_positions, similarities[query_index]
        )
        ranking = [
            image_ids[index]
            for index in np.argsort(-scores, kind="stable")
        ]
        relevant = set(query["relevant_image_ids"])
        per_k = {}
        for k in k_values:
            retrieved = set(ranking[:k])
            true_positives = len(retrieved & relevant)
            false_positives = len(retrieved - relevant)
            false_negatives = len(relevant - retrieved)
            precision = (
                true_positives / (true_positives + false_positives)
                if true_positives + false_positives
                else 0.0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if true_positives + false_negatives
                else 0.0
            )
            row = {
                "query_id": query["query_id"],
                "k": k,
                "gt_images": len(relevant),
                "tp": true_positives,
                "fp": false_positives,
                "fn": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                ),
            }
            metric_rows.append(row)
            per_k[str(k)] = row
        hits = 0
        precision_sum = 0.0
        for rank, image_id in enumerate(ranking, 1):
            if image_id in relevant:
                hits += 1
                precision_sum += hits / rank
        details.append(
            {
                **query,
                "average_precision": (
                    precision_sum / len(relevant) if relevant else 0.0
                ),
                "metrics_by_k": per_k,
                "retrieved_image_ids_at_50": ranking[:50],
                "unretrieved_gt_image_ids_at_50": sorted(
                    relevant - set(ranking[:50])
                ),
            }
        )

    frame = pd.DataFrame(metric_rows)
    aggregates = []
    for k, group in frame.groupby("k"):
        true_positives = int(group.tp.sum())
        false_positives = int(group.fp.sum())
        false_negatives = int(group.fn.sum())
        micro_precision = true_positives / (
            true_positives + false_positives
        )
        micro_recall = true_positives / (
            true_positives + false_negatives
        )
        aggregates.append(
            {
                "k": int(k),
                "macro_precision": float(group.precision.mean()),
                "macro_recall": float(group.recall.mean()),
                "macro_f1": float(group.f1.mean()),
                "micro_precision": float(micro_precision),
                "micro_recall": float(micro_recall),
                "micro_f1": float(
                    2
                    * micro_precision
                    * micro_recall
                    / (micro_precision + micro_recall)
                    if micro_precision + micro_recall
                    else 0.0
                ),
            }
        )
    return {
        "metrics": metric_rows,
        "details": details,
        "metrics_by_k": aggregates,
        "mean_average_precision": float(
            np.mean([row["average_precision"] for row in details])
        ),
        "chunks": len(chunks),
        "images": len(image_ids),
        "queries": len(queries),
    }


def evaluate_gold_retrieval(
    dataset_root: Path,
    *,
    gold: list[dict[str, Any]] | None = None,
    k_values: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, Any]:
    """Evaluate trusted fact chunks on the canonical answerable gold queries."""
    gold = gold or load_gold()
    queries = [
        {
            "query_id": row["question_id"],
            "question": row["question"],
            "relevant_image_ids": row["relevant_image_ids"],
            "split": row["split"],
            "question_type": row["question_type"],
        }
        for row in gold
        if row.get("answerable", True) and row["relevant_image_ids"]
    ]
    chunks = read_jsonl(
        dataset_root / "offline_chunking/fact_chunks_trusted.jsonl"
    )
    result = evaluate_queries(
        chunks=chunks, queries=queries, k_values=k_values
    )
    result["retriever"] = (
        "TF-IDF unigram+bigram, chunk-to-image max score"
    )
    result["corpus"] = (
        f"hf/{DATASET_VERSION}/offline_chunking/"
        "fact_chunks_trusted.jsonl"
    )
    result["gold"] = "gigaschool_rag.versions.v0 package resource"
    return result


def evaluate_label_retrieval(
    dataset_root: Path,
    *,
    k_values: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, Any]:
    """Evaluate one ontology-derived query per label present in trusted facts."""
    chunks = read_jsonl(
        dataset_root / "offline_chunking/fact_chunks_trusted.jsonl"
    )
    ontology_document = yaml.safe_load(
        (
            dataset_root / "annotations/gt/ontology/labels.yaml"
        ).read_text(encoding="utf-8")
    )
    ontology = {
        row["label_id"]: row for row in ontology_document["labels"]
    }
    images_by_label: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        images_by_label[chunk["label_id"]].add(chunk["image_id"])
    queries = []
    for label_id in sorted(images_by_label):
        metadata = ontology[label_id]
        aliases = [
            alias["text"]
            for alias in metadata.get("aliases", [])
            if alias.get("lang") == "en" and alias.get("text")
        ]
        names = list(
            dict.fromkeys([metadata["display_name"], *aliases])
        )
        queries.append(
            {
                "query_id": label_id,
                "question": "Which images show "
                + " or ".join(names)
                + "?",
                "relevant_image_ids": sorted(
                    images_by_label[label_id]
                ),
            }
        )
    result = evaluate_queries(
        chunks=chunks, queries=queries, k_values=k_values
    )
    result["retriever"] = (
        "TF-IDF unigram+bigram, chunk-to-image max score"
    )
    result["corpus"] = (
        f"hf/{DATASET_VERSION}/offline_chunking/"
        "fact_chunks_trusted.jsonl"
    )
    result["labels"] = len(queries)
    return result
