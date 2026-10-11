"""Control Point 2: preprocessing, chunking and retrieval experiments.

The module intentionally keeps experiment logic outside the notebook so every
reported number can be regenerated from the packaged v0 dataset and gold set.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import unicodedata
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import tiktoken
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .evaluation import load_gold, read_jsonl


SEED = 20261007
TOKENIZER_NAME = "cl100k_base"
ENCODER = tiktoken.get_encoding(TOKENIZER_NAME)
DEFAULT_K = (1, 5, 10, 50)
HYDE_PROMPT_VERSION = "v2-three-beams"


def token_count(text: str) -> int:
    """Count tokens using the tokenizer shared by all reported statistics."""
    return len(ENCODER.encode(text))


def load_documents(dataset_root: Path) -> list[dict[str, Any]]:
    """Build the uncleaned image-parent documents defined at Control Point 1.

    A document contains the producer summary followed by every producer fact
    and its visual evidence. Verifier decisions are metadata, not a destructive
    filter: this is deliberately the "retrieval dataset before cleaning".
    """
    descriptions = read_jsonl(
        dataset_root / "annotations/vlm_inferred/descriptions.jsonl"
    )
    verifications = {
        row["image_id"]: row
        for row in read_jsonl(
            dataset_root / "annotations/vlm_inferred/verifications.jsonl"
        )
    }
    documents = []
    for row in descriptions:
        image_id = row["image_id"]
        payload = row["description"]
        facts = payload["facts"]
        text = payload["description"] + "\n" + "\n".join(
            f"{fact['sentence']} Evidence: {fact['evidence']}"
            for fact in facts
        )
        verification = verifications.get(image_id, {})
        decisions = verification.get("verification", {}).get(
            "label_verdicts", []
        )
        accepted = sum(
            decision.get("verdict") == "supported" for decision in decisions
        )
        total = len(decisions)
        producer = row.get("producer", {})
        documents.append(
            {
                "document_id": image_id,
                "image_id": image_id,
                "represented_image_ids": [image_id],
                "text": text,
                "summary": payload["description"],
                "source": "VLM producer+verifier",
                "published_at": producer.get("created_at"),
                "content_type": "camera-image description",
                "language": "en",
                "facts": len(facts),
                "trusted_ratio": accepted / total if total else 0.0,
                "producer_usage": sum(
                    (attempt.get("usage") or {}).get("total_tokens", 0)
                    for attempt in producer.get("attempts", [])
                ),
                "producer_duration_ms": sum(
                    attempt.get("duration_ms") or 0
                    for attempt in producer.get("attempts", [])
                ),
            }
        )
    return documents


def validate_dataset_consistency(dataset_root: Path) -> dict[str, Any]:
    """Check final JSONL counts and gold references before any experiment.

    ``manifest.json`` describes the released normalized dataset. The agreement
    summary is retained as provenance from an earlier pipeline stage and can
    therefore contain pre-normalization counts.
    """
    manifest = json.loads(
        (dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    inferred = dataset_root / "annotations/vlm_inferred"
    facts_all = read_jsonl(inferred / "facts_all.jsonl")
    facts_trusted = read_jsonl(inferred / "facts_trusted.jsonl")
    facts_untrusted = read_jsonl(inferred / "facts_untrusted.jsonl")
    descriptions = read_jsonl(inferred / "descriptions.jsonl")
    actual = {
        "images": len(descriptions),
        "unique_facts": len(facts_all),
        "trusted_facts": len(facts_trusted),
        "untrusted_facts": len(facts_untrusted),
    }
    expected = {
        key: manifest["counts"][key]
        for key in actual
    }

    agreement_path = dataset_root / "provenance/agreement_summary.json"
    agreement = {}
    if agreement_path.exists():
        raw_agreement = json.loads(
            agreement_path.read_text(encoding="utf-8")
        )
        agreement = {
            "images": raw_agreement.get("images"),
            "unique_facts": raw_agreement.get("facts"),
            "trusted_facts": raw_agreement.get("trusted_facts"),
            "untrusted_facts": raw_agreement.get("untrusted_facts"),
        }

    image_ids = {row["image_id"] for row in descriptions}
    trusted_fact_ids = {row["fact_id"] for row in facts_trusted}
    gold = load_gold()
    gold_image_ids = {
        image_id
        for row in gold
        for image_id in row.get("relevant_image_ids", [])
    }
    gold_fact_ids = {
        fact_id
        for row in gold
        for fact_id in row.get("relevant_fact_ids", [])
    }
    return {
        "actual": actual,
        "manifest": expected,
        "agreement_provenance": agreement,
        "manifest_matches_actual": expected == actual,
        "agreement_matches_actual": agreement == actual,
        "missing_gold_image_ids": sorted(gold_image_ids - image_ids),
        "missing_gold_fact_ids": sorted(gold_fact_ids - trusted_fact_ids),
        "source_of_truth": (
            "final annotations/vlm_inferred/*.jsonl; manifest.json is the "
            "released count contract"
        ),
    }


def strip_markup(text: str) -> str:
    """Remove executable/navigation markup while preserving visible text."""
    text = re.sub(
        r"(?is)<(?:script|style|nav)[^>]*>.*?</(?:script|style|nav)>",
        " ",
        text,
    )
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}(?:#{1,6}|>|[-*+]\s)\s*", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def normalize_unicode_whitespace(text: str) -> str:
    """Apply NFKC and deterministic whitespace/newline normalization."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize_symbols(text: str) -> str:
    """Keep punctuation and verbalise emoji/symbols instead of deleting them."""
    output = []
    for character in text:
        category = unicodedata.category(character)
        if category in {"Cc", "Cf", "Cs"} and character not in "\n\t":
            continue
        codepoint = ord(character)
        is_emoji = (
            0x1F300 <= codepoint <= 0x1FAFF
            or 0x2600 <= codepoint <= 0x27BF
        )
        if is_emoji:
            name = unicodedata.name(character, "symbol").lower().replace(" ", "_")
            output.append(f" {name} ")
        else:
            output.append(character)
    return normalize_unicode_whitespace("".join(output))


def apply_text_stage(
    documents: list[dict[str, Any]],
    transform: Callable[[str], str],
) -> list[dict[str, Any]]:
    result = deepcopy(documents)
    for document in result:
        document["text"] = transform(document["text"])
    return result


def _normalized_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def deduplicate_documents(
    documents: list[dict[str, Any]],
    *,
    threshold: float = 0.88,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse exact/near duplicate texts while retaining all image aliases."""
    exact: dict[str, dict[str, Any]] = {}
    exact_removed = 0
    for document in documents:
        key = normalize_unicode_whitespace(document["text"]).casefold()
        if key in exact:
            exact[key]["represented_image_ids"].extend(
                document["represented_image_ids"]
            )
            exact_removed += 1
        else:
            exact[key] = deepcopy(document)
    unique = list(exact.values())

    signatures: dict[str, MinHash] = {}
    index = MinHashLSH(threshold=threshold, num_perm=128)
    kept: list[dict[str, Any]] = []
    near_removed = 0
    for position, document in enumerate(unique):
        signature = MinHash(num_perm=128, seed=SEED)
        for word in sorted(_normalized_words(document["text"])):
            signature.update(word.encode("utf-8"))
        candidates = index.query(signature)
        if candidates:
            best_key = max(
                candidates,
                key=lambda key: signature.jaccard(signatures[key]),
            )
            score = signature.jaccard(signatures[best_key])
            if score >= threshold:
                kept[int(best_key[1:])]["represented_image_ids"].extend(
                    document["represented_image_ids"]
                )
                near_removed += 1
                continue
        key = f"d{len(kept)}"
        signatures[key] = signature
        index.insert(key, signature)
        kept.append(deepcopy(document))

    for document in kept:
        document["represented_image_ids"] = sorted(
            set(document["represented_image_ids"])
        )
    return kept, {
        "input_documents": len(documents),
        "output_documents": len(kept),
        "exact_removed": exact_removed,
        "near_removed": near_removed,
        "removed_percent": 100
        * (len(documents) - len(kept))
        / max(len(documents), 1),
        "near_duplicate_threshold": threshold,
        "image_aliases_preserved": True,
    }


def noise_score(text: str) -> float:
    """Share of whitespace-delimited tokens with no alphabetic character."""
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(not any(character.isalpha() for character in token) for token in tokens) / len(tokens)


def corpus_statistics(documents: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [token_count(document["text"]) for document in documents]
    return {
        "documents": len(documents),
        "represented_images": len(
            {
                image_id
                for document in documents
                for image_id in document["represented_image_ids"]
            }
        ),
        "tokens": int(sum(lengths)),
        "mean_tokens": float(np.mean(lengths)) if lengths else 0.0,
        "noise_score": float(
            np.mean([noise_score(document["text"]) for document in documents])
        )
        if documents
        else 0.0,
    }


def gold_queries(
    *,
    split: str | None = None,
    question_types: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    allowed_types = set(question_types or [])
    return [
        {
            "query_id": row["question_id"],
            "question": row["question"],
            "relevant_image_ids": row["relevant_image_ids"],
            "split": row["split"],
            "question_type": row["question_type"],
        }
        for row in load_gold()
        if row.get("answerable", True)
        and row.get("relevant_image_ids")
        and (split is None or row["split"] == split)
        and (not allowed_types or row["question_type"] in allowed_types)
    ]


def _chunks_from_chonkie(
    documents: list[dict[str, Any]],
    chunker: Any,
    *,
    strategy: str,
) -> list[dict[str, Any]]:
    chunks = []
    for document in documents:
        parts = chunker.chunk(document["text"])
        if not parts:
            parts = [type("FallbackChunk", (), {"text": document["text"]})()]
        for position, part in enumerate(parts):
            text = part.text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "chunk_id": f"{document['document_id']}:{strategy}:{position}",
                    "document_id": document["document_id"],
                    "image_id": document["image_id"],
                    "represented_image_ids": document["represented_image_ids"],
                    "text": text,
                    "position": position,
                    "strategy": strategy,
                    "token_count": token_count(text),
                    "source": document["source"],
                    "published_at": document["published_at"],
                    "content_type": document["content_type"],
                    "language": document["language"],
                    "trusted_ratio": document["trusted_ratio"],
                    "noise_score": noise_score(text),
                }
            )
    return chunks


def chunk_documents(
    documents: list[dict[str, Any]],
    *,
    strategy: str,
    chunk_size: int = 512,
    threshold: float = 0.7,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[dict[str, Any]]:
    """Chunk documents with one of the required Chonkie strategies."""
    from chonkie import (
        NeuralChunker,
        SemanticChunker,
        SentenceChunker,
        TokenChunker,
    )

    if strategy == "token":
        chunker = TokenChunker(
            tokenizer=TOKENIZER_NAME,
            chunk_size=chunk_size,
            chunk_overlap=0,
        )
    elif strategy == "sentence":
        chunker = SentenceChunker(
            tokenizer=TOKENIZER_NAME,
            chunk_size=chunk_size,
            chunk_overlap=0,
            min_sentences_per_chunk=1,
        )
    elif strategy == "semantic":
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            threshold=threshold,
            chunk_size=chunk_size,
            min_sentences_per_chunk=1,
        )
    elif strategy == "neural":
        chunker = NeuralChunker(device_map="cpu")
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    return _chunks_from_chonkie(documents, chunker, strategy=strategy)


def chunk_statistics(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = np.asarray([chunk["token_count"] for chunk in chunks])
    return {
        "chunks": len(chunks),
        "mean_chunk_tokens": float(lengths.mean()) if len(lengths) else 0.0,
        "median_chunk_tokens": float(np.median(lengths)) if len(lengths) else 0.0,
        "std_chunk_tokens": float(lengths.std()) if len(lengths) else 0.0,
        "share_lt_50": float((lengths < 50).mean()) if len(lengths) else 0.0,
        "share_gt_500": float((lengths > 500).mean()) if len(lengths) else 0.0,
    }


def evaluate_rankings(
    rankings: list[list[str]],
    queries: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...] = DEFAULT_K,
) -> dict[str, Any]:
    """Calculate image-level Hit@k, Recall@k, Precision@k and MAP."""
    rows = []
    details = []
    average_precisions = []
    for query, ranking in zip(queries, rankings, strict=True):
        relevant = set(query["relevant_image_ids"])
        deduplicated_ranking = list(dict.fromkeys(ranking))
        hits_seen = 0
        precision_sum = 0.0
        for rank, image_id in enumerate(deduplicated_ranking, 1):
            if image_id in relevant:
                hits_seen += 1
                precision_sum += hits_seen / rank
        average_precision = precision_sum / len(relevant) if relevant else 0.0
        average_precisions.append(average_precision)
        per_k = {}
        for k in k_values:
            retrieved = set(deduplicated_ranking[:k])
            tp = len(retrieved & relevant)
            row = {
                "query_id": query["query_id"],
                "split": query.get("split"),
                "question_type": query.get("question_type"),
                "k": k,
                "hit": float(tp > 0),
                "precision": tp / k,
                "recall": tp / len(relevant) if relevant else 0.0,
            }
            rows.append(row)
            per_k[str(k)] = row
        details.append(
            {
                **query,
                "average_precision": average_precision,
                "metrics_by_k": per_k,
                "ranking": deduplicated_ranking[: max(k_values)],
            }
        )
    frame = pd.DataFrame(rows)
    aggregates = []
    for k, group in frame.groupby("k"):
        aggregates.append(
            {
                "k": int(k),
                "hit_rate": float(group["hit"].mean()),
                "macro_precision": float(group["precision"].mean()),
                "macro_recall": float(group["recall"].mean()),
            }
        )
    return {
        "metrics_by_k": aggregates,
        "details": details,
        "mean_average_precision": float(np.mean(average_precisions)),
        "queries": len(queries),
    }


def _rank_from_scores(
    scores: np.ndarray,
    chunks: list[dict[str, Any]],
) -> list[list[str]]:
    rankings = []
    for query_scores in scores:
        image_scores: dict[str, float] = defaultdict(lambda: -np.inf)
        for score, chunk in zip(query_scores, chunks, strict=True):
            for image_id in chunk["represented_image_ids"]:
                image_scores[image_id] = max(image_scores[image_id], float(score))
        rankings.append(
            [
                image_id
                for image_id, _ in sorted(
                    image_scores.items(), key=lambda item: (-item[1], item[0])
                )
            ]
        )
    return rankings


def evaluate_tfidf(
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...] = DEFAULT_K,
) -> dict[str, Any]:
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=20000,
    )
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    query_matrix = vectorizer.transform([query["question"] for query in queries])
    index_seconds = time.perf_counter() - started
    scores = cosine_similarity(query_matrix, matrix)
    result = evaluate_rankings(
        _rank_from_scores(scores, chunks), queries, k_values=k_values
    )
    result.update(
        {
            "retriever": "TF-IDF unigram+bigram; max chunk score per image",
            "index_seconds": index_seconds,
            "chunks": len(chunks),
        }
    )
    return result


def _metric_at(result: dict[str, Any], metric: str, k: int) -> float:
    return next(
        row[metric] for row in result["metrics_by_k"] if row["k"] == k
    )


def result_row(
    name: str,
    chunks: list[dict[str, Any]],
    result: dict[str, Any],
    **configuration: Any,
) -> dict[str, Any]:
    return {
        "configuration": name,
        **configuration,
        **chunk_statistics(chunks),
        "hit@5": _metric_at(result, "hit_rate", 5),
        "precision@5": _metric_at(result, "macro_precision", 5),
        "recall@5": _metric_at(result, "macro_recall", 5),
        "hit@10": _metric_at(result, "hit_rate", 10),
        "precision@10": _metric_at(result, "macro_precision", 10),
        "recall@10": _metric_at(result, "macro_recall", 10),
        "map": result["mean_average_precision"],
        "index_seconds": result["index_seconds"],
    }


def preprocessing_experiment(
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Measure corpus and retrieval metrics after every ordered cleaning step."""
    states: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = [
        ("raw", deepcopy(documents), {})
    ]
    current = apply_text_stage(documents, strip_markup)
    states.append(("strip_markup", current, {}))
    current = apply_text_stage(current, normalize_unicode_whitespace)
    states.append(("unicode_whitespace", current, {}))
    current = apply_text_stage(current, normalize_symbols)
    states.append(("emoji_symbols", current, {}))
    current, duplicate_report = deduplicate_documents(current)
    states.append(("deduplicate", current, duplicate_report))

    rows = []
    for stage, stage_documents, stage_metadata in states:
        chunks = chunk_documents(
            stage_documents, strategy="token", chunk_size=512
        )
        result = evaluate_tfidf(chunks, queries)
        rows.append(
            {
                "stage": stage,
                **corpus_statistics(stage_documents),
                **stage_metadata,
                "hit@5": _metric_at(result, "hit_rate", 5),
                "precision@5": _metric_at(result, "macro_precision", 5),
                "recall@5": _metric_at(result, "macro_recall", 5),
                "hit@10": _metric_at(result, "hit_rate", 10),
                "precision@10": _metric_at(result, "macro_precision", 10),
                "recall@10": _metric_at(result, "macro_recall", 10),
                "map": result["mean_average_precision"],
            }
        )
    return pd.DataFrame(rows), current, duplicate_report


def compare_chunkers(
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    include_neural: bool = True,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    configurations = {
        "TokenChunker-512": {"strategy": "token", "chunk_size": 512},
        "SentenceChunker-160": {"strategy": "sentence", "chunk_size": 160},
        "SemanticChunker-0.70": {
            "strategy": "semantic",
            "chunk_size": 512,
            "threshold": 0.70,
        },
    }
    if include_neural:
        configurations["NeuralChunker"] = {"strategy": "neural"}
    rows = []
    chunks_by_name = {}
    for name, parameters in configurations.items():
        started = time.perf_counter()
        chunks = _cached_chunks(
            cache_dir,
            f"chunker-{name}",
            documents,
            lambda parameters=parameters: chunk_documents(
                documents, **parameters
            ),
        )
        chunk_seconds = time.perf_counter() - started
        result = evaluate_tfidf(chunks, queries)
        chunks_by_name[name] = chunks
        rows.append(
            result_row(
                name,
                chunks,
                result,
                preprocessing="all documented stages",
                chunker=parameters["strategy"],
                chunk_seconds=chunk_seconds,
                metadata="none",
                llm_processing="none",
            )
        )
    return pd.DataFrame(rows), chunks_by_name


def semantic_ablation(
    documents: list[dict[str, Any]],
    validation_queries: list[dict[str, Any]],
    *,
    thresholds: tuple[float, ...] = (0.55, 0.70, 0.85),
    models: tuple[str, ...] = (
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L3-v2",
    ),
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    rows = []
    outputs = {}
    for model in models:
        for threshold in thresholds:
            name = f"{model.rsplit('/', 1)[-1]}@{threshold:.2f}"
            started = time.perf_counter()
            chunks = _cached_chunks(
                cache_dir,
                f"semantic-{name}",
                documents,
                lambda model=model, threshold=threshold: chunk_documents(
                    documents,
                    strategy="semantic",
                    chunk_size=512,
                    threshold=threshold,
                    embedding_model=model,
                ),
            )
            seconds = time.perf_counter() - started
            result = evaluate_tfidf(chunks, validation_queries)
            outputs[name] = chunks
            rows.append(
                result_row(
                    name,
                    chunks,
                    result,
                    embedding_model=model,
                    threshold=threshold,
                    chunk_seconds=seconds,
                )
            )
    return pd.DataFrame(rows), outputs


def semantic_split_examples(
    outputs: dict[str, list[dict[str, Any]]],
    document_ids: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for configuration, chunks in outputs.items():
        grouped: dict[str, list[str]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk["document_id"]].append(chunk["text"])
        for document_id in document_ids:
            for position, text in enumerate(grouped.get(document_id, [])):
                rows.append(
                    {
                        "configuration": configuration,
                        "document_id": document_id,
                        "chunk_position": position,
                        "text": text,
                    }
                )
    return pd.DataFrame(rows)


def add_metadata(
    chunks: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add structural, content and quality metadata to every chunk."""
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=5000, ngram_range=(1, 2)
    )
    document_texts = [document["text"] for document in documents]
    matrix = vectorizer.fit_transform(document_texts)
    terms = np.asarray(vectorizer.get_feature_names_out())
    keywords: dict[str, list[str]] = {}
    for row_index, document in enumerate(documents):
        weights = matrix.getrow(row_index).toarray().ravel()
        top = weights.argsort()[-5:][::-1]
        keywords[document["document_id"]] = [
            str(terms[index]) for index in top if weights[index] > 0
        ]
    document_lengths = {
        document["document_id"]: token_count(document["text"])
        for document in documents
    }
    enriched = deepcopy(chunks)
    totals = defaultdict(int)
    for chunk in enriched:
        totals[chunk["document_id"]] += 1
    for chunk in enriched:
        chunk.update(
            {
                "document_length": document_lengths[chunk["document_id"]],
                "chunk_count": totals[chunk["document_id"]],
                "keywords": keywords[chunk["document_id"]],
            }
        )
    return enriched


def prepend_metadata(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = deepcopy(chunks)
    for chunk in enriched:
        prefix = (
            f"Source: {chunk['source']}. "
            f"Content type: {chunk['content_type']}. "
            f"Language: {chunk['language']}. "
            f"Keywords: {', '.join(chunk.get('keywords', []))}.\n"
        )
        chunk["text"] = prefix + chunk["text"]
        chunk["token_count"] = token_count(chunk["text"])
    return enriched


def metadata_experiments(
    chunks: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    enriched = add_metadata(chunks, documents)
    threshold = float(np.median([chunk["trusted_ratio"] for chunk in enriched]))
    filtered = [
        chunk for chunk in enriched if chunk["trusted_ratio"] >= threshold
    ]
    prepended = prepend_metadata(enriched)
    configurations = {
        "metadata-control": enriched,
        "metadata-quality-filter": filtered,
        "metadata-prepend": prepended,
    }
    rows = []
    for name, configured_chunks in configurations.items():
        result = evaluate_tfidf(configured_chunks, queries)
        rows.append(
            result_row(
                name,
                configured_chunks,
                result,
                metadata=(
                    f"trusted_ratio >= {threshold:.3f}"
                    if name.endswith("filter")
                    else "prepend"
                    if name.endswith("prepend")
                    else "stored only"
                ),
                llm_processing="none",
            )
        )
    return pd.DataFrame(rows), configurations


def summary_chunks(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_documents = deepcopy(documents)
    for document in summary_documents:
        document["text"] = document["summary"]
    return chunk_documents(summary_documents, strategy="token", chunk_size=512)


def _cache_key(model_name: str, documents: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode())
    for document in documents:
        digest.update(
            json.dumps(
                document, ensure_ascii=False, sort_keys=True
            ).encode()
        )
    return digest.hexdigest()[:16]


def _cached_chunks(
    cache_dir: Path | None,
    name: str,
    documents: list[dict[str, Any]],
    builder: Callable[[], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if cache_dir is None:
        return builder()
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    path = cache_dir / (
        f"{safe_name}-{_cache_key(name, documents)}.jsonl"
    )
    if path.exists():
        return read_jsonl(path)
    rows = builder()
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def generate_hypothetical_questions(
    documents: list[dict[str, Any]],
    *,
    cache_dir: Path,
    model_name: str = "google/flan-t5-small",
    batch_size: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate three retrieval questions per document with a local text2text LLM."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"hyde-{_cache_key(model_name + HYDE_PROMPT_VERSION, documents)}.jsonl"
    )
    if cache_path.exists():
        rows = read_jsonl(cache_path)
        return rows, {
            "model": model_name,
            "cache_hit": True,
            "seconds": 0.0,
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "output_tokens": sum(row["output_tokens"] for row in rows),
        }

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    prompts = [
        "Generate three short search questions answered by this camera-image "
        f"description. Separate questions with ' | '.\n{document['text']}"
        for document in documents
    ]
    rows = []
    started = time.perf_counter()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        generated = model.generate(
            **encoded,
            max_new_tokens=48,
            num_beams=3,
            num_return_sequences=3,
            do_sample=False,
        )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        batch_documents = documents[start : start + batch_size]
        for offset, (document, prompt) in enumerate(
            zip(batch_documents, batch_prompts, strict=True)
        ):
            generated_texts = texts[offset * 3 : (offset + 1) * 3]
            token_ids = generated[offset * 3 : (offset + 1) * 3]
            questions = list(
                dict.fromkeys(
                    value.strip()
                    for generated_text in generated_texts
                    for value in re.split(
                        r"\s*\|\s*|\n+", generated_text
                    )
                    if value.strip()
                )
            )[:3]
            if not questions:
                questions = [document["summary"]]
            rows.append(
                {
                    "chunk_id": f"{document['document_id']}:hyde",
                    "document_id": document["document_id"],
                    "image_id": document["image_id"],
                    "represented_image_ids": document["represented_image_ids"],
                    "text": "\n".join(questions),
                    "questions": questions,
                    "token_count": token_count("\n".join(questions)),
                    "input_tokens": len(tokenizer.encode(prompt)),
                    "output_tokens": int(
                        (token_ids != tokenizer.pad_token_id).sum().item()
                    ),
                    "position": 0,
                    "strategy": "hyde-local-llm",
                }
            )
    seconds = time.perf_counter() - started
    with cache_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows, {
        "model": model_name,
        "cache_hit": False,
        "seconds": seconds,
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
    }


def llm_experiments(
    documents: list[dict[str, Any]],
    control_chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    cache_dir: Path,
    run_hyde: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    configurations: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {
        "llm-control-rich-document": (
            control_chunks,
            {"llm_processing": "none", "llm_seconds": 0.0, "llm_tokens": 0},
        ),
        "llm-producer-summary": (
            summary_chunks(documents),
            {
                "llm_processing": "existing GPT producer summary",
                "llm_seconds": sum(
                    document["producer_duration_ms"] for document in documents
                )
                / 1000,
                "llm_tokens": sum(
                    document["producer_usage"] for document in documents
                ),
            },
        ),
    }
    generation_stats: dict[str, Any] = {}
    if run_hyde:
        hyde_chunks, generation_stats = generate_hypothetical_questions(
            documents, cache_dir=cache_dir
        )
        configurations["llm-hyde-questions"] = (
            hyde_chunks,
            {
                "llm_processing": generation_stats["model"],
                "llm_seconds": generation_stats["seconds"],
                "llm_tokens": generation_stats["input_tokens"]
                + generation_stats["output_tokens"],
            },
        )

    rows = []
    for name, (chunks, metadata) in configurations.items():
        result = evaluate_tfidf(chunks, queries)
        rows.append(result_row(name, chunks, result, **metadata))
    return pd.DataFrame(rows), generation_stats


def parent_child_chunks(
    documents: list[dict[str, Any]],
    *,
    parent_size: int,
    child_size: int,
) -> list[dict[str, Any]]:
    parents = chunk_documents(
        documents, strategy="sentence", chunk_size=parent_size
    )
    children = []
    from chonkie import TokenChunker

    chunker = TokenChunker(
        tokenizer=TOKENIZER_NAME, chunk_size=child_size, chunk_overlap=0
    )
    for parent_position, parent in enumerate(parents):
        for child_position, part in enumerate(chunker.chunk(parent["text"])):
            text = part.text.strip()
            if not text:
                continue
            children.append(
                {
                    **parent,
                    "chunk_id": (
                        f"{parent['document_id']}:p{parent_position}:"
                        f"c{child_position}"
                    ),
                    "parent_id": parent["chunk_id"],
                    "text": text,
                    "token_count": token_count(text),
                    "position": child_position,
                    "strategy": f"parent-{parent_size}-child-{child_size}",
                }
            )
    return children


@lru_cache(maxsize=2)
def _embedding_model(model_name: str) -> Any:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def evaluate_dense(
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    model_name: str = "BAAI/bge-m3",
    k_values: tuple[int, ...] = DEFAULT_K,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = _embedding_model(model_name)
    embedding_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        embedding_key = hashlib.sha256(
            (
                model_name
                + "\n"
                + "\n".join(chunk["text"] for chunk in chunks)
            ).encode()
        ).hexdigest()[:16]
        embedding_path = cache_dir / f"dense-corpus-{embedding_key}.npy"
    if embedding_path is not None and embedding_path.exists():
        corpus_embeddings = np.load(embedding_path)
    else:
        corpus_embeddings = model.encode(
            [chunk["text"] for chunk in chunks],
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if embedding_path is not None:
            np.save(embedding_path, corpus_embeddings)
    query_embeddings = model.encode(
        [query["question"] for query in queries],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    index_seconds = time.perf_counter() - started
    scores = np.asarray(query_embeddings) @ np.asarray(corpus_embeddings).T
    result = evaluate_rankings(
        _rank_from_scores(scores, chunks), queries, k_values=k_values
    )
    result.update(
        {
            "retriever": f"dense cosine, {model_name}",
            "index_seconds": index_seconds,
            "chunks": len(chunks),
        }
    )
    return result


def load_fact_chunks(
    dataset_root: Path,
    *,
    trusted_only: bool = True,
) -> list[dict[str, Any]]:
    """Load deterministic one-fact-per-chunk records for retrieval comparison."""
    filename = (
        "fact_chunks_trusted.jsonl"
        if trusted_only
        else "fact_chunks_all.jsonl"
    )
    rows = read_jsonl(dataset_root / "offline_chunking" / filename)
    chunks = []
    for row in rows:
        text = row["text"]
        chunks.append(
            {
                "chunk_id": row["chunk_id"],
                "document_id": row.get("parent_document_id", row["image_id"]),
                "image_id": row["image_id"],
                "represented_image_ids": [row["image_id"]],
                "text": text,
                "position": 0,
                "strategy": (
                    "fact-trusted" if trusted_only else "fact-all"
                ),
                "token_count": token_count(text),
                "label_id": row.get("label_id"),
                "trusted_fact": row.get("trusted_fact"),
                "agreement_state": row.get("agreement_state"),
                "noise_score": noise_score(text),
            }
        )
    return chunks


def fact_vs_chunking_experiment(
    dataset_root: Path,
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    raw_chunks: dict[str, list[dict[str, Any]]],
    run_dense: bool = True,
    model_name: str = "BAAI/bge-m3",
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare extracted fact chunks against raw-description chunking.

    Fact chunks are the deterministic one-fact-per-chunk projection from CP1.
    Raw chunkers operate on the rich producer document (summary + facts +
    evidence). Both TF-IDF and optional BGE-M3 dense retrieval are reported on
    the same query set so the corpus representation is the only varying factor.
    """
    corpora: dict[str, list[dict[str, Any]]] = {
        "fact-chunks-trusted": load_fact_chunks(
            dataset_root, trusted_only=True
        ),
        "fact-chunks-all": load_fact_chunks(
            dataset_root, trusted_only=False
        ),
    }
    for name, chunks in raw_chunks.items():
        if chunks:
            corpora[f"raw-{name}"] = chunks

    rows = []
    by_type = []
    results: dict[str, Any] = {}
    for corpus_name, chunks in corpora.items():
        if not chunks:
            continue
        for retriever_name, evaluator in (
            ("tfidf", lambda c=chunks: evaluate_tfidf(c, queries)),
            (
                "bge-m3",
                (
                    lambda c=chunks: evaluate_dense(
                        c,
                        queries,
                        model_name=model_name,
                        cache_dir=cache_dir,
                    )
                )
                if run_dense
                else None,
            ),
        ):
            if evaluator is None:
                continue
            result = evaluator()
            configuration = f"{corpus_name}/{retriever_name}"
            results[configuration] = result
            rows.append(
                result_row(
                    configuration,
                    chunks,
                    result,
                    corpus=corpus_name,
                    retriever=retriever_name,
                    chunker=(
                        "fact-extracted"
                        if corpus_name.startswith("fact-")
                        else "raw-description"
                    ),
                    embedding_model=(
                        model_name if retriever_name == "bge-m3" else "tfidf"
                    ),
                    metadata="none",
                    llm_processing="none",
                )
            )
            details = result["details"]
            for question_type in sorted(
                {detail["question_type"] for detail in details}
            ):
                selected = [
                    detail
                    for detail in details
                    if detail["question_type"] == question_type
                ]
                for k in (5, 10):
                    by_type.append(
                        {
                            "configuration": configuration,
                            "corpus": corpus_name,
                            "retriever": retriever_name,
                            "question_type": question_type,
                            "k": k,
                            "hit_rate": float(
                                np.mean(
                                    [
                                        detail["metrics_by_k"][str(k)]["hit"]
                                        for detail in selected
                                    ]
                                )
                            ),
                            "macro_recall": float(
                                np.mean(
                                    [
                                        detail["metrics_by_k"][str(k)][
                                            "recall"
                                        ]
                                        for detail in selected
                                    ]
                                )
                            ),
                        }
                    )
    return pd.DataFrame(rows), {
        "by_question_type": pd.DataFrame(by_type),
        "results": results,
        "corpora_sizes": {
            name: len(chunks) for name, chunks in corpora.items()
        },
    }


def parent_child_experiment(
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    simple_chunks: list[dict[str, Any]] | None = None,
    model_name: str = "BAAI/bge-m3",
    combinations: tuple[tuple[int, int], ...] = (
        (256, 64),
        (384, 96),
        (512, 128),
    ),
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    by_type = []
    all_results = {}
    configurations: list[
        tuple[str, list[dict[str, Any]], int | None, int | None]
    ] = []
    if simple_chunks is not None:
        configurations.append(
            ("dense-simple-best-semantic", simple_chunks, None, None)
        )
    for parent_size, child_size in combinations:
        name = f"parent-{parent_size}/child-{child_size}"
        configurations.append(
            (
                name,
                _cached_chunks(
                    cache_dir,
                    name,
                    documents,
                    lambda parent_size=parent_size, child_size=child_size: parent_child_chunks(
                        documents,
                        parent_size=parent_size,
                        child_size=child_size,
                    ),
                ),
                parent_size,
                child_size,
            )
        )

    for name, chunks, parent_size, child_size in configurations:
        result = evaluate_dense(
            chunks,
            queries,
            model_name=model_name,
            cache_dir=cache_dir,
        )
        all_results[name] = result
        rows.append(
            result_row(
                name,
                chunks,
                result,
                chunker=(
                    "simple-semantic"
                    if parent_size is None
                    else "parent-child"
                ),
                embedding_model=model_name,
                parent_size=parent_size,
                child_size=child_size,
                metadata="stored only",
                llm_processing="none",
            )
        )
        details = result["details"]
        for question_type in sorted(
            {detail["question_type"] for detail in details}
        ):
            selected = [
                detail
                for detail in details
                if detail["question_type"] == question_type
            ]
            for k in (5, 10):
                by_type.append(
                    {
                        "configuration": name,
                        "question_type": question_type,
                        "k": k,
                        "hit_rate": float(
                            np.mean(
                                [
                                    detail["metrics_by_k"][str(k)]["hit"]
                                    for detail in selected
                                ]
                            )
                        ),
                        "macro_recall": float(
                            np.mean(
                                [
                                    detail["metrics_by_k"][str(k)]["recall"]
                                    for detail in selected
                                ]
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows), {
        "by_question_type": pd.DataFrame(by_type),
        "results": all_results,
    }


def run_control_point_2(
    dataset_root: Path,
    *,
    output_dir: Path,
    run_neural: bool = True,
    run_dense: bool = True,
    run_hyde: bool = True,
) -> dict[str, Any]:
    """Run the complete sequential CP2 experiment and persist CSV artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    validation = gold_queries(split="validation")
    test = gold_queries(split="test")
    all_queries = validation + test
    consistency = validate_dataset_consistency(dataset_root)
    if not consistency["manifest_matches_actual"]:
        raise ValueError(
            "manifest.json counts do not match final annotation JSONL files"
        )
    if consistency["missing_gold_image_ids"]:
        raise ValueError("Gold references images absent from descriptions")
    if consistency["missing_gold_fact_ids"]:
        raise ValueError("Gold references facts absent from trusted facts")
    documents = load_documents(dataset_root)

    preprocessing, cleaned, duplicate_report = preprocessing_experiment(
        documents, all_queries
    )
    chunkers, chunks_by_name = compare_chunkers(
        cleaned,
        validation,
        include_neural=run_neural,
        cache_dir=cache_dir,
    )
    semantic, semantic_outputs = semantic_ablation(
        cleaned, validation, cache_dir=cache_dir
    )
    best_semantic_name = semantic.sort_values(
        ["recall@10", "hit@10", "map"], ascending=False
    ).iloc[0]["configuration"]
    best_chunks = semantic_outputs[best_semantic_name]
    examples = semantic_split_examples(
        semantic_outputs,
        [document["document_id"] for document in cleaned[:3]],
    )
    metadata, metadata_chunks = metadata_experiments(
        best_chunks, cleaned, test
    )
    llm, generation_stats = llm_experiments(
        cleaned,
        best_chunks,
        test,
        cache_dir=cache_dir,
        run_hyde=run_hyde,
    )
    if run_dense:
        parent_child, parent_details = parent_child_experiment(
            cleaned,
            test,
            simple_chunks=best_chunks,
            cache_dir=cache_dir,
        )
    else:
        parent_child = pd.DataFrame()
        parent_details = {
            "by_question_type": pd.DataFrame(),
            "results": {},
        }

    raw_for_comparison = {
        "TokenChunker-512": chunks_by_name.get("TokenChunker-512", []),
        "SentenceChunker-160": chunks_by_name.get("SentenceChunker-160", []),
        f"best-semantic-{best_semantic_name}": best_chunks,
    }
    fact_vs_chunking, fact_details = fact_vs_chunking_experiment(
        dataset_root,
        cleaned,
        test,
        raw_chunks=raw_for_comparison,
        run_dense=run_dense,
        cache_dir=cache_dir,
    )

    final = pd.concat(
        [
            chunkers.assign(experiment_family="chunker-screening"),
            semantic.assign(experiment_family="semantic-ablation"),
            metadata.assign(experiment_family="metadata"),
            llm.assign(experiment_family="llm-processing"),
            parent_child.assign(experiment_family="parent-child"),
            fact_vs_chunking.assign(
                experiment_family="fact-vs-raw-chunking"
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    frames = {
        "preprocessing": preprocessing,
        "chunkers": chunkers,
        "semantic_ablation": semantic,
        "semantic_examples": examples,
        "metadata": metadata,
        "llm": llm,
        "parent_child": parent_child,
        "parent_by_question_type": parent_details["by_question_type"],
        "fact_vs_chunking": fact_vs_chunking,
        "fact_vs_chunking_by_type": fact_details["by_question_type"],
        "final": final,
    }
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset_documents": len(documents),
                "validation_queries": len(validation),
                "test_queries": len(test),
                "dataset_consistency": consistency,
                "duplicate_report": duplicate_report,
                "selected_semantic_configuration": best_semantic_name,
                "hyde": generation_stats,
                "fact_vs_chunking_corpora": fact_details["corpora_sizes"],
                "design": (
                    "validation selects chunker/semantic parameters; test is "
                    "used for metadata, LLM, parent-child and "
                    "fact-vs-raw-chunking reporting"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        **frames,
        "documents": documents,
        "dataset_consistency": consistency,
        "cleaned_documents": cleaned,
        "chunks_by_name": chunks_by_name,
        "semantic_outputs": semantic_outputs,
        "metadata_chunks": metadata_chunks,
        "parent_details": parent_details,
        "fact_details": fact_details,
        "duplicate_report": duplicate_report,
        "selected_semantic_configuration": best_semantic_name,
        "generation_stats": generation_stats,
    }


__all__ = [
    "add_metadata",
    "chunk_documents",
    "chunk_statistics",
    "compare_chunkers",
    "corpus_statistics",
    "deduplicate_documents",
    "evaluate_dense",
    "evaluate_tfidf",
    "fact_vs_chunking_experiment",
    "generate_hypothetical_questions",
    "gold_queries",
    "llm_experiments",
    "load_documents",
    "load_fact_chunks",
    "metadata_experiments",
    "normalize_symbols",
    "normalize_unicode_whitespace",
    "parent_child_chunks",
    "parent_child_experiment",
    "prepend_metadata",
    "preprocessing_experiment",
    "run_control_point_2",
    "semantic_ablation",
    "semantic_split_examples",
    "strip_markup",
    "summary_chunks",
    "token_count",
    "validate_dataset_consistency",
]
