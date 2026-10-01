"""Reusable exploratory data analysis for complete v0 image documents."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import tiktoken
from datasketch import MinHash, MinHashLSH
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


SEED = 20260930


def read_jsonl_with_raw(
    path: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    lines = [
        line.rstrip("\n")
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]
    return lines, [json.loads(line) for line in lines]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(
        r"\s+", " ", re.sub(r"[^\w\s]", "", text)
    ).strip()


def analyze_dataset(dataset_root: Path) -> dict[str, Any]:
    """Calculate all Control Point 1 EDA artifacts in memory."""
    raw_lines, rows = read_jsonl_with_raw(
        dataset_root
        / "annotations/vlm_inferred/descriptions.jsonl"
    )
    encoder = tiktoken.get_encoding("cl100k_base")
    summaries = [
        row["description"]["description"] for row in rows
    ]
    semantic_documents = [
        row["description"]["description"]
        + "\n"
        + "\n".join(
            f"{fact['sentence']} Evidence: {fact['evidence']}"
            for fact in row["description"]["facts"]
        )
        for row in rows
    ]
    documents = pd.DataFrame(
        {
            "image_id": [row["image_id"] for row in rows],
            "raw_tokens": [
                len(encoder.encode(text)) for text in raw_lines
            ],
            "summary_tokens": [
                len(encoder.encode(text)) for text in summaries
            ],
            "semantic_tokens": [
                len(encoder.encode(text))
                for text in semantic_documents
            ],
            "facts": [
                len(row["description"]["facts"]) for row in rows
            ],
        }
    )

    def statistics(column: str) -> dict[str, Any]:
        values = documents[column]
        return {
            "documents": len(values),
            "tokens_total": int(values.sum()),
            "min": int(values.min()),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "p95": float(values.quantile(0.95)),
            "max": int(values.max()),
            "short_lt_50": int((values < 50).sum()),
            "long_gt_2000": int((values > 2000).sum()),
        }

    lengths = {
        "raw_json": statistics("raw_tokens"),
        "image_summary": statistics("summary_tokens"),
        "summary_facts_evidence": statistics("semantic_tokens"),
    }

    all_text = "".join(raw_lines)
    token_total = lengths["raw_json"]["tokens_total"]
    service_pattern = re.compile(
        r"\b(?:schema_version|image_id|run_id|cache_key|"
        r"provenance_hash|allowed_labels|input_tokens|"
        r"output_tokens)\b"
    )
    number_pattern = re.compile(r"\b\d+(?:[.,]\d+)?\b")
    url_pattern = re.compile(r"https?://\S+|www\.\S+", re.I)

    def is_emoji(character: str) -> bool:
        codepoint = ord(character)
        return (
            0x1F300 <= codepoint <= 0x1FAFF
            or 0x2600 <= codepoint <= 0x27BF
        )

    noise = {
        "token_denominator": token_total,
        "url_per_1000_tokens": len(url_pattern.findall(all_text))
        / token_total
        * 1000,
        "number_per_1000_tokens": len(
            number_pattern.findall(all_text)
        )
        / token_total
        * 1000,
        "service_terms_per_1000_tokens": len(
            service_pattern.findall(all_text)
        )
        / token_total
        * 1000,
        "emoji_per_1000_tokens": sum(
            is_emoji(character) for character in all_text
        )
        / token_total
        * 1000,
        "special_character_ratio": sum(
            not character.isalnum()
            and not character.isspace()
            and not is_emoji(character)
            for character in all_text
        )
        / len(all_text),
    }

    normalized = [normalize(text) for text in summaries]
    exact_counts = Counter(normalized)
    exact_groups = [
        {"normalized_summary": text, "documents": count}
        for text, count in exact_counts.most_common()
        if count > 1
    ]
    signatures: dict[str, tuple[str, MinHash]] = {}
    index = MinHashLSH(threshold=0.82, num_perm=64)
    for position, text in enumerate(sorted(set(normalized))):
        signature = MinHash(num_perm=64, seed=SEED)
        for token in set(re.findall(r"\w+", text)):
            signature.update(token.encode())
        key = f"d{position:04d}"
        signatures[key] = (text, signature)
        index.insert(key, signature)
    near = set()
    for key, (text, signature) in signatures.items():
        for other in index.query(signature):
            if other <= key:
                continue
            score = signature.jaccard(signatures[other][1])
            if score >= 0.82:
                near.add(
                    (
                        text,
                        signatures[other][0],
                        round(float(score), 4),
                    )
                )
    near_pairs = [
        {
            "summary_a": left,
            "summary_b": right,
            "estimated_jaccard": score,
        }
        for left, right, score in sorted(
            near, key=lambda row: -row[2]
        )
    ]
    duplicates = {
        "raw_json_exact_duplicates": len(raw_lines)
        - len(set(raw_lines)),
        "summary_exact_duplicate_groups": len(exact_groups),
        "summary_documents_in_exact_groups": sum(
            row["documents"] for row in exact_groups
        ),
        "near_duplicate_summary_pairs": len(near_pairs),
        "minhash_threshold": 0.82,
    }

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=4000,
    )
    matrix = vectorizer.fit_transform(semantic_documents)
    projection = TruncatedSVD(
        n_components=min(50, matrix.shape[1] - 1),
        random_state=SEED,
    ).fit_transform(matrix)
    candidates = []
    for cluster_count in range(4, 11):
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=SEED,
            n_init=20,
        ).fit_predict(projection)
        candidates.append(
            {
                "k": cluster_count,
                "silhouette": float(
                    silhouette_score(projection, labels)
                ),
            }
        )
    selected_count = max(
        candidates, key=lambda row: row["silhouette"]
    )["k"]
    labels = KMeans(
        n_clusters=selected_count,
        random_state=SEED,
        n_init=30,
    ).fit_predict(projection)
    terms = np.asarray(vectorizer.get_feature_names_out())
    cluster_rows = []
    for cluster in range(selected_count):
        mask = labels == cluster
        centroid = np.asarray(matrix[mask].mean(axis=0)).ravel()
        cluster_rows.append(
            {
                "cluster": cluster,
                "documents": int(mask.sum()),
                "top_terms": terms[
                    np.argsort(centroid)[-12:][::-1]
                ].tolist(),
            }
        )
    clusters = {
        "selected_k": selected_count,
        "silhouette_candidates": candidates,
        "clusters": cluster_rows,
    }
    return {
        "documents": documents,
        "raw_lines": raw_lines,
        "summaries": summaries,
        "semantic_documents": semantic_documents,
        "lengths": lengths,
        "noise": noise,
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_pairs": near_pairs,
        "duplicates": duplicates,
        "clusters": clusters,
    }


def write_analysis(result: dict[str, Any], output: Path) -> None:
    """Persist an in-memory analysis using the legacy report filenames."""
    output.mkdir(parents=True, exist_ok=True)
    result["documents"].to_csv(
        output / "documents.csv", index=False
    )
    for name, key in (
        ("length_summary.json", "lengths"),
        ("noise_summary.json", "noise"),
        ("duplicate_summary.json", "duplicates"),
        ("cluster_summary.json", "clusters"),
    ):
        (output / name).write_text(
            json.dumps(
                result[key], ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
    for name, key in (
        (
            "exact_duplicate_groups.jsonl",
            "exact_duplicate_groups",
        ),
        ("near_duplicate_pairs.jsonl", "near_duplicate_pairs"),
    ):
        with (output / name).open("w", encoding="utf-8") as stream:
            for row in result[key]:
                stream.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
    plot_frame = result["documents"].melt(
        id_vars=["image_id"],
        value_vars=[
            "raw_tokens",
            "summary_tokens",
            "semantic_tokens",
        ],
        var_name="projection",
        value_name="tokens",
    )
    figure = px.histogram(
        plot_frame,
        x="tokens",
        color="projection",
        barmode="overlay",
        nbins=60,
        title="Длина исходных image documents",
    )
    figure.add_vline(x=50, line_dash="dash")
    figure.add_vline(x=2000, line_dash="dash")
    figure.write_html(
        output / "length_distribution.html",
        include_plotlyjs="cdn",
    )
