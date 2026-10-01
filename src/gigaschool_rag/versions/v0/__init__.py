"""Paths and schema identifiers fixed for dataset version v0."""

import os
from pathlib import Path


DATASET_VERSION = "v0"
FACT_SCHEMA_VERSION = "v0-fact-1"
FACT_CHUNK_SCHEMA_VERSION = "v0-derived-fact-chunk-1"


def dataset_root(course_root: Path) -> Path:
    """Return the local root of the v0 Hugging Face dataset."""
    return course_root / "hf" / DATASET_VERSION


def image_root(course_root: Path, image_dataset: str = "pilot-500-v1") -> Path:
    """Return the image directory shared by annotation versions."""
    return course_root / "hf" / "images" / image_dataset


def eda_output_root(course_root: Path) -> Path:
    """Return the regenerable EDA output directory for v0."""
    return course_root / "eda" / DATASET_VERSION / "output"


def evaluation_output_root(course_root: Path) -> Path:
    """Return the evaluation output directory for v0."""
    return course_root / "evaluation" / DATASET_VERSION


def _dataset_from(candidate: Path) -> Path | None:
    """Return a v0 root when candidate points to a project, hf root or release."""
    options = (
        candidate,
        candidate / DATASET_VERSION,
        candidate / "hf" / DATASET_VERSION,
    )
    return next(
        (
            path.resolve()
            for path in options
            if (path / "manifest.json").is_file()
        ),
        None,
    )


def resolve_dataset_root(
    *,
    repo_id: str = "mairon2300/video_retrieval",
) -> Path:
    """Locate local v0 data or download its metadata from Hugging Face.

    The optional ``GIGASCHOOL_RAG_DATASET_ROOT`` environment variable remains
    available for non-standard deployments, but normal local and Colab runs do
    not need to set it.
    """
    explicit = os.getenv("GIGASCHOOL_RAG_DATASET_ROOT")
    if explicit:
        resolved = _dataset_from(Path(explicit).expanduser())
        if resolved:
            return resolved
        raise FileNotFoundError(
            "GIGASCHOOL_RAG_DATASET_ROOT does not contain manifest.json"
        )

    search_roots = [Path.cwd(), *Path.cwd().parents]
    package_path = Path(__file__).resolve()
    search_roots.extend([package_path, *package_path.parents])
    checked: set[Path] = set()
    for root in search_roots:
        if root in checked:
            continue
        checked.add(root)
        resolved = _dataset_from(root)
        if resolved:
            return resolved

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[f"{DATASET_VERSION}/**"],
        )
    )
    resolved = _dataset_from(snapshot)
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"Dataset {repo_id} does not contain {DATASET_VERSION}/manifest.json"
    )


def resolve_image_root(root: Path) -> Path | None:
    """Return local pilot images when available; images are optional in Colab."""
    explicit = os.getenv("GIGASCHOOL_RAG_IMAGES_ROOT")
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        root.parent / "images" / "pilot-500-v1",
    ]
    return next(
        (
            path.resolve()
            for path in candidates
            if path is not None and path.is_dir()
        ),
        None,
    )


__all__ = [
    "DATASET_VERSION",
    "FACT_CHUNK_SCHEMA_VERSION",
    "FACT_SCHEMA_VERSION",
    "dataset_root",
    "eda_output_root",
    "evaluation_output_root",
    "image_root",
    "resolve_dataset_root",
    "resolve_image_root",
]
