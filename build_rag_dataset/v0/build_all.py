#!/usr/bin/env python3
"""Run the complete v0 build, chunking and validation sequence."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=HERE / "input")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-labels-per-image", type=int)
    parser.add_argument(
        "--generate-vlm",
        action="store_true",
        help="Run producer and verifier from images instead of saved records.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        help="Placeholder path containing image subdirectories such as 100/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep generated VLM records and resume an interrupted online run.",
    )
    parser.add_argument(
        "--producer-run-id", default="v0-producer-rebuild"
    )
    parser.add_argument(
        "--verifier-run-id", default="v0-verifier-rebuild"
    )
    parser.add_argument(
        "--agreement-run-id", default="v0-agreement-rebuild"
    )
    args = parser.parse_args()
    if args.generate_vlm and args.images_root is None:
        parser.error("--generate-vlm requires --images-root")
    if args.resume and not args.generate_vlm:
        parser.error("--resume is only valid with --generate-vlm")

    if args.output.exists() and not args.resume:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    descriptions = args.input_root / "annotations/descriptions.jsonl"
    verifications = args.input_root / "annotations/verifications.jsonl"

    if args.generate_vlm:
        descriptions = (
            args.output
            / "annotations/vlm_inferred/descriptions.jsonl"
        )
        verifications = (
            args.output
            / "annotations/vlm_inferred/verifications.jsonl"
        )
        producer_command = [
            sys.executable,
            str(HERE / "generate_descriptions.py"),
            "--images-root",
            str(args.images_root),
            "--input-root",
            str(args.input_root),
            "--output",
            str(descriptions),
            "--run-id",
            args.producer_run_id,
        ]
        if args.max_images is not None:
            producer_command.extend(
                ["--max-images", str(args.max_images)]
            )
        if args.max_labels_per_image is not None:
            producer_command.extend(
                [
                    "--max-labels-per-image",
                    str(args.max_labels_per_image),
                ]
            )
        run(producer_command)
        run(
            [
                sys.executable,
                str(HERE / "verify_descriptions.py"),
                "--images-root",
                str(args.images_root),
                "--input-root",
                str(args.input_root),
                "--descriptions",
                str(descriptions),
                "--output",
                str(verifications),
                "--producer-run-id",
                args.producer_run_id,
                "--run-id",
                args.verifier_run_id,
            ]
        )

    build_command = [
        sys.executable,
        str(HERE / "build_dataset.py"),
        "--input-root",
        str(args.input_root),
        "--output",
        str(args.output),
        "--descriptions",
        str(descriptions),
        "--verifications",
        str(verifications),
    ]
    if args.generate_vlm:
        build_command.extend(
            [
                "--producer-run-id",
                args.producer_run_id,
                "--verifier-run-id",
                args.verifier_run_id,
                "--agreement-run-id",
                args.agreement_run_id,
            ]
        )
    if args.max_images is not None:
        build_command.extend(["--max-images", str(args.max_images)])
    if args.max_labels_per_image is not None:
        build_command.extend(
            ["--max-labels-per-image", str(args.max_labels_per_image)]
        )
    run(build_command)
    run(
        [
            sys.executable,
            str(HERE / "build_fact_chunks.py"),
            "--dataset-root",
            str(args.output),
        ]
    )
    run(
        [
            sys.executable,
            str(HERE / "validate_dataset.py"),
            "--dataset-root",
            str(args.output),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
