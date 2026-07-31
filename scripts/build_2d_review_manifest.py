#!/usr/bin/env python3
"""Build the non-destructive, logical-plane expert-review manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfree_detection.review_dataset import (
    build_review_2d_samples,
    write_review_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/standardized/review_2d_manifest_v1.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "datasets/standardized/review_2d_manifest_v1.summary.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    samples = build_review_2d_samples(root)
    summary = write_review_manifest(
        samples,
        root / args.output,
        root / args.summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
