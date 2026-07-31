#!/usr/bin/env python3
"""Export current durable expert reviews into training-ready annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfree_detection.review_cases import load_review_cases
from tfree_detection.review_dataset import load_review_manifest
from tfree_detection.review_export import build_review_export
from tfree_detection.review_prediction import discover_review_predictions
from tfree_detection.review_store import ReviewStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("datasets/expert_annotations/review_store_v1.sqlite3"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("datasets/standardized/expert_review_cases_v1.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/standardized/expert_review_queue_v1.jsonl"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/expert_review_2d_predictions_v1"),
    )
    parser.add_argument("--reviewer-id")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT.resolve()
    store_path = args.store if args.store.is_absolute() else root / args.store
    store = ReviewStore(store_path)
    revisions = store.list_current(reviewer_id=args.reviewer_id)
    summary = build_review_export(
        revisions,
        cases=load_review_cases(root / args.cases),
        samples=load_review_manifest(root / args.manifest),
        predictions=discover_review_predictions(
            root / args.predictions, project_root=root
        ),
        project_root=root,
        output_directory=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
