#!/usr/bin/env python3
"""Group the unlabeled plane queue into non-redundant review cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfree_detection.review_cases import (
    build_expert_review_cases,
    summarize_review_cases,
)
from tfree_detection.review_dataset import load_review_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path(
            "datasets/standardized/expert_review_queue_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/standardized/expert_review_cases_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "datasets/standardized/expert_review_cases_v1.summary.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("Refusing to overwrite expert review cases")
    samples = load_review_manifest(args.queue)
    cases = build_expert_review_cases(samples)
    with args.output.open("x", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
    summary = summarize_review_cases(cases)
    with args.summary.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
