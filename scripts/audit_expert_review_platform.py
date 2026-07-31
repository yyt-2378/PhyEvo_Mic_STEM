#!/usr/bin/env python3
"""Audit the real 154-case queue and sealed prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfree_detection.review_cases import load_review_cases
from tfree_detection.review_dataset import load_review_manifest
from tfree_detection.review_platform import (
    audit_review_platform,
    file_sha256,
    require_review_platform,
)
from tfree_detection.review_prediction import discover_review_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/standardized/expert_review_queue_v1.jsonl"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("datasets/standardized/expert_review_cases_v1.jsonl"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/expert_review_2d_predictions_v1"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT.resolve()
    manifest = (root / args.manifest).resolve()
    cases = (root / args.cases).resolve()
    report = audit_review_platform(
        load_review_manifest(manifest),
        load_review_cases(cases),
        discover_review_predictions(root / args.predictions, project_root=root),
    )
    report["expert_queue_sha256"] = file_sha256(manifest)
    report["expert_cases_sha256"] = file_sha256(cases)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        target = (root / args.output).resolve()
        target.relative_to(root)
        if target.exists():
            raise FileExistsError("Refusing to overwrite an audit report")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    require_review_platform(report)


if __name__ == "__main__":
    main()
