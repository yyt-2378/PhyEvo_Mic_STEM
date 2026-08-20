#!/usr/bin/env python3
"""Export current durable expert reviews into training-ready annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tfree_detection.review_cases import load_review_cases
from tfree_detection.review_dataset import (
    load_review_manifest,
    records_sha256 as review_records_sha256,
)
from tfree_detection.review_export import build_review_export
from tfree_detection.review_prediction import discover_review_predictions
from tfree_detection.review_store import ReviewStore
from tfree_detection.sample_manifest import sha256_records as asset_records_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument(
        "--registry-summary",
        type=Path,
        default=Path("datasets/standardized/sample_manifest_v7.summary.json"),
    )
    parser.add_argument(
        "--queue-summary",
        type=Path,
        default=Path(
            "datasets/standardized/expert_review_queue_v1.summary.json"
        ),
    )
    parser.add_argument(
        "--source-review-manifest",
        type=Path,
        default=Path("datasets/standardized/review_2d_manifest_v1.jsonl"),
    )
    parser.add_argument(
        "--source-review-summary",
        type=Path,
        default=Path("datasets/standardized/review_2d_manifest_v1.summary.json"),
    )
    parser.add_argument(
        "--source-asset-manifest",
        type=Path,
        default=Path("datasets/standardized/sample_manifest_v7.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT.resolve()
    store_path = args.store if args.store.is_absolute() else root / args.store
    store = ReviewStore(store_path)
    revisions = store.list_current(reviewer_id=args.reviewer_id)
    cases_path = root / args.cases
    queue_path = root / args.manifest
    queue_summary = json.loads(
        (root / args.queue_summary).read_text(encoding="utf-8")
    )
    source_review_path = root / args.source_review_manifest
    source_review_summary = json.loads(
        (root / args.source_review_summary).read_text(encoding="utf-8")
    )
    source_asset_path = root / args.source_asset_manifest
    registry_summary = json.loads(
        (root / args.registry_summary).read_text(encoding="utf-8")
    )
    source_review_sha256 = _sha256_file(source_review_path)
    expected_source_path = source_review_path.relative_to(root).as_posix()
    if queue_summary["source_manifest"] != expected_source_path:
        raise ValueError("Review queue names a different source review manifest")
    if queue_summary["source_manifest_sha256"] != source_review_sha256:
        raise ValueError("Review queue is not linked to the source review manifest")
    source_review_samples = load_review_manifest(source_review_path)
    if source_review_summary["records_sha256"] != review_records_sha256(
        source_review_samples
    ):
        raise ValueError("Source review records differ from their summary")
    source_asset_records = [
        json.loads(line)
        for line in source_asset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if registry_summary["records_sha256"] != asset_records_sha256(
        source_asset_records
    ):
        raise ValueError("Source asset records differ from their summary")
    summary = build_review_export(
        revisions,
        cases=load_review_cases(cases_path),
        samples=load_review_manifest(queue_path),
        source_review_samples=source_review_samples,
        source_asset_records=source_asset_records,
        predictions=discover_review_predictions(
            root / args.predictions, project_root=root
        ),
        project_root=root,
        output_directory=args.output,
        case_manifest_sha256=_sha256_file(cases_path),
        review_queue_sha256=_sha256_file(queue_path),
        source_review_manifest_sha256=source_review_sha256,
        source_asset_manifest_sha256=_sha256_file(source_asset_path),
        source_registry_sha256=str(registry_summary["registry_sha256"]),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
