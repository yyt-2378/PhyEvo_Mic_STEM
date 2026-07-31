#!/usr/bin/env python3
"""Build an expert queue containing only samples without pixel labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tfree_detection.review_dataset import (
    load_review_manifest,
    split_annotation_and_reference_pools,
)


QUEUE_FORMAT = "phyevo-mic-expert-annotation-queue-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/standardized/review_2d_manifest_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/standardized/expert_review_queue_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "datasets/standardized/expert_review_queue_v1.summary.json"
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("Refusing to overwrite an expert queue")
    samples = load_review_manifest(args.manifest)
    annotation, reference = split_annotation_and_reference_pools(samples)
    with args.output.open("x", encoding="utf-8") as stream:
        for sample in annotation:
            payload = sample.to_dict()
            payload.update(
                {
                    "queue_format": QUEUE_FORMAT,
                    "queue_role": "expert_pixel_annotation",
                    "annotation_target": {
                        "point": "point_set",
                        "line": "line_trace",
                        "interface": "boundary_trace",
                        "region": "region_contour",
                    }.get(sample.geometry_hint, "open_set_geometry"),
                    "existing_pixel_label": False,
                }
            )
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    summary = {
        "queue_format": QUEUE_FORMAT,
        "source_manifest": args.manifest.as_posix(),
        "source_manifest_sha256": _sha256(args.manifest),
        "source_sample_count": len(samples),
        "expert_annotation_count": len(annotation),
        "reference_evaluation_count": len(reference),
        "existing_pixel_label_in_queue_count": sum(
            item.label_path is not None for item in annotation
        ),
        "dataset_counts": dict(
            sorted(Counter(item.dataset_id for item in annotation).items())
        ),
        "split_counts": dict(
            sorted(Counter(item.review_split for item in annotation).items())
        ),
        "geometry_counts": dict(
            sorted(Counter(item.geometry_hint for item in annotation).items())
        ),
        "reference_dataset_counts": dict(
            sorted(Counter(item.dataset_id for item in reference).items())
        ),
        "policy": {
            "include_label_kinds": ["none", "source_prior_only"],
            "exclude_label_kinds": [
                "pixel_mask",
                "implicit_empty_mask",
            ],
            "labeled_samples_are_evaluation_only": True,
            "locked_external_test_is_reviewed": False,
        },
    }
    with args.summary.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
