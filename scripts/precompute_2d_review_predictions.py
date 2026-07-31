#!/usr/bin/env python3
"""Precompute compact prediction masks for the 2-D review application."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from collections import Counter
from pathlib import Path

from tfree_detection.review_dataset import load_review_manifest
from tfree_detection.review_prediction import (
    ReviewPrediction,
    build_abstention_prediction,
    build_review_prediction,
    discover_review_predictions,
    write_prediction_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/standardized/review_2d_manifest_v1.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/expert_review_2d_predictions_v1"),
    )
    parser.add_argument(
        "--task-mode",
        choices=("primary", "all"),
        default="primary",
    )
    parser.add_argument("--dataset-id", action="append", default=[])
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent CPU workers; each writes a distinct sample/task.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse checksum-validated artifacts in an incomplete output tree.",
    )
    return parser.parse_args()


def _run_prediction(
    sample,
    root: Path,
    output: Path,
    task: str,
) -> dict:
    try:
        return build_review_prediction(
            sample, root, output, task=task
        ).to_dict()
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        return build_abstention_prediction(
            sample,
            root,
            output,
            task=task,
            reason=reason,
        ).to_dict()


def _new_versioned_path(output: Path, name: str) -> Path:
    target = output / name
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    for index in range(1, 1000):
        candidate = output / f"{stem}.resume-{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Too many finalized resume attempts")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = (root / args.output).resolve()
    output.relative_to(root)
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists; use a new version or --resume: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    existing = discover_review_predictions(output, project_root=root)
    samples = load_review_manifest(root / args.manifest)
    if args.dataset_id:
        samples = [
            item for item in samples if item.dataset_id in args.dataset_id
        ]
    if args.split:
        samples = [
            item for item in samples if item.review_split in args.split
        ]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        samples = samples[: args.limit]
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    started = time.monotonic()
    created = []
    abstentions = []
    failures = []
    jobs = []
    for sample in samples:
        tasks = (
            sample.prediction_tasks
            if args.task_mode == "all"
            else (sample.primary_task,)
        )
        for task in tasks:
            key = (sample.sample_id, task)
            if key in existing:
                continue
            jobs.append((sample, task))

    def register_result(sample, task, payload=None, error=None) -> None:
        if error is None:
            record = ReviewPrediction.from_dict(payload)
            existing[(sample.sample_id, task)] = record
            created.append(record)
            if record.status == "abstained":
                abstentions.append(
                    {
                        "sample_id": sample.sample_id,
                        "task": task,
                        "reason": record.abstention_reason,
                    }
                )
        else:
            failures.append(
                {
                    "sample_id": sample.sample_id,
                    "task": task,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        print(
            json.dumps(
                {
                    "progress": f"{len(created) + len(failures)}/{len(jobs)}",
                    "sample_id": sample.sample_id,
                    "created": len(created),
                    "abstained": len(abstentions),
                    "failure_count": len(failures),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if args.workers == 1:
        for sample, task in jobs:
            try:
                register_result(
                    sample, task, _run_prediction(sample, root, output, task)
                )
            except Exception as error:
                register_result(sample, task, error=error)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(
                    _run_prediction, sample, root, output, task
                ): (sample, task)
                for sample, task in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                sample, task = futures[future]
                try:
                    register_result(sample, task, future.result())
                except Exception as error:
                    register_result(sample, task, error=error)

    report = {
        "report_format": "phyevo-mic-2d-prediction-run-v1",
        "manifest": args.manifest.as_posix(),
        "task_mode": args.task_mode,
        "workers": args.workers,
        "sample_count": len(samples),
        "prediction_count": len(existing),
        "created_count": len(created),
        "abstention_count": len(abstentions),
        "abstentions": abstentions,
        "failure_count": len(failures),
        "failure_types": dict(
            sorted(Counter(item["error_type"] for item in failures).items())
        ),
        "elapsed_seconds": time.monotonic() - started,
        "failures": failures,
    }
    report_path = _new_versioned_path(output, "run_report.json")
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    write_prediction_index(
        existing.values(),
        _new_versioned_path(output, "prediction_index.jsonl"),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
