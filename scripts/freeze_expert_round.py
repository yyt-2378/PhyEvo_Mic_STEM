#!/usr/bin/env python3
"""Freeze an expert round without modifying the live ledger or data partitions.

The output is an immutable input/audit bundle, not permission to train on a
development or locked-test source. Geometry overlap is descriptive: assisted
labels, mask thickness and incomplete negatives preclude an accuracy claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from tfree_detection.review_cases import load_review_cases
from tfree_detection.review_dataset import load_review_manifest
from tfree_detection.review_export import review_training_mask, review_usage_policy
from tfree_detection.review_prediction import (
    discover_review_predictions, load_prediction_mask, sample_sha256,
)
from tfree_detection.review_store import (
    CaseReviewAction, REVIEW_PROJECT_ID, ReviewRevision, ReviewSubmission,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def readonly_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def snapshot_database(source: Path, target: Path) -> None:
    # backup() includes committed WAL contents; copying just the sqlite file
    # does not. Reserve a new file and never overwrite an earlier snapshot.
    with target.open("xb"):
        pass
    source_connection = readonly_database(source)
    destination = sqlite3.connect(target)
    try:
        source_connection.backup(destination)
    finally:
        destination.close()
        source_connection.close()


def verified_revisions(snapshot: Path, reviewers: tuple[str, ...]):
    if not reviewers or len(set(reviewers)) != len(reviewers):
        raise ValueError("Reviewer selection must be non-empty and unique")
    connection = readonly_database(snapshot)
    try:
        if [row[0] for row in connection.execute("PRAGMA integrity_check")] != ["ok"]:
            raise ValueError("SQLite integrity check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("SQLite foreign-key check failed")
        events = connection.execute("SELECT * FROM review_events").fetchall()
        for row in events:
            payload = json.loads(row["payload_json"])
            canonical = json.dumps(
                payload, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            )
            if hashlib.sha256(canonical.encode()).hexdigest() != row["payload_sha256"]:
                raise ValueError(f"Payload checksum mismatch: {row['revision_id']}")
            for field in ("project_id", "reviewer_id", "case_id"):
                if payload[field] != row[field]:
                    raise ValueError(f"Ledger column mismatch: {field}")
        current = connection.execute(
            "SELECT e.* FROM current_reviews c JOIN review_events e "
            "ON e.revision_id=c.revision_id WHERE c.project_id=?",
            (REVIEW_PROJECT_ID,),
        ).fetchall()
    finally:
        connection.close()
    selected = []
    seen = set()
    for row in current:
        if row["reviewer_id"] not in reviewers:
            continue
        if row["case_id"] in seen:
            raise ValueError("Overlapping reviewers need an explicit assignment; never auto-merge")
        seen.add(row["case_id"])
        payload = json.loads(row["payload_json"])
        submission = ReviewSubmission(
            **{**payload, "action": CaseReviewAction(payload["action"]),
               "annotation_duration_seconds": row["annotation_duration_seconds"],
               "client_submission_id": row["client_submission_id"]},
        )
        submission.validate()
        selected.append(ReviewRevision(
            revision_id=row["revision_id"], revision_number=row["revision_number"],
            submitted_at_utc=row["submitted_at_utc"],
            client_submission_id=row["client_submission_id"],
            payload_sha256=row["payload_sha256"],
            annotation_duration_seconds=row["annotation_duration_seconds"],
            submission=payload,
        ))
    if set(reviewers) != {r.reviewer_id for r in selected}:
        raise ValueError("At least one selected reviewer has no current records")
    return selected, len(events), len(current) - len(selected)


def freeze_round(root: Path, output: Path, reviewers: tuple[str, ...]) -> dict:
    root = root.resolve()
    output = project_path(root, output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an expert-round bundle")
    inputs = {
        "case_manifest": "datasets/standardized/expert_review_cases_v1.jsonl",
        "review_manifest": "datasets/standardized/expert_review_queue_v1.jsonl",
    }
    input_hashes = {key: file_sha256(project_path(root, value)) for key, value in inputs.items()}
    cases = load_review_cases(project_path(root, inputs["case_manifest"]))
    samples = load_review_manifest(project_path(root, inputs["review_manifest"]))
    sample_by_id = {s.sample_id: s for s in samples}
    predictions = discover_review_predictions(
        project_path(root, "outputs/expert_review_2d_predictions_v1"), project_root=root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    snapshot = output / "review_store_snapshot.sqlite3"
    snapshot_database(project_path(root, "datasets/expert_annotations/review_store_v1.sqlite3"), snapshot)
    revisions, history_count, excluded_count = verified_revisions(snapshot, reviewers)
    by_case = {r.case_id: r for r in revisions}
    if set(by_case) != {case.case_id for case in cases}:
        raise ValueError("Selected reviewers must cover exactly the frozen case inventory")
    records, groups = [], defaultdict(list)
    source_hashes = {}
    for index, case in enumerate(cases, 1):
        revision = by_case[case.case_id]
        payload = revision.submission
        if payload["case_manifest_sha256"] != input_hashes["case_manifest"]:
            raise ValueError("Review refers to a different case manifest")
        for field in ("dataset_id", "review_split", "geometry_hint", "annotation_scope"):
            if payload[field] != getattr(case, field):
                raise ValueError(f"Case/revision mismatch: {field}")
        if tuple(payload["applies_to_sample_ids"]) != case.sample_ids:
            raise ValueError("Case/revision sample scope differs")
        sample = sample_by_id[payload["selected_sample_id"]]
        member_groups = {sample_by_id[s].split_group_id for s in case.sample_ids}
        member_partitions = {sample_by_id[s].source_partition for s in case.sample_ids}
        if len(member_groups) != 1 or len(member_partitions) != 1:
            raise ValueError("Case crosses original source groups/partitions")
        source = project_path(root, sample.source_path)
        if sample.source_path not in source_hashes:
            source_hashes[sample.source_path] = file_sha256(source)
        prediction = predictions[(sample.sample_id, sample.primary_task)]
        if prediction.sample_sha256 != sample_sha256(sample):
            raise ValueError("Prediction refers to a different review sample")
        if payload["prediction_sha256"] not in (None, prediction.mask_sha256):
            raise ValueError("Prediction differs from the reviewed version")
        teacher = load_prediction_mask(prediction, root)
        if tuple(payload["image_shape"]) != teacher.shape:
            raise ValueError("Expert/prediction image shapes differ")
        human = review_training_mask(revision, prediction_mask=teacher)
        usage = review_usage_policy(sample, compiled_mask_available=human is not None)
        points = (payload.get("geometry") or {}).get("points_xy", [])
        point_coverage = None
        if case.geometry_hint == "point" and points:
            xy = np.rint(np.asarray(points)).astype(int)
            point_coverage = {
                "marked_points": len(points),
                "inside_teacher_mask": int(teacher[xy[:, 1], xy[:, 0]].sum()),
                "semantics": "marked-point coverage, not one-to-one recall or precision",
            }
        record = {
            "case_number": index, "case_id": case.case_id,
            "revision": revision.to_dict(), "sample": sample.to_dict(),
            "case": asdict(case), "source_file_sha256": source_hashes[sample.source_path],
            "prediction": prediction.to_dict(), "existing_usage_policy": usage,
            "label_tier": "expert_single_silver",
            "human_geometry_available": human is not None,
            "teacher_positive_pixels": int(teacher.sum()),
            "human_raster_positive_pixels": None if human is None else int(human.sum()),
            "empty_teacher_with_positive_expert": bool(
                human is not None and human.any() and not teacher.any()
            ),
            "marked_point_coverage": point_coverage,
            "raster_iou_descriptive_only": (
                None if human is None or not np.any(human | teacher)
                else float(np.sum(human & teacher) / np.sum(human | teacher))
            ),
            "negative_label_completeness_verified": False,
        }
        records.append(record)
        groups[sample.split_group_id].append(record)
    for key, value in inputs.items():
        if file_sha256(project_path(root, value)) != input_hashes[key]:
            raise ValueError("Input manifest changed during freeze")
    geometry_summary = {}
    for geometry in sorted({r["case"]["geometry_hint"] for r in records}):
        rr = [r for r in records if r["case"]["geometry_hint"] == geometry]
        covered = [r["marked_point_coverage"] for r in rr if r["marked_point_coverage"]]
        geometry_summary[geometry] = {
            "cases": len(rr),
            "human_geometry_available": sum(r["human_geometry_available"] for r in rr),
            "empty_teacher_positive_expert": [r["case_number"] for r in rr if r["empty_teacher_with_positive_expert"]],
            "expert_marked_points": sum(c["marked_points"] for c in covered),
            "expert_marked_points_inside_teacher_mask": sum(c["inside_teacher_mask"] for c in covered),
        }
    summary = {
        "format": "phyevo-expert-round-freeze-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewers": list(reviewers), "case_count": len(records),
        "all_history_events_verified": history_count,
        "excluded_nonselected_current_records": excluded_count,
        "geometry_summary": geometry_summary,
        "action_counts": dict(Counter(r["revision"]["action"] for r in records)),
        "assistance_counts": dict(Counter(str(r["revision"]["model_assistance_shown"]) for r in records)),
        "current_training_eligible_count": sum(r["existing_usage_policy"]["training_eligible"] for r in records),
        "groups": {
            group: {
                "case_numbers": [r["case_number"] for r in rr],
                "source_partition": sorted({r["sample"]["source_partition"] for r in rr}),
                "review_split": sorted({r["sample"]["review_split"] for r in rr}),
                "geometry_counts": dict(Counter(r["case"]["geometry_hint"] for r in rr)),
            }
            for group, rr in sorted(groups.items())
        },
        "input_manifest_hashes": input_hashes,
        "source_file_hashes": source_hashes,
        "source_partitions_changed": False,
        "independent_accuracy_claim_allowed": False,
        "notes": [
            "No extra expert annotation requested.",
            "Assisted audit is not independent evaluation or consensus Gold.",
            "Region/line raster width differs from prediction-mask semantics.",
            "Explicit versioned source-use admission is required before training.",
        ],
    }
    with (output / "cases.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "summary.json", summary)
    write_json(output / "seal.json", {
        "format": "phyevo-expert-round-seal-v1",
        "files": {name: file_sha256(output / name) for name in (
            "review_store_snapshot.sqlite3", "cases.jsonl", "summary.json",
        )},
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reviewers", nargs="+", default=["tsj", "zyp"])
    args = parser.parse_args()
    summary = freeze_round(PROJECT_ROOT, args.output, tuple(args.reviewers))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
