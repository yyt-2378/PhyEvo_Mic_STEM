#!/usr/bin/env python
"""Validate independent exports and compile two-reviewer consensus outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tfree_detection.review import consensus_outcomes
from tfree_detection.review_ui import load_and_validate_review_responses


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        nargs=2,
        action="append",
        metavar=("RESPONSES_JSON", "PACKETS_JSON"),
        required=True,
        help="repeat once per independent reviewer export",
    )
    parser.add_argument("--annotation-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite consensus output: {arguments.output}"
        )
    responses = []
    inputs = []
    for response_name, packet_name in arguments.submission:
        response_path = Path(response_name)
        packet_path = Path(packet_name)
        validated = load_and_validate_review_responses(
            response_path, packet_path
        )
        responses.extend(validated)
        inputs.append(
            {
                "response_path": str(response_path),
                "response_sha256": _sha256(response_path),
                "packet_path": str(packet_path),
                "packet_sha256": _sha256(packet_path),
                "response_count": len(validated),
            }
        )
    outcomes = consensus_outcomes(
        responses,
        annotation_version=arguments.annotation_version,
    )
    status_counts = Counter(outcome.status for outcome in outcomes)
    payload = {
        "consensus_version": "1.0",
        "annotation_version": arguments.annotation_version,
        "input_submissions": inputs,
        "response_count": len(responses),
        "candidate_count": len(outcomes),
        "status_counts": dict(sorted(status_counts.items())),
        "outcomes": [
            {
                "candidate_id": outcome.candidate_id,
                "status": outcome.status,
                "annotation": (
                    outcome.annotation.to_dict()
                    if outcome.annotation is not None
                    else None
                ),
                "reviewer_count": outcome.reviewer_count,
                "mean_annotation_duration_seconds": (
                    outcome.mean_annotation_duration_seconds
                ),
                "assistance_conditions": list(
                    outcome.assistance_conditions
                ),
                "disagreement_reason": outcome.disagreement_reason,
            }
            for outcome in outcomes
        ],
    }
    payload["consensus_content_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
