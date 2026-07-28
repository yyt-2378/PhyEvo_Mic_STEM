#!/usr/bin/env python
"""Validate an offline review export before consensus conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfree_detection.review_ui import load_and_validate_review_responses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    arguments = parser.parse_args()
    responses = load_and_validate_review_responses(
        arguments.responses, arguments.packets
    )
    print(
        json.dumps(
            {
                "valid": True,
                "response_count": len(responses),
                "reviewer_ids": sorted(
                    {response.reviewer_id for response in responses}
                ),
                "candidate_ids": sorted(
                    response.candidate_id for response in responses
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
