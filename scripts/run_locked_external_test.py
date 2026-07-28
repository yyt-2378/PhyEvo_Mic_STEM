#!/usr/bin/env python
"""Seal, predict, and score a locked external benchmark in distinct stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfree_detection.locked_evaluation import (
    create_protocol_seal,
    generate_locked_predictions,
    score_locked_predictions,
    verify_protocol_seal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "benchmarks" / "locked_dennler_v1_protocol.yaml"
)
DEFAULT_PROTOCOL_SEAL = (
    PROJECT_ROOT / "benchmarks" / "locked_dennler_v1_protocol_seal.json"
)
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "public"
    / "dennler_2021_hrstem_defects"
    / "locked_extracted"
)
DEFAULT_LEDGER = (
    PROJECT_ROOT / "benchmarks" / "locked_runs" / "unblinding_ledger.jsonl"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser(
        "seal", help="freeze protocol before labels are unpacked"
    )
    seal.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    seal.add_argument("--seal", type=Path, default=DEFAULT_PROTOCOL_SEAL)
    seal.add_argument(
        "--forbidden-unpacked-path",
        type=Path,
        default=DEFAULT_DATASET_ROOT.relative_to(PROJECT_ROOT),
    )

    verify = subparsers.add_parser(
        "verify", help="verify the protocol and all frozen inputs"
    )
    verify.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    verify.add_argument("--seal", type=Path, default=DEFAULT_PROTOCOL_SEAL)

    predict = subparsers.add_parser(
        "predict", help="generate label-blind sealed predictions"
    )
    predict.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    predict.add_argument("--seal", type=Path, default=DEFAULT_PROTOCOL_SEAL)
    predict.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    predict.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser(
        "score", help="unblind one immutable prediction set"
    )
    score.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    score.add_argument("--seal", type=Path, default=DEFAULT_PROTOCOL_SEAL)
    score.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "seal":
        result = create_protocol_seal(
            protocol_path=arguments.protocol,
            seal_path=arguments.seal,
            project_root=PROJECT_ROOT,
            forbidden_unpacked_path=arguments.forbidden_unpacked_path,
        )
    elif arguments.command == "verify":
        protocol, seal = verify_protocol_seal(
            protocol_path=arguments.protocol,
            seal_path=arguments.seal,
            project_root=PROJECT_ROOT,
        )
        result = {
            "valid": True,
            "protocol_id": protocol.protocol_id,
            "asset_id": protocol.asset_id,
            "protocol_file_sha256": protocol.file_sha256,
            "seal_content_sha256": seal["seal_content_sha256"],
        }
    elif arguments.command == "predict":
        result = generate_locked_predictions(
            protocol_path=arguments.protocol,
            seal_path=arguments.seal,
            project_root=PROJECT_ROOT,
            dataset_root=arguments.dataset_root,
            output_directory=arguments.output,
        )
        result = {
            "output_directory": result["output_directory"],
            "counts": result["manifest"]["counts"],
            "prediction_set_sha256": result["prediction_seal"][
                "prediction_set_sha256"
            ],
        }
    else:
        result = score_locked_predictions(
            protocol_path=arguments.protocol,
            seal_path=arguments.seal,
            project_root=PROJECT_ROOT,
            dataset_root=arguments.dataset_root,
            prediction_directory=arguments.predictions,
            output_directory=arguments.output,
            ledger_path=arguments.ledger,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
