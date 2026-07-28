#!/usr/bin/env python
"""Run a development detector and package candidates for offline review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfree_detection.config import load_config
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.governance import DataPartition
from tfree_detection.review_ui import build_review_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("assisted", "blinded"),
        required=True,
    )
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument(
        "--partition",
        choices=(
            DataPartition.TRAIN.value,
            DataPartition.CALIBRATION.value,
            DataPartition.DEVELOPMENT_TEST.value,
        ),
        required=True,
    )
    parser.add_argument("--padding-pixels", type=int, default=32)
    arguments = parser.parse_args()
    result = TrainingFreeDetectionFramework(
        load_config(arguments.config)
    ).detect(arguments.input)
    manifest = build_review_bundle(
        result,
        arguments.output,
        show_model_assistance=arguments.condition == "assisted",
        dataset_id=arguments.dataset_id,
        partition=DataPartition(arguments.partition),
        padding_pixels=arguments.padding_pixels,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
