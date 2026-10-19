#!/usr/bin/env python3
"""Bound research routing alongside the unchanged physics detector; no training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tfree_detection.config import load_config
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.rsi_runtime import detect_rsi, inside, save_rsi_detection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--binding", help="SHA-bound nested router/protocol/outer-checkpoint JSON")
    parser.add_argument("--card", action="append", default=[],
                        help="Optional production card; must currently be active in --ledger")
    parser.add_argument("--ledger", help="Project-local activation ledger events directory; required for production routing")
    parser.add_argument("--experimental-research", action="store_true",
                        help="Explicitly select a development router's main candidate stream; not production")
    parser.add_argument("--shadow", action="store_true", help="Compute candidates but keep native physics primary")
    parser.add_argument("--student-device", default="cpu")
    parser.add_argument("--skeleton")
    parser.add_argument("--exemplar")
    parser.add_argument("--dataset-path")
    stack = parser.add_mutually_exclusive_group()
    stack.add_argument("--frame-index", type=int)
    stack.add_argument("--projection", choices=("mean", "median", "max"))
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = Path(args.project_root).resolve(strict=True)
    output = inside(root, args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an RSI detection")
    config = load_config(inside(root, args.config))
    # This entry point does not provision external SAM workers or services.
    if config.stage2.sam3.enabled:
        raise ValueError("Use a physics-only config here; external SAM workers require a separate explicit workflow")
    result = detect_rsi(TrainingFreeDetectionFramework(config), inside(root, args.input),
        project_root=root, binding=args.binding, cards=args.card, production_ledger=args.ledger,
        experimental_research=args.experimental_research, shadow=args.shadow, student_device=args.student_device,
        skeleton=args.skeleton, exemplar=args.exemplar, frame_index=args.frame_index,
        projection=args.projection, dataset_path=args.dataset_path)
    summary = save_rsi_detection(result, config, output, project_root=root)
    print(json.dumps({"output": str(output), "primary_branch": result.primary_branch,
                      "status": summary["status"], "reason": summary["reason"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
