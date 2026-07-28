#!/usr/bin/env python
"""Check current four-task performance against the frozen v0.2 baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "benchmarks" / "baseline_v0_2_manifest.json"
BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "run_multitask_benchmark.py"


def _load_benchmark_module() -> Any:
    specification = importlib.util.spec_from_file_location(
        "tfree_multitask_benchmark", BENCHMARK_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load benchmark module from {BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _run_cases() -> list[dict[str, Any]]:
    module = _load_benchmark_module()
    functions: list[Callable[[], dict[str, Any]]] = [
        module.amorphous_case,
        module.atomic_case,
        module.stacking_case,
        module.phase_boundary_case,
    ]
    return [function() for function in functions]


def compare(
    current_cases: list[dict[str, Any]],
    baseline: dict[str, Any],
    minimum_fraction: float,
) -> dict[str, Any]:
    reference = baseline["synthetic_multitask_benchmark"]["tasks"]
    results = []
    passed = True
    for current in current_cases:
        task = current["task"]
        old = reference[task]
        metric_ratios = {}
        metric_pass = True
        for metric in ("iou", "precision", "recall"):
            ratio = float(current[metric]) / max(float(old[metric]), 1e-12)
            metric_ratios[metric] = ratio
            metric_pass &= ratio >= minimum_fraction
        results.append(
            {
                "task": task,
                "baseline": {
                    metric: float(old[metric])
                    for metric in ("iou", "precision", "recall")
                },
                "current": {
                    metric: float(current[metric])
                    for metric in ("iou", "precision", "recall")
                },
                "retained_fraction": metric_ratios,
                "passed": metric_pass,
            }
        )
        passed &= metric_pass
    return {
        "baseline_id": baseline["baseline_id"],
        "minimum_retained_performance_fraction": minimum_fraction,
        "tasks": results,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minimum-fraction",
        type=float,
        default=None,
        help="Override the retention gate recorded in the baseline manifest.",
    )
    arguments = parser.parse_args()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    minimum = (
        arguments.minimum_fraction
        if arguments.minimum_fraction is not None
        else float(baseline["minimum_retained_performance_fraction"])
    )
    if not 0 < minimum <= 1:
        raise ValueError("minimum fraction must be in (0, 1]")
    report = compare(_run_cases(), baseline, minimum)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
