#!/usr/bin/env python3
"""One-command bounded RSI cycle, from baseline freeze through audited report.

Accepts only a NEW output directory. On failure it stops and preserves every
artifact; it never silently restarts an incomplete/live run or promotes a model
to production. Candidate generation receives no interactive roundwise advice.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def stage_commands(output, gpus):
    out = str(output); release = str(output / "release_v1")
    runner = "scripts/run_autonomous_rsi.py"
    verifier = "scripts/verify_autonomous_rsi.py"
    reporter = "scripts/report_autonomous_rsi.py"
    return [
        ("freeze", ["scripts/freeze_autonomous_rsi_baseline.py", "--output", str(output / "baseline")]),
        ("plan", [runner, "plan", "--output", out]),
        ("cache", [runner, "cache", "--output", out, "--gpus", gpus]),
        ("integrity", [verifier, "integrity", "--output", out]),
        ("search", [runner, "search", "--output", out, "--gpus", gpus]),
        ("seal-searches", [runner, "seal-searches", "--output", out]),
        ("audit", [runner, "audit", "--output", out, "--gpus", gpus]),
        ("verify", [verifier, "launch", "--output", out, "--gpus", gpus, "--attempt", "pipeline"]),
        ("release", [reporter, "release", "--study", out, "--output", release]),
        ("report", [reporter, "report", "--study", out, "--output", release]),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True); parser.add_argument("--gpus", default="1,2,3,4,5,6,7")
    args = parser.parse_args(); output = (ROOT / args.output).resolve(); output.relative_to(ROOT)
    output.parent.resolve().relative_to(ROOT)
    if output.exists() or output == ROOT:
        raise FileExistsError("A new project-local output directory is required; originals and prior attempts are protected")
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("Explicit distinct GPU IDs required")
    temp = (ROOT / ".codex-tmp").resolve(); temp.relative_to(ROOT); temp.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
           "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(temp), "OMP_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    for name, command in stage_commands(output, args.gpus):
        print("PIPELINE", name, flush=True)
        # No shell interpolation, no external paths, no deletion or replacement.
        proc = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env)
        print("PIPELINE PID", proc.pid, flush=True)
        code = proc.wait()
        if output.exists():
            with (output / "pipeline_events.jsonl").open("a") as f:
                f.write(json.dumps({"stage": name, "pid": proc.pid, "exit_code": code}) + "\n")
        if code:
            raise RuntimeError(f"Stage {name} failed with {code}; all artifacts retained, no automatic retry")
    print("CYCLE COMPLETE; inspect final audit admission. Production defaults unchanged.", flush=True)


if __name__ == "__main__":
    main()
