"""Create an additive experiment record and verify the real OS/GPU boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .contracts import inside, protected_snapshot, verify_snapshot, write_new
from .sandbox import Sandbox


ROOT = Path(__file__).resolve().parents[2]


REQUIREMENTS = [
    ("R01", "Protect existing source, best releases, expert labels and input assets"),
    ("R02", "Generated program execution in networkless filesystem isolation; optional GPU"),
    ("R03", "Versioned executable analysis/learning/routing/research-policy candidates"),
    ("R04", "Evidence-derived specific proposal causes actual program execution"),
    ("R05", "Generate, tune, evolve, repair with exact ancestry and retained alternatives"),
    ("R06", "Persistent resumable experiment archive plus evidence-linked research memory"),
    ("R07", "Frozen source-disjoint evaluation; audit excluded from search"),
    ("R08", "Existing experts' annotations reused without requiring new annotation"),
    ("R09", "Actual next-round program/state inheritance, not retrospective text linkage"),
    ("R10", "Editable research strategy with separate same-budget improvement evaluation"),
    ("R11", "Matched model/data/tool/budget full versus generic and ablated controls"),
    ("R12", "Real microscopy demonstration, cost/result report, reproducible tests and usage"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--resume-probe", action="store_true")
    args = parser.parse_args()
    output = inside(ROOT, args.output)
    inputs, program = output / "probe_input", output / "probe_program"
    if args.resume_probe:
        before = json.loads((output / "protected_before.json").read_text())
        verify_snapshot(ROOT, before)
    else:
        output.mkdir(parents=True, exist_ok=False)
        before = protected_snapshot(ROOT)
        write_new(output / "protected_before.json", before)
        git = subprocess.run(["git", "status", "--short", "--branch"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        with (output / "initial_worktree.txt").open("x") as f:
            f.write(git)
        write_new(output / "requirements.json", [{"id": key, "requirement": text,
                    "status": "pending_verification"} for key, text in REQUIREMENTS])
        inputs.mkdir(); program.mkdir()
        write_new(inputs / "request.json", {"mode": "probe", "production_path": str(ROOT / "src"),
                                            "cuda": args.gpu is not None})
    sandbox = Sandbox(ROOT, ROOT / ".venvs/sam2", timeout=45, gpu=args.gpu)
    destination = output / "isolation_probe"
    attempt = 1
    while destination.exists():
        attempt += 1
        destination = output / f"isolation_probe_attempt_{attempt:02d}"
    execution = sandbox.run(program, inputs, destination)
    if execution["status"] != "ok":
        print(json.dumps(execution, ensure_ascii=False))
        raise RuntimeError("Sandbox failed; inspect preserved logs. No fallback to unrestricted execution.")
    checks = json.loads((destination / "result.json").read_text())
    required = ["credentials_absent", "host_home_absent", "production_absent", "input_readonly", "network_blocked"]
    if args.gpu is not None:
        required.append("cuda_available")
    if not all(checks.get(key) is True for key in required):
        raise RuntimeError("Isolation/GPU check did not pass")
    receipt = {"isolation": checks, "protected": verify_snapshot(ROOT, before),
               "requirements": "requirements.json", "status": "infrastructure_verified_not_goal_complete"}
    write_new(output / ("preflight.json" if not (output / "preflight.json").exists()
                        else f"preflight_attempt_{attempt:02d}.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
