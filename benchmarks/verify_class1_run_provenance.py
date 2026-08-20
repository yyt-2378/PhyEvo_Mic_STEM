#!/usr/bin/env python
"""Link the sealed Class-1 result to the exact source revision that made it.

The working tree has evolved since the post-development Class-1 run.  This
audit proves which committed source snapshot matches the implementation digest
stored in that run and verifies the manifest/seal/config chain.  It does not
claim to rerun the detector or replace the public image archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "0d692f26b43b0d4fc9604ce1ef82a1d49ae39a98"
IMPLEMENTATION_PATHS = (
    "src/tfree_detection/symmetry/group.py",
    "src/tfree_detection/symmetry/references.py",
    "src/tfree_detection/symmetry/task.py",
    "src/tfree_detection/config.py",
)
RUNNER_PATH = "scripts/run_class1_symmetry_benchmark.py"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs/class1_symmetry_final_v2/predictions/prediction_manifest.json"
)
SEAL_PATH = (
    PROJECT_ROOT
    / "outputs/class1_symmetry_final_v2/predictions/prediction_seal.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "benchmarks/paper_baselines/class1_run_provenance_v1.json"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_hash(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )


def _git_blob(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def audit() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    source_hashes = {
        path: _sha256_bytes(_git_blob(SOURCE_COMMIT, path))
        for path in IMPLEMENTATION_PATHS
    }
    implementation_sha256 = _json_hash(source_hashes)
    runner_sha256 = _sha256_bytes(_git_blob(SOURCE_COMMIT, RUNNER_PATH))
    config_sha256 = _json_hash(manifest["config"])
    checks = {
        "source_implementation_matches_manifest": (
            implementation_sha256 == manifest["implementation_sha256"]
        ),
        "serialized_config_matches_manifest": (
            config_sha256 == manifest["config_sha256"]
        ),
        "prediction_manifest_matches_seal": (
            _sha256_file(MANIFEST_PATH)
            == seal["prediction_manifest_sha256"]
        ),
        "prediction_content_matches_seal": (
            _json_hash(manifest) == seal["prediction_content_sha256"]
        ),
        "runner_matches_current_checkout": (
            runner_sha256 == _sha256_file(PROJECT_ROOT / RUNNER_PATH)
        ),
    }
    return {
        "audit_format": "phyevo-class1-source-provenance-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "scientific_status": manifest["scientific_status"],
        "source_commit": SOURCE_COMMIT,
        "source_file_sha256": source_hashes,
        "implementation_sha256": implementation_sha256,
        "runner_path": RUNNER_PATH,
        "runner_sha256": runner_sha256,
        "config_sha256": config_sha256,
        "prediction_manifest_sha256": _sha256_file(MANIFEST_PATH),
        "reproduction_note": (
            "The sealed Class-1 implementation is the source at source_commit, "
            "not the evolving default configuration in the current checkout. "
            "Recreate the historical v0.3 masks first, then run the Class-1 "
            "predict and score stages at that commit against the checksum-fixed "
            "public Dennler archive."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(PROJECT_ROOT.resolve())
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    result = audit()
    if not result["passed"]:
        raise RuntimeError(json.dumps(result["checks"], indent=2))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
