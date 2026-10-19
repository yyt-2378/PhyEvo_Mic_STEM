#!/usr/bin/env python3
"""Exclusive, recoverable baseline snapshot before autonomous experiments.

No existing file is modified. Dataset originals stay in place, with hashes;
the already frozen expert round and selected models are also archived.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
LATEST = "outputs/interface_improvement_20261015_v1/release_v2"
BEST = "outputs/best_release_20261014_v1"
DATA = "outputs/expert_cycle_001/learning_dataset_v1"


def path(value):
    p = (ROOT / value).resolve()
    p.relative_to(ROOT)
    return p


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def main(output):
    out = path(output)
    if out.exists():
        raise FileExistsError(out)
    protected, archived, sources = {}, set(), set()

    def protect(p, expected=None, archive=False):
        p = path(p)
        d = digest(p)
        if expected is not None and expected != d:
            raise ValueError(f"Existing seal mismatch: {p.relative_to(ROOT)}")
        rel = str(p.relative_to(ROOT))
        protected[rel] = d
        if archive:
            archived.add(rel)

    for folder, seal in [(LATEST, "seal_final.json"), (BEST, "seal.json"), (BEST, "replay_seal.json")]:
        protect(path(folder) / seal, archive=True)
        for name, d in read(path(folder) / seal)["files"].items():
            protect(path(folder) / name, d, archive=True)
    for p in sorted((path(LATEST) / "bindings").glob("*.json")):
        b = read(p)
        checkpoint = path(b["checkpoint"])
        protect(checkpoint / "checkpoint.json", b["checkpoint_sha256"], archive=True)
        meta = read(checkpoint / "checkpoint.json")
        protect(checkpoint / "state.npz", meta["state_sha256"], archive=True)
        for name, d in b["runtime_source_hashes"].items():
            protect(name, d, archive=True)
    manifest = read(path(DATA) / "manifest.json")
    for name in ("manifest.json", "seal.json"):
        protect(path(DATA) / name, archive=True)
    for entry in manifest["entries"]:
        protect(path(DATA) / entry["plane"], entry["plane_sha256"])
        protect(entry["source_path"], entry["source_file_sha256"])
    bundle = path(manifest["frozen_bundle"])
    protect(bundle / "seal.json", archive=True)
    for name, d in read(bundle / "seal.json")["files"].items():
        protect(bundle / name, d, archive=True)
    for folder in ("src", "scripts", "tests", "configs", "benchmarks", "environments"):
        for p in path(folder).rglob("*"):
            if p.is_symlink():
                continue
            if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".toml", ".txt", ".md"}:
                rel = str(path(p).relative_to(ROOT))
                sources.add(rel)
    sources.update(("pyproject.toml", "README.md"))
    source_hashes = {s: digest(path(s)) for s in sorted(sources)}
    out.mkdir(parents=True, exist_ok=False)
    with ZipFile(out / "baseline_artifacts.zip", "x", compression=ZIP_DEFLATED) as z:
        for name in sorted(archived):
            z.write(path(name), name)
    with ZipFile(out / "baseline_source.zip", "x", compression=ZIP_DEFLATED) as z:
        for name in sorted(sources):
            z.write(path(name), name)
    status = subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True)
    receipt = {
        "format": "phyevo-autonomous-rsi-baseline-v1",
        "current_interface_release": LATEST, "historical_best_collection": BEST,
        "protected_files": protected, "source_files": source_hashes,
        "archives": {n: digest(out / n) for n in ("baseline_artifacts.zip", "baseline_source.zip")},
        "preexisting_worktree": status, "datasets_modified": False,
        "baseline_selection_previously_development_informed": True,
        "restoration": "Extract selected files into a NEW project-local recovery directory, never overwrite current files.",
    }
    with (out / "manifest.json").open("x") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print(json.dumps({"snapshot": str(out.relative_to(ROOT)), "protected": len(protected),
                      "source_files": len(sources), "archive_bytes": sum((out / n).stat().st_size for n in receipt["archives"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/autonomous_rsi_20261015_v1/baseline")
    main(parser.parse_args().output)
