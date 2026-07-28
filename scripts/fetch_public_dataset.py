#!/usr/bin/env python
"""Fetch checksum-pinned public microscopy inputs without overwriting data."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "datasets" / "manifest.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "public"


def checksum(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def safe_destination(root: Path, dataset_id: str, filename: str) -> Path:
    if Path(filename).name != filename or not filename:
        raise ValueError(f"Unsafe manifest filename: {filename!r}")
    destination = root / dataset_id / filename
    root_resolved = root.resolve()
    parent_resolved = destination.parent.resolve()
    if root_resolved != parent_resolved and root_resolved not in parent_resolved.parents:
        raise ValueError("Dataset destination escaped the configured output root")
    return destination


def fetch_file(entry: dict, destination: Path) -> str:
    expected = str(entry["md5"]).lower()
    if destination.exists():
        actual = checksum(destination)
        if actual != expected:
            raise ValueError(
                f"Existing dataset file has checksum {actual}, expected {expected}: "
                f"{destination}"
            )
        return "verified-existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and checksum(partial) == expected:
        partial.rename(destination)
        return "verified-complete-partial"
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "tfree-detection-public-benchmark/0.2"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(
        str(entry["url"]),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        status = int(getattr(response, "status", response.getcode()))
        if offset and status != 206:
            raise ValueError(
                f"Server did not honor safe resume for {partial}; "
                "the preserved partial file was not modified"
            )
        mode = "ab" if offset else "xb"
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    expected_size = int(entry["bytes"])
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Downloaded size {actual_size}, expected {expected_size}; "
            f"the preserved file is {partial}"
        )
    actual = checksum(partial)
    if actual != expected:
        raise ValueError(
            f"Downloaded checksum {actual}, expected {expected}; "
            f"the preserved file is {partial}"
        )
    partial.rename(destination)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = yaml.safe_load(
        arguments.manifest.read_text(encoding="utf-8")
    )
    matches = [
        dataset
        for dataset in payload["datasets"]
        if dataset["id"] == arguments.dataset
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate dataset id: {arguments.dataset}")
    dataset = matches[0]
    for entry in dataset["files"]:
        destination = safe_destination(
            arguments.output, dataset["id"], entry["name"]
        )
        state = fetch_file(entry, destination)
        print(f"{state}: {destination} md5={entry['md5']}")


if __name__ == "__main__":
    main()
