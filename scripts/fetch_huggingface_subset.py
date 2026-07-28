#!/usr/bin/env python
"""Fetch a revision-pinned Hugging Face dataset subset without overwrites."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import shutil
import time
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = PROJECT_ROOT / "datasets" / "public"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Path escapes {root}: {path}") from error
    return resolved


def _api_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "phyevo-mic-data-intake/0.3"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download(
    *,
    url: str,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> str:
    if destination.exists():
        if destination.stat().st_size != expected_bytes:
            raise ValueError(
                f"Existing dataset byte size differs: {destination}"
            )
        observed = _sha256(destination)
        if observed != expected_sha256:
            raise ValueError(
                f"Existing dataset checksum differs: {destination}"
            )
        return "verified-existing"

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_bytes:
        raise ValueError(
            f"Preserved partial is larger than expected: {partial}"
        )
    if offset == expected_bytes:
        observed = _sha256(partial)
        if observed != expected_sha256:
            raise ValueError(
                f"Complete partial has wrong checksum: {partial}"
            )
        partial.rename(destination)
        return "verified-complete-partial"

    headers = {"User-Agent": "phyevo-mic-data-intake/0.3"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = int(getattr(response, "status", response.getcode()))
        if offset and status != 206:
            raise ValueError(
                f"Server refused safe resume; partial preserved: {partial}"
            )
        mode = "ab" if partial.exists() else "xb"
        with partial.open(mode) as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)

    if partial.stat().st_size != expected_bytes:
        raise ValueError(
            f"Incomplete download preserved at {partial}: "
            f"{partial.stat().st_size} / {expected_bytes} bytes"
        )
    observed = _sha256(partial)
    if observed != expected_sha256:
        raise ValueError(
            f"Checksum mismatch; file preserved at {partial}: "
            f"{observed} != {expected_sha256}"
        )
    partial.rename(destination)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=("image", "label", "bbox", "metadata"),
        default=("image", "label"),
    )
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PUBLIC_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 16:
        raise ValueError("--workers must be between 1 and 16")

    output_root = _inside(PUBLIC_ROOT, arguments.output_root)
    manifest_path = _inside(
        PROJECT_ROOT / "datasets", arguments.manifest_output
    )
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite source manifest: {manifest_path}"
        )
    dataset_root = _inside(
        output_root, output_root / arguments.dataset_id
    )
    encoded_repo = urllib.parse.quote(arguments.repo_id, safe="/")
    encoded_revision = urllib.parse.quote(arguments.revision, safe="")
    encoded_prefix = urllib.parse.quote(arguments.prefix.strip("/"), safe="/")
    tree_url = (
        f"https://huggingface.co/api/datasets/{encoded_repo}/tree/"
        f"{encoded_revision}/{encoded_prefix}"
        "?recursive=true&expand=false"
    )
    tree = _api_json(tree_url)
    if not isinstance(tree, list):
        raise ValueError("Unexpected Hugging Face tree response")

    selected: list[dict[str, Any]] = []
    role_set = set(arguments.roles)
    prefix = arguments.prefix.strip("/") + "/"
    for entry in tree:
        if entry.get("type") != "file":
            continue
        remote_path = str(entry["path"])
        if not remote_path.startswith(prefix):
            raise ValueError(
                f"API returned a path outside the prefix: {remote_path}"
            )
        suffix = remote_path[len(prefix) :]
        parts = Path(suffix).parts
        if len(parts) != 2 or parts[0] not in role_set:
            continue
        lfs = entry.get("lfs")
        expected_sha256 = (
            str(lfs["oid"]).lower()
            if isinstance(lfs, dict) and lfs.get("oid")
            else ""
        )
        if len(expected_sha256) != 64:
            raise ValueError(
                f"Selected file lacks a pinned LFS SHA-256: {remote_path}"
            )
        expected_bytes = int(
            lfs.get("size", entry.get("size", 0))
        )
        selected.append(
            {
                "remote_path": remote_path,
                "local_suffix": suffix,
                "role": parts[0],
                "bytes": expected_bytes,
                "sha256": expected_sha256,
            }
        )
    if not selected:
        raise ValueError("No files matched the requested roles")
    selected.sort(key=lambda item: item["remote_path"])

    def fetch_entry(entry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        destination = _inside(
            dataset_root, dataset_root / entry["local_suffix"]
        )
        encoded_path = urllib.parse.quote(
            entry["remote_path"], safe="/"
        )
        url = (
            f"https://huggingface.co/datasets/{encoded_repo}/resolve/"
            f"{encoded_revision}/{encoded_path}"
        )
        state = ""
        for attempt in range(4):
            try:
                state = _download(
                    url=url,
                    destination=destination,
                    expected_bytes=entry["bytes"],
                    expected_sha256=entry["sha256"],
                )
                break
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.IncompleteRead,
            ):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return state, {
            "remote_path": entry["remote_path"],
            "relative_path": str(
                destination.relative_to(PROJECT_ROOT)
            ),
            "role": entry["role"],
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
            "source_url": url,
        }

    records = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=arguments.workers
    ) as executor:
        results = executor.map(fetch_entry, selected)
        for index, (state, record) in enumerate(results, start=1):
            print(
                f"[{index}/{len(selected)}] {state}: "
                f"{record['relative_path']}",
                flush=True,
            )
            records.append(record)

    payload = {
        "manifest_format": "phyevo-mic-hf-source-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": arguments.repo_id,
        "revision": arguments.revision,
        "prefix": arguments.prefix.strip("/"),
        "dataset_id": arguments.dataset_id,
        "selected_roles": sorted(role_set),
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "files": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "file_count": len(records),
                "total_bytes": payload["total_bytes"],
                "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
