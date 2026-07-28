#!/usr/bin/env python
"""Download a frozen source manifest with resumable checksum verification."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import re
import shutil
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(record: dict) -> dict:
    target = _inside(PROJECT_ROOT / record["relative_path"])
    expected_bytes = int(record["bytes"])
    expected_hash = str(record["sha256"]).lower()
    if target.exists():
        if (
            target.stat().st_size != expected_bytes
            or _sha256(target) != expected_hash
        ):
            raise ValueError(f"Existing file differs from manifest: {target}")
        return {"relative_path": record["relative_path"], "state": "verified"}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    transient_errors = (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        http.client.IncompleteRead,
        ssl.SSLError,
    )
    consecutive_failures = 0
    for segment_attempt in range(256):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_bytes:
            raise ValueError(f"Partial file is too large: {partial}")
        if offset == expected_bytes:
            break
        headers = {"User-Agent": "phyevo-mic-materializer/0.4"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(
            record["source_url"], headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = int(
                    getattr(response, "status", response.getcode())
                )
                if offset and status != 206:
                    raise ValueError(
                        "Server refused safe resume; partial preserved: "
                        f"{partial}"
                    )
                if offset:
                    content_range = response.headers.get(
                        "Content-Range", ""
                    )
                    matched = re.match(
                        r"bytes\s+(\d+)-\d+/\d+", content_range
                    )
                    if (
                        matched is None
                        or int(matched.group(1)) != offset
                    ):
                        raise ValueError(
                            "Server returned an unsafe range; partial "
                            f"preserved: {partial}"
                        )
                mode = "ab" if partial.exists() else "xb"
                with partial.open(mode) as stream:
                    shutil.copyfileobj(
                        response, stream, length=1024 * 1024
                    )
        except transient_errors:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                raise
            time.sleep(2 ** (consecutive_failures - 1))
        else:
            new_size = partial.stat().st_size
            if new_size <= offset:
                raise ValueError(
                    f"Download made no progress; partial preserved: {partial}"
                )
            consecutive_failures = 0
    else:
        raise ValueError(
            f"Download exceeded 256 safe resume segments: {partial}"
        )
    if partial.stat().st_size != expected_bytes:
        raise ValueError(f"Incomplete file preserved: {partial}")
    observed = _sha256(partial)
    if observed != expected_hash:
        raise ValueError(
            f"Checksum mismatch; partial preserved: {partial}"
        )
    partial.rename(target)
    return {"relative_path": record["relative_path"], "state": "downloaded"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    manifest_path = _inside(arguments.manifest)
    report_path = _inside(arguments.report)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = list(manifest["files"])
    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=arguments.workers
    ) as executor:
        for index, result in enumerate(
            executor.map(_fetch, records), start=1
        ):
            results.append(result)
            print(
                f"[{index}/{len(records)}] {result['state']}: "
                f"{result['relative_path']}",
                flush=True,
            )
    report = {
        "report_format": "phyevo-mic-materialization-report-v1",
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": _sha256(manifest_path),
        "file_count": len(records),
        "total_bytes": sum(int(item["bytes"]) for item in records),
        "downloaded_count": sum(
            item["state"] == "downloaded" for item in results
        ),
        "verified_existing_count": sum(
            item["state"] == "verified" for item in results
        ),
        "all_verified": True,
        "files": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
