#!/usr/bin/env python
"""Download the official DINO ViT-S/8 checkpoint with resumable ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import urllib.request
from pathlib import Path


URL = (
    "https://dl.fbaipublicfiles.com/dino/"
    "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
)
EXPECTED_SIZE = 86_728_949
EXPECTED_SHA256 = (
    "55c8b267f479b614ec20858947ce361ba804846c193e20be74263a096bb6d75e"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download_part(
    part: int, start: int, end: int, part_directory: Path
) -> Path:
    path = part_directory / f"part-{part:02d}.bin"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return path
    request = urllib.request.Request(
        URL, headers={"Range": f"bytes={start}-{end}"}
    )
    temporary = path.with_suffix(".tmp")
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
    if temporary.stat().st_size != expected:
        raise RuntimeError(
            f"Part {part} has {temporary.stat().st_size} bytes; expected {expected}"
        )
    temporary.replace(path)
    return path


def download(output: Path, workers: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == EXPECTED_SIZE:
        digest = _sha256(output)
        if digest != EXPECTED_SHA256:
            raise RuntimeError(
                f"Existing checkpoint checksum mismatch: {digest}"
            )
        print(f"Checkpoint already complete: {output}")
        print(f"sha256={digest}")
        return
    parts = output.parent / ".dino-download-parts"
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = (EXPECTED_SIZE + workers - 1) // workers
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for part in range(workers):
            start = part * chunk_size
            end = min(EXPECTED_SIZE - 1, (part + 1) * chunk_size - 1)
            jobs.append(
                executor.submit(_download_part, part, start, end, parts)
            )
        paths = [job.result() for job in jobs]

    temporary_output = output.with_suffix(".tmp")
    digest = hashlib.sha256()
    with temporary_output.open("wb") as target:
        for path in sorted(paths):
            with path.open("rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    digest.update(chunk)
    if temporary_output.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError("Assembled checkpoint size is incorrect")
    if digest.hexdigest() != EXPECTED_SHA256:
        raise RuntimeError(
            f"Assembled checkpoint checksum mismatch: {digest.hexdigest()}"
        )
    temporary_output.replace(output)
    for path in paths:
        path.unlink()
    parts.rmdir()
    print(f"Downloaded {output} ({EXPECTED_SIZE} bytes)")
    print(f"sha256={digest.hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vendor/models/dino-vits8/dino_deitsmall8_pretrain.pth"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    download(args.output, max(1, args.workers))


if __name__ == "__main__":
    main()
