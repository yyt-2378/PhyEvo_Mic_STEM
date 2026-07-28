#!/usr/bin/env python
"""Safely extract the checksum-frozen Dennler HRSTEM locked test archive.

Extraction is intentionally separate from download/registration. Running this
script prepares evaluation inputs but does not run a detector or write metrics.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

from tfree_detection.locked_evaluation import verify_protocol_seal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    PROJECT_ROOT
    / "datasets"
    / "public"
    / "dennler_2021_hrstem_defects"
    / "hrstem_defects_dataset.zip"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "datasets"
    / "public"
    / "dennler_2021_hrstem_defects"
    / "locked_extracted"
)
EXPECTED_SHA256 = (
    "abef46a0183ba08b925c64807d5e914cd"
    "97967444d9fd3e0799924896f0614ec"
)
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "benchmarks" / "locked_dennler_v1_protocol.yaml"
)
DEFAULT_PROTOCOL_SEAL = (
    PROJECT_ROOT / "benchmarks" / "locked_dennler_v1_protocol_seal.json"
)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="new directory for the locked image/label tree",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
        help="pre-registered locked-test protocol",
    )
    parser.add_argument(
        "--seal",
        type=Path,
        default=DEFAULT_PROTOCOL_SEAL,
        help="seal created before labels were unpacked",
    )
    arguments = parser.parse_args()
    verify_protocol_seal(
        protocol_path=arguments.protocol,
        seal_path=arguments.seal,
        project_root=PROJECT_ROOT,
    )
    if _sha256(ARCHIVE) != EXPECTED_SHA256:
        raise ValueError("Locked HRSTEM archive checksum mismatch")
    output = arguments.output.resolve()
    try:
        output.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("Extraction output must remain inside project") from error
    if output.exists():
        raise FileExistsError(
            f"Refusing to merge with or overwrite existing dataset: {output}"
        )

    with ZipFile(ARCHIVE) as archive:
        members = archive.infolist()
        for member in members:
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"Unsafe archive member path: {member.filename!r}"
                )
            destination = (output / relative).resolve()
            try:
                destination.relative_to(output)
            except ValueError as error:
                raise ValueError(
                    f"Archive member escapes output: {member.filename!r}"
                ) from error
        output.mkdir(parents=True)
        archive.extractall(output)
    print(
        f"Extracted {len(members)} locked members to {output}. "
        "No detector or metric was run."
    )


if __name__ == "__main__":
    main()
