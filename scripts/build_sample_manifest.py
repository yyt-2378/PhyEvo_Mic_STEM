#!/usr/bin/env python
"""Build an append-only, per-asset executable sample manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfree_detection.governance import DataRegistry
from tfree_detection.sample_manifest import (
    build_sample_records,
    sha256_records,
    summarize_sample_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    arguments = parser.parse_args()

    registry_path = _inside_project(arguments.registry)
    output_path = _inside_project(arguments.output)
    summary_path = _inside_project(arguments.summary)
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(
            "Refusing to overwrite a versioned sample manifest or summary"
        )

    registry = DataRegistry.from_yaml(registry_path)
    records = build_sample_records(registry, PROJECT_ROOT)
    summary = summarize_sample_records(records, registry=registry)
    summary["records_sha256"] = sha256_records(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
            )
            stream.write("\n")
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                **summary,
                "output": str(output_path.relative_to(PROJECT_ROOT)),
                "summary": str(summary_path.relative_to(PROJECT_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
