#!/usr/bin/env python
"""Build a conservative one-group training intake draft from an HF manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tfree_detection.data_intake import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Path must remain inside the project: {path}"
        ) from error
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--license-spdx", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--lab-id", required=True)
    parser.add_argument("--microscope-id", default="unknown")
    parser.add_argument("--tasks", nargs="+", required=True)
    arguments = parser.parse_args()

    manifest_path = _project_path(arguments.source_manifest)
    output_path = _project_path(arguments.output)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite intake draft: {output_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_format") != "phyevo-mic-hf-source-v1":
        raise ValueError("Unsupported source manifest format")
    records = manifest.get("files", [])
    images = {
        Path(item["relative_path"]).stem: item
        for item in records
        if item["role"] == "image"
    }
    annotations = {
        Path(item["relative_path"]).stem: item
        for item in records
        if item["role"] == "label"
    }
    if not images or set(images) != set(annotations):
        raise ValueError(
            "Every source image must have exactly one same-stem label"
        )

    files = []
    for pair_id in sorted(images):
        for role, record in (
            ("image", images[pair_id]),
            ("annotation", annotations[pair_id]),
        ):
            files.append(
                {
                    "file_id": f"{role}-{pair_id}",
                    "relative_path": record["relative_path"],
                    "role": role,
                    "sha256": record["sha256"],
                    "bytes": int(record["bytes"]),
                    "pair_id": pair_id,
                }
            )
    files.append(
        {
            "file_id": "source-manifest",
            "relative_path": str(
                manifest_path.relative_to(PROJECT_ROOT)
            ),
            "role": "metadata",
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "pair_id": None,
        }
    )
    payload = {
        "plan_format": "phyevo-mic-data-intake-v1",
        "plan_id": arguments.plan_id,
        "dataset_id": manifest["dataset_id"],
        "source_uri": (
            f"https://huggingface.co/datasets/{manifest['repo_id']}"
        ),
        "source_revision": manifest["revision"],
        "license_spdx": arguments.license_spdx,
        "admission_state": "training_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "groups": [
            {
                "group_id": arguments.group_id,
                "partition": "train",
                "acquisition_id": arguments.group_id,
                "sample_id": arguments.group_id,
                "lab_id": arguments.lab_id,
                "microscope_id": arguments.microscope_id,
                "material": arguments.material,
                "tasks": arguments.tasks,
                "grouping_evidence": "single_group_conservative",
                "annotation_qualification": "author_curated",
                "reviewer_count": 0,
                "control_role": "mixed",
                "eligible_for_active_learning": True,
                "files": files,
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "created": True,
                "pair_count": len(images),
                "file_count": len(files),
                "grouping": "single_group_conservative",
                "output": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
