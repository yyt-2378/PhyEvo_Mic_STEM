#!/usr/bin/env python
"""Freeze a label-blind, material-balanced subset from the full EMSeek plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from tfree_detection.data_intake import (
    DatasetIntakePlan,
    FileRole,
    IntakeFile,
    IntakeGroup,
    seal_plan,
    sha256_file,
    write_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def _selection_key(group_id: str, pair_id: str) -> str:
    return hashlib.sha256(
        f"{group_id}:{pair_id}".encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-plan", type=Path, required=True)
    parser.add_argument("--full-source-manifest", type=Path, required=True)
    parser.add_argument("--pairs-per-group", type=int, default=2)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.pairs_per_group < 1:
        raise ValueError("--pairs-per-group must be positive")
    plan_path = _inside(arguments.full_plan)
    source_path = _inside(arguments.full_source_manifest)
    manifest_path = _inside(arguments.manifest_output)
    output_plan_path = _inside(arguments.plan_output)
    if manifest_path.exists() or output_plan_path.exists():
        raise FileExistsError(
            "Refusing to overwrite subset manifest or sealed plan"
        )
    full = DatasetIntakePlan.from_yaml(plan_path)
    full.validate(require_seal=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_by_file_id = {
        item["file_id"]: item for item in source["files"]
    }

    groups = []
    selected_source_records = []
    group_summary = []
    for group in full.groups:
        paired: dict[str, list[IntakeFile]] = {}
        for item in group.files:
            if item.pair_id is None:
                continue
            if Path(item.relative_path).name.startswith("._"):
                continue
            paired.setdefault(item.pair_id, []).append(item)
        eligible = [
            pair_id
            for pair_id, files in paired.items()
            if {item.role for item in files}
            >= {FileRole.IMAGE, FileRole.ANNOTATION}
        ]
        if not eligible:
            continue
        selected_pairs = sorted(
            eligible,
            key=lambda pair_id: _selection_key(
                group.group_id, pair_id
            ),
        )[: arguments.pairs_per_group]
        selected_files = tuple(
            item
            for pair_id in selected_pairs
            for item in sorted(
                paired[pair_id], key=lambda candidate: candidate.file_id
            )
        )
        groups.append(replace(group, files=selected_files))
        for item in selected_files:
            selected_source_records.append(source_by_file_id[item.file_id])
        group_summary.append(
            {
                "group_id": group.group_id,
                "selected_pair_ids": selected_pairs,
                "file_count": len(selected_files),
                "bytes": sum(item.bytes for item in selected_files),
            }
        )
    if not groups:
        raise ValueError("No complete image/annotation pairs were available")

    manifest = {
        "manifest_format": "phyevo-mic-hf-balanced-subset-v1",
        "selection_policy": {
            "kind": "sha256_ranked_complete_pairs",
            "pairs_per_group": arguments.pairs_per_group,
            "labels_opened_during_selection": False,
            "macos_resource_forks_excluded": True,
        },
        "parent_plan_id": full.plan_id,
        "parent_plan_content_sha256": full.content_hash(),
        "parent_source_manifest_sha256": sha256_file(source_path),
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "dataset_id": full.dataset_id,
        "group_count": len(groups),
        "pair_count": sum(
            len(item["selected_pair_ids"]) for item in group_summary
        ),
        "file_count": len(selected_source_records),
        "total_bytes": sum(
            int(item["bytes"]) for item in selected_source_records
        ),
        "groups": group_summary,
        "files": selected_source_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    manifest_asset = IntakeFile(
        file_id="emseek-balanced-subset-v1-source-manifest",
        relative_path=manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        role=FileRole.METADATA,
        sha256=sha256_file(manifest_path),
        bytes=manifest_path.stat().st_size,
        pair_id=None,
    )
    metadata_template = full.groups[-1]
    groups.append(
        replace(
            metadata_template,
            group_id="emseek-balanced-subset-v1-source-manifest-group",
            acquisition_id="emseek-balanced-subset-v1-source-manifest",
            sample_id="emseek-balanced-subset-v1-source-manifest",
            eligible_for_active_learning=False,
            files=(manifest_asset,),
        )
    )
    subset = DatasetIntakePlan(
        plan_format=full.plan_format,
        plan_id="emseek-balanced-subset-train-v1",
        dataset_id=full.dataset_id,
        source_uri=full.source_uri,
        source_revision=full.source_revision,
        license_spdx=full.license_spdx,
        admission_state=full.admission_state,
        created_at_utc=full.created_at_utc,
        groups=tuple(groups),
    )
    sealed = seal_plan(subset)
    write_plan(sealed, output_plan_path)
    print(
        json.dumps(
            {
                "plan_id": sealed.plan_id,
                "content_sha256": sealed.content_hash(),
                "group_count": len(sealed.groups),
                "pair_count": manifest["pair_count"],
                "file_count": sum(
                    len(group.files) for group in sealed.groups
                ),
                "download_bytes": manifest["total_bytes"],
                "labels_opened_during_selection": False,
                "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
                "plan": str(output_plan_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
