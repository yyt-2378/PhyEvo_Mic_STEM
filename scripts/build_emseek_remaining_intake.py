#!/usr/bin/env python
"""Freeze metadata for the remaining EMSeek subsets without bulk download.

Hugging Face LFS object ids are SHA-256 content hashes.  This script walks the
revision-pinned repository with pagination, records only LFS-backed scientific
assets, and creates a sealed training-only plan.  The plan can be verified
without local files now; admission still requires ``--verify-files`` after a
separate checksum-verified download.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tfree_detection.data_intake import (
    AdmissionState,
    AnnotationQualification,
    ControlRole,
    DatasetIntakePlan,
    FileRole,
    IntakeFile,
    IntakeGroup,
    seal_plan,
    sha256_file,
    write_plan,
)
from tfree_detection.governance import DataPartition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "gary23ai/EMSeek"
DEFAULT_REVISION = "8f4a987c78af743098459e5d467f2c1b5f5c11b3"
DATASET_ID = "emseek_remaining_v1"
LEAVES = (
    "atom_columns/DyScO3",
    "atom_columns/LiCoO2",
    "atom_columns/MoS2",
    "atom_columns/NiO",
    "atom_columns/Pt",
    "atom_columns/Si",
    "atom_columns/SrTiO3",
    "atom_columns/graphene",
    "metal_alloy_defects/FeCrAl",
    "nanoparticles/Au@C",
    "nanoparticles/Au@Ge",
    "nanoparticles/Au@SiN",
    "nanoparticles/CdSe@C",
    "nanoparticles/Fe3O4",
    "nanoparticles/Fe3O4@SiO2",
    "nanoparticles/Pd@C",
)
ROLE_MAP = {
    "image": FileRole.IMAGE,
    "label": FileRole.ANNOTATION,
    "bbox": FileRole.ANNOTATION,
}


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path must remain inside project: {path}") from error
    return resolved


def _request_json(url: str) -> tuple[Any, str | None]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "phyevo-mic-intake/0.4"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
        return payload, response.headers.get("Link")


def _next_link(link: str | None) -> str | None:
    if not link:
        return None
    for target, relation in re.findall(r'<([^>]+)>;\s*rel="([^"]+)"', link):
        if relation == "next":
            return target
    return None


def _paginated_tree(
    repo_id: str, revision: str, prefix: str
) -> Iterable[dict[str, Any]]:
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_prefix = urllib.parse.quote(prefix.strip("/"), safe="/")
    url: str | None = (
        f"https://huggingface.co/api/datasets/{encoded_repo}/tree/"
        f"{encoded_revision}/{encoded_prefix}"
        "?recursive=true&expand=false&limit=1000"
    )
    seen_urls: set[str] = set()
    while url is not None:
        if url in seen_urls:
            raise ValueError("Hugging Face pagination repeated a cursor")
        seen_urls.add(url)
        payload, link = _request_json(url)
        if not isinstance(payload, list):
            raise ValueError("Unexpected Hugging Face tree response")
        for entry in payload:
            if isinstance(entry, dict):
                yield entry
        url = _next_link(link)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _tasks_for_leaf(leaf: str) -> tuple[str, ...]:
    if leaf.startswith("atom_columns/"):
        return ("atomic_column_localization", "atomic_site_anomaly")
    if leaf == "metal_alloy_defects/FeCrAl":
        return (
            "irradiation_damage_region",
            "artifact_or_damage_disambiguation",
        )
    return (
        "particle_and_surface_morphology",
        "particle_support_interface",
    )


def _collect_leaf(
    repo_id: str, revision: str, leaf: str
) -> tuple[list[IntakeFile], list[dict[str, Any]], int]:
    group_slug = _slug(leaf)
    records: list[dict[str, Any]] = []
    omitted_non_lfs = 0
    image_stems: set[str] = set()
    raw_entries: list[tuple[str, str, dict[str, Any]]] = []
    prefix = leaf.rstrip("/") + "/"
    for entry in _paginated_tree(repo_id, revision, leaf):
        if entry.get("type") != "file":
            continue
        remote_path = str(entry.get("path", ""))
        if not remote_path.startswith(prefix):
            raise ValueError(f"Tree path escaped requested leaf: {remote_path}")
        relative = remote_path[len(prefix) :]
        parts = Path(relative).parts
        if len(parts) != 2 or parts[0] not in ROLE_MAP:
            continue
        lfs = entry.get("lfs")
        if not isinstance(lfs, dict) or not lfs.get("oid"):
            omitted_non_lfs += 1
            continue
        oid = str(lfs["oid"]).lower()
        if len(oid) != 64:
            raise ValueError(f"Invalid LFS SHA-256 for {remote_path}")
        try:
            int(oid, 16)
        except ValueError as error:
            raise ValueError(
                f"Non-hex LFS SHA-256 for {remote_path}"
            ) from error
        role_name = parts[0]
        stem = Path(parts[1]).stem
        if role_name == "image":
            image_stems.add(stem)
        raw_entries.append((role_name, stem, entry))

    files: list[IntakeFile] = []
    for index, (role_name, stem, entry) in enumerate(
        sorted(raw_entries, key=lambda item: str(item[2]["path"])),
        start=1,
    ):
        remote_path = str(entry["path"])
        lfs = entry["lfs"]
        role = ROLE_MAP[role_name]
        pair_id = (
            f"{group_slug}:{stem}"
            if role_name in {"image", "label"} and stem in image_stems
            else None
        )
        local_relative = (
            Path("datasets") / "public" / DATASET_ID / remote_path
        )
        item = IntakeFile(
            file_id=f"{group_slug}:{role_name}:{index:05d}",
            relative_path=local_relative.as_posix(),
            role=role,
            sha256=str(lfs["oid"]).lower(),
            bytes=int(lfs.get("size", entry.get("size", 0))),
            pair_id=pair_id,
        )
        files.append(item)
        records.append(
            {
                **item.to_dict(),
                "remote_path": remote_path,
                "source_url": (
                    f"https://huggingface.co/datasets/{repo_id}/resolve/"
                    f"{revision}/{urllib.parse.quote(remote_path, safe='/')}"
                ),
                "leaf": leaf,
                "source_role": role_name,
            }
        )
    if not any(item.role == FileRole.IMAGE for item in files):
        raise ValueError(f"No LFS images found for EMSeek leaf {leaf}")
    return files, records, omitted_non_lfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(
            "datasets/intake/emseek_remaining_v1_source_manifest.json"
        ),
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path("datasets/intake/emseek_remaining_train_v1.sealed.yaml"),
    )
    arguments = parser.parse_args()

    manifest_path = _inside_project(arguments.manifest_output)
    plan_path = _inside_project(arguments.plan_output)
    if manifest_path.exists() or plan_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an EMSeek source manifest or sealed plan"
        )

    groups: list[IntakeGroup] = []
    manifest_records: list[dict[str, Any]] = []
    omitted_non_lfs = 0
    group_summaries: list[dict[str, Any]] = []
    for leaf in LEAVES:
        files, records, omitted = _collect_leaf(
            arguments.repo_id, arguments.revision, leaf
        )
        omitted_non_lfs += omitted
        manifest_records.extend(records)
        slug = _slug(leaf)
        groups.append(
            IntakeGroup(
                group_id=f"emseek-{slug}-single-conservative-group",
                partition=DataPartition.TRAIN,
                acquisition_id=f"emseek-{slug}-unresolved-acquisition",
                sample_id=f"emseek-{slug}-unresolved-sample",
                lab_id="emseek_source_metadata_unresolved",
                microscope_id="unresolved",
                material=leaf.rsplit("/", 1)[-1],
                tasks=_tasks_for_leaf(leaf),
                grouping_evidence="single_group_conservative",
                annotation_qualification=(
                    AnnotationQualification.AUTHOR_CURATED
                ),
                reviewer_count=0,
                control_role=ControlRole.MIXED,
                eligible_for_active_learning=True,
                files=tuple(files),
            )
        )
        group_summaries.append(
            {
                "leaf": leaf,
                "group_id": groups[-1].group_id,
                "file_count": len(files),
                "bytes": sum(item.bytes for item in files),
            }
        )

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "manifest_format": "phyevo-mic-hf-remote-source-v1",
        "created_at_utc": created_at,
        "repo_id": arguments.repo_id,
        "revision": arguments.revision,
        "dataset_id": DATASET_ID,
        "bulk_download_performed": False,
        "lfs_sha256_semantics": "Hugging Face LFS oid is content SHA-256",
        "group_count": len(groups),
        "file_count": len(manifest_records),
        "total_bytes": sum(item["bytes"] for item in manifest_records),
        "omitted_non_lfs_or_non_scientific_files": omitted_non_lfs,
        "excluded_prefixes": [
            "atomic_defects/WSe2 (already admitted separately)",
            "single_atom_catalysts (absent at pinned revision)",
        ],
        "groups": group_summaries,
        "files": manifest_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    manifest_file = IntakeFile(
        file_id="emseek-remaining-source-manifest",
        relative_path=manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        role=FileRole.METADATA,
        sha256=sha256_file(manifest_path),
        bytes=manifest_path.stat().st_size,
        pair_id=None,
    )
    groups.append(
        IntakeGroup(
            group_id="emseek-remaining-source-manifest-group",
            partition=DataPartition.TRAIN,
            acquisition_id="emseek-remaining-source-manifest",
            sample_id="emseek-remaining-source-manifest",
            lab_id="emseek_repository",
            microscope_id="not_applicable",
            material="mixed",
            tasks=("provenance",),
            grouping_evidence="repository_metadata",
            annotation_qualification=AnnotationQualification.NONE,
            reviewer_count=0,
            control_role=ControlRole.UNLABELED,
            eligible_for_active_learning=False,
            files=(manifest_file,),
        )
    )
    plan = seal_plan(
        DatasetIntakePlan(
            plan_format="phyevo-mic-data-intake-v1",
            plan_id="emseek-remaining-train-v1",
            dataset_id=DATASET_ID,
            source_uri=(
                f"https://huggingface.co/datasets/{arguments.repo_id}"
            ),
            source_revision=arguments.revision,
            license_spdx="Apache-2.0",
            admission_state=AdmissionState.TRAINING_ONLY,
            created_at_utc=created_at,
            groups=tuple(groups),
        )
    )
    write_plan(plan, plan_path)
    print(
        json.dumps(
            {
                "sealed": True,
                "bulk_download_performed": False,
                "group_count": len(plan.groups),
                "remote_file_count": len(manifest_records),
                "remote_total_bytes": manifest["total_bytes"],
                "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
                "plan": str(plan_path.relative_to(PROJECT_ROOT)),
                "content_sha256": plan.content_hash(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
