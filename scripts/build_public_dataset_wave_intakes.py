#!/usr/bin/env python
"""Seal acquisition-safe intake plans for the compact public-data wave."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

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
DATA_ROOT = PROJECT_ROOT / "datasets" / "public"
SPECS: dict[str, dict[str, Any]] = {
    "zhang_2026_wse2_3d_defects_coordinates": {
        "plan_id": "wse2-3d-raw-1657-and-coordinates-development-v2",
        "output": (
            "datasets/intake/"
            "wse2_3d_raw_1657_development_v2.sealed.yaml"
        ),
        "group_id": "wse2-3d-raw-1657-and-coordinate-archive",
        "acquisition_id": "wse2-3d-study-plus-raw-1657",
        "sample_id": "twisted-bilayer-wse2-published-study",
        "lab_id": "zhang_2026_archive",
        "microscope_id": "EMPAD_published_metadata",
        "material": "twisted_bilayer_WSe2",
        "grouping_evidence": "single_group_conservative",
        "qualification": AnnotationQualification.AUTHOR_CURATED,
        "control_role": ControlRole.MIXED,
    },
    "strauch_2021_srtio3_high_resolution_4dstem": {
        "plan_id": "srtio3-high-resolution-4dstem-development-v1",
        "output": (
            "datasets/intake/"
            "srtio3_high_resolution_4dstem_development_v1.sealed.yaml"
        ),
        "group_id": "srtio3-high-resolution-single-acquisition",
        "acquisition_id": "strauch-2021-srtio3-20200518-165148",
        "sample_id": "strauch-2021-srtio3-crystal",
        "lab_id": "strauch_2021_archive",
        "microscope_id": "FEI_Titan_80_300_Medipix_Merlin",
        "material": "SrTiO3",
        "grouping_evidence": "repository_metadata",
        "qualification": AnnotationQualification.NONE,
        "control_role": ControlRole.UNLABELED,
    },
    "mills_2022_irradiated_al_4dstem": {
        "plan_id": "irradiated-al-4dstem-development-v1",
        "output": (
            "datasets/intake/"
            "irradiated_al_4dstem_development_v1.sealed.yaml"
        ),
        "group_id": "irradiated-al-raw-and-calibration",
        "acquisition_id": "mills-2022-irradiated-al",
        "sample_id": "mills-2022-irradiated-al-sample",
        "lab_id": "mills_2022_archive",
        "microscope_id": "FEI_TitanX_Gatan_Orius_830",
        "material": "irradiated_Al",
        "grouping_evidence": "repository_metadata",
        "qualification": AnnotationQualification.AUTHOR_CURATED,
        "control_role": ControlRole.POSITIVE,
    },
    "wang_2026_polymer_lamella_4dstem_rt": {
        "plan_id": "polymer-lamella-4dstem-rt-development-v1",
        "output": (
            "datasets/intake/"
            "polymer_lamella_4dstem_rt_development_v1.sealed.yaml"
        ),
        "group_id": "polymer-lamella-room-temperature-acquisition",
        "acquisition_id": "wang-2026-polymer-4dstem-rt",
        "sample_id": "wang-2026-polymer-lamella-sample",
        "lab_id": "wang_2026_archive",
        "microscope_id": "published_300kV_4dstem_instrument",
        "material": "polymer_lamellar_clusters",
        "grouping_evidence": "repository_metadata",
        "qualification": AnnotationQualification.NONE,
        "control_role": ControlRole.UNLABELED,
    },
}


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_role(value: str) -> FileRole:
    if value == "image":
        return FileRole.IMAGE
    if value == "annotation":
        return FileRole.ANNOTATION
    return FileRole.METADATA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/manifest.yaml"),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(SPECS),
        help="Repeat to select datasets; default seals the complete wave",
    )
    arguments = parser.parse_args()
    manifest_path = _inside(arguments.manifest)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in payload["datasets"]}
    selected = arguments.dataset or list(SPECS)

    prepared: list[tuple[DatasetIntakePlan, Path]] = []
    for dataset_id in selected:
        spec = SPECS[dataset_id]
        dataset = by_id[dataset_id]
        output = _inside(Path(spec["output"]))
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite a sealed intake plan: {output}"
            )
        intake_files = []
        tasks: set[str] = set()
        for entry in dataset["files"]:
            source = _inside(DATA_ROOT / dataset_id / entry["name"])
            if (
                not source.is_file()
                or source.stat().st_size != int(entry["bytes"])
            ):
                raise ValueError(f"Missing or incomplete source: {source}")
            if _md5(source) != str(entry["md5"]).lower():
                raise ValueError(f"MD5 mismatch: {source}")
            tasks.update(str(item) for item in entry.get("tasks", []))
            intake_files.append(
                IntakeFile(
                    file_id=f"{dataset_id}:{entry['name']}",
                    relative_path=source.relative_to(PROJECT_ROOT).as_posix(),
                    role=_file_role(str(entry.get("role", "metadata"))),
                    sha256=sha256_file(source),
                    bytes=int(entry["bytes"]),
                )
            )
        plan = DatasetIntakePlan(
            plan_format="phyevo-mic-data-intake-v1",
            plan_id=str(spec["plan_id"]),
            dataset_id=dataset_id,
            source_uri=str(dataset["record_url"]),
            source_revision=str(dataset["source_revision"]),
            license_spdx=str(dataset["license"]),
            admission_state=AdmissionState.DEVELOPMENT_CANDIDATE,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            groups=(
                IntakeGroup(
                    group_id=str(spec["group_id"]),
                    partition=DataPartition.DEVELOPMENT_TEST,
                    acquisition_id=str(spec["acquisition_id"]),
                    sample_id=str(spec["sample_id"]),
                    lab_id=str(spec["lab_id"]),
                    microscope_id=str(spec["microscope_id"]),
                    material=str(spec["material"]),
                    tasks=tuple(sorted(tasks)),
                    grouping_evidence=str(spec["grouping_evidence"]),
                    annotation_qualification=spec["qualification"],
                    reviewer_count=0,
                    control_role=spec["control_role"],
                    eligible_for_active_learning=False,
                    files=tuple(intake_files),
                ),
            ),
        )
        sealed = seal_plan(plan)
        sealed.validate(
            project_root=PROJECT_ROOT,
            verify_files=True,
            require_seal=True,
        )
        prepared.append((sealed, output))

    for sealed, output in prepared:
        write_plan(sealed, output)
    print(
        json.dumps(
            {
                "sealed_plan_count": len(prepared),
                "plans": [
                    {
                        "plan_id": plan.plan_id,
                        "dataset_id": plan.dataset_id,
                        "content_sha256": plan.content_hash(),
                        "output": output.relative_to(
                            PROJECT_ROOT
                        ).as_posix(),
                    }
                    for plan, output in prepared
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
