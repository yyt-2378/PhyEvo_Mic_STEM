#!/usr/bin/env python
"""Seal the small WSe2 coordinate/README pilot as one conservative group."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

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
DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets/public/zhang_2026_wse2_3d_defects_coordinates"
)
FILES = (
    ("README.md", 1750, "3210f45e2dbc4264451fbdab8826fb9e", FileRole.METADATA),
    (
        "coordinates.zip",
        168745,
        "f32c5556f8f9251a6f342b239cd37a80",
        FileRole.ANNOTATION,
    ),
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/intake/wse2_3d_coordinates_development_v1.sealed.yaml"
        ),
    )
    arguments = parser.parse_args()
    output = _inside(arguments.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a sealed intake plan")
    intake_files = []
    for filename, size, expected_md5, role in FILES:
        path = _inside(DATASET_ROOT / filename)
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or incomplete source: {path}")
        if _md5(path) != expected_md5:
            raise ValueError(f"MD5 mismatch: {path}")
        intake_files.append(
            IntakeFile(
                file_id=f"wse2-3d-{path.stem}",
                relative_path=path.relative_to(PROJECT_ROOT).as_posix(),
                role=role,
                sha256=sha256_file(path),
                bytes=size,
            )
        )
    plan = DatasetIntakePlan(
        plan_format="phyevo-mic-data-intake-v1",
        plan_id="wse2-3d-coordinates-development-v1",
        dataset_id="zhang_2026_wse2_3d_defects_coordinates",
        source_uri="https://zenodo.org/records/18260255",
        source_revision="10.5281/zenodo.18260255:2026-05-29",
        license_spdx="CC-BY-4.0",
        admission_state=AdmissionState.DEVELOPMENT_CANDIDATE,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        groups=(
            IntakeGroup(
                group_id="wse2-3d-coordinate-archive-single-group",
                partition=DataPartition.DEVELOPMENT_TEST,
                acquisition_id="wse2-3d-1657-1815-and-simulation-archive",
                sample_id="twisted-bilayer-wse2-published-study",
                lab_id="zhang_2026_archive",
                microscope_id="EMPAD_published_metadata",
                material="twisted_bilayer_WSe2",
                tasks=(
                    "atomic_point_defect",
                    "topological_defect",
                    "moire_corrugation",
                ),
                grouping_evidence="single_group_conservative",
                annotation_qualification=(
                    AnnotationQualification.AUTHOR_CURATED
                ),
                reviewer_count=0,
                control_role=ControlRole.MIXED,
                eligible_for_active_learning=False,
                files=tuple(intake_files),
            ),
        ),
    )
    sealed = seal_plan(plan)
    sealed.validate(
        project_root=PROJECT_ROOT, verify_files=True, require_seal=True
    )
    write_plan(sealed, output)
    print(
        json.dumps(
            {
                "sealed": True,
                "plan_id": sealed.plan_id,
                "content_sha256": sealed.content_hash(),
                "output": output.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
