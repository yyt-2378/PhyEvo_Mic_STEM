#!/usr/bin/env python
"""Build a sealed acquisition-safe intake plan for the fast 4D-STEM data."""

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
    / "datasets/public/werbrouck_2026_fast_4dstem_phase"
)
SPECS = (
    (
        "tio2",
        "werbrouck-tio2-sio2",
        "werbrouck-tio2-on-sio2",
        "TiO2_on_SiO2",
        "TiO2_dataset.h5",
        142469777,
        "45bb27a45a8c9568a7896d8cec8faa13",
    ),
    (
        "nmc-lgps",
        "werbrouck-nmc-lgps",
        "werbrouck-nmc-lgps-interface",
        "NMC_LGPS_interface",
        "NMC_LGPS_dataset.h5",
        159258737,
        "c3d63ca5a29e5e471245626a65a139d0",
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
            "datasets/intake/"
            "werbrouck_fast_4dstem_development_v1.sealed.yaml"
        ),
    )
    parser.add_argument(
        "--plan-id",
        default="werbrouck-fast-4dstem-development-v1",
        help="Versioned plan identifier stored inside the sealed plan",
    )
    parser.add_argument(
        "--acquisition",
        choices=("tio2", "nmc-lgps", "all"),
        default="tio2",
        help=(
            "Default freezes the smaller published acquisition before "
            "detector execution"
        ),
    )
    arguments = parser.parse_args()
    output = _inside(arguments.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a sealed intake plan")

    groups = []
    selected_specs = [
        spec
        for spec in SPECS
        if arguments.acquisition == "all"
        or spec[0] == arguments.acquisition
    ]
    for (
        _,
        acquisition_id,
        sample_id,
        material,
        filename,
        size,
        md5,
    ) in selected_specs:
        source = _inside(DATASET_ROOT / filename)
        if not source.is_file() or source.stat().st_size != size:
            raise ValueError(f"Missing or incomplete source: {source}")
        if _md5(source) != md5:
            raise ValueError(f"MD5 mismatch: {source}")
        groups.append(
            IntakeGroup(
                group_id=acquisition_id,
                partition=DataPartition.DEVELOPMENT_TEST,
                acquisition_id=acquisition_id,
                sample_id=sample_id,
                lab_id="werbrouck_2026_archive",
                microscope_id="published_4dstem_instrument",
                material=material,
                tasks=(
                    "amorphous_region",
                    "phase_boundary",
                    "virtual_detector",
                ),
                grouping_evidence="repository_metadata",
                annotation_qualification=AnnotationQualification.NONE,
                reviewer_count=0,
                control_role=ControlRole.MIXED,
                eligible_for_active_learning=False,
                files=(
                    IntakeFile(
                        file_id=f"{acquisition_id}-raw-4dstem",
                        relative_path=source.relative_to(
                            PROJECT_ROOT
                        ).as_posix(),
                        role=FileRole.IMAGE,
                        sha256=sha256_file(source),
                        bytes=size,
                    ),
                ),
            )
        )
    plan = DatasetIntakePlan(
        plan_format="phyevo-mic-data-intake-v1",
        plan_id=arguments.plan_id,
        dataset_id="werbrouck_2026_fast_4dstem_phase",
        source_uri="https://zenodo.org/records/20366781",
        source_revision="10.5281/zenodo.20366781:2026-06-04",
        license_spdx="CC-BY-4.0",
        admission_state=AdmissionState.DEVELOPMENT_CANDIDATE,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        groups=tuple(groups),
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
                "group_count": len(sealed.groups),
                "content_sha256": sealed.content_hash(),
                "output": output.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
