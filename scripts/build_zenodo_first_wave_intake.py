#!/usr/bin/env python
"""Build verified sealed intake plans for downloaded first-wave Zenodo data."""

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
PUBLIC_ROOT = PROJECT_ROOT / "datasets" / "public"


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path must remain inside project: {path}") from error
    return resolved


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(
    *,
    dataset_id: str,
    name: str,
    file_id: str,
    role: FileRole,
    expected_bytes: int,
    expected_md5: str,
) -> IntakeFile:
    path = _inside_project(PUBLIC_ROOT / dataset_id / name)
    if not path.is_file():
        raise FileNotFoundError(f"First-wave asset is missing: {path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Byte-size mismatch for {path}")
    if _md5(path) != expected_md5:
        raise ValueError(f"MD5 mismatch for {path}")
    return IntakeFile(
        file_id=file_id,
        relative_path=path.relative_to(PROJECT_ROOT).as_posix(),
        role=role,
        sha256=sha256_file(path),
        bytes=expected_bytes,
        pair_id=None,
    )


def _write_sealed(plan: DatasetIntakePlan, output: Path) -> dict[str, object]:
    sealed = seal_plan(plan)
    sealed.validate(
        project_root=PROJECT_ROOT,
        verify_files=True,
        require_seal=True,
    )
    write_plan(sealed, _inside_project(output))
    return {
        "plan_id": sealed.plan_id,
        "groups": len(sealed.groups),
        "files": sum(len(group.files) for group in sealed.groups),
        "content_sha256": sealed.content_hash(),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--guo-output",
        type=Path,
        default=Path(
            "datasets/intake/guo_superscan_development_v1.sealed.yaml"
        ),
    )
    parser.add_argument(
        "--sigma-output",
        type=Path,
        default=Path(
            "datasets/intake/al_sigma3_development_v1.sealed.yaml"
        ),
    )
    arguments = parser.parse_args()
    if _inside_project(arguments.guo_output).exists() or _inside_project(
        arguments.sigma_output
    ).exists():
        raise FileExistsError("Refusing to overwrite a sealed intake plan")

    created_at = datetime.now(timezone.utc).isoformat()
    superscan = _verified_file(
        dataset_id="guo_2021_atomic_defects",
        name="Recording_of_SuperScan_img_stack.ndata1",
        file_id="guo-superscan-movie",
        role=FileRole.IMAGE,
        expected_bytes=104862360,
        expected_md5="06ef37dd2e9dcf36bcab8b360da244c2",
    )
    guo = DatasetIntakePlan(
        plan_format="phyevo-mic-data-intake-v1",
        plan_id="guo-superscan-development-v1",
        dataset_id="guo_2021_superscan",
        source_uri="https://zenodo.org/records/5520169",
        source_revision="10.5281/zenodo.5520169:2024-07-17",
        license_spdx="CC-BY-4.0",
        admission_state=AdmissionState.DEVELOPMENT_CANDIDATE,
        created_at_utc=created_at,
        groups=(
            IntakeGroup(
                group_id="guo-superscan-single-acquisition",
                partition=DataPartition.DEVELOPMENT_TEST,
                acquisition_id="guo-superscan-img-stack",
                sample_id="guo-superscan-sample-unresolved",
                lab_id="guo_2021_archive",
                microscope_id="unresolved_public_metadata",
                material="unresolved_atomic_crystal",
                tasks=(
                    "atomic_point_defect",
                    "stacking_fault",
                    "temporal_defect_evolution",
                ),
                grouping_evidence="repository_metadata",
                annotation_qualification=AnnotationQualification.NONE,
                reviewer_count=0,
                control_role=ControlRole.UNLABELED,
                eligible_for_active_learning=False,
                files=(superscan,),
            ),
        ),
    )

    sigma_specs = (
        (
            "sigma3-orientation-i",
            "Sigma3_ORI.emd",
            FileRole.IMAGE,
            28883920,
            "986f7d23a80d5b1f27b40489a96c7c02",
            ("twin_boundary", "interface_defect"),
            AnnotationQualification.NONE,
        ),
        (
            "sigma3-orientation-ii",
            "Sigma3_ORII.emd",
            FileRole.IMAGE,
            58721072,
            "cacbf850efd8a733071d2c9c9956de74",
            ("twin_boundary", "interface_defect"),
            AnnotationQualification.NONE,
        ),
        (
            "sigma3-ebsd",
            "EBSD_data.osc",
            FileRole.METADATA,
            68773183,
            "0f22fa7d439b8d31a31c595ccdfd6eab",
            ("twin_boundary", "grain_boundary", "orientation_evidence"),
            AnnotationQualification.NONE,
        ),
        (
            "sigma3-simulation",
            "simulation.zip",
            FileRole.METADATA,
            122054770,
            "8595b02d66fb1a9abf79f1bbb68604fa",
            ("twin_boundary", "digital_twin"),
            AnnotationQualification.SIMULATION_TRUTH,
        ),
        (
            "sigma3-readme",
            "README.md",
            FileRole.METADATA,
            5897,
            "909d52a16d60c4aa21b52862f98b264c",
            ("provenance",),
            AnnotationQualification.NONE,
        ),
    )
    sigma_groups: list[IntakeGroup] = []
    for group_id, name, role, size, md5, tasks, qualification in sigma_specs:
        item = _verified_file(
            dataset_id="al_sigma3_twin_boundaries",
            name=name,
            file_id=f"{group_id}-file",
            role=role,
            expected_bytes=size,
            expected_md5=md5,
        )
        sigma_groups.append(
            IntakeGroup(
                group_id=group_id,
                partition=DataPartition.DEVELOPMENT_TEST,
                acquisition_id=group_id,
                sample_id=(
                    "al-sigma3-experimental-sample"
                    if group_id != "sigma3-simulation"
                    else "al-sigma3-digital-twin"
                ),
                lab_id="sigma3_2022_archive",
                microscope_id=(
                    "unresolved_public_metadata"
                    if role == FileRole.IMAGE
                    else "not_applicable_or_metadata"
                ),
                material="Al",
                tasks=tasks,
                grouping_evidence="repository_metadata",
                annotation_qualification=qualification,
                reviewer_count=0,
                control_role=ControlRole.UNLABELED,
                eligible_for_active_learning=False,
                files=(item,),
            )
        )
    sigma = DatasetIntakePlan(
        plan_format="phyevo-mic-data-intake-v1",
        plan_id="al-sigma3-development-v1",
        dataset_id="al_sigma3_twin_boundaries",
        source_uri="https://zenodo.org/records/6414955",
        source_revision="10.5281/zenodo.6414955:2024-09-30",
        license_spdx="CC-BY-4.0",
        admission_state=AdmissionState.DEVELOPMENT_CANDIDATE,
        created_at_utc=created_at,
        groups=tuple(sigma_groups),
    )

    results = [
        _write_sealed(guo, arguments.guo_output),
        _write_sealed(sigma, arguments.sigma_output),
    ]
    print(json.dumps({"sealed": True, "plans": results}, indent=2))


if __name__ == "__main__":
    main()
