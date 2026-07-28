from pathlib import Path

import pytest

from tfree_detection.data_intake import (
    AdmissionState,
    AnnotationQualification,
    ControlRole,
    DatasetIntakePlan,
    FileRole,
    IntakeFile,
    IntakeGroup,
    merge_plan_into_registry,
    merge_plans_into_registry,
    seal_plan,
)
from tfree_detection.governance import DataPartition, DataRegistry


def _file(
    file_id: str,
    role: FileRole,
    *,
    pair_id: str = "pair-1",
    checksum: str | None = None,
) -> IntakeFile:
    return IntakeFile(
        file_id=file_id,
        relative_path=f"datasets/{file_id}.png",
        role=role,
        sha256=checksum or hashlib_sha256(file_id),
        bytes=10,
        pair_id=pair_id,
    )


def hashlib_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _group(
    partition: DataPartition = DataPartition.TRAIN,
    *,
    qualification: AnnotationQualification = (
        AnnotationQualification.AUTHOR_CURATED
    ),
    reviewers: int = 0,
    control_role: ControlRole = ControlRole.MIXED,
    active: bool = True,
    group_id: str = "one-acquisition",
) -> IntakeGroup:
    return IntakeGroup(
        group_id=group_id,
        partition=partition,
        acquisition_id=group_id,
        sample_id="sample",
        lab_id="lab",
        microscope_id="scope",
        material="WSe2",
        tasks=("atomic_point_defect",),
        grouping_evidence="single_group_conservative",
        annotation_qualification=qualification,
        reviewer_count=reviewers,
        control_role=control_role,
        eligible_for_active_learning=active,
        files=(
            _file(f"{group_id}-image", FileRole.IMAGE),
            _file(f"{group_id}-label", FileRole.ANNOTATION),
        ),
    )


def _plan(
    groups: tuple[IntakeGroup, ...],
    state: AdmissionState = AdmissionState.TRAINING_ONLY,
) -> DatasetIntakePlan:
    return DatasetIntakePlan(
        plan_format="phyevo-mic-data-intake-v1",
        plan_id="plan-v1",
        dataset_id="dataset",
        source_uri="https://example.org/dataset",
        source_revision="immutable-revision",
        license_spdx="CC-BY-4.0",
        admission_state=state,
        created_at_utc="2026-07-26T00:00:00Z",
        groups=groups,
    )


def test_training_plan_seals_and_only_images_enter_active_learning() -> None:
    plan = seal_plan(_plan((_group(),)))
    plan.validate()
    merged = merge_plan_into_registry(
        base=DataRegistry(
            registry_version="1",
            split_unit="explicit_group",
            assets=[],
        ),
        plan=plan,
        registry_version="2",
    )
    assert len(merged.assets) == 2
    assert [asset.asset_id for asset in merged.active_learning_assets()] == [
        "dataset:one-acquisition-image"
    ]
    annotation = next(
        asset
        for asset in merged.assets
        if asset.asset_id.endswith("label")
    )
    assert annotation.parent_asset_id == "dataset:one-acquisition-image"


def test_calibration_rejects_author_curated_labels() -> None:
    plan = _plan(
        (
            _group(
                DataPartition.CALIBRATION,
                qualification=AnnotationQualification.AUTHOR_CURATED,
                active=False,
            ),
        ),
        AdmissionState.CALIBRATION_CANDIDATE,
    )
    with pytest.raises(ValueError, match="two-expert consensus"):
        seal_plan(plan)


def test_calibration_requires_positive_and_negative_controls() -> None:
    only_positive = _group(
        DataPartition.CALIBRATION,
        qualification=AnnotationQualification.EXPERT_CONSENSUS,
        reviewers=2,
        control_role=ControlRole.POSITIVE,
        active=False,
    )
    with pytest.raises(ValueError, match="positive and negative"):
        seal_plan(
            _plan(
                (only_positive,),
                AdmissionState.CALIBRATION_CANDIDATE,
            )
        )
    negative = _group(
        DataPartition.CALIBRATION,
        qualification=AnnotationQualification.EXPERT_CONSENSUS,
        reviewers=2,
        control_role=ControlRole.NEGATIVE,
        active=False,
        group_id="negative-acquisition",
    )
    sealed = seal_plan(
        _plan(
            (only_positive, negative),
            AdmissionState.CALIBRATION_CANDIDATE,
        )
    )
    sealed.validate()


def test_identical_content_cannot_cross_partitions() -> None:
    train = _group(active=False)
    shared_hash = train.files[0].sha256
    calibration = _group(
        DataPartition.CALIBRATION,
        qualification=AnnotationQualification.EXPERT_CONSENSUS,
        reviewers=2,
        control_role=ControlRole.MIXED,
        active=False,
        group_id="calibration-acquisition",
    )
    files = list(calibration.files)
    files[0] = IntakeFile(
        **{
            **files[0].__dict__,
            "sha256": shared_hash,
        }
    )
    calibration = IntakeGroup(
        **{
            **calibration.__dict__,
            "files": tuple(files),
        }
    )
    mixed_state = DatasetIntakePlan(
        **{
            **_plan(
                (train, calibration),
                AdmissionState.CALIBRATION_CANDIDATE,
            ).__dict__,
        }
    )
    with pytest.raises(ValueError, match="Identical file content"):
        seal_plan(mixed_state)


def test_verify_files_detects_post_seal_change(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    label = tmp_path / "label.png"
    image.write_bytes(b"image")
    label.write_bytes(b"label")
    import hashlib

    group = IntakeGroup(
        **{
            **_group(active=False).__dict__,
            "files": (
                IntakeFile(
                    file_id="image",
                    relative_path="image.png",
                    role=FileRole.IMAGE,
                    sha256=hashlib.sha256(b"image").hexdigest(),
                    bytes=5,
                    pair_id="pair",
                ),
                IntakeFile(
                    file_id="label",
                    relative_path="label.png",
                    role=FileRole.ANNOTATION,
                    sha256=hashlib.sha256(b"label").hexdigest(),
                    bytes=5,
                    pair_id="pair",
                ),
            ),
        }
    )
    plan = seal_plan(_plan((group,)))
    plan.validate(project_root=tmp_path, verify_files=True)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Byte-size mismatch"):
        plan.validate(project_root=tmp_path, verify_files=True)


def test_development_candidate_accepts_only_development_groups() -> None:
    development = _group(DataPartition.DEVELOPMENT_TEST, active=False)
    sealed = seal_plan(
        _plan(
            (development,),
            AdmissionState.DEVELOPMENT_CANDIDATE,
        )
    )
    sealed.validate()

    with pytest.raises(ValueError, match="development_test groups"):
        seal_plan(
            _plan(
                (_group(DataPartition.TRAIN, active=False),),
                AdmissionState.DEVELOPMENT_CANDIDATE,
            )
        )


def test_multiple_plans_are_merged_only_after_cross_plan_checks() -> None:
    first = seal_plan(_plan((_group(),)))
    second_group = _group(
        DataPartition.DEVELOPMENT_TEST,
        active=False,
        group_id="development-acquisition",
    )
    second = seal_plan(
        DatasetIntakePlan(
            **{
                **_plan(
                    (second_group,),
                    AdmissionState.DEVELOPMENT_CANDIDATE,
                ).__dict__,
                "plan_id": "plan-v2",
                "dataset_id": "dataset-two",
            }
        )
    )
    merged = merge_plans_into_registry(
        base=DataRegistry(
            registry_version="1",
            split_unit="explicit_group",
            assets=[],
        ),
        plans=(first, second),
        registry_version="3",
    )
    assert merged.registry_version == "3"
    assert len(merged.assets) == 4

    copied_files = tuple(
        IntakeFile(
            **{
                **item.__dict__,
                "file_id": f"copied-{item.file_id}",
                "relative_path": f"datasets/copied-{item.file_id}.png",
            }
        )
        for item in first.groups[0].files
    )
    leaking_group = IntakeGroup(
        **{
            **second_group.__dict__,
            "files": copied_files,
        }
    )
    leaking = seal_plan(
        DatasetIntakePlan(
            **{
                **_plan(
                    (leaking_group,),
                    AdmissionState.DEVELOPMENT_CANDIDATE,
                ).__dict__,
                "plan_id": "leaking-plan",
                "dataset_id": "leaking-dataset",
            }
        )
    )
    with pytest.raises(ValueError, match="cross registry partitions"):
        merge_plans_into_registry(
            base=merge_plans_into_registry(
                base=DataRegistry(
                    registry_version="1",
                    split_unit="explicit_group",
                    assets=[],
                ),
                plans=(first,),
                registry_version="2",
            ),
            plans=(leaking,),
            registry_version="3",
        )


def test_exact_existing_assets_can_be_reused_for_incremental_intake() -> None:
    plan = seal_plan(_plan((_group(),)))
    first = merge_plan_into_registry(
        base=DataRegistry(
            registry_version="1",
            split_unit="explicit_group",
            assets=[],
        ),
        plan=plan,
        registry_version="2",
    )
    reused = merge_plan_into_registry(
        base=first,
        plan=plan,
        registry_version="3",
        allow_exact_existing=True,
    )
    assert len(reused.assets) == len(first.assets)
    with pytest.raises(ValueError, match="already registered"):
        merge_plan_into_registry(
            base=first,
            plan=plan,
            registry_version="3",
        )
