"""Prospective, group-safe admission of external microscopy datasets.

The intake plan is deliberately separate from the model-facing data registry.
It records why files belong to the same acquisition group and what level of
annotation provenance is actually supported before any file can be promoted to
training, calibration, or a locked external test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

from .governance import DataAsset, DataPartition, DataRegistry


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class FileRole(_StringEnum):
    IMAGE = "image"
    ANNOTATION = "annotation"
    METADATA = "metadata"


class AnnotationQualification(_StringEnum):
    NONE = "none"
    AUTHOR_CURATED = "author_curated"
    EXPERT_SINGLE = "expert_single"
    EXPERT_CONSENSUS = "expert_consensus"
    SIMULATION_TRUTH = "simulation_truth"


class AdmissionState(_StringEnum):
    QUARANTINE = "quarantine"
    TRAINING_ONLY = "training_only"
    DEVELOPMENT_CANDIDATE = "development_candidate"
    CALIBRATION_CANDIDATE = "calibration_candidate"
    LOCKED_CANDIDATE = "locked_candidate"


class ControlRole(_StringEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNLABELED = "unlabeled"


ALLOWED_GROUPING_EVIDENCE = {
    "laboratory_log",
    "repository_metadata",
    "author_statement",
    "single_group_conservative",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class IntakeFile:
    file_id: str
    relative_path: str
    role: FileRole
    sha256: str
    bytes: int
    pair_id: str | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "IntakeFile":
        return cls(
            file_id=str(values["file_id"]),
            relative_path=str(values["relative_path"]),
            role=FileRole(str(values["role"])),
            sha256=str(values["sha256"]).lower(),
            bytes=int(values["bytes"]),
            pair_id=(
                str(values["pair_id"])
                if values.get("pair_id") is not None
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["role"] = self.role.value
        return payload


@dataclass(frozen=True)
class IntakeGroup:
    group_id: str
    partition: DataPartition
    acquisition_id: str
    sample_id: str
    lab_id: str
    microscope_id: str
    material: str
    tasks: tuple[str, ...]
    grouping_evidence: str
    annotation_qualification: AnnotationQualification
    reviewer_count: int
    control_role: ControlRole
    eligible_for_active_learning: bool
    files: tuple[IntakeFile, ...]

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "IntakeGroup":
        return cls(
            group_id=str(values["group_id"]),
            partition=DataPartition(str(values["partition"])),
            acquisition_id=str(values["acquisition_id"]),
            sample_id=str(values["sample_id"]),
            lab_id=str(values["lab_id"]),
            microscope_id=str(values.get("microscope_id", "unknown")),
            material=str(values.get("material", "unknown")),
            tasks=tuple(str(item) for item in values["tasks"]),
            grouping_evidence=str(values["grouping_evidence"]),
            annotation_qualification=AnnotationQualification(
                str(values["annotation_qualification"])
            ),
            reviewer_count=int(values.get("reviewer_count", 0)),
            control_role=ControlRole(
                str(values.get("control_role", "unlabeled"))
            ),
            eligible_for_active_learning=bool(
                values.get("eligible_for_active_learning", False)
            ),
            files=tuple(
                IntakeFile.from_dict(item) for item in values.get("files", [])
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "partition": self.partition.value,
            "acquisition_id": self.acquisition_id,
            "sample_id": self.sample_id,
            "lab_id": self.lab_id,
            "microscope_id": self.microscope_id,
            "material": self.material,
            "tasks": list(self.tasks),
            "grouping_evidence": self.grouping_evidence,
            "annotation_qualification": self.annotation_qualification.value,
            "reviewer_count": self.reviewer_count,
            "control_role": self.control_role.value,
            "eligible_for_active_learning": self.eligible_for_active_learning,
            "files": [item.to_dict() for item in self.files],
        }


@dataclass(frozen=True)
class DatasetIntakePlan:
    plan_format: str
    plan_id: str
    dataset_id: str
    source_uri: str
    source_revision: str
    license_spdx: str
    admission_state: AdmissionState
    created_at_utc: str
    groups: tuple[IntakeGroup, ...]
    seal_sha256: str | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DatasetIntakePlan":
        seal = values.get("seal") or {}
        return cls(
            plan_format=str(values["plan_format"]),
            plan_id=str(values["plan_id"]),
            dataset_id=str(values["dataset_id"]),
            source_uri=str(values["source_uri"]),
            source_revision=str(values["source_revision"]),
            license_spdx=str(values["license_spdx"]),
            admission_state=AdmissionState(str(values["admission_state"])),
            created_at_utc=str(values["created_at_utc"]),
            groups=tuple(
                IntakeGroup.from_dict(item) for item in values.get("groups", [])
            ),
            seal_sha256=(
                str(seal["content_sha256"]).lower()
                if seal.get("content_sha256")
                else None
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetIntakePlan":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, Mapping):
            raise ValueError("Intake plan must be a YAML mapping")
        return cls.from_dict(values)

    def content_dict(self) -> Dict[str, Any]:
        return {
            "plan_format": self.plan_format,
            "plan_id": self.plan_id,
            "dataset_id": self.dataset_id,
            "source_uri": self.source_uri,
            "source_revision": self.source_revision,
            "license_spdx": self.license_spdx,
            "admission_state": self.admission_state.value,
            "created_at_utc": self.created_at_utc,
            "groups": [group.to_dict() for group in self.groups],
        }

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.content_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self, *, include_seal: bool = True) -> Dict[str, Any]:
        payload = self.content_dict()
        if include_seal and self.seal_sha256 is not None:
            payload["seal"] = {
                "algorithm": "sha256",
                "content_sha256": self.seal_sha256,
            }
        return payload

    def validate(
        self,
        *,
        project_root: str | Path | None = None,
        verify_files: bool = False,
        require_seal: bool = True,
    ) -> None:
        if self.plan_format != "phyevo-mic-data-intake-v1":
            raise ValueError(f"Unsupported intake format: {self.plan_format!r}")
        if not all(
            (
                self.plan_id,
                self.dataset_id,
                self.source_uri,
                self.source_revision,
                self.license_spdx,
                self.created_at_utc,
            )
        ):
            raise ValueError("Intake provenance fields cannot be empty")
        if self.license_spdx.lower() in {"unknown", "none", "unlicensed"}:
            if self.admission_state != AdmissionState.QUARANTINE:
                raise ValueError(
                    "Unknown or absent license can only remain in quarantine"
                )
        if not self.groups:
            raise ValueError("Intake plan must contain at least one group")
        if require_seal:
            if self.seal_sha256 is None:
                raise ValueError("Intake plan is not sealed")
            if self.seal_sha256 != self.content_hash():
                raise ValueError("Intake seal does not match plan contents")

        group_ids: set[str] = set()
        file_ids: set[str] = set()
        relative_paths: set[str] = set()
        hashes_by_partition: dict[str, DataPartition] = {}
        calibration_roles: set[ControlRole] = set()
        for group in self.groups:
            if not group.group_id or group.group_id in group_ids:
                raise ValueError(
                    f"Group ids must be non-empty and unique: {group.group_id!r}"
                )
            group_ids.add(group.group_id)
            if group.grouping_evidence not in ALLOWED_GROUPING_EVIDENCE:
                raise ValueError(
                    f"Unsupported grouping evidence: "
                    f"{group.grouping_evidence!r}"
                )
            if not all(
                (
                    group.acquisition_id,
                    group.sample_id,
                    group.lab_id,
                    group.tasks,
                )
            ):
                raise ValueError(
                    f"Group {group.group_id!r} lacks acquisition provenance"
                )
            if (
                group.eligible_for_active_learning
                and group.partition != DataPartition.TRAIN
            ):
                raise ValueError(
                    "Only train groups may be active-learning candidates"
                )
            if group.partition == DataPartition.CALIBRATION:
                calibration_roles.add(group.control_role)
                if self.admission_state != AdmissionState.CALIBRATION_CANDIDATE:
                    raise ValueError(
                        "Calibration assets require calibration_candidate state"
                    )
                if (
                    group.annotation_qualification
                    != AnnotationQualification.EXPERT_CONSENSUS
                    or group.reviewer_count < 2
                ):
                    raise ValueError(
                        "Experimental calibration requires at least two-expert "
                        "consensus; author-curated labels are not Gold"
                    )
            if group.partition == DataPartition.LOCKED_EXTERNAL_TEST:
                if self.admission_state != AdmissionState.LOCKED_CANDIDATE:
                    raise ValueError(
                        "Locked assets require locked_candidate state"
                    )
                if group.eligible_for_active_learning:
                    raise ValueError(
                        "Locked assets can never enter active learning"
                    )
            if (
                self.admission_state == AdmissionState.TRAINING_ONLY
                and group.partition != DataPartition.TRAIN
            ):
                raise ValueError(
                    "training_only plans may contain only train groups"
                )
            if (
                self.admission_state == AdmissionState.DEVELOPMENT_CANDIDATE
                and group.partition != DataPartition.DEVELOPMENT_TEST
            ):
                raise ValueError(
                    "development_candidate plans may contain only "
                    "development_test groups"
                )
            if not group.files:
                raise ValueError(f"Group {group.group_id!r} contains no files")
            pair_roles: dict[str, set[FileRole]] = {}
            for item in group.files:
                if not item.file_id or item.file_id in file_ids:
                    raise ValueError(
                        f"File ids must be non-empty and unique: "
                        f"{item.file_id!r}"
                    )
                file_ids.add(item.file_id)
                relative = Path(item.relative_path)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"Unsafe intake path: {item.relative_path!r}"
                    )
                if item.relative_path in relative_paths:
                    raise ValueError(
                        f"File path appears more than once: "
                        f"{item.relative_path!r}"
                    )
                relative_paths.add(item.relative_path)
                if item.bytes < 0 or len(item.sha256) != 64:
                    raise ValueError(
                        f"Invalid file size/checksum for {item.file_id!r}"
                    )
                try:
                    int(item.sha256, 16)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid SHA-256 for {item.file_id!r}"
                    ) from error
                previous_partition = hashes_by_partition.setdefault(
                    item.sha256, group.partition
                )
                if previous_partition != group.partition:
                    raise ValueError(
                        "Identical file content cannot cross partitions"
                    )
                if item.pair_id:
                    pair_roles.setdefault(item.pair_id, set()).add(item.role)
                if verify_files:
                    if project_root is None:
                        raise ValueError(
                            "project_root is required when verify_files=True"
                        )
                    self._verify_file(project_root, item)
            for pair_id, roles in pair_roles.items():
                if FileRole.ANNOTATION in roles and FileRole.IMAGE not in roles:
                    raise ValueError(
                        f"Pair {pair_id!r} has annotation but no image"
                    )

        if any(
            group.partition == DataPartition.CALIBRATION
            for group in self.groups
        ):
            has_positive = bool(
                calibration_roles & {ControlRole.POSITIVE, ControlRole.MIXED}
            )
            has_negative = bool(
                calibration_roles & {ControlRole.NEGATIVE, ControlRole.MIXED}
            )
            if not (has_positive and has_negative):
                raise ValueError(
                    "Calibration intake must include Gold positive and negative "
                    "controls"
                )

    @staticmethod
    def _verify_file(
        project_root: str | Path, item: IntakeFile
    ) -> None:
        root = Path(project_root).resolve()
        target = (root / item.relative_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"Intake file resolves outside project: {item.relative_path}"
            ) from error
        if not target.is_file():
            raise FileNotFoundError(
                f"Intake file is missing: {item.relative_path}"
            )
        if target.stat().st_size != item.bytes:
            raise ValueError(
                f"Byte-size mismatch for {item.relative_path}"
            )
        observed = sha256_file(target)
        if observed != item.sha256:
            raise ValueError(
                f"Checksum mismatch for {item.relative_path}: "
                f"expected {item.sha256}, observed {observed}"
            )


def seal_plan(draft: DatasetIntakePlan) -> DatasetIntakePlan:
    """Return a sealed copy after validating a draft's non-file semantics."""

    draft.validate(require_seal=False, verify_files=False)
    return DatasetIntakePlan(
        **{
            **draft.__dict__,
            "seal_sha256": draft.content_hash(),
        }
    )


def write_plan(plan: DatasetIntakePlan, path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite an intake plan: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            plan.to_dict(),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def merge_plan_into_registry(
    *,
    base: DataRegistry,
    plan: DatasetIntakePlan,
    registry_version: str,
    allow_exact_existing: bool = False,
) -> DataRegistry:
    """Materialize a sealed intake plan without mutating the base registry."""

    plan.validate(require_seal=True)
    existing_by_id = {asset.asset_id: asset for asset in base.assets}
    existing_ids = set(existing_by_id)
    partitions_by_checksum = {
        asset.checksum: asset.partition for asset in base.assets
    }
    new_assets: list[DataAsset] = []
    for group in plan.groups:
        images_by_pair = {
            item.pair_id: item
            for item in group.files
            if item.role == FileRole.IMAGE and item.pair_id
        }
        for item in group.files:
            asset_id = f"{plan.dataset_id}:{item.file_id}"
            previous_partition = partitions_by_checksum.setdefault(
                item.sha256, group.partition
            )
            if previous_partition != group.partition:
                raise ValueError(
                    "Identical file content cannot cross registry partitions: "
                    f"{item.relative_path!r} would enter "
                    f"{group.partition.value}, but the same checksum is already "
                    f"in {previous_partition.value}"
                )
            parent_asset_id = None
            if item.role == FileRole.ANNOTATION and item.pair_id:
                parent = images_by_pair.get(item.pair_id)
                if parent is None:
                    raise ValueError(
                        f"Annotation pair has no image: {item.pair_id}"
                    )
                parent_asset_id = (
                    f"{plan.dataset_id}:{parent.file_id}"
                )
            candidate = DataAsset(
                asset_id=asset_id,
                dataset_id=plan.dataset_id,
                relative_path=item.relative_path,
                partition=group.partition,
                split_group_id=group.group_id,
                acquisition_id=group.acquisition_id,
                sample_id=group.sample_id,
                lab_id=group.lab_id,
                tasks=group.tasks,
                checksum=item.sha256,
                checksum_algorithm="sha256",
                material=group.material,
                microscope_id=group.microscope_id,
                parent_asset_id=parent_asset_id,
                eligible_for_active_learning=(
                    group.eligible_for_active_learning
                    and item.role == FileRole.IMAGE
                ),
            )
            if asset_id in existing_ids:
                if (
                    allow_exact_existing
                    and existing_by_id[asset_id] == candidate
                ):
                    continue
                raise ValueError(f"Asset already registered: {asset_id}")
            existing_ids.add(asset_id)
            existing_by_id[asset_id] = candidate
            new_assets.append(candidate)
    merged = DataRegistry(
        registry_version=registry_version,
        split_unit="explicit_group",
        assets=[*base.assets, *new_assets],
    )
    merged.validate()
    return merged


def merge_plans_into_registry(
    *,
    base: DataRegistry,
    plans: Sequence[DatasetIntakePlan],
    registry_version: str,
    allow_exact_existing: bool = False,
) -> DataRegistry:
    """Atomically assemble a registry candidate from multiple sealed plans.

    Nothing is written until every plan has passed validation and every
    sequential merge has passed cross-plan id/checksum leakage checks.
    """

    if not plans:
        raise ValueError("At least one intake plan is required")
    merged = base
    for index, plan in enumerate(plans, start=1):
        merged = merge_plan_into_registry(
            base=merged,
            plan=plan,
            registry_version=(
                registry_version
                if index == len(plans)
                else f"{registry_version}.candidate-{index}"
            ),
            allow_exact_existing=allow_exact_existing,
        )
    merged.registry_version = registry_version
    merged.validate()
    return merged


def write_registry(registry: DataRegistry, path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite a versioned registry: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "registry_version": registry.registry_version,
        "split_unit": registry.split_unit,
        "assets": [asset.to_dict() for asset in registry.assets],
    }
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
