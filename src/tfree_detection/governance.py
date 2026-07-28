"""Data partitions, candidate provenance, and append-only label governance."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Sequence

import yaml

from .evidence import LabelTier

if TYPE_CHECKING:
    from .models import DetectionResult


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class DataPartition(_StringEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    DEVELOPMENT_TEST = "development_test"
    LOCKED_EXTERNAL_TEST = "locked_external_test"


class AnnotationSource(_StringEnum):
    PHYSICS_PSEUDO = "physics_pseudo"
    DIGITAL_TWIN = "digital_twin"
    EXPERT_CONSENSUS = "expert_consensus"
    EXPERT_SINGLE = "expert_single"
    STUDENT_PROPOSAL = "student_proposal"


GOLD_ALLOWED_SOURCES = {
    AnnotationSource.EXPERT_CONSENSUS,
    AnnotationSource.DIGITAL_TWIN,
}


@dataclass(frozen=True)
class DataAsset:
    """One indivisible acquisition/sample unit in the partition registry."""

    asset_id: str
    dataset_id: str
    relative_path: str
    partition: DataPartition
    split_group_id: str
    acquisition_id: str
    sample_id: str
    lab_id: str
    tasks: tuple[str, ...]
    checksum: str
    checksum_algorithm: str = "sha256"
    material: str = "unknown"
    microscope_id: str = "unknown"
    parent_asset_id: str | None = None
    eligible_for_active_learning: bool = False

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DataAsset":
        return cls(
            asset_id=str(values["asset_id"]),
            dataset_id=str(values["dataset_id"]),
            relative_path=str(values["relative_path"]),
            partition=DataPartition(str(values["partition"])),
            split_group_id=str(values["split_group_id"]),
            acquisition_id=str(values["acquisition_id"]),
            sample_id=str(values["sample_id"]),
            lab_id=str(values["lab_id"]),
            tasks=tuple(str(item) for item in values["tasks"]),
            checksum=str(values["checksum"]),
            checksum_algorithm=str(values.get("checksum_algorithm", "sha256")),
            material=str(values.get("material", "unknown")),
            microscope_id=str(values.get("microscope_id", "unknown")),
            parent_asset_id=(
                str(values["parent_asset_id"])
                if values.get("parent_asset_id") is not None
                else None
            ),
            eligible_for_active_learning=bool(
                values.get("eligible_for_active_learning", False)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["partition"] = self.partition.value
        payload["tasks"] = list(self.tasks)
        return payload


@dataclass
class DataRegistry:
    """Versioned assets with leakage and active-learning safety checks."""

    registry_version: str
    split_unit: str
    assets: list[DataAsset]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataRegistry":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        registry = cls(
            registry_version=str(payload["registry_version"]),
            split_unit=str(payload["split_unit"]),
            assets=[
                DataAsset.from_dict(item) for item in payload.get("assets", [])
            ],
        )
        registry.validate()
        return registry

    def validate(self) -> None:
        if self.split_unit not in {
            "acquisition",
            "sample",
            "lab",
            "explicit_group",
        }:
            raise ValueError(
                "split_unit must be acquisition, sample, lab, or explicit_group"
            )
        ids: set[str] = set()
        partitions_by_group: Dict[str, DataPartition] = {}
        assets_by_id: Dict[str, DataAsset] = {}
        for asset in self.assets:
            if not asset.asset_id or asset.asset_id in ids:
                raise ValueError(
                    f"Asset ids must be non-empty and unique: {asset.asset_id!r}"
                )
            ids.add(asset.asset_id)
            assets_by_id[asset.asset_id] = asset
            if not asset.tasks:
                raise ValueError(f"Asset {asset.asset_id!r} has no tasks")
            old_partition = partitions_by_group.setdefault(
                asset.split_group_id, asset.partition
            )
            if old_partition != asset.partition:
                raise ValueError(
                    f"Split leakage: group {asset.split_group_id!r} occurs in "
                    f"{old_partition.value} and {asset.partition.value}"
                )
            if (
                asset.eligible_for_active_learning
                and asset.partition != DataPartition.TRAIN
            ):
                raise ValueError(
                    f"Only train assets may enter active learning; "
                    f"{asset.asset_id!r} is {asset.partition.value}"
                )
            if asset.partition == DataPartition.LOCKED_EXTERNAL_TEST and (
                asset.eligible_for_active_learning
            ):
                raise ValueError(
                    "Locked external test assets can never enter active learning"
                )
        for asset in self.assets:
            if asset.parent_asset_id is None:
                continue
            parent = assets_by_id.get(asset.parent_asset_id)
            if parent is None:
                raise ValueError(
                    f"Asset {asset.asset_id!r} has an unknown parent"
                )
            if parent.partition != asset.partition:
                raise ValueError(
                    f"Patch/derivative {asset.asset_id!r} must inherit its "
                    "parent partition"
                )
            if parent.split_group_id != asset.split_group_id:
                raise ValueError(
                    f"Patch/derivative {asset.asset_id!r} must inherit its "
                    "parent split_group_id"
                )

    def assets_for_partition(
        self, partition: DataPartition
    ) -> list[DataAsset]:
        return [asset for asset in self.assets if asset.partition == partition]

    def active_learning_assets(self) -> list[DataAsset]:
        selected = [
            asset for asset in self.assets if asset.eligible_for_active_learning
        ]
        if any(
            asset.partition == DataPartition.LOCKED_EXTERNAL_TEST
            for asset in selected
        ):
            raise RuntimeError("Locked external test leaked into active learning")
        return selected

    def content_hash(self) -> str:
        canonical = {
            "registry_version": self.registry_version,
            "split_unit": self.split_unit,
            "assets": [
                asset.to_dict()
                for asset in sorted(self.assets, key=lambda item: item.asset_id)
            ],
        }
        return hashlib.sha256(
            json.dumps(
                canonical, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def verify_files(self, project_root: str | Path) -> Dict[str, Any]:
        """Verify registered inputs without modifying or normalizing them."""

        root = Path(project_root).resolve()
        verified = []
        for asset in self.assets:
            relative = Path(asset.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"Asset {asset.asset_id!r} path must stay relative to project"
                )
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"Asset {asset.asset_id!r} resolves outside project root"
                ) from error
            if not target.is_file():
                raise FileNotFoundError(
                    f"Registered asset {asset.asset_id!r} is missing: {target}"
                )
            try:
                hasher = hashlib.new(asset.checksum_algorithm)
            except ValueError as error:
                raise ValueError(
                    f"Unsupported checksum algorithm "
                    f"{asset.checksum_algorithm!r}"
                ) from error
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            observed = hasher.hexdigest()
            if observed.lower() != asset.checksum.lower():
                raise ValueError(
                    f"Checksum mismatch for asset {asset.asset_id!r}: "
                    f"expected {asset.checksum}, observed {observed}"
                )
            verified.append(
                {
                    "asset_id": asset.asset_id,
                    "relative_path": asset.relative_path,
                    "checksum_algorithm": asset.checksum_algorithm,
                    "checksum": observed,
                    "bytes": target.stat().st_size,
                }
            )
        return {
            "verified": True,
            "asset_count": len(verified),
            "assets": verified,
        }


@dataclass(frozen=True)
class AnnotationRecord:
    """One immutable annotation event in an append-only ledger."""

    annotation_id: str
    candidate_id: str
    label_tier: LabelTier
    source: AnnotationSource
    annotation_version: str
    defect_labels: tuple[str, ...]
    accepted: bool | None
    reviewers: tuple[str, ...] = ()
    parent_annotation_id: str | None = None
    notes: str = ""

    def validate(self) -> None:
        if not self.annotation_id or not self.candidate_id:
            raise ValueError("Annotation and candidate ids are required")
        if self.label_tier == LabelTier.GOLD:
            if self.source not in GOLD_ALLOWED_SOURCES:
                raise ValueError(
                    "Gold labels require expert consensus or verifiable "
                    "digital-twin truth"
                )
            if (
                self.source == AnnotationSource.EXPERT_CONSENSUS
                and len(self.reviewers) < 2
            ):
                raise ValueError(
                    "Expert-consensus Gold requires at least two reviewers"
                )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["label_tier"] = self.label_tier.value
        payload["source"] = self.source.value
        payload["defect_labels"] = list(self.defect_labels)
        payload["reviewers"] = list(self.reviewers)
        return payload


@dataclass
class AnnotationLedger:
    """In-memory validator for an append-only annotation history."""

    records: list[AnnotationRecord] = field(default_factory=list)

    def append(self, record: AnnotationRecord) -> None:
        record.validate()
        if any(
            old.annotation_id == record.annotation_id for old in self.records
        ):
            raise ValueError(
                f"Annotation id {record.annotation_id!r} already exists"
            )
        candidate_history = [
            old
            for old in self.records
            if old.candidate_id == record.candidate_id
        ]
        if candidate_history and record.parent_annotation_id is not None:
            if record.parent_annotation_id not in {
                old.annotation_id for old in candidate_history
            }:
                raise ValueError(
                    "parent_annotation_id must reference the same candidate"
                )
        if any(
            old.label_tier == LabelTier.GOLD for old in candidate_history
        ) and record.label_tier != LabelTier.GOLD:
            raise ValueError("Gold labels cannot be overwritten by weaker labels")
        self.records.append(record)

    def effective(self, candidate_id: str) -> AnnotationRecord | None:
        history = [
            record
            for record in self.records
            if record.candidate_id == candidate_id
        ]
        if not history:
            return None
        gold = [record for record in history if record.label_tier == LabelTier.GOLD]
        return (gold or history)[-1]

    def content_hash(self) -> str:
        canonical = [record.to_dict() for record in self.records]
        return hashlib.sha256(
            json.dumps(
                canonical, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class CandidateRecord:
    """Candidate-level table row linking geometry, evidence, and provenance."""

    candidate_id: str
    task: str
    source_id: str
    dataset_id: str | None
    partition: str | None
    geometry: str
    violation_types: tuple[str, ...]
    defect_type: str
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area_pixels: int
    teacher_score: float
    legacy_confidence: float
    reliability: float
    decision_state: str
    label_tier: str
    evidence_vector: Mapping[str, Mapping[str, float]]
    quality_flags: tuple[str, ...]
    provenance: Mapping[str, Any]
    resolved_family: str = ""
    resolved_label: str = ""
    epistemic_state: str = "unresolved"
    hypothesis_ranking: tuple[Mapping[str, Any], ...] = ()
    missing_auxiliary_evidence: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["violation_types"] = list(self.violation_types)
        payload["bbox_xyxy"] = list(self.bbox_xyxy)
        payload["centroid_xy"] = list(self.centroid_xy)
        payload["quality_flags"] = list(self.quality_flags)
        payload["hypothesis_ranking"] = [
            dict(item) for item in self.hypothesis_ranking
        ]
        payload["missing_auxiliary_evidence"] = list(
            self.missing_auxiliary_evidence
        )
        return payload


def candidate_records_from_result(
    result: "DetectionResult",
    *,
    dataset_id: str | None = None,
    partition: DataPartition | None = None,
) -> list[CandidateRecord]:
    if result.evidence_bundle is None:
        raise ValueError(
            "DetectionResult has no EvidenceBundle; run through the framework "
            "with evidence_protocol.enabled=true"
        )
    source_id = str(result.evidence_bundle.provenance.get("source_id", "unknown"))
    return [
        CandidateRecord(
            candidate_id=region.candidate_id,
            task=result.task,
            source_id=source_id,
            dataset_id=dataset_id,
            partition=partition.value if partition is not None else None,
            geometry=region.spatial_support,
            violation_types=tuple(region.violation_types),
            defect_type=region.defect_type,
            bbox_xyxy=region.bbox_xyxy,
            centroid_xy=region.centroid_xy,
            area_pixels=region.area_pixels,
            teacher_score=region.mean_score,
            legacy_confidence=region.confidence,
            reliability=region.reliability,
            decision_state=region.decision_state,
            label_tier=region.label_tier,
            evidence_vector=region.evidence_vector,
            quality_flags=tuple(region.quality_flags),
            provenance=region.provenance,
            resolved_family=region.resolved_family,
            resolved_label=region.resolved_label,
            epistemic_state=region.epistemic_state,
            hypothesis_ranking=tuple(region.hypothesis_ranking),
            missing_auxiliary_evidence=tuple(
                region.missing_auxiliary_evidence
            ),
        )
        for region in result.regions
    ]


def write_candidate_jsonl(
    records: Iterable[CandidateRecord], path: str | Path
) -> None:
    """Write a new candidate table; refuse accidental provenance overwrite."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing candidate table: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
            )
            stream.write("\n")


def write_candidate_csv(
    records: Sequence[CandidateRecord], path: str | Path
) -> None:
    """Write a compact, human-auditable candidate index."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing candidate table: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "candidate_id",
        "task",
        "source_id",
        "dataset_id",
        "partition",
        "geometry",
        "violation_types",
        "defect_type",
        "bbox_xyxy",
        "centroid_xy",
        "area_pixels",
        "teacher_score",
        "legacy_confidence",
        "reliability",
        "decision_state",
        "label_tier",
        "quality_flags",
    ]
    with target.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            values = record.to_dict()
            for key in (
                "violation_types",
                "bbox_xyxy",
                "centroid_xy",
                "quality_flags",
            ):
                values[key] = json.dumps(values[key], ensure_ascii=False)
            writer.writerow({key: values[key] for key in columns})
