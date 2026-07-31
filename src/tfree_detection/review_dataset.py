"""Two-dimensional microscopy review manifest and leakage-safe splits.

The asset registry intentionally records source files and containers.  Expert
review needs a different unit: one directly viewable 2-D plane.  This module
expands the currently admitted 2-D sources without copying or modifying the
source datasets.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from skimage.io import imread

from .preprocessing import (
    list_emd_datasets,
    load_grayscale,
    load_microscopy_stack,
)


REVIEW_MANIFEST_FORMAT = "phyevo-mic-2d-review-manifest-v1"
REVIEW_SPLIT_POLICY = "source-group-isolated-v1"


class ReviewSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class LabelKind(str, Enum):
    PIXEL_MASK = "pixel_mask"
    IMPLICIT_EMPTY_MASK = "implicit_empty_mask"
    SOURCE_PRIOR_ONLY = "source_prior_only"
    NONE = "none"


@dataclass(frozen=True)
class ReviewSample:
    """One logical 2-D plane presented to an expert."""

    sample_id: str
    dataset_id: str
    source_path: str
    source_format: str
    source_partition: str
    review_split: str
    split_group_id: str
    acquisition_id: str
    material: str
    primary_task: str
    prediction_tasks: tuple[str, ...]
    geometry_hint: str
    frame_index: int | None = None
    dataset_path: str | None = None
    channel_name: str | None = None
    label_kind: str = LabelKind.NONE.value
    label_path: str | None = None
    label_qualification: str = "none"
    source_prior: str | None = None
    review_priority: str = "normal"
    review_eligible: bool = True

    def validate(self) -> None:
        required = (
            self.sample_id,
            self.dataset_id,
            self.source_path,
            self.source_format,
            self.source_partition,
            self.review_split,
            self.split_group_id,
            self.acquisition_id,
            self.material,
            self.primary_task,
            self.geometry_hint,
        )
        if not all(required):
            raise ValueError("Review sample fields cannot be empty")
        ReviewSplit(self.review_split)
        kind = LabelKind(self.label_kind)
        if self.source_format == "emd" and self.dataset_path is None:
            raise ValueError("EMD samples require an explicit dataset_path")
        if self.frame_index is not None and self.frame_index < 0:
            raise ValueError("frame_index cannot be negative")
        if kind == LabelKind.PIXEL_MASK and not self.label_path:
            raise ValueError("Pixel-mask samples require label_path")
        if kind != LabelKind.PIXEL_MASK and self.label_path is not None:
            raise ValueError("Only pixel-mask samples may declare label_path")
        if not self.prediction_tasks:
            raise ValueError("At least one prediction task is required")
        if self.primary_task not in self.prediction_tasks:
            raise ValueError("primary_task must occur in prediction_tasks")
        if (
            self.source_partition == "locked_external_test"
            and self.review_split != ReviewSplit.TEST.value
        ):
            raise ValueError("Locked external assets must remain in test")
        if (
            self.source_partition == "locked_external_test"
            and self.review_eligible
        ):
            raise ValueError(
                "Locked external-test assets cannot enter model-assisted review"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["prediction_tasks"] = list(self.prediction_tasks)
        payload["manifest_format"] = REVIEW_MANIFEST_FORMAT
        payload["split_policy"] = REVIEW_SPLIT_POLICY
        return payload

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ReviewSample":
        sample = cls(
            sample_id=str(values["sample_id"]),
            dataset_id=str(values["dataset_id"]),
            source_path=str(values["source_path"]),
            source_format=str(values["source_format"]),
            source_partition=str(values["source_partition"]),
            review_split=str(values["review_split"]),
            split_group_id=str(values["split_group_id"]),
            acquisition_id=str(values["acquisition_id"]),
            material=str(values["material"]),
            primary_task=str(values["primary_task"]),
            prediction_tasks=tuple(
                str(item) for item in values["prediction_tasks"]
            ),
            geometry_hint=str(values["geometry_hint"]),
            frame_index=(
                None
                if values.get("frame_index") is None
                else int(values["frame_index"])
            ),
            dataset_path=(
                None
                if values.get("dataset_path") is None
                else str(values["dataset_path"])
            ),
            channel_name=(
                None
                if values.get("channel_name") is None
                else str(values["channel_name"])
            ),
            label_kind=str(values.get("label_kind", LabelKind.NONE.value)),
            label_path=(
                None
                if values.get("label_path") is None
                else str(values["label_path"])
            ),
            label_qualification=str(
                values.get("label_qualification", "none")
            ),
            source_prior=(
                None
                if values.get("source_prior") is None
                else str(values["source_prior"])
            ),
            review_priority=str(values.get("review_priority", "normal")),
            review_eligible=bool(values.get("review_eligible", True)),
        )
        sample.validate()
        return sample


def _stable_bucket(group_id: str, modulus: int = 10) -> int:
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def split_for_group(
    dataset_id: str,
    group_id: str,
    source_partition: str,
) -> ReviewSplit:
    """Return a deterministic group-level split.

    Known single-acquisition datasets use explicit conservative assignments.
    The multi-material EMSeek subset is split by material group.  The locked
    Dennler archive remains an external test and is never exposed to assisted
    review.
    """

    if source_partition == "locked_external_test":
        return ReviewSplit.TEST
    explicit = {
        "emseek_wse2_single_conservative_group": ReviewSplit.TRAIN,
        "sigma3-orientation-i": ReviewSplit.TRAIN,
        "sigma3-orientation-ii": ReviewSplit.VALIDATION,
        "guo-superscan-single-acquisition": ReviewSplit.TRAIN,
        "guo_2021_mowte2_acquisition": ReviewSplit.TRAIN,
        "guo_2021_zro2_twin_acquisition": ReviewSplit.VALIDATION,
        "guo_2021_al_partial_dislocations_acquisition": (
            ReviewSplit.VALIDATION
        ),
        "local_zeolite_reference_stretch_series": ReviewSplit.VALIDATION,
    }
    if group_id in explicit:
        return explicit[group_id]
    if dataset_id == "emseek_remaining_v1":
        bucket = _stable_bucket(group_id)
        if bucket < 7:
            return ReviewSplit.TRAIN
        if bucket < 9:
            return ReviewSplit.VALIDATION
        return ReviewSplit.TEST
    raise ValueError(
        f"No review split policy for dataset={dataset_id!r}, "
        f"group={group_id!r}"
    )


def _sample(
    *,
    sample_id: str,
    dataset_id: str,
    source_path: Path,
    source_partition: str,
    split_group_id: str,
    acquisition_id: str,
    material: str,
    primary_task: str,
    prediction_tasks: Sequence[str],
    geometry_hint: str,
    frame_index: int | None = None,
    dataset_path: str | None = None,
    channel_name: str | None = None,
    label_kind: LabelKind = LabelKind.NONE,
    label_path: Path | None = None,
    label_qualification: str = "none",
    source_prior: str | None = None,
    review_priority: str = "normal",
    review_eligible: bool = True,
) -> ReviewSample:
    review_split = split_for_group(
        dataset_id, split_group_id, source_partition
    )
    return ReviewSample(
        sample_id=sample_id,
        dataset_id=dataset_id,
        source_path=source_path.as_posix(),
        source_format=source_path.suffix.lower().lstrip("."),
        source_partition=source_partition,
        review_split=review_split.value,
        split_group_id=split_group_id,
        acquisition_id=acquisition_id,
        material=material,
        primary_task=primary_task,
        prediction_tasks=tuple(prediction_tasks),
        geometry_hint=geometry_hint,
        frame_index=frame_index,
        dataset_path=dataset_path,
        channel_name=channel_name,
        label_kind=label_kind.value,
        label_path=None if label_path is None else label_path.as_posix(),
        label_qualification=label_qualification,
        source_prior=source_prior,
        review_priority=review_priority,
        review_eligible=review_eligible,
    )


def _emseek_wse2_records(root: Path) -> list[ReviewSample]:
    base = Path("datasets/public/emseek_wse2_atomic_defects")
    records = []
    for image in sorted(
        (root / base / "image").glob("*.png"),
        key=lambda path: int(path.stem),
    ):
        relative = image.relative_to(root)
        label = base / "label" / image.name
        if not (root / label).is_file():
            raise FileNotFoundError(f"Missing EMSeek label: {label}")
        records.append(
            _sample(
                sample_id=f"emseek-wse2-{image.stem}",
                dataset_id="emseek_wse2_atomic_defects",
                source_path=relative,
                source_partition="train",
                split_group_id="emseek_wse2_single_conservative_group",
                acquisition_id="emseek_wse2_single_conservative_group",
                material="WSe2",
                primary_task="atomic_point_defect",
                prediction_tasks=("atomic_point_defect",),
                geometry_hint="point",
                label_kind=LabelKind.PIXEL_MASK,
                label_path=label,
                label_qualification="author_curated_silver",
                source_prior="atomic_site_anomaly",
                review_priority="reference_audit",
            )
        )
    return records


def _emseek_remaining_records(root: Path) -> list[ReviewSample]:
    base = Path("datasets/public/emseek_remaining_v1")
    records = []
    for image in sorted((root / base).glob("**/image/*.png")):
        relative = image.relative_to(root)
        label = relative.parent.parent / "label" / image.name
        if not (root / label).is_file():
            raise FileNotFoundError(f"Missing EMSeek label: {label}")
        category, material = relative.parts[-4], relative.parts[-3]
        slug = (
            f"{category}-{material}".lower()
            .replace("_", "-")
            .replace("@", "-")
        )
        group = f"emseek-{slug}-single-conservative-group"
        if category == "atom_columns":
            primary_task = "atomic_point_defect"
            tasks = ("atomic_point_defect",)
            geometry = "point"
            prior = "atomic_column_reference"
        elif category == "metal_alloy_defects":
            primary_task = "amorphous_region"
            tasks = ("amorphous_region", "stacking_fault")
            geometry = "region"
            prior = "damage_or_artifact_region"
        else:
            primary_task = "phase_boundary"
            tasks = ("phase_boundary",)
            geometry = "interface"
            prior = "particle_or_support_interface"
        records.append(
            _sample(
                sample_id=f"emseek-{slug}-{image.stem}",
                dataset_id="emseek_remaining_v1",
                source_path=relative,
                source_partition="train",
                split_group_id=group,
                acquisition_id=f"emseek-{slug}-unresolved-acquisition",
                material=material,
                primary_task=primary_task,
                prediction_tasks=tasks,
                geometry_hint=geometry,
                label_kind=LabelKind.PIXEL_MASK,
                label_path=label,
                label_qualification="author_curated_silver",
                source_prior=prior,
                review_priority="reference_audit",
            )
        )
    return records


def _dennler_records(root: Path) -> list[ReviewSample]:
    base = Path(
        "datasets/public/dennler_2021_hrstem_defects/locked_extracted/"
        "hrstem_defects_dataset"
    )
    records = []
    defective = sorted((root / base / "defective/images").glob("*.tif"))
    for index, image in enumerate(defective, start=1):
        relative = image.relative_to(root)
        label = base / "defective/labels" / image.name
        if not (root / label).is_file():
            raise FileNotFoundError(f"Missing Dennler label: {label}")
        records.append(
            _sample(
                sample_id=f"dennler-defective-{index:03d}",
                dataset_id="dennler_2021_hrstem_defects",
                source_path=relative,
                source_partition="locked_external_test",
                split_group_id=(
                    "dennler_2021_iii_v_jeol_arm200f_archive"
                ),
                acquisition_id="dennler_2021_complete_archive",
                material="InP_GaAs_InGaAs_InAlGaAs",
                primary_task="stacking_fault",
                prediction_tasks=(
                    "amorphous_region",
                    "stacking_fault",
                    "phase_boundary",
                ),
                geometry_hint="region",
                label_kind=LabelKind.PIXEL_MASK,
                label_path=label,
                label_qualification="author_curated_silver",
                source_prior="defective_region",
                review_priority="locked_benchmark",
                review_eligible=False,
            )
        )
    normal = sorted((root / base / "non_defective/images").glob("*.tif"))
    for index, image in enumerate(normal, start=1):
        records.append(
            _sample(
                sample_id=f"dennler-normal-{index:03d}",
                dataset_id="dennler_2021_hrstem_defects",
                source_path=image.relative_to(root),
                source_partition="locked_external_test",
                split_group_id=(
                    "dennler_2021_iii_v_jeol_arm200f_archive"
                ),
                acquisition_id="dennler_2021_complete_archive",
                material="InP_GaAs_InGaAs_InAlGaAs",
                primary_task="stacking_fault",
                prediction_tasks=(
                    "amorphous_region",
                    "stacking_fault",
                    "phase_boundary",
                ),
                geometry_hint="region",
                label_kind=LabelKind.IMPLICIT_EMPTY_MASK,
                label_qualification="author_curated_negative_control",
                source_prior="normal_crystalline_control",
                review_priority="locked_benchmark",
                review_eligible=False,
            )
        )
    return records


def _guo_records(root: Path) -> list[ReviewSample]:
    base = Path("datasets/public/guo_2021_atomic_defects")
    records = [
        _sample(
            sample_id="guo-mowte2-large-fov",
            dataset_id="guo_2021_atomic_defects",
            source_path=base
            / "2H_MoWTe2_large_field_of_view_acquisition.ndata1",
            source_partition="development_test",
            split_group_id="guo_2021_mowte2_acquisition",
            acquisition_id="guo_2021_mowte2_large_fov",
            material="2H_MoWTe2",
            primary_task="atomic_point_defect",
            prediction_tasks=("atomic_point_defect", "stacking_fault"),
            geometry_hint="point",
            label_kind=LabelKind.SOURCE_PRIOR_ONLY,
            source_prior="mixed_atomic_and_line_anomaly",
            label_qualification="source_description_only",
            review_priority="high_unlabeled",
        ),
        _sample(
            sample_id="guo-zro2-twin-boundary",
            dataset_id="guo_2021_atomic_defects",
            source_path=base / "ZrO2_monoclinic_twin_boundary.tif",
            source_partition="development_test",
            split_group_id="guo_2021_zro2_twin_acquisition",
            acquisition_id="guo_2021_zro2_twin_boundary",
            material="ZrO2",
            primary_task="phase_boundary",
            prediction_tasks=("phase_boundary",),
            geometry_hint="interface",
            label_kind=LabelKind.SOURCE_PRIOR_ONLY,
            source_prior="twin_boundary",
            label_qualification="source_description_only",
            review_priority="high_unlabeled",
        ),
        _sample(
            sample_id="guo-al-partial-dislocations",
            dataset_id="guo_2021_atomic_defects",
            source_path=base
            / "relaxed_single_Al_partial_dislocations_verified.tif",
            source_partition="development_test",
            split_group_id=(
                "guo_2021_al_partial_dislocations_acquisition"
            ),
            acquisition_id="guo_2021_al_partial_dislocations",
            material="Al",
            primary_task="stacking_fault",
            prediction_tasks=("stacking_fault",),
            geometry_hint="line",
            label_kind=LabelKind.SOURCE_PRIOR_ONLY,
            source_prior="partial_dislocation_or_stacking_fault",
            label_qualification="source_description_only",
            review_priority="high_unlabeled",
        ),
    ]
    stack_path = base / "Recording_of_SuperScan_img_stack.ndata1"
    frame_count = int(load_microscopy_stack(root / stack_path).shape[0])
    for frame_index in range(frame_count):
        records.append(
            _sample(
                sample_id=f"guo-superscan-frame-{frame_index:03d}",
                dataset_id="guo_2021_superscan",
                source_path=stack_path,
                source_partition="development_test",
                split_group_id="guo-superscan-single-acquisition",
                acquisition_id="guo-superscan-img-stack",
                material="unresolved_atomic_crystal",
                primary_task="atomic_point_defect",
                prediction_tasks=("atomic_point_defect", "stacking_fault"),
                geometry_hint="point",
                frame_index=frame_index,
                label_kind=LabelKind.NONE,
                label_qualification="none",
                source_prior="unlabeled_temporal_sequence",
                review_priority="high_unlabeled",
            )
        )
    return records


def _sigma3_records(root: Path) -> list[ReviewSample]:
    base = Path("datasets/public/al_sigma3_twin_boundaries")
    records = []
    for filename, group in (
        ("Sigma3_ORI.emd", "sigma3-orientation-i"),
        ("Sigma3_ORII.emd", "sigma3-orientation-ii"),
    ):
        relative = base / filename
        for channel_index, channel in enumerate(
            list_emd_datasets(root / relative)
        ):
            dataset_path = str(channel["dataset_path"])
            frame_count = int(channel["shape"][2])
            detector = str(channel.get("detector_name", "unknown"))
            for frame_index in range(frame_count):
                records.append(
                    _sample(
                        sample_id=(
                            f"{group}-channel-{channel_index:02d}-"
                            f"frame-{frame_index:03d}"
                        ),
                        dataset_id="al_sigma3_twin_boundaries",
                        source_path=relative,
                        source_partition="development_test",
                        split_group_id=group,
                        acquisition_id=group,
                        material="Al",
                        primary_task="phase_boundary",
                        prediction_tasks=("phase_boundary",),
                        geometry_hint="interface",
                        frame_index=frame_index,
                        dataset_path=dataset_path,
                        channel_name=detector,
                        label_kind=LabelKind.SOURCE_PRIOR_ONLY,
                        source_prior="sigma3_twin_boundary_present",
                        label_qualification="source_description_only",
                        review_priority="high_boundary_trace",
                    )
                )
    return records


def _local_reference_record() -> ReviewSample:
    return _sample(
        sample_id="local-zeolite-reference",
        dataset_id="local_zeolite_reference",
        source_path=Path("reference_scale_radial_wiener.png"),
        source_partition="development_test",
        split_group_id="local_zeolite_reference_stretch_series",
        acquisition_id="reference_scale_radial_wiener",
        material="zeolite",
        primary_task="amorphous_region",
        prediction_tasks=("amorphous_region",),
        geometry_hint="region",
        label_kind=LabelKind.SOURCE_PRIOR_ONLY,
        source_prior="local_amorphous_region",
        label_qualification="source_description_only",
        review_priority="high_unlabeled",
    )


def validate_review_samples(
    samples: Iterable[ReviewSample],
    *,
    project_root: str | Path | None = None,
) -> list[ReviewSample]:
    materialized = list(samples)
    sample_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    root = None if project_root is None else Path(project_root).resolve()
    for sample in materialized:
        sample.validate()
        if sample.sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
        sample_ids.add(sample.sample_id)
        group_splits[sample.split_group_id].add(sample.review_split)
        if root is not None:
            source = (root / sample.source_path).resolve()
            source.relative_to(root)
            if not source.is_file():
                raise FileNotFoundError(source)
            if sample.label_path is not None:
                label = (root / sample.label_path).resolve()
                label.relative_to(root)
                if not label.is_file():
                    raise FileNotFoundError(label)
    leaking = {
        group: sorted(splits)
        for group, splits in group_splits.items()
        if len(splits) != 1
    }
    if leaking:
        raise ValueError(f"Split-group leakage detected: {leaking}")
    return materialized


def build_review_2d_samples(project_root: str | Path) -> list[ReviewSample]:
    root = Path(project_root).resolve()
    records = [
        *_emseek_wse2_records(root),
        *_emseek_remaining_records(root),
        *_dennler_records(root),
        *_guo_records(root),
        *_sigma3_records(root),
        _local_reference_record(),
    ]
    return validate_review_samples(records, project_root=root)


def summarize_review_samples(
    samples: Iterable[ReviewSample],
) -> dict[str, Any]:
    materialized = list(samples)
    summary: dict[str, Any] = {
        "manifest_format": REVIEW_MANIFEST_FORMAT,
        "split_policy": REVIEW_SPLIT_POLICY,
        "sample_count": len(materialized),
        "dataset_counts": dict(
            sorted(Counter(item.dataset_id for item in materialized).items())
        ),
        "split_counts": dict(
            sorted(Counter(item.review_split for item in materialized).items())
        ),
        "label_kind_counts": dict(
            sorted(Counter(item.label_kind for item in materialized).items())
        ),
        "priority_counts": dict(
            sorted(
                Counter(item.review_priority for item in materialized).items()
            )
        ),
        "task_counts": dict(
            sorted(Counter(item.primary_task for item in materialized).items())
        ),
        "review_eligible_count": sum(
            item.review_eligible for item in materialized
        ),
        "pixel_mask_count": sum(
            item.label_kind == LabelKind.PIXEL_MASK.value
            for item in materialized
        ),
        "implicit_empty_mask_count": sum(
            item.label_kind == LabelKind.IMPLICIT_EMPTY_MASK.value
            for item in materialized
        ),
        "source_prior_only_count": sum(
            item.label_kind == LabelKind.SOURCE_PRIOR_ONLY.value
            for item in materialized
        ),
        "unlabeled_count": sum(
            item.label_kind
            in {LabelKind.NONE.value, LabelKind.SOURCE_PRIOR_ONLY.value}
            for item in materialized
        ),
    }
    summary["records_sha256"] = records_sha256(materialized)
    return summary


def records_sha256(samples: Iterable[ReviewSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            json.dumps(
                sample.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def write_review_manifest(
    samples: Iterable[ReviewSample],
    manifest_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    materialized = validate_review_samples(samples)
    manifest = Path(manifest_path)
    summary_target = Path(summary_path)
    if manifest.exists() or summary_target.exists():
        raise FileExistsError("Refusing to overwrite an existing review manifest")
    with manifest.open("x", encoding="utf-8") as stream:
        for sample in materialized:
            stream.write(
                json.dumps(sample.to_dict(), ensure_ascii=False) + "\n"
            )
    summary = summarize_review_samples(materialized)
    with summary_target.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def load_review_manifest(path: str | Path) -> list[ReviewSample]:
    records = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("manifest_format") != REVIEW_MANIFEST_FORMAT:
            raise ValueError(
                f"Line {line_number} has unsupported manifest format"
            )
        records.append(ReviewSample.from_dict(payload))
    return validate_review_samples(records)


def requires_pixel_annotation(sample: ReviewSample) -> bool:
    """Return whether a sample belongs in the human annotation queue."""

    return (
        sample.review_eligible
        and sample.label_kind
        in {LabelKind.NONE.value, LabelKind.SOURCE_PRIOR_ONLY.value}
    )


def split_annotation_and_reference_pools(
    samples: Iterable[ReviewSample],
) -> tuple[list[ReviewSample], list[ReviewSample]]:
    """Separate unlabeled annotation work from already labeled evaluation."""

    annotation = []
    reference = []
    for sample in samples:
        if requires_pixel_annotation(sample):
            annotation.append(sample)
        else:
            reference.append(sample)
    return annotation, reference


def load_review_plane(
    sample: ReviewSample,
    project_root: str | Path,
) -> np.ndarray:
    root = Path(project_root).resolve()
    source = (root / sample.source_path).resolve()
    source.relative_to(root)
    return load_grayscale(
        source,
        frame_index=sample.frame_index,
        dataset_path=sample.dataset_path,
    )


def load_review_label(
    sample: ReviewSample,
    project_root: str | Path,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    kind = LabelKind(sample.label_kind)
    if kind in {LabelKind.NONE, LabelKind.SOURCE_PRIOR_ONLY}:
        return None
    if kind == LabelKind.IMPLICIT_EMPTY_MASK:
        return np.zeros(image_shape, dtype=np.uint8)
    root = Path(project_root).resolve()
    assert sample.label_path is not None
    label_path = (root / sample.label_path).resolve()
    label_path.relative_to(root)
    label = np.asarray(imread(str(label_path)))
    if label.shape[:2] != image_shape:
        raise ValueError(
            f"Label shape {label.shape[:2]} does not match image {image_shape}"
        )
    return label
