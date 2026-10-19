"""Versioned, positive/unknown expert supervision; original ledgers stay immutable.

An explicit development-data retirement is required for each learning cycle.
This does not change the source registry or promote single-assisted labels to Gold.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.draw import polygon

from .review_dataset import ReviewSample, load_review_plane
from .review_export import rasterize_review_geometry
from .review_prediction import ReviewPrediction, load_prediction_mask, sample_sha256


FORMAT = "phyevo-expert-learning-dataset-v1"
GEOMETRY_INDEX = {"point": 1, "line": 2, "region": 3, "interface": 4}


def contained_path(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path: Path, values: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(values, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 4 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2-D microscopy plane")
    low, high = np.percentile(image, [0.5, 99.5])
    return np.clip((image - low) / max(float(high - low), 1e-8), 0, 1).astype(np.float32)


def closed_interface(points: np.ndarray, shape: tuple[int, int]) -> bool:
    """Detect an outlined ROI, not an exact interface centerline."""
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        return False
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    gap = float(np.linalg.norm(points[-1] - points[0]))
    area = abs(float(np.dot(points[:, 0], np.roll(points[:, 1], 1))
                     - np.dot(points[:, 1], np.roll(points[:, 0], 1)))) / 2
    return length > 0 and gap <= min(0.1 * length, 0.08 * max(shape)) and area > 9


@dataclass
class ExpertSupervision:
    positive: np.ndarray
    known: np.ndarray
    teacher_weight: np.ndarray
    semantic_kind: str
    geometry_index: int


def supervision_from_record(record: Mapping[str, Any], teacher: np.ndarray) -> ExpertSupervision:
    revision = record["revision"]
    shape = tuple(revision["image_shape"])
    teacher = np.asarray(teacher, dtype=bool)
    if teacher.shape != shape:
        raise ValueError("Teacher/expert shape mismatch")
    positive = np.zeros(shape, dtype=bool)
    known = np.zeros(shape, dtype=np.float32)
    # An empty/abstained teacher is NOT an all-background label.
    teacher_weight = np.full(shape, float(teacher.any()), dtype=np.float32)
    if record["prediction"].get("status") == "abstained":
        teacher_weight[:] = 0
    hint = revision["geometry_hint"]
    action = revision["action"]
    semantic = "unresolved"
    if action == "correct_geometry":
        geometry = revision["geometry"]
        points = np.asarray(geometry["points_xy"], dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                or (points < 0).any() or (points[:, 0] > shape[1] - 1).any()
                or (points[:, 1] > shape[0] - 1).any()):
            raise ValueError("Expert geometry outside source pixel bounds")
        if hint == "interface" and closed_interface(points, shape):
            rr, cc = polygon(points[:, 1], points[:, 0], shape=shape)
            positive[rr, cc] = True
            semantic = "interface_band_roi_not_centerline"
        else:
            positive = rasterize_review_geometry(geometry, geometry_hint=hint, image_shape=shape)
            semantic = {"point": "sparse_positive_points", "region": "positive_region",
                        "line": "positive_trace", "interface": "positive_interface_trace"}[hint]
        known[positive] = 1.0
        # Human positives outrank the teacher, including near invalid borders.
        # The nearby unknown halo must not pull a corrected positive back to zero.
        teacher_weight[binary_dilation(positive, iterations=4)] = 0
    elif action == "accept_prediction":
        positive = teacher.copy()
        known[positive] = 0.5  # Assisted accepted positives, not exhaustive negatives.
        semantic = "accepted_teacher_positive"
    elif action == "no_defect":
        known[:] = 1
        teacher_weight[:] = 0
        semantic = "explicit_no_defect"
    elif action in {"reject_prediction", "artifact", "unknown"}:
        teacher_weight[:] = 0
    else:
        raise ValueError(f"Unsupported expert action: {action}")
    return ExpertSupervision(positive, known, teacher_weight, semantic, GEOMETRY_INDEX[hint])


def verify_frozen_round(root: Path, bundle: Path) -> list[dict]:
    bundle = contained_path(root, bundle)
    seal = json.loads((bundle / "seal.json").read_text())
    if seal.get("format") != "phyevo-expert-round-seal-v1":
        raise ValueError("Unsupported expert freeze seal")
    required = {"cases.jsonl", "summary.json", "review_store_snapshot.sqlite3"}
    if set(seal["files"]) != required:
        raise ValueError("Incomplete frozen round")
    for name, digest in seal["files"].items():
        if sha256_file(contained_path(bundle, name)) != digest:
            raise ValueError(f"Frozen round checksum mismatch: {name}")
    records = [json.loads(line) for line in (bundle / "cases.jsonl").read_text().splitlines() if line]
    ids = [record["case_id"] for record in records]
    if not records or len(set(ids)) != len(ids):
        raise ValueError("Frozen cases must be nonempty and unique")
    return records


def validate_admission(records: list[dict], *, retire_development: bool) -> None:
    for record in records:
        partition = record["sample"]["source_partition"]
        if partition == "development_test" and not retire_development:
            raise ValueError("Explicit development-data retirement is required")
        if partition not in {"train", "development_test"}:
            raise ValueError(f"Cannot repurpose protected source partition {partition}")


def group_folds(entries: list[dict]) -> list[dict]:
    """Leave one acquisition group out; never split frames/channels/patches."""
    eligible = [entry for entry in entries if entry["human_known_pixels"] > 0]
    groups_by_task: dict[str, set[str]] = {}
    for entry in eligible:
        groups_by_task.setdefault(entry["geometry"], set()).add(entry["group"])
    folds = []
    for group in sorted({entry["group"] for entry in eligible}):
        validation = [entry for entry in eligible if entry["group"] == group]
        # A single-source task can be fitted, but cannot demonstrate transfer.
        if not all(len(groups_by_task[entry["geometry"]]) >= 2 for entry in validation):
            continue
        training = [entry for entry in eligible if entry["group"] != group]
        folds.append({"fold_id": f"group-{len(folds):02d}", "held_out_group": group,
                      "train_case_numbers": [e["case_number"] for e in training],
                      "validation_case_numbers": [e["case_number"] for e in validation],
                      "train_groups": sorted({e["group"] for e in training})})
    return folds


def build_learning_dataset(root: Path, bundle: Path, output: Path, *,
                           retire_development: bool = False) -> dict:
    root = root.resolve()
    bundle, output = contained_path(root, bundle), contained_path(root, output)
    records = verify_frozen_round(root, bundle)
    validate_admission(records, retire_development=retire_development)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a learning dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    (output / "planes").mkdir()
    entries, source_hashes = [], {}
    for record in records:
        sample = ReviewSample.from_dict(record["sample"])
        source = contained_path(root, sample.source_path)
        if sample.source_path not in source_hashes:
            source_hashes[sample.source_path] = sha256_file(source)
        if source_hashes[sample.source_path] != record["source_file_sha256"]:
            raise ValueError("Source changed since expert annotation")
        prediction = ReviewPrediction.from_dict(record["prediction"])
        if prediction.sample_sha256 != sample_sha256(sample):
            raise ValueError("Prediction/sample provenance mismatch")
        image = load_review_plane(sample, root)
        if image.shape != tuple(record["revision"]["image_shape"]):
            raise ValueError("Image/annotation shape mismatch")
        contiguous = np.ascontiguousarray(image)
        pixel_digest = hashlib.sha256()
        pixel_digest.update(str(contiguous.dtype).encode("utf-8"))
        pixel_digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        pixel_digest.update(contiguous.view(np.uint8).tobytes())
        if pixel_digest.hexdigest() != prediction.pixel_sha256:
            raise ValueError("Loaded plane differs from the reviewed image pixels")
        teacher = load_prediction_mask(prediction, root)
        supervision = supervision_from_record(record, teacher)
        number = record["case_number"]
        filename = f"planes/case-{number:03d}.npz"
        with (output / filename).open("xb") as stream:
            np.savez_compressed(stream, image=normalize_image(image), teacher=teacher,
                                human=supervision.positive, human_weight=supervision.known,
                                teacher_weight=supervision.teacher_weight)
        points = (record["revision"].get("geometry") or {}).get("points_xy", [])
        entries.append({
            "case_number": number, "case_id": record["case_id"], "plane": filename,
            "plane_sha256": sha256_file(output / filename), "group": sample.split_group_id,
            "geometry": sample.geometry_hint, "geometry_index": supervision.geometry_index,
            "semantic_kind": supervision.semantic_kind, "shape": list(image.shape),
            "human_known_pixels": int((supervision.known > 0).sum()),
            "human_positive_pixels": int(supervision.positive.sum()),
            "teacher_positive_pixels": int(teacher.sum()), "points_xy": points,
            "source_path": sample.source_path, "source_file_sha256": record["source_file_sha256"],
            "original_source_partition": sample.source_partition,
            "original_review_split": sample.review_split,
            "revision_id": record["revision"]["revision_id"],
            "revision_sha256": record["revision"]["payload_sha256"],
            "action": record["revision"]["action"], "assisted": record["revision"]["model_assistance_shown"],
        })
        if number % 25 == 0:
            print(f"prepared {number}/{len(records)} expert cases", flush=True)
    folds = group_folds(entries)
    manifest = {
        "format": FORMAT, "frozen_bundle": str(bundle.relative_to(root)),
        "frozen_seal_sha256": sha256_file(bundle / "seal.json"), "entries": entries, "folds": folds,
        "admission": {
            "purpose": "user_authorized_expert_learning_cycle",
            "development_retirement_authorized": retire_development,
            "source_registry_modified": False,
            "retired_from_student_independent_test_groups": sorted({e["group"] for e in entries}),
            "original_teacher_historical_results_preserved": True,
            "evaluation_kind": "assisted_silver_group_cross_validation_not_locked_external_test",
            "gold_calibration_eligible": False, "production_release_eligible": False,
        },
        "supervision_policy": {
            "unmarked_pixels": "unknown_not_expert_negative",
            "teacher_background": "weak_pseudo_prior_only_when_teacher_nonempty",
            "teacher_empty": "unknown_not_no_defect",
            "closed_interface": "weak_positive_band_not_exact_centerline",
            "defect_subtypes": "case_level_only_not_per_point_truth",
            "channels": "selected_plane_only_no_automatic_label_propagation",
        },
        "unvalidated_geometries": sorted({e["geometry"] for e in entries} -
            {e["geometry"] for e in entries if any(e["case_number"] in f["validation_case_numbers"] for f in folds)}),
    }
    write_new_json(output / "manifest.json", manifest)
    write_new_json(output / "seal.json", {"manifest_sha256": sha256_file(output / "manifest.json")})
    return manifest


def load_learning_dataset(root: Path, directory: Path) -> tuple[dict, dict[int, dict[str, np.ndarray]]]:
    directory = contained_path(root, directory)
    seal = json.loads((directory / "seal.json").read_text())
    if sha256_file(directory / "manifest.json") != seal["manifest_sha256"]:
        raise ValueError("Learning manifest checksum mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported expert learning format")
    arrays = {}
    for entry in manifest["entries"]:
        path = contained_path(directory, entry["plane"])
        if sha256_file(path) != entry["plane_sha256"]:
            raise ValueError("Learning plane checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            arrays[entry["case_number"]] = {key: archive[key] for key in archive.files}
    return manifest, arrays
