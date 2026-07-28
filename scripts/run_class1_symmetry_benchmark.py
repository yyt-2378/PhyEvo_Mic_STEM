#!/usr/bin/env python
"""Predict and score the final group-aware Class1 Dennler diagnostic.

Dennler labels were previously unblinded.  This runner keeps prediction and
label scoring in separate sealed stages, but all results remain post-hoc and
are not eligible as independent external validation.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from scipy.ndimage import binary_dilation
import tifffile

from tfree_detection.config import DetectorConfig
from tfree_detection.evaluation import boundary_metrics, region_metrics
from tfree_detection.framework import TrainingFreeDetectionFramework


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    PROJECT_ROOT
    / "datasets/public/dennler_2021_hrstem_defects/"
    "hrstem_defects_dataset.zip"
)
HISTORICAL_ROOT = (
    PROJECT_ROOT / "benchmarks/locked_runs/dennler_v1_predictions"
)
HISTORICAL_MANIFEST = HISTORICAL_ROOT / "prediction_manifest.json"
IMPLEMENTATION_FILES = (
    PROJECT_ROOT / "src/tfree_detection/symmetry/group.py",
    PROJECT_ROOT / "src/tfree_detection/symmetry/references.py",
    PROJECT_ROOT / "src/tfree_detection/symmetry/task.py",
    PROJECT_ROOT / "src/tfree_detection/config.py",
)
ARCHIVE_SHA256 = (
    "abef46a0183ba08b925c64807d5e914cd"
    "97967444d9fd3e0799924896f0614ec"
)


def _sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _inside_project(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def _read_image(values: bytes) -> np.ndarray:
    image = np.asarray(tifffile.imread(BytesIO(values)))
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        image = np.mean(image[..., :3].astype(float), axis=-1)
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise ValueError(f"Invalid STEM image shape/content: {image.shape}")
    return image


def _historical_mask(image_id: str, shape: tuple[int, int]) -> np.ndarray:
    masks = []
    for detector in ("stacking_fault", "phase_boundary"):
        path = (
            HISTORICAL_ROOT
            / "predictions"
            / image_id
            / f"{detector}.npz"
        )
        with np.load(path, allow_pickle=False) as values:
            mask = np.asarray(values["mask"], dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"Historical mask shape mismatch: {path}")
        masks.append(mask)
    return masks[0] | masks[1]


def _implementation_sha256() -> str:
    return _json_hash(
        {
            str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
            for path in IMPLEMENTATION_FILES
        }
    )


def predict(output: str | Path) -> None:
    destination = _inside_project(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite output: {destination}")
    if _sha256_file(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("Dennler archive checksum changed")
    destination.mkdir(parents=True, exist_ok=False)
    arrays = destination / "arrays"
    arrays.mkdir()

    historical = json.loads(
        HISTORICAL_MANIFEST.read_text(encoding="utf-8")
    )
    config = DetectorConfig(task="symmetry_relation")
    config.evidence_protocol.enabled = False
    config.resolver.enabled = False
    framework = TrainingFreeDetectionFramework(config)
    records: list[dict[str, Any]] = []

    with ZipFile(ARCHIVE) as archive:
        for index, source in enumerate(historical["images"], start=1):
            image_id = str(source["image_id"])
            relative_path = str(source["relative_path"])
            if "/labels/" in relative_path:
                raise RuntimeError("Prediction stage attempted label access")
            raw = archive.read(relative_path)
            if _sha256_bytes(raw) != source["source_sha256"]:
                raise ValueError(
                    f"Source checksum mismatch: {source['filename']}"
                )
            record: dict[str, Any] = {
                "image_id": image_id,
                "category": source["category"],
                "filename": source["filename"],
                "source_relative_path": relative_path,
                "source_sha256": source["source_sha256"],
                "status": "failed",
            }
            try:
                image = _read_image(raw)
                result = framework.detect(image)
                feature = result.feature_maps
                domain = feature["symmetry_domain_region"] >= 0.5
                interface = feature["interface_line"] >= 0.5
                phase_slip = feature["phase_slip_line"] >= 0.5
                core = feature["dislocation_core_point"] >= 0.5
                multi_geometry = domain | interface | phase_slip | core
                historical_mask = _historical_mask(
                    image_id, tuple(image.shape)
                )
                # A label-free conservative expansion: new evidence can only
                # expand within one lattice-scale neighborhood of an old seed.
                radius = max(
                    1,
                    int(round(result.metadata["lattice_spacing_pixels"])),
                )
                guided = historical_mask | (
                    multi_geometry
                    & binary_dilation(
                        historical_mask,
                        iterations=radius,
                    )
                )
                path = arrays / f"{image_id}.npz"
                np.savez_compressed(
                    path,
                    symmetry_domain_region=domain.astype(np.uint8),
                    interface_line=interface.astype(np.uint8),
                    phase_slip_line=phase_slip.astype(np.uint8),
                    dislocation_core_point=core.astype(np.uint8),
                    multigeometry_class1=multi_geometry.astype(np.uint8),
                    historical_class1=historical_mask.astype(np.uint8),
                    conservative_union=(
                        historical_mask | multi_geometry
                    ).astype(np.uint8),
                    guided_expansion=guided.astype(np.uint8),
                    reliability=result.evidence_maps[
                        "symmetry_relation_validity"
                    ].astype(np.float32),
                    motif_score=result.evidence_maps[
                        "group_phase_motif_residual"
                    ].astype(np.float32),
                    displacement_score=result.evidence_maps[
                        "group_displacement_discontinuity"
                    ].astype(np.float32),
                    artifact_score=result.evidence_maps[
                        "group_scan_time_artifact"
                    ].astype(np.float32),
                )
                record.update(
                    {
                        "status": "success",
                        "prediction_path": str(
                            path.relative_to(destination)
                        ),
                        "prediction_sha256": _sha256_file(path),
                        "image_shape": list(image.shape),
                        "lattice_spacing_pixels": result.metadata[
                            "lattice_spacing_pixels"
                        ],
                        "reference_selection": result.metadata[
                            "reference_selection"
                        ],
                        "geometry_counts": result.metadata[
                            "geometry_counts"
                        ],
                        "positive_fractions": {
                            "domain": float(domain.mean()),
                            "interface": float(interface.mean()),
                            "phase_slip": float(phase_slip.mean()),
                            "core": float(core.mean()),
                            "multigeometry": float(
                                multi_geometry.mean()
                            ),
                        },
                    }
                )
            except Exception as error:
                record["error_type"] = type(error).__name__
                record["error_message"] = str(error)
            records.append(record)
            print(
                f"[{index:02d}/{len(historical['images'])}] "
                f"{record['status']}: {record['filename']}",
                flush=True,
            )

    manifest = {
        "prediction_format": "class1-symmetry-posthoc-v1",
        "protocol_id": "dennler-class1-symmetry-final-posthoc-v1",
        "scientific_status": "posthoc_diagnostic_not_locked",
        "labels_read_during_prediction": False,
        "training_used": False,
        "historical_locked_prediction_replaced": False,
        "archive_sha256": _sha256_file(ARCHIVE),
        "implementation_sha256": _implementation_sha256(),
        "config": config.to_dict(),
        "config_sha256": _json_hash(config.to_dict()),
        "counts": {
            "images": len(records),
            "successful": sum(
                row["status"] == "success" for row in records
            ),
            "failed": sum(row["status"] != "success" for row in records),
        },
        "images": records,
    }
    manifest_path = destination / "prediction_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        destination / "prediction_seal.json",
        {
            "prediction_manifest_sha256": _sha256_file(manifest_path),
            "prediction_content_sha256": _json_hash(manifest),
            "labels_read_during_prediction": False,
        },
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


def _decode_labels(
    archive: ZipFile,
    records: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    raw_labels: dict[str, np.ndarray] = {}
    unique_values: set[float] = set()
    for record in records:
        if record["category"] != "defective":
            continue
        label_path = record["source_relative_path"].replace(
            "/images/", "/labels/"
        )
        label_image = np.asarray(
            tifffile.imread(BytesIO(archive.read(label_path)))
        )
        if label_image.ndim == 3:
            label_image = label_image[..., 0]
        unique_values.update(
            float(value) for value in np.unique(label_image)
        )
        raw_labels[record["filename"]] = label_image
    ordered = sorted(unique_values)
    if ordered != [0.0, 127.0, 255.0]:
        raise ValueError(f"Unexpected Dennler labels: {ordered}")
    return {
        name: np.searchsorted(ordered, np.asarray(values, dtype=float))
        for name, values in raw_labels.items()
    }


def _micro(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": f1,
    }


def score(predictions: str | Path, output: str | Path) -> None:
    prediction_root = _inside_project(predictions)
    destination = _inside_project(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite output: {destination}")
    manifest_path = prediction_root / "prediction_manifest.json"
    seal_path = prediction_root / "prediction_seal.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if _sha256_file(manifest_path) != seal[
        "prediction_manifest_sha256"
    ]:
        raise ValueError("Prediction manifest seal mismatch")
    if manifest["counts"]["failed"]:
        raise RuntimeError("Cannot score a prediction set with failures")
    targets = {
        "historical_class1": "historical_class1",
        "symmetry_domain_region": "symmetry_domain_region",
        "interface_line": "interface_line",
        "phase_slip_line": "phase_slip_line",
        "dislocation_core_point": "dislocation_core_point",
        "multigeometry_class1": "multigeometry_class1",
        "guided_expansion": "guided_expansion",
        "conservative_union": "conservative_union",
    }
    accumulators = {
        name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "per_image": []}
        for name in targets
    }
    with ZipFile(ARCHIVE) as archive:
        labels = _decode_labels(archive, manifest["images"])
        for record in manifest["images"]:
            path = prediction_root / record["prediction_path"]
            if _sha256_file(path) != record["prediction_sha256"]:
                raise ValueError(
                    f"Prediction checksum mismatch: {record['filename']}"
                )
            if record["category"] == "defective":
                truth = labels[record["filename"]] == 1
            else:
                truth = np.zeros(
                    tuple(record["image_shape"]), dtype=bool
                )
            with np.load(path, allow_pickle=False) as values:
                for target_name, key in targets.items():
                    predicted = np.asarray(values[key], dtype=bool)
                    tp = int(np.sum(predicted & truth))
                    fp = int(np.sum(predicted & ~truth))
                    fn = int(np.sum(~predicted & truth))
                    tn = int(np.sum(~predicted & ~truth))
                    accumulator = accumulators[target_name]
                    accumulator["tp"] += tp
                    accumulator["fp"] += fp
                    accumulator["fn"] += fn
                    accumulator["tn"] += tn
                    metrics = region_metrics(predicted, truth)
                    boundary = boundary_metrics(
                        predicted, truth, tolerance_pixels=10.0
                    )
                    accumulator["per_image"].append(
                        {
                            "image_id": record["image_id"],
                            "filename": record["filename"],
                            "category": record["category"],
                            "truth_positive_fraction": float(
                                truth.mean()
                            ),
                            "predicted_positive_fraction": float(
                                predicted.mean()
                            ),
                            **metrics,
                            **boundary,
                        }
                    )
    scored = {}
    for name, values in accumulators.items():
        positive = [
            row
            for row in values["per_image"]
            if row["truth_positive_fraction"] > 0
        ]
        normal = [
            row
            for row in values["per_image"]
            if row["category"] == "normal"
        ]
        scored[name] = {
            "micro_pixel_metrics": _micro(
                values["tp"],
                values["fp"],
                values["fn"],
                values["tn"],
            ),
            "truth_positive_images": len(positive),
            "positive_image_mean_iou": float(
                np.mean([row["iou"] for row in positive])
            ),
            "normal_control_mean_false_positive_area_fraction": float(
                np.mean(
                    [
                        row["predicted_positive_fraction"]
                        for row in normal
                    ]
                )
            ),
            "per_image": values["per_image"],
        }
    historical = scored["historical_class1"]["micro_pixel_metrics"]
    deltas = {
        name: {
            metric: float(
                target["micro_pixel_metrics"][metric]
                - historical[metric]
            )
            for metric in ("iou", "precision", "recall", "f1")
        }
        for name, target in scored.items()
        if name != "historical_class1"
    }
    payload = {
        "score_format": "class1-symmetry-posthoc-score-v1",
        "protocol_id": manifest["protocol_id"],
        "scientific_status": "posthoc_diagnostic_not_locked",
        "labels_accessed_for_scoring": True,
        "eligible_as_new_external_validation": False,
        "targets": scored,
        "descriptive_delta_vs_historical": deltas,
        "claim_boundary": (
            "Dennler was previously unblinded. Metrics are diagnostic only "
            "and cannot establish independent zero-shot generalization."
        ),
    }
    destination.mkdir(parents=True, exist_ok=False)
    metrics_path = destination / "class1_metrics.json"
    _write_json(metrics_path, payload)
    _write_json(
        destination / "score_seal.json",
        {
            "score_file_sha256": _sha256_file(metrics_path),
            "score_content_sha256": _json_hash(payload),
        },
    )
    print(
        json.dumps(
            {
                name: values["micro_pixel_metrics"]
                for name, values in scored.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--output", required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "predict":
        predict(args.output)
    else:
        score(args.predictions, args.output)


if __name__ == "__main__":
    main()
