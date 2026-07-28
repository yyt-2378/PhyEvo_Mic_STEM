#!/usr/bin/env python
"""Predict and score the explicitly post-hoc Dennler v0.4 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import tifffile

from tfree_detection.diagnostic_v4 import (
    PosthocV4Config,
    analyze_posthoc_v4,
    compose_posthoc_v4,
)
from tfree_detection.evaluation import boundary_metrics, region_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    PROJECT_ROOT
    / "datasets/public/dennler_2021_hrstem_defects/"
    "hrstem_defects_dataset.zip"
)
PROTOCOL = (
    PROJECT_ROOT
    / "benchmarks/dennler_posthoc_v4_diagnostic_protocol.yaml"
)
HISTORICAL_ROOT = (
    PROJECT_ROOT / "benchmarks/locked_runs/dennler_v1_predictions"
)
HISTORICAL_MANIFEST = HISTORICAL_ROOT / "prediction_manifest.json"
HISTORICAL_METRICS = (
    PROJECT_ROOT
    / "benchmarks/locked_runs/dennler_v1_score/locked_metrics.json"
)
MODULE_PATH = (
    PROJECT_ROOT / "src/tfree_detection/diagnostic_v4.py"
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
    values = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(values)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inside_project(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("Output must stay inside the project") from error
    return resolved


def _read_image(stream: bytes) -> np.ndarray:
    image = np.asarray(tifffile.imread(BytesIO(stream)))
    if image.ndim == 3:
        if image.shape[-1] not in {3, 4}:
            raise ValueError(f"Unsupported RGB image shape: {image.shape}")
        image = np.mean(image[..., :3].astype(np.float64), axis=-1)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D STEM image, got {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("STEM image contains non-finite values")
    return image


def _historical_masks(
    image_id: str,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    masks = []
    for detector in ("stacking_fault", "phase_boundary"):
        path = (
            HISTORICAL_ROOT
            / "predictions"
            / image_id
            / f"{detector}.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            mask = np.asarray(archive["mask"], dtype=bool)
        if mask.shape != shape:
            raise ValueError(
                f"Historical {detector} shape {mask.shape} != {shape}"
            )
        masks.append(mask)
    return masks[0], masks[1]


def predict(output: str | Path) -> None:
    destination = _inside_project(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite output: {destination}")
    if _sha256_file(ARCHIVE) != (
        "abef46a0183ba08b925c64807d5e914cd97967444d9fd3e0799924896f0614ec"
    ):
        raise ValueError("Dennler archive checksum changed")
    destination.mkdir(parents=True, exist_ok=False)
    prediction_directory = destination / "arrays"
    prediction_directory.mkdir()

    historical = _read_json(HISTORICAL_MANIFEST)
    config = PosthocV4Config()
    config.validate()
    records: list[dict[str, Any]] = []
    successful = 0
    failed = 0
    with ZipFile(ARCHIVE) as archive:
        for index, source in enumerate(historical["images"], start=1):
            image_id = str(source["image_id"])
            filename = str(source["filename"])
            relative_path = str(source["relative_path"])
            if "/labels/" in relative_path:
                raise RuntimeError("Prediction stage attempted label access")
            raw = archive.read(relative_path)
            if _sha256_bytes(raw) != source["source_sha256"]:
                raise ValueError(f"Source checksum mismatch: {filename}")
            record: dict[str, Any] = {
                "image_id": image_id,
                "category": source["category"],
                "filename": filename,
                "source_relative_path": relative_path,
                "source_sha256": source["source_sha256"],
                "status": "failed",
                "error_type": None,
                "error_message": None,
            }
            try:
                image = _read_image(raw)
                result = analyze_posthoc_v4(image, config)
                stacking, phase = _historical_masks(
                    image_id,
                    tuple(image.shape),
                )
                historical_secondary = stacking | phase
                composite = compose_posthoc_v4(
                    result,
                    historical_secondary,
                )
                array_path = prediction_directory / f"{image_id}.npz"
                np.savez_compressed(
                    array_path,
                    coherence_loss_region=(
                        composite.coherence_loss_region.astype(np.uint8)
                    ),
                    domain_symmetry_region=(
                        composite.domain_symmetry_region.astype(np.uint8)
                    ),
                    historical_secondary_region=(
                        historical_secondary.astype(np.uint8)
                    ),
                    secondary_symmetry_region=(
                        composite.secondary_symmetry_region.astype(
                            np.uint8
                        )
                    ),
                    structural_boundary=(
                        composite.structural_boundary.astype(np.uint8)
                    ),
                    augmented_secondary_symmetry_region=(
                        composite.augmented_secondary_symmetry_region.astype(
                            np.uint8
                        )
                    ),
                    augmented_any_defect_region=(
                        composite.augmented_any_defect_region.astype(
                            np.uint8
                        )
                    ),
                    any_defect_region=(
                        composite.any_defect_region.astype(np.uint8)
                    ),
                    reliability=result.reliability_map.astype(np.float32),
                    coherence_loss_score=result.evidence_maps[
                        "coherence_loss_score"
                    ].astype(np.float32),
                    domain_membership_margin=result.evidence_maps[
                        "domain_membership_margin"
                    ].astype(np.float32),
                    deformation_gradient_z=result.evidence_maps[
                        "deformation_gradient_z"
                    ].astype(np.float32),
                )
                relative_array = array_path.relative_to(destination)
                record.update(
                    {
                        "status": "success",
                        "prediction_path": str(relative_array),
                        "prediction_sha256": _sha256_file(array_path),
                        "image_shape": list(image.shape),
                        "lattice_spacing_pixels": result.metadata[
                            "lattice_spacing_pixels"
                        ],
                        "domain_symmetry": result.metadata[
                            "domain_symmetry"
                        ],
                        "positive_fractions": {
                            "coherence_loss_region": float(
                                result.coherence_loss_mask.mean()
                            ),
                            "domain_symmetry_region": float(
                                result.secondary_symmetry_region_mask.mean()
                            ),
                            "historical_secondary_region": float(
                                historical_secondary.mean()
                            ),
                            "secondary_symmetry_region": float(
                                composite.secondary_symmetry_region.mean()
                            ),
                            "structural_boundary": float(
                                composite.structural_boundary.mean()
                            ),
                            "augmented_secondary_symmetry_region": float(
                                composite
                                .augmented_secondary_symmetry_region
                                .mean()
                            ),
                            "augmented_any_defect_region": float(
                                composite.augmented_any_defect_region.mean()
                            ),
                            "any_defect_region": float(
                                composite.any_defect_region.mean()
                            ),
                        },
                    }
                )
                successful += 1
            except Exception as error:  # diagnostic manifest retains failures
                record["error_type"] = type(error).__name__
                record["error_message"] = str(error)
                failed += 1
            records.append(record)
            print(
                f"[{index:02d}/{len(historical['images'])}] "
                f"{record['status']}: {filename}",
                flush=True,
            )

    manifest = {
        "prediction_format": "phyevo-mic-posthoc-v0.4-predictions-v1",
        "protocol_id": "dennler-2021-posthoc-v0.4-diagnostic-v1",
        "scientific_status": "posthoc_diagnostic_not_locked",
        "labels_read_during_prediction": False,
        "historical_locked_prediction_replaced": False,
        "primary_composition_policy": (
            "coherence_loss_v0.4_plus_read_only_historical_structure; "
            "experimental_domain_proposals_are_ablation_only"
        ),
        "archive_sha256": _sha256_file(ARCHIVE),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "implementation_sha256": _sha256_file(MODULE_PATH),
        "config": config.to_dict(),
        "config_sha256": _json_hash(config.to_dict()),
        "historical_prediction_set_sha256": historical[
            "prediction_set_sha256"
        ]
        if "prediction_set_sha256" in historical
        else "f7b126bb10691d81c3e08d677325e2e36dac5d0f1619436bd9e220ef773232de",
        "counts": {
            "images": len(records),
            "successful": successful,
            "failed": failed,
        },
        "images": records,
    }
    manifest_path = destination / "prediction_manifest.json"
    _write_json(manifest_path, manifest)
    seal = {
        "seal_format": "phyevo-mic-posthoc-prediction-seal-v1",
        "protocol_id": manifest["protocol_id"],
        "scientific_status": manifest["scientific_status"],
        "prediction_manifest_sha256": _sha256_file(manifest_path),
        "prediction_content_sha256": _json_hash(manifest),
        "labels_read_during_prediction": False,
        "historical_locked_prediction_replaced": False,
    }
    _write_json(destination / "prediction_seal.json", seal)
    print(
        json.dumps(
            {
                "output": str(destination.relative_to(PROJECT_ROOT)),
                "counts": manifest["counts"],
                "seal": seal,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _decode_labels(
    archive: ZipFile,
    image_records: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    raw: dict[str, np.ndarray] = {}
    values: set[float] = set()
    for record in image_records:
        if record["category"] != "defective":
            continue
        label_path = str(record["source_relative_path"]).replace(
            "/images/",
            "/labels/",
        )
        label = np.asarray(
            tifffile.imread(BytesIO(archive.read(label_path)))
        )
        if label.ndim == 3:
            label = label[..., 0]
        values.update(float(value) for value in np.unique(label))
        raw[record["filename"]] = label
    ordered = sorted(values)
    if ordered != [0.0, 127.0, 255.0]:
        raise ValueError(f"Unexpected Dennler label values: {ordered}")
    decoded = {}
    for name, label in raw.items():
        class_map = np.full(label.shape, 255, dtype=np.uint8)
        for class_id, value in enumerate(ordered):
            class_map[np.asarray(label, dtype=float) == value] = class_id
        if np.any(class_map == 255):
            raise ValueError(f"Could not decode label: {name}")
        decoded[name] = class_map
    return decoded


def _micro_metrics(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> dict[str, float | int]:
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    iou = true_positive / max(
        true_positive + false_positive + false_negative,
        1,
    )
    dice = 2 * true_positive / max(
        2 * true_positive + false_positive + false_negative,
        1,
    )
    return {
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "true_negative_pixels": true_negative,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def score(predictions: str | Path, output: str | Path) -> None:
    prediction_root = _inside_project(predictions)
    destination = _inside_project(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite output: {destination}")
    manifest_path = prediction_root / "prediction_manifest.json"
    seal_path = prediction_root / "prediction_seal.json"
    manifest = _read_json(manifest_path)
    seal = _read_json(seal_path)
    if _sha256_file(manifest_path) != seal[
        "prediction_manifest_sha256"
    ]:
        raise ValueError("Post-hoc prediction manifest seal mismatch")
    if manifest["counts"]["failed"]:
        raise RuntimeError("Cannot score a prediction set with failures")
    for record in manifest["images"]:
        array_path = prediction_root / record["prediction_path"]
        if _sha256_file(array_path) != record["prediction_sha256"]:
            raise ValueError(
                f"Prediction checksum mismatch: {record['filename']}"
            )

    targets = {
        "coherence_loss_region": {
            "truth_classes": (2,),
            "prediction_key": "coherence_loss_region",
            "geometry": "region",
        },
        "secondary_symmetry_region": {
            "truth_classes": (1,),
            "prediction_key": "secondary_symmetry_region",
            "geometry": "region_and_boundary",
        },
        "structural_boundary": {
            "truth_classes": (1,),
            "prediction_key": "structural_boundary",
            "geometry": "boundary_only",
        },
        "any_defect_region": {
            "truth_classes": (1, 2),
            "prediction_key": "any_defect_region",
            "geometry": "region",
        },
        "domain_symmetry_region_ablation": {
            "truth_classes": (1,),
            "prediction_key": "domain_symmetry_region",
            "geometry": "region_ablation",
        },
        "augmented_secondary_symmetry_region_ablation": {
            "truth_classes": (1,),
            "prediction_key": "augmented_secondary_symmetry_region",
            "geometry": "region_ablation",
        },
        "augmented_any_defect_region_ablation": {
            "truth_classes": (1, 2),
            "prediction_key": "augmented_any_defect_region",
            "geometry": "region_ablation",
        },
        "historical_secondary_region_control": {
            "truth_classes": (1,),
            "prediction_key": "historical_secondary_region",
            "geometry": "historical_control",
        },
    }
    accumulators = {
        name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "per_image": [],
        }
        for name in targets
    }
    with ZipFile(ARCHIVE) as archive:
        labels = _decode_labels(archive, manifest["images"])
        for record in manifest["images"]:
            array_path = prediction_root / record["prediction_path"]
            with np.load(array_path, allow_pickle=False) as values:
                predictions_by_key = {
                    target["prediction_key"]: np.asarray(
                        values[target["prediction_key"]],
                        dtype=bool,
                    )
                    for target in targets.values()
                }
            shape = tuple(record["image_shape"])
            if record["category"] == "defective":
                class_map = labels[record["filename"]]
            else:
                class_map = np.zeros(shape, dtype=np.uint8)
            for name, target in targets.items():
                predicted = predictions_by_key[
                    target["prediction_key"]
                ]
                truth = np.isin(class_map, target["truth_classes"])
                tp = int(np.sum(predicted & truth))
                fp = int(np.sum(predicted & ~truth))
                fn = int(np.sum(~predicted & truth))
                tn = int(np.sum(~predicted & ~truth))
                values = accumulators[name]
                values["tp"] += tp
                values["fp"] += fp
                values["fn"] += fn
                values["tn"] += tn
                region = region_metrics(predicted, truth)
                boundary = boundary_metrics(
                    predicted,
                    truth,
                    tolerance_pixels=2.0,
                )
                values["per_image"].append(
                    {
                        "image_id": record["image_id"],
                        "category": record["category"],
                        "filename": record["filename"],
                        "truth_positive_fraction": float(truth.mean()),
                        "predicted_positive_fraction": float(
                            predicted.mean()
                        ),
                        **region,
                        **boundary,
                    }
                )

    scored_targets: dict[str, Any] = {}
    for name, target in targets.items():
        values = accumulators[name]
        micro = _micro_metrics(
            values["tp"],
            values["fp"],
            values["fn"],
            values["tn"],
        )
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
        scored_targets[name] = {
            **target,
            "micro_pixel_metrics": micro,
            "truth_positive_images": len(positive),
            "truth_positive_image_mean_iou": (
                float(np.mean([row["iou"] for row in positive]))
                if positive
                else float("nan")
            ),
            "truth_positive_image_median_iou": (
                float(np.median([row["iou"] for row in positive]))
                if positive
                else float("nan")
            ),
            "truth_positive_image_mean_boundary_f1": (
                float(
                    np.mean(
                        [
                            row["boundary_f1"]
                            for row in positive
                            if np.isfinite(row["boundary_f1"])
                        ]
                    )
                )
                if positive
                else float("nan")
            ),
            "normal_control_mean_false_positive_area_fraction": (
                float(
                    np.mean(
                        [
                            row["predicted_positive_fraction"]
                            for row in normal
                        ]
                    )
                )
                if normal
                else float("nan")
            ),
            "per_image": values["per_image"],
        }

    historical = _read_json(HISTORICAL_METRICS)
    comparisons = {
        "coherence_loss_region": (
            "blurred_amorphous_or_beam_composite"
        ),
        "secondary_symmetry_region": (
            "secondary_crystal_symmetry_composite"
        ),
        "any_defect_region": "any_expert_marked_defect",
    }
    deltas = {}
    for current_name, historical_name in comparisons.items():
        current = scored_targets[current_name]["micro_pixel_metrics"]
        previous = historical["targets"][historical_name][
            "micro_pixel_metrics"
        ]
        deltas[current_name] = {
            metric: float(current[metric] - previous[metric])
            for metric in ("iou", "precision", "recall", "f1")
        }
    payload = {
        "score_format": "phyevo-mic-posthoc-v0.4-score-v1",
        "protocol_id": manifest["protocol_id"],
        "scientific_status": "posthoc_diagnostic_not_locked",
        "labels_accessed_for_scoring": True,
        "eligible_as_new_external_validation": False,
        "historical_locked_prediction_replaced": False,
        "prediction_manifest_sha256": _sha256_file(manifest_path),
        "prediction_seal_sha256": _sha256_file(seal_path),
        "targets": scored_targets,
        "descriptive_delta_vs_historical_v0_3": deltas,
        "claim_boundary": (
            "Dennler labels were already unblinded before v0.4 development. "
            "All deltas are post-hoc diagnostics and cannot support a new "
            "zero-shot, locked, or independent-generalization claim."
        ),
    }
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "posthoc_metrics.json", payload)
    _write_json(
        destination / "score_seal.json",
        {
            "score_format": payload["score_format"],
            "scientific_status": payload["scientific_status"],
            "score_content_sha256": _json_hash(payload),
            "score_file_sha256": _sha256_file(
                destination / "posthoc_metrics.json"
            ),
        },
    )
    summary = {
        name: scored_targets[name]["micro_pixel_metrics"]
        for name in (
            "coherence_loss_region",
            "secondary_symmetry_region",
            "structural_boundary",
            "any_defect_region",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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
    elif args.command == "score":
        score(args.predictions, args.output)
    else:  # pragma: no cover
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
