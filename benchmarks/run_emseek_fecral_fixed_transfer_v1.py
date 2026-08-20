#!/usr/bin/env python
"""Run the two-image EMSeek FeCrAl fixed-method transfer stress test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from analyze_v04_score_sensitivity_v2 import (
    _postprocess_score,
    _weighted_score,
)
from run_dennler_reference_baseline import predict_fourier
from tfree_detection.diagnostic_v4 import PosthocV4Config, analyze_posthoc_v4


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "benchmarks/emseek_fecral_fixed_transfer_v1"
PROTOCOL = RUN_ROOT / "protocol.yaml"
DETECTOR = ROOT / "src/tfree_detection/diagnostic_v4.py"
FOURIER = ROOT / "benchmarks/run_dennler_reference_baseline.py"
SENSITIVITY_ANALYZER = ROOT / "benchmarks/analyze_v04_score_sensitivity_v2.py"

CONFIG_SHA256 = "1b4ba9063837e6f620a130e86fea72bca7342adab48a29ab233a64ccdcf53d6c"
DETECTOR_SHA256 = "ff24b8e368b915560bf3f0ed83da9d1a35a8436b8b6f3ba2da97e918ce406187"
SOURCE_RECORDS = (
    {
        "sample_id": "fecral-148",
        "image_path": "datasets/public/emseek_remaining_v1/metal_alloy_defects/FeCrAl/image/148.png",
        "image_sha256": "dce375bd1a3d56b4f1a713ada34033de03a2b34795e02bac384e6da62eea77d4",
        "label_path": "datasets/public/emseek_remaining_v1/metal_alloy_defects/FeCrAl/label/148.png",
        "label_sha256": "8df3cb0805899aa88db103cb82d1b6dda012ca7084289e2fb542fdd86e6507a0",
    },
    {
        "sample_id": "fecral-46",
        "image_path": "datasets/public/emseek_remaining_v1/metal_alloy_defects/FeCrAl/image/46.png",
        "image_sha256": "7a95df67c34fff594bb7d450eac3fd736638fa47a36fdf967373cf2cdf77e17a",
        "label_path": "datasets/public/emseek_remaining_v1/metal_alloy_defects/FeCrAl/label/46.png",
        "label_sha256": "5f0f5ffdcf15075df5b290c31dd83f801602fb9b9684e65f628ed85fe19dbaa5",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _inside_root(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Output must remain inside the project") from error
    return resolved


def _load_grayscale(path: Path) -> np.ndarray:
    values = np.asarray(Image.open(path))
    if values.ndim == 3 and values.shape[-1] in {3, 4}:
        values = np.mean(values[..., :3].astype(float), axis=-1)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected finite 2-D image: {path}")
    return values.astype(float, copy=False)


def _config() -> PosthocV4Config:
    config = PosthocV4Config()
    config.validate()
    if _canonical_hash(config.to_dict()) != CONFIG_SHA256:
        raise ValueError("Current v0.4 defaults differ from frozen configuration")
    if _sha256(DETECTOR) != DETECTOR_SHA256:
        raise ValueError("v0.4 implementation differs from the frozen implementation")
    return config


def _component_mask(
    carrier_loss: np.ndarray,
    texture_loss: np.ndarray,
    spacing: float,
    config: PosthocV4Config,
    carrier_weight: float,
    texture_weight: float,
) -> np.ndarray:
    score = _weighted_score(
        carrier_loss,
        texture_loss,
        spacing,
        carrier_weight,
        texture_weight,
    )
    return _postprocess_score(
        score,
        spacing,
        threshold=config.coherence_threshold,
        morphology_radius_cells=config.morphology_radius_cells,
        minimum_region_cells=config.minimum_coherence_region_cells,
    )


def predict(output: str | Path) -> None:
    destination = _inside_root(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    config = _config()
    destination.mkdir(parents=True, exist_ok=False)
    arrays = destination / "arrays"
    arrays.mkdir()
    records: list[dict[str, Any]] = []

    for source in SOURCE_RECORDS:
        image_path = ROOT / source["image_path"]
        if _sha256(image_path) != source["image_sha256"]:
            raise ValueError(f"Source image changed: {image_path}")
        image = _load_grayscale(image_path)
        started = time.perf_counter()
        result = analyze_posthoc_v4(image, config)
        v04_seconds = time.perf_counter() - started
        spacing = float(result.metadata["lattice_spacing_pixels"])
        carrier_loss = result.evidence_maps["carrier_loss"]
        texture_loss = result.evidence_maps["highpass_loss"]
        methods = {
            "v04_score_only": np.asarray(result.coherence_loss_mask, dtype=bool),
            "carrier_contribution_only": _component_mask(
                carrier_loss, texture_loss, spacing, config, 0.68, 0.0
            ),
            "texture_contribution_only": _component_mask(
                carrier_loss, texture_loss, spacing, config, 0.0, 0.32
            ),
        }
        fourier_started = time.perf_counter()
        methods["fourier_bragg_class2"] = predict_fourier(image) == 2
        fourier_seconds = time.perf_counter() - fourier_started
        array_path = arrays / f"{source['sample_id']}.npz"
        np.savez_compressed(
            array_path,
            **{name: mask.astype(np.uint8) for name, mask in methods.items()},
            v04_continuous_score=result.evidence_maps["coherence_loss_score"].astype(np.float32),
        )
        records.append(
            {
                "sample_id": source["sample_id"],
                "image_path": source["image_path"],
                "image_sha256": source["image_sha256"],
                "image_shape": list(image.shape),
                "array_path": str(array_path.relative_to(destination)),
                "array_sha256": _sha256(array_path),
                "lattice_spacing_pixels": spacing,
                "runtime_seconds": {
                    "v04_analysis_including_component_evidence": v04_seconds,
                    "fourier_bragg_class2": fourier_seconds,
                },
                "positive_fractions": {
                    name: float(mask.mean()) for name, mask in methods.items()
                },
                "status": "success",
            }
        )
        print(f"predicted {source['sample_id']}", flush=True)

    manifest = {
        "prediction_format": "phyevo-mic-emseek-fecral-fixed-transfer-v1",
        "protocol_id": "emseek-fecral-fixed-transfer-v1",
        "scientific_status": "secondary_same_group_author_mask_transfer_stress",
        "labels_read_during_prediction": False,
        "external_generalization_claimed": False,
        "experimental_gold_claimed": False,
        "protocol_sha256": _sha256(PROTOCOL),
        "implementation": {
            "diagnostic_v4_sha256": _sha256(DETECTOR),
            "fourier_port_sha256": _sha256(FOURIER),
            "sensitivity_helpers_sha256": _sha256(SENSITIVITY_ANALYZER),
            "runner_sha256": _sha256(Path(__file__)),
            "config_sha256": CONFIG_SHA256,
        },
        "grouping": {
            "split_group_id": "emseek-metal-alloy-defects-fecral-single-conservative-group",
            "acquisition_id": "emseek-metal-alloy-defects-fecral-unresolved-acquisition",
            "independent_group_count": 1,
            "source_partition": "train",
        },
        "counts": {"images": len(records), "successful": len(records), "failed": 0},
        "images": records,
    }
    manifest_path = destination / "prediction_manifest.json"
    _write_json(manifest_path, manifest)
    seal = {
        "seal_format": "phyevo-mic-emseek-fecral-prediction-seal-v1",
        "prediction_manifest_sha256": _sha256(manifest_path),
        "prediction_content_sha256": _canonical_hash(manifest),
        "labels_read_during_prediction": False,
        "failed": 0,
    }
    _write_json(destination / "prediction_seal.json", seal)


def _confusion(prediction: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(truth, dtype=bool)
    return {
        "tp": int(np.sum(pred & ref)),
        "fp": int(np.sum(pred & ~ref)),
        "fn": int(np.sum(~pred & ref)),
        "tn": int(np.sum(~pred & ~ref)),
    }


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    denominator = tp + fp + fn
    return {
        **counts,
        "iou": float(tp / denominator if denominator else 1.0),
        "f1": float(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0),
        "precision": float(tp / (tp + fp) if tp + fp else 0.0),
        "recall": float(tp / (tp + fn) if tp + fn else 0.0),
    }


def score(predictions: str | Path, output: str | Path) -> None:
    prediction_root = _inside_root(predictions)
    destination = _inside_root(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    manifest_path = prediction_root / "prediction_manifest.json"
    prediction_seal_path = prediction_root / "prediction_seal.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(prediction_seal_path.read_text(encoding="utf-8"))
    if seal["prediction_manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("Prediction manifest seal mismatch")
    if seal["prediction_content_sha256"] != _canonical_hash(manifest):
        raise ValueError("Prediction content seal mismatch")
    if manifest["labels_read_during_prediction"] is not False:
        raise ValueError("Prediction stage was not label-isolated")

    source_by_id = {record["sample_id"]: record for record in SOURCE_RECORDS}
    methods = (
        "v04_score_only",
        "carrier_contribution_only",
        "texture_contribution_only",
        "fourier_bragg_class2",
    )
    results: list[dict[str, Any]] = []
    for record in manifest["images"]:
        source = source_by_id[record["sample_id"]]
        label_path = ROOT / source["label_path"]
        if _sha256(label_path) != source["label_sha256"]:
            raise ValueError(f"Annotation changed: {label_path}")
        label = np.asarray(Image.open(label_path))
        values = {int(value) for value in np.unique(label)}
        if not values.issubset({0, 85, 170, 255}) or tuple(label.shape) != tuple(record["image_shape"]):
            raise ValueError(f"Unexpected FeCrAl label encoding: {label_path}")
        composite = label > 0
        array_path = prediction_root / record["array_path"]
        if _sha256(array_path) != record["array_sha256"]:
            raise ValueError(f"Prediction array changed: {array_path}")
        with np.load(array_path, allow_pickle=False) as arrays:
            method_results = {}
            for method in methods:
                prediction = np.asarray(arrays[method], dtype=bool)
                method_results[method] = {
                    "composite_nonzero_author_mask": _metrics(
                        _confusion(prediction, composite)
                    ),
                    "positive_area_fraction": float(prediction.mean()),
                    "palette_value_coverage": {
                        str(value): float(np.mean(prediction[label == value]))
                        for value in (85, 170, 255)
                        if np.any(label == value)
                    },
                }
        results.append(
            {
                "sample_id": record["sample_id"],
                "image_sha256": record["image_sha256"],
                "label_sha256": source["label_sha256"],
                "label_values": sorted(values),
                "label_positive_area_fraction": float(composite.mean()),
                "methods": method_results,
            }
        )

    payload = {
        "score_format": "phyevo-mic-emseek-fecral-fixed-transfer-score-v1",
        "protocol_id": manifest["protocol_id"],
        "scientific_status": manifest["scientific_status"],
        "claim_boundary": (
            "Two author-curated composite masks from one unresolved acquisition group; "
            "per-image fixed-method stress only, not Gold or external generalization."
        ),
        "prediction_manifest_sha256": _sha256(manifest_path),
        "prediction_seal_sha256": _sha256(prediction_seal_path),
        "aggregate_inferential_statistics_reported": False,
        "palette_semantics_inferred": False,
        "images": results,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, payload)
    output_seal = {
        "seal_format": "phyevo-mic-emseek-fecral-score-seal-v1",
        "score_sha256": _sha256(destination),
        "score_content_sha256": _canonical_hash(payload),
        "analyzer_sha256": _sha256(Path(__file__)),
        "prediction_manifest_sha256": _sha256(manifest_path),
    }
    _write_json(destination.with_suffix(".seal.json"), output_seal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--output", required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.output)
    else:
        score(args.predictions, args.output)


if __name__ == "__main__":
    main()
