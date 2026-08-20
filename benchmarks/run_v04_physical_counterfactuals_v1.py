#!/usr/bin/env python
"""Run a label-isolated physical-degradation audit of the frozen v0.4 score.

The prediction command never opens a Dennler label member. It perturbs source
images using fixed, lattice-relative operators and seals the resulting masks.
Only the separate score command opens labels. All claims remain retrospective
because the Dennler archive was consumed before this audit was designed.
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
from scipy.ndimage import gaussian_filter, map_coordinates
import tifffile

from tfree_detection.diagnostic_v4 import PosthocV4Config, analyze_posthoc_v4


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "datasets/public/dennler_2021_hrstem_defects/hrstem_defects_dataset.zip"
PROTOCOL = ROOT / "benchmarks/v04_physical_counterfactual_protocol_v1.yaml"
STORED_ROOT = ROOT / "outputs/dennler_posthoc_v4_diagnostic_v2/predictions"
STORED_MANIFEST = STORED_ROOT / "prediction_manifest.json"
DETECTOR = ROOT / "src/tfree_detection/diagnostic_v4.py"

ARCHIVE_SHA256 = "abef46a0183ba08b925c64807d5e914cd97967444d9fd3e0799924896f0614ec"
STORED_MANIFEST_SHA256 = "33652927a60b1be467c458cef86bb2a4dbac56ef1dc3d18436147743bfe83951"
DETECTOR_SHA256 = "ff24b8e368b915560bf3f0ed83da9d1a35a8436b8b6f3ba2da97e918ce406187"
CONFIG_SHA256 = "1b4ba9063837e6f620a130e86fea72bca7342adab48a29ab233a64ccdcf53d6c"
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_RESAMPLES = 20_000

LEVELS: dict[str, dict[str, float]] = {
    "transfer_loss_proxy": {"low": 0.05, "nominal": 0.10, "high": 0.20},
    "count_noise_proxy": {"low": 200.0, "nominal": 50.0, "high": 12.5},
    "linear_drift_proxy": {"low": 0.05, "nominal": 0.10, "high": 0.20},
    "scan_jitter_proxy": {"low": 0.05, "nominal": 0.10, "high": 0.20},
    "field_edge_truncation": {"low": 0.50, "nominal": 1.00, "high": 2.00},
}


def _sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
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


def _inside_root(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Output must remain inside the project") from error
    return resolved


def _read_image(payload: bytes) -> np.ndarray:
    image = np.asarray(tifffile.imread(BytesIO(payload)))
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        image = np.mean(image[..., :3].astype(np.float64), axis=-1)
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise ValueError(f"Expected finite 2-D image, got {image.shape}")
    return image.astype(np.float64, copy=False)


def _variant_name(family: str, level: str) -> str:
    return f"{family}__{level}"


def _variant_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for family, levels in LEVELS.items():
        for level, value in levels.items():
            specs[_variant_name(family, level)] = {
                "family": family,
                "level": level,
                "value": value,
                "scope": "all_archive" if level == "nominal" else "ladder_subset",
            }
    return specs


def _seed(image_id: str, variant: str) -> int:
    digest = hashlib.sha256(f"{image_id}:{variant}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _row_warp(values: np.ndarray, displacement: np.ndarray, *, order: int) -> np.ndarray:
    height, width = values.shape
    yy, xx = np.mgrid[:height, :width]
    coordinates = np.asarray(
        [yy.astype(np.float64), xx.astype(np.float64) - displacement[:, None]]
    )
    return map_coordinates(
        values,
        coordinates,
        order=order,
        mode="reflect",
        prefilter=order > 1,
    )


def apply_variant(
    values: np.ndarray,
    *,
    family: str,
    value: float,
    spacing: float,
    seed: int,
    is_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply one frozen transform and return values, valid support, metadata."""

    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("Physical counterfactuals require 2-D arrays")
    height, width = array.shape
    valid = np.ones((height, width), dtype=bool)

    if family == "transfer_loss_proxy":
        if is_mask:
            transformed = array.astype(bool, copy=True)
        else:
            sigma = max(float(value * spacing), 1e-6)
            transformed = gaussian_filter(array.astype(float), sigma=sigma, mode="reflect")
        metadata = {"sigma_pixels": float(value * spacing)}
    elif family == "count_noise_proxy":
        if is_mask:
            transformed = array.astype(bool, copy=True)
        else:
            low, high = np.percentile(array, (0.5, 99.5))
            scale = max(float(high - low), 1e-12)
            normalized = np.clip((array - low) / scale, 0.0, 1.0)
            rng = np.random.default_rng(seed)
            sampled = rng.poisson(normalized * value) / float(value)
            transformed = sampled * scale + low
        metadata = {"unit_intensity_count_equivalents": float(value)}
    elif family in {"linear_drift_proxy", "scan_jitter_proxy"}:
        amplitude = float(value * spacing)
        rows = np.arange(height, dtype=float)
        if family == "linear_drift_proxy":
            displacement = np.linspace(-amplitude, amplitude, height)
            period = None
        else:
            period = max(8.0 * spacing, 2.0)
            displacement = amplitude * np.sin(2.0 * np.pi * rows / period)
        transformed = _row_warp(
            array.astype(float), displacement, order=0 if is_mask else 1
        )
        if is_mask:
            transformed = transformed >= 0.5
        margin = min(int(np.ceil(amplitude)) + 2, max(width // 4, 1))
        if margin:
            valid[:, :margin] = False
            valid[:, width - margin :] = False
        metadata = {
            "peak_displacement_pixels": amplitude,
            "period_pixels": period,
            "excluded_horizontal_margin_pixels": margin,
        }
    elif family == "field_edge_truncation":
        crop = min(max(int(round(value * spacing)), 1), max(height // 4, 1))
        transformed = array[crop:, :].copy()
        valid = np.ones(transformed.shape, dtype=bool)
        if is_mask:
            transformed = transformed.astype(bool)
        metadata = {"top_crop_pixels": crop}
    else:
        raise ValueError(f"Unknown perturbation family: {family}")
    return np.asarray(transformed), valid, metadata


def _confusion(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(truth, dtype=bool)
    use = np.asarray(valid, dtype=bool)
    if pred.shape != ref.shape or pred.shape != use.shape:
        raise ValueError("Prediction, truth and valid masks must share shape")
    return np.asarray(
        [
            np.sum(pred & ref & use),
            np.sum(pred & ~ref & use),
            np.sum(~pred & ref & use),
            np.sum(~pred & ~ref & use),
        ],
        dtype=np.int64,
    )


def _metrics(confusion: np.ndarray) -> dict[str, float | int]:
    tp, fp, fn, tn = (int(value) for value in confusion)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "f1": float(f1),
    }


def _validate_frozen_inputs() -> tuple[dict[str, Any], PosthocV4Config]:
    checks = {
        ARCHIVE: ARCHIVE_SHA256,
        STORED_MANIFEST: STORED_MANIFEST_SHA256,
        DETECTOR: DETECTOR_SHA256,
    }
    for path, expected in checks.items():
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(f"Frozen input changed: {path} ({observed})")
    stored = json.loads(STORED_MANIFEST.read_text(encoding="utf-8"))
    config = PosthocV4Config()
    config.validate()
    if _canonical_hash(config.to_dict()) != CONFIG_SHA256:
        raise ValueError("Current v0.4 defaults differ from the frozen configuration")
    if stored["config_sha256"] != CONFIG_SHA256:
        raise ValueError("Stored manifest has an unexpected v0.4 configuration")
    if stored["counts"] != {"images": 61, "successful": 61, "failed": 0}:
        raise ValueError("Expected the complete successful 61-image v0.4 run")
    return stored, config


def _stored_mask(record: dict[str, Any]) -> np.ndarray:
    path = STORED_ROOT / str(record["prediction_path"])
    if _sha256_file(path) != record["prediction_sha256"]:
        raise ValueError(f"Stored prediction changed: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["coherence_loss_region"], dtype=bool)


def predict(output: str | Path) -> None:
    destination = _inside_root(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    stored, config = _validate_frozen_inputs()
    destination.mkdir(parents=True, exist_ok=False)
    arrays = destination / "arrays"
    arrays.mkdir()
    specs = _variant_specs()
    records: list[dict[str, Any]] = []
    failures = 0

    with ZipFile(ARCHIVE) as archive:
        for index, source in enumerate(stored["images"]):
            relative = str(source["source_relative_path"])
            if "/labels/" in relative:
                raise RuntimeError("Prediction attempted to access a label member")
            raw = archive.read(relative)
            if _sha256_bytes(raw) != source["source_sha256"]:
                raise ValueError(f"Source image changed: {relative}")
            image = _read_image(raw)
            baseline = _stored_mask(source)
            spacing = float(source["lattice_spacing_pixels"])
            ladder = index % 5 == 0
            output_arrays: dict[str, np.ndarray] = {}
            variants: list[dict[str, Any]] = []

            for variant, spec in specs.items():
                if spec["scope"] == "ladder_subset" and not ladder:
                    continue
                entry: dict[str, Any] = {
                    "variant": variant,
                    **spec,
                    "status": "failed",
                    "error_type": None,
                    "error_message": None,
                }
                try:
                    transformed, valid, transform_meta = apply_variant(
                        image,
                        family=spec["family"],
                        value=float(spec["value"]),
                        spacing=spacing,
                        seed=_seed(str(source["image_id"]), variant),
                    )
                    aligned_baseline, baseline_valid, _ = apply_variant(
                        baseline,
                        family=spec["family"],
                        value=float(spec["value"]),
                        spacing=spacing,
                        seed=_seed(str(source["image_id"]), variant),
                        is_mask=True,
                    )
                    if valid.shape != baseline_valid.shape or not np.array_equal(valid, baseline_valid):
                        raise RuntimeError("Image and mask transform supports differ")
                    result = analyze_posthoc_v4(transformed, config)
                    prediction = np.asarray(result.coherence_loss_mask, dtype=bool)
                    stability = _confusion(prediction, aligned_baseline, valid)
                    key = f"mask__{variant}"
                    output_arrays[key] = prediction.astype(np.uint8)
                    entry.update(
                        {
                            "status": "success",
                            "array_key": key,
                            "output_shape": list(prediction.shape),
                            "valid_pixels": int(valid.sum()),
                            "positive_pixels": int(np.sum(prediction & valid)),
                            "positive_fraction_valid": float(np.mean(prediction[valid])),
                            "reestimated_lattice_spacing_pixels": float(
                                result.metadata["lattice_spacing_pixels"]
                            ),
                            "transform": transform_meta,
                            "stability_confusion_tp_fp_fn_tn": stability.tolist(),
                            "stability": _metrics(stability),
                        }
                    )
                except Exception as error:
                    entry["error_type"] = type(error).__name__
                    entry["error_message"] = str(error)
                    failures += 1
                variants.append(entry)

            array_path = arrays / f"{source['image_id']}.npz"
            np.savez_compressed(array_path, **output_arrays)
            records.append(
                {
                    "manifest_index": index,
                    "image_id": source["image_id"],
                    "category": source["category"],
                    "filename": source["filename"],
                    "source_relative_path": relative,
                    "source_sha256": source["source_sha256"],
                    "baseline_lattice_spacing_pixels": spacing,
                    "severity_ladder_member": ladder,
                    "array_path": str(array_path.relative_to(destination)),
                    "array_sha256": _sha256_file(array_path),
                    "variants": variants,
                }
            )
            print(
                f"[{index + 1:02d}/61] {source['filename']}: "
                f"{sum(v['status'] == 'success' for v in variants)}/{len(variants)}",
                flush=True,
            )

    manifest = {
        "prediction_format": "phyevo-mic-v04-physical-counterfactual-predictions-v1",
        "protocol_id": "dennler-v04-physical-counterfactual-audit-v1",
        "scientific_status": "retrospective_consumed_archive_stress_audit",
        "labels_read_during_prediction": False,
        "external_generalization_claimed": False,
        "calibrated_microscope_simulation_claimed": False,
        "archive_sha256": _sha256_file(ARCHIVE),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "detector_implementation_sha256": _sha256_file(DETECTOR),
        "detector_config_sha256": CONFIG_SHA256,
        "stored_prediction_manifest_sha256": _sha256_file(STORED_MANIFEST),
        "variant_specs": specs,
        "selection": {
            "all_archive": "every stored manifest record",
            "ladder_subset": "zero-based manifest index modulo 5 equals 0",
            "ladder_subset_count": sum(index % 5 == 0 for index in range(61)),
            "selection_uses_labels": False,
        },
        "counts": {
            "images": len(records),
            "variant_attempts": sum(len(record["variants"]) for record in records),
            "successful_variants": sum(
                variant["status"] == "success"
                for record in records
                for variant in record["variants"]
            ),
            "failed_variants": failures,
        },
        "images": records,
    }
    manifest_path = destination / "prediction_manifest.json"
    _write_json(manifest_path, manifest)
    seal = {
        "seal_format": "phyevo-mic-v04-physical-counterfactual-prediction-seal-v1",
        "prediction_manifest_sha256": _sha256_file(manifest_path),
        "prediction_content_sha256": _canonical_hash(manifest),
        "protocol_sha256": _sha256_file(PROTOCOL),
        "labels_read_during_prediction": False,
        "failed_variants": failures,
    }
    _write_json(destination / "prediction_seal.json", seal)
    print(json.dumps({"output": str(destination), "counts": manifest["counts"]}, indent=2))


def _decode_truth(archive: ZipFile, source: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    if source["category"] != "defective":
        return np.zeros(shape, dtype=np.uint8)
    label_path = str(source["source_relative_path"]).replace("/images/", "/labels/")
    label = np.asarray(tifffile.imread(BytesIO(archive.read(label_path))))
    if label.ndim == 3:
        label = label[..., 0]
    values = {float(value) for value in np.unique(label)}
    if not values.issubset({0.0, 127.0, 255.0}) or label.shape != shape:
        raise ValueError(f"Unexpected label encoding or shape: {label_path}")
    decoded = np.zeros(shape, dtype=np.uint8)
    decoded[np.asarray(label, dtype=float) == 127.0] = 1
    decoded[np.asarray(label, dtype=float) == 255.0] = 2
    return decoded


def _bootstrap_delta(
    current: np.ndarray,
    baseline: np.ndarray,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    if current.shape != baseline.shape or current.ndim != 2 or current.shape[1] != 4:
        raise ValueError("Bootstrap inputs must be matching N x 4 confusion arrays")
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    n = current.shape[0]
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for start in range(0, BOOTSTRAP_RESAMPLES, 500):
        size = min(500, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, n, size=(size, n))
        c = current[indices].sum(axis=1)
        b = baseline[indices].sum(axis=1)
        c_den = c[:, 0] + c[:, 1] + c[:, 2]
        b_den = b[:, 0] + b[:, 1] + b[:, 2]
        c_iou = np.divide(c[:, 0], c_den, out=np.ones(size), where=c_den > 0)
        b_iou = np.divide(b[:, 0], b_den, out=np.ones(size), where=b_den > 0)
        values[start : start + size] = c_iou - b_iou
    observed = _metrics(current.sum(axis=0))["iou"] - _metrics(baseline.sum(axis=0))["iou"]
    low, high = np.percentile(values, (2.5, 97.5))
    return {
        "observed_delta_iou": float(observed),
        "interval_95": [float(low), float(high)],
        "resamples": BOOTSTRAP_RESAMPLES,
        "resampling_unit": "whole_image",
    }


def score(predictions: str | Path, output: str | Path) -> None:
    source_root = _inside_root(predictions)
    destination = _inside_root(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    manifest_path = source_root / "prediction_manifest.json"
    seal_path = source_root / "prediction_seal.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["prediction_manifest_sha256"] != _sha256_file(manifest_path):
        raise ValueError("Prediction manifest seal mismatch")
    if seal["prediction_content_sha256"] != _canonical_hash(manifest):
        raise ValueError("Prediction content seal mismatch")
    if manifest["labels_read_during_prediction"] is not False:
        raise ValueError("Prediction manifest does not assert label isolation")
    if manifest["counts"]["failed_variants"]:
        raise ValueError("Cannot score an incomplete perturbation run")

    specs = manifest["variant_specs"]
    stored_manifest = json.loads(STORED_MANIFEST.read_text(encoding="utf-8"))
    stored_by_id = {
        str(record["image_id"]): record for record in stored_manifest["images"]
    }
    by_variant: dict[str, list[dict[str, Any]]] = {key: [] for key in specs}
    with ZipFile(ARCHIVE) as archive:
        for source in manifest["images"]:
            array_path = source_root / source["array_path"]
            if _sha256_file(array_path) != source["array_sha256"]:
                raise ValueError(f"Prediction array changed: {array_path}")
            baseline = _stored_mask(stored_by_id[str(source["image_id"])])
            truth = _decode_truth(archive, source, baseline.shape)
            with np.load(array_path, allow_pickle=False) as arrays:
                for variant_record in source["variants"]:
                    variant = variant_record["variant"]
                    spec = specs[variant]
                    prediction = np.asarray(arrays[variant_record["array_key"]], dtype=bool)
                    aligned_any, valid, _ = apply_variant(
                        truth > 0,
                        family=spec["family"],
                        value=float(spec["value"]),
                        spacing=float(source["baseline_lattice_spacing_pixels"]),
                        seed=_seed(str(source["image_id"]), variant),
                        is_mask=True,
                    )
                    aligned_c2, valid_c2, _ = apply_variant(
                        truth == 2,
                        family=spec["family"],
                        value=float(spec["value"]),
                        spacing=float(source["baseline_lattice_spacing_pixels"]),
                        seed=_seed(str(source["image_id"]), variant),
                        is_mask=True,
                    )
                    aligned_baseline, valid_baseline, _ = apply_variant(
                        baseline,
                        family=spec["family"],
                        value=float(spec["value"]),
                        spacing=float(source["baseline_lattice_spacing_pixels"]),
                        seed=_seed(str(source["image_id"]), variant),
                        is_mask=True,
                    )
                    if not np.array_equal(valid, valid_c2) or not np.array_equal(valid, valid_baseline):
                        raise RuntimeError("Scoring supports disagree")
                    record = {
                        "image_id": source["image_id"],
                        "category": source["category"],
                        "filename": source["filename"],
                        "any_confusion_tp_fp_fn_tn": _confusion(prediction, aligned_any, valid).tolist(),
                        "class2_confusion_tp_fp_fn_tn": _confusion(prediction, aligned_c2, valid).tolist(),
                        "baseline_any_confusion_tp_fp_fn_tn": _confusion(aligned_baseline, aligned_any, valid).tolist(),
                        "baseline_class2_confusion_tp_fp_fn_tn": _confusion(aligned_baseline, aligned_c2, valid).tolist(),
                        "stability_confusion_tp_fp_fn_tn": _confusion(prediction, aligned_baseline, valid).tolist(),
                        "valid_pixels": int(valid.sum()),
                        "positive_pixels": int(np.sum(prediction & valid)),
                    }
                    by_variant[variant].append(record)

    summaries: dict[str, Any] = {}
    for offset, (variant, records) in enumerate(sorted(by_variant.items())):
        if not records:
            raise ValueError(f"No records for variant {variant}")
        any_c = np.asarray([r["any_confusion_tp_fp_fn_tn"] for r in records], dtype=np.int64)
        c2_c = np.asarray([r["class2_confusion_tp_fp_fn_tn"] for r in records], dtype=np.int64)
        any_b = np.asarray([r["baseline_any_confusion_tp_fp_fn_tn"] for r in records], dtype=np.int64)
        c2_b = np.asarray([r["baseline_class2_confusion_tp_fp_fn_tn"] for r in records], dtype=np.int64)
        stability = np.asarray([r["stability_confusion_tp_fp_fn_tn"] for r in records], dtype=np.int64)
        normal = [r for r in records if r["category"] != "defective"]
        normal_pixels = sum(r["valid_pixels"] for r in normal)
        normal_positive = sum(r["positive_pixels"] for r in normal)
        summaries[variant] = {
            "family": specs[variant]["family"],
            "level": specs[variant]["level"],
            "value": specs[variant]["value"],
            "scope": specs[variant]["scope"],
            "image_count": len(records),
            "defective_image_count": sum(r["category"] == "defective" for r in records),
            "normal_image_count": len(normal),
            "any_defect": _metrics(any_c.sum(axis=0)),
            "class_2": _metrics(c2_c.sum(axis=0)),
            "mask_stability": _metrics(stability.sum(axis=0)),
            "normal_positive_area_fraction": float(normal_positive / max(normal_pixels, 1)),
            "normal_nonempty_fraction": float(
                sum(r["positive_pixels"] > 0 for r in normal) / max(len(normal), 1)
            ),
            "paired_delta_vs_aligned_baseline": {
                "any_defect": _bootstrap_delta(any_c, any_b, seed_offset=2 * offset),
                "class_2": _bootstrap_delta(c2_c, c2_b, seed_offset=2 * offset + 1),
            },
            "per_image": records,
        }

    result = {
        "result_format": "phyevo-mic-v04-physical-counterfactual-score-v1",
        "protocol_id": manifest["protocol_id"],
        "scientific_status": manifest["scientific_status"],
        "claim_boundary": (
            "Retrospective consumed-archive stress audit using generic imaging-degradation "
            "proxies; not calibrated defocus, thickness, dose, or external generalization."
        ),
        "prediction_manifest_sha256": _sha256_file(manifest_path),
        "prediction_seal_sha256": _sha256_file(seal_path),
        "archive_sha256": _sha256_file(ARCHIVE),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "resampling_unit": "whole_image",
        },
        "variants": summaries,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, result)
    score_seal = {
        "seal_format": "phyevo-mic-v04-physical-counterfactual-score-seal-v1",
        "score_path": str(destination.relative_to(ROOT)),
        "score_sha256": _sha256_file(destination),
        "score_content_sha256": _canonical_hash(result),
        "analyzer_sha256": _sha256_file(Path(__file__)),
        "prediction_manifest_sha256": _sha256_file(manifest_path),
    }
    seal_output = destination.with_suffix(".seal.json")
    _write_json(seal_output, score_seal)
    print(json.dumps({"score": str(destination), "seal": str(seal_output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prediction = subparsers.add_parser("predict")
    prediction.add_argument("--output", required=True)
    scoring = subparsers.add_parser("score")
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.output)
    else:
        score(args.predictions, args.output)


if __name__ == "__main__":
    main()
