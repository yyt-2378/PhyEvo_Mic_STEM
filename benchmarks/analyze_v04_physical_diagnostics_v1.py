#!/usr/bin/env python3
"""Audit edge exposure, spectral-basis quality and archive heterogeneity.

This analysis reads only the consumed Dennler archive and sealed v0.4 arrays.
It does not select or alter a detector.  Its role is to expose where the
reported score mask lies, how the two selected reciprocal carriers vary, and
how performance differs across filename-derived acquisition strata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_dennler_reference_baseline import (  # noqa: E402
    DEFAULT_ARCHIVE,
    EXPECTED_ARCHIVE_SHA256,
    _decode_grayscale_tiff,
    _decode_label,
)


PREDICTION_ROOT = ROOT / "outputs/dennler_posthoc_v4_diagnostic_v2/predictions"
SENSITIVITY = ROOT / "benchmarks/paper_baselines/v04_score_sensitivity_v2.json"
SENSITIVITY_SEAL = ROOT / "benchmarks/paper_baselines/v04_score_sensitivity_v2.seal.json"
DEFAULT_OUTPUT = ROOT / "benchmarks/paper_baselines/v04_physical_diagnostics_v1.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _verify_analysis(path: Path, seal_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _read_json(path)
    seal = _read_json(seal_path)
    if _sha256_file(path) != seal["analysis_sha256"]:
        raise RuntimeError(f"Analysis raw hash mismatch: {path}")
    if _json_hash(result) != seal["analysis_content_sha256"]:
        raise RuntimeError(f"Analysis content hash mismatch: {path}")
    analyzer = ROOT / seal["analyzer_path"]
    if _sha256_file(analyzer) != seal["analyzer_sha256"]:
        raise RuntimeError(f"Analysis source hash mismatch: {analyzer}")
    return result, seal


def _counts(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and truth shapes differ")
    return np.asarray(
        [
            np.sum(prediction & truth),
            np.sum(prediction & ~truth),
            np.sum(~prediction & truth),
            np.sum(~prediction & ~truth),
        ],
        dtype=np.int64,
    )


def _metrics(counts: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = [float(value) for value in counts]
    return {
        "iou": tp / max(tp + fp + fn, 1.0),
        "f1": 2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "specificity": tn / max(tn + fp, 1.0),
    }


def _stratum(filename: str) -> tuple[str, str]:
    modality = filename.strip().split()[0]
    match = re.search(r"x(\d+)M", filename)
    magnification = f"x{match.group(1)}M" if match else "unknown"
    return modality, magnification


def _basis_quality(basis: list[list[float]], spacing: float) -> dict[str, Any]:
    vectors = np.asarray(basis, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    cosine = float(np.dot(vectors[0], vectors[1]) / np.prod(norms))
    acute_angle = float(np.degrees(np.arccos(np.clip(abs(cosine), 0.0, 1.0))))
    return {
        "acute_angle_degrees": acute_angle,
        "carrier_norm_ratio": float(norms.max() / norms.min()),
        "reciprocal_basis_condition_number": float(np.linalg.cond(vectors)),
        "lattice_relative_spacing_pixels": float(spacing),
        "spacing_at_configured_clip_bound": bool(
            np.isclose(spacing, 2.5) or np.isclose(spacing, 28.0)
        ),
    }


def analyze() -> dict[str, Any]:
    if _sha256_file(DEFAULT_ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("Dennler archive checksum changed")
    sensitivity, sensitivity_seal = _verify_analysis(SENSITIVITY, SENSITIVITY_SEAL)
    reconstruction = {
        row["filename"]: row for row in sensitivity["per_image_reconstruction"]
    }
    manifest_path = PREDICTION_ROOT / "prediction_manifest.json"
    prediction_seal_path = PREDICTION_ROOT / "prediction_seal.json"
    manifest = _read_json(manifest_path)
    prediction_seal = _read_json(prediction_seal_path)
    if _sha256_file(manifest_path) != prediction_seal["prediction_manifest_sha256"]:
        raise RuntimeError("Prediction manifest seal mismatch")
    if _json_hash(manifest) != prediction_seal["prediction_content_sha256"]:
        raise RuntimeError("Prediction manifest content mismatch")

    per_image: list[dict[str, Any]] = []
    support_counts = {
        target: {band: np.zeros(4, dtype=np.int64) for band in ("edge_ramp", "low_support", "interior")}
        for target in ("class_2", "any_defect")
    }
    stratum_counts: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "modality": {},
        "magnification": {},
    }
    total_positive_pixels = 0
    edge_positive_pixels = 0
    low_positive_pixels = 0
    nonempty_count = 0
    border_touch_count = 0

    with ZipFile(DEFAULT_ARCHIVE) as archive:
        for record in manifest["images"]:
            raw = archive.read(record["source_relative_path"])
            if _sha256_bytes(raw) != record["source_sha256"]:
                raise RuntimeError(f"Source hash mismatch: {record['filename']}")
            image = _decode_grayscale_tiff(raw)
            expected_shape = tuple(record["image_shape"])
            if image.shape != expected_shape:
                raise RuntimeError(f"Source shape mismatch: {record['filename']}")
            if record["category"] == "defective":
                label_path = record["source_relative_path"].replace("/images/", "/labels/")
                truth = _decode_label(archive.read(label_path), expected_shape)
            else:
                truth = np.zeros(expected_shape, dtype=np.uint8)

            prediction_path = PREDICTION_ROOT / record["prediction_path"]
            if _sha256_file(prediction_path) != record["prediction_sha256"]:
                raise RuntimeError(f"Prediction hash mismatch: {record['filename']}")
            with np.load(prediction_path, allow_pickle=False) as values:
                prediction = np.asarray(values["coherence_loss_region"], dtype=bool)
                support = np.asarray(values["reliability"], dtype=np.float32)
            if prediction.shape != expected_shape or support.shape != expected_shape:
                raise RuntimeError(f"Prediction shape mismatch: {record['filename']}")

            edge_ramp = support < np.float32(1.0)
            low_support = support < np.float32(0.5)
            interior = ~edge_ramp
            positive_pixels = int(prediction.sum())
            touches_border = bool(
                prediction[0].any()
                or prediction[-1].any()
                or prediction[:, 0].any()
                or prediction[:, -1].any()
            )
            total_positive_pixels += positive_pixels
            edge_positive_pixels += int(np.sum(prediction & edge_ramp))
            low_positive_pixels += int(np.sum(prediction & low_support))
            nonempty_count += int(positive_pixels > 0)
            border_touch_count += int(touches_border)

            targets = {"class_2": truth == 2, "any_defect": truth > 0}
            for target_name, target in targets.items():
                for band_name, selector in (
                    ("edge_ramp", edge_ramp),
                    ("low_support", low_support),
                    ("interior", interior),
                ):
                    support_counts[target_name][band_name] += _counts(
                        prediction[selector], target[selector]
                    )
            modality, magnification = _stratum(record["filename"])
            for family, value in (("modality", modality), ("magnification", magnification)):
                target_store = stratum_counts[family].setdefault(
                    value,
                    {target: np.zeros(4, dtype=np.int64) for target in targets},
                )
                for target_name, target in targets.items():
                    target_store[target_name] += _counts(prediction, target)

            replay = reconstruction[record["filename"]]
            quality = _basis_quality(
                replay["reciprocal_basis_fy_fx"],
                replay["lattice_spacing_pixels"],
            )
            per_image.append(
                {
                    "filename": record["filename"],
                    "category": record["category"],
                    "modality": modality,
                    "magnification": magnification,
                    "score_mask_nonempty": bool(positive_pixels),
                    "score_mask_touches_border": touches_border,
                    "score_positive_pixel_count": positive_pixels,
                    "score_positive_fraction_in_edge_ramp": (
                        float(np.sum(prediction & edge_ramp) / positive_pixels)
                        if positive_pixels
                        else 0.0
                    ),
                    "score_positive_fraction_in_low_support": (
                        float(np.sum(prediction & low_support) / positive_pixels)
                        if positive_pixels
                        else 0.0
                    ),
                    **quality,
                }
            )

    quality_fields = (
        "acute_angle_degrees",
        "carrier_norm_ratio",
        "reciprocal_basis_condition_number",
        "lattice_relative_spacing_pixels",
    )
    quality_summary = {
        field: {
            "minimum": float(min(row[field] for row in per_image)),
            "median": float(np.median([row[field] for row in per_image])),
            "maximum": float(max(row[field] for row in per_image)),
        }
        for field in quality_fields
    }
    quality_summary["spacing_clip_bound_image_count"] = int(
        sum(row["spacing_at_configured_clip_bound"] for row in per_image)
    )
    subgroup_summary = {
        family: {
            value: {
                "image_count": sum(row[family] == value for row in per_image),
                "targets": {
                    target: _metrics(counts)
                    for target, counts in targets.items()
                },
            }
            for value, targets in groups.items()
        }
        for family, groups in stratum_counts.items()
    }

    return {
        "analysis_format": "phyevo-v04-physical-diagnostics-v1",
        "scientific_status": "retrospective_same_consumed_archive_diagnostic",
        "external_generalization_claim_allowed": False,
        "labels_read_by_analysis": True,
        "training_or_fitting_used": False,
        "purpose": (
            "Expose edge exposure, selected-carrier geometry and filename-derived "
            "archive heterogeneity without altering the sealed score masks."
        ),
        "input_provenance": {
            "archive_path": DEFAULT_ARCHIVE.relative_to(ROOT).as_posix(),
            "archive_sha256": _sha256_file(DEFAULT_ARCHIVE),
            "prediction_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "prediction_manifest_sha256": _sha256_file(manifest_path),
            "prediction_seal_path": prediction_seal_path.relative_to(ROOT).as_posix(),
            "prediction_seal_sha256": _sha256_file(prediction_seal_path),
            "sensitivity_analysis_path": SENSITIVITY.relative_to(ROOT).as_posix(),
            "sensitivity_analysis_sha256": sensitivity_seal["analysis_sha256"],
            "all_61_source_hashes_and_shapes_verified": True,
            "all_61_prediction_hashes_and_shapes_verified": True,
        },
        "dataset": {
            "image_count": len(per_image),
            "defective_image_count": sum(row["category"] == "defective" for row in per_image),
            "archive_nominal_control_image_count": sum(row["category"] == "normal" for row in per_image),
        },
        "edge_exposure": {
            "nonempty_score_mask_count": nonempty_count,
            "border_touching_nonempty_score_mask_count": border_touch_count,
            "score_positive_pixel_count": total_positive_pixels,
            "score_positive_fraction_in_edge_ramp": edge_positive_pixels / max(total_positive_pixels, 1),
            "score_positive_fraction_in_low_support": low_positive_pixels / max(total_positive_pixels, 1),
            "support_prior_is_not_used_for_mask_decision": True,
            "interpretation": (
                "These are exposure diagnostics, not calibrated reliability or proof "
                "that edge-touching positives are errors."
            ),
        },
        "support_stratified_metrics": {
            target: {
                band: _metrics(counts)
                for band, counts in bands.items()
            }
            for target, bands in support_counts.items()
        },
        "selected_carrier_quality": quality_summary,
        "filename_derived_subgroups": subgroup_summary,
        "per_image": per_image,
        "limitations": [
            "Modality and magnification strata are parsed from filenames and are not independent acquisition groups.",
            "The edge-distance support field is uncalibrated and does not alter the score mask.",
            "Selected spectral carriers are not crystallographically indexed reciprocal-lattice vectors.",
            "All metrics use the same post-unblinding consumed archive.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT.resolve())
    seal_path = output.with_name(output.stem + ".seal.json")
    if output.exists() or seal_path.exists():
        raise FileExistsError("Refusing to overwrite a versioned analysis or seal")
    result = analyze()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    analyzer = Path(__file__).resolve()
    seal = {
        "seal_format": "phyevo-analysis-seal-v1",
        "analysis_path": output.relative_to(ROOT).as_posix(),
        "analysis_sha256": _sha256_file(output),
        "analysis_content_sha256": _json_hash(result),
        "analyzer_path": analyzer.relative_to(ROOT).as_posix(),
        "analyzer_sha256": _sha256_file(analyzer),
        "scientific_status": result["scientific_status"],
    }
    _write_json(seal_path, seal)
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {seal_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
