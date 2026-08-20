#!/usr/bin/env python3
"""Audit primary per-image failures and edge/interior dependence for v0.4.

This is a post-unblinding descriptive analysis of the same consumed Dennler
archive. It does not tune or alter the detector. Border widths are expressed
in multiples of each image's stored algorithm-selected carrier scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.analyze_v04_physical_diagnostics_v1 import (  # noqa: E402
    PREDICTION_ROOT,
    _counts,
    _json_hash,
    _metrics,
    _sha256_bytes,
    _sha256_file,
)
from benchmarks.run_dennler_reference_baseline import (  # noqa: E402
    DEFAULT_ARCHIVE,
    EXPECTED_ARCHIVE_SHA256,
    _decode_grayscale_tiff,
    _decode_label,
)


DEFAULT_OUTPUT = ROOT / "benchmarks/paper_baselines/v04_primary_failure_and_edge_v1.json"
MARGINS_IN_CARRIER_SCALES = (0.0, 0.5, 1.0, 2.0)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _edge_distance(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices(shape)
    return np.minimum.reduce((yy, xx, height - 1 - yy, width - 1 - xx))


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Positive-image distribution is empty")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q3": float(np.quantile(array, 0.75)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "below_0_50": int(np.sum(array < 0.50)),
        "below_0_25": int(np.sum(array < 0.25)),
    }


def analyze() -> dict[str, Any]:
    if _sha256_file(DEFAULT_ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("Dennler archive checksum changed")
    manifest_path = PREDICTION_ROOT / "prediction_manifest.json"
    seal_path = PREDICTION_ROOT / "prediction_seal.json"
    manifest = _read_json(manifest_path)
    seal = _read_json(seal_path)
    if _sha256_file(manifest_path) != seal["prediction_manifest_sha256"]:
        raise RuntimeError("Prediction manifest seal mismatch")
    if _json_hash(manifest) != seal["prediction_content_sha256"]:
        raise RuntimeError("Prediction manifest content mismatch")

    targets = ("any_defect", "class_2")
    margin_counts = {
        target: {str(margin): np.zeros(4, dtype=np.int64) for margin in MARGINS_IN_CARRIER_SCALES}
        for target in targets
    }
    one_scale_band_counts = {
        target: {"edge": np.zeros(4, dtype=np.int64), "interior": np.zeros(4, dtype=np.int64)}
        for target in targets
    }
    positive_iou: dict[str, list[dict[str, Any]]] = {target: [] for target in targets}

    with ZipFile(DEFAULT_ARCHIVE) as archive:
        for record in manifest["images"]:
            source = archive.read(record["source_relative_path"])
            if _sha256_bytes(source) != record["source_sha256"]:
                raise RuntimeError(f"Source hash mismatch: {record['filename']}")
            image = _decode_grayscale_tiff(source)
            shape = tuple(record["image_shape"])
            if image.shape != shape:
                raise RuntimeError(f"Source shape mismatch: {record['filename']}")
            if record["category"] == "defective":
                label_path = record["source_relative_path"].replace("/images/", "/labels/")
                truth = _decode_label(archive.read(label_path), shape)
            else:
                truth = np.zeros(shape, dtype=np.uint8)

            prediction_path = PREDICTION_ROOT / record["prediction_path"]
            if _sha256_file(prediction_path) != record["prediction_sha256"]:
                raise RuntimeError(f"Prediction hash mismatch: {record['filename']}")
            with np.load(prediction_path, allow_pickle=False) as values:
                prediction = np.asarray(values["coherence_loss_region"], dtype=bool)
            if prediction.shape != shape:
                raise RuntimeError(f"Prediction shape mismatch: {record['filename']}")

            scale = float(record["lattice_spacing_pixels"])
            distance = _edge_distance(shape)
            truth_masks = {"any_defect": truth > 0, "class_2": truth == 2}
            for target_name, target in truth_masks.items():
                if np.any(target):
                    counts = _counts(prediction, target)
                    positive_iou[target_name].append(
                        {
                            "filename": record["filename"],
                            "iou": _metrics(counts)["iou"],
                            "carrier_scale_pixels": scale,
                        }
                    )
                for margin in MARGINS_IN_CARRIER_SCALES:
                    width = int(np.ceil(margin * scale))
                    selector = distance >= width
                    margin_counts[target_name][str(margin)] += _counts(
                        prediction[selector], target[selector]
                    )
                width = int(np.ceil(scale))
                edge = distance < width
                one_scale_band_counts[target_name]["edge"] += _counts(prediction[edge], target[edge])
                one_scale_band_counts[target_name]["interior"] += _counts(prediction[~edge], target[~edge])

    primary = {}
    for target, rows in positive_iou.items():
        ordered = sorted(rows, key=lambda row: (row["iou"], row["filename"]))
        primary[target] = {
            "positive_image_iou": _distribution([float(row["iou"]) for row in rows]),
            "five_lowest_images": ordered[:5],
            "per_image": sorted(rows, key=lambda row: row["filename"]),
        }

    return {
        "analysis_format": "phyevo-v04-primary-failure-and-edge-v1",
        "scientific_status": "post_unblinding_same_archive_descriptive_audit",
        "external_generalization_claim_allowed": False,
        "labels_read_by_analysis": True,
        "training_or_fitting_used": False,
        "detector_changed": False,
        "terminology": {
            "stored_scale": "algorithm-selected carrier scale in pixels; not a crystallographically indexed lattice spacing",
            "edge_width": "ceil(multiple × stored carrier scale) measured from the image border",
        },
        "input_provenance": {
            "archive_path": DEFAULT_ARCHIVE.relative_to(ROOT).as_posix(),
            "archive_sha256": _sha256_file(DEFAULT_ARCHIVE),
            "prediction_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "prediction_manifest_sha256": _sha256_file(manifest_path),
            "prediction_seal_path": seal_path.relative_to(ROOT).as_posix(),
            "prediction_seal_sha256": _sha256_file(seal_path),
            "all_61_sources_predictions_hash_and_shape_verified": True,
        },
        "primary_positive_image_failure": primary,
        "eroded_field_of_view_sensitivity": {
            target: {
                multiple: {
                    "border_excluded_in_carrier_scales": float(multiple),
                    "counts_tp_fp_fn_tn": [int(value) for value in counts],
                    "metrics": _metrics(counts),
                }
                for multiple, counts in margins.items()
            }
            for target, margins in margin_counts.items()
        },
        "one_carrier_scale_edge_vs_interior": {
            target: {
                band: {
                    "counts_tp_fp_fn_tn": [int(value) for value in counts],
                    "metrics": _metrics(counts),
                }
                for band, counts in bands.items()
            }
            for target, bands in one_scale_band_counts.items()
        },
        "limitations": [
            "All statistics are post-unblinding and use the same consumed 61-image archive.",
            "Border erosion changes both the evaluated prediction and target support; it is a sensitivity analysis, not a corrected performance estimate.",
            "Edge and interior pixels within one image are not independent statistical units.",
            "The stored scale is selected by the algorithm and can switch spectral branch under image operators.",
            "No alternative padding or detector rerun is evaluated here.",
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
    analysis_seal = {
        "seal_format": "phyevo-analysis-seal-v1",
        "analysis_path": output.relative_to(ROOT).as_posix(),
        "analysis_sha256": _sha256_file(output),
        "analysis_content_sha256": _json_hash(result),
        "analyzer_path": analyzer.relative_to(ROOT).as_posix(),
        "analyzer_sha256": _sha256_file(analyzer),
        "scientific_status": result["scientific_status"],
    }
    _write_json(seal_path, analysis_seal)
    print(f"Wrote {output.relative_to(ROOT)}")
    print(f"Wrote {seal_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
