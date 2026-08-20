#!/usr/bin/env python
"""Audit the retrospective Class-1 hybrid against local-expansion nulls.

This analysis uses the already consumed Dennler archive and sealed prediction
arrays.  It does not establish external validation.  Its purpose is narrower:
separate the increment of the historical-seeded hybrid from gains that arise
merely by expanding the historical mask within the same lattice-scale ring.
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
from scipy.ndimage import binary_dilation, distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_dennler_reference_baseline import (  # noqa: E402
    DEFAULT_ARCHIVE,
    EXPECTED_ARCHIVE_SHA256,
    _decode_grayscale_tiff,
    _decode_label,
)


PREDICTION_ROOT = (
    PROJECT_ROOT / "outputs/class1_symmetry_final_v2/predictions"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "benchmarks/paper_baselines/class1_local_null_models_v1.json"
)
DEFAULT_SEED = 20260811
DEFAULT_RANDOM_REPLICATES = 20_000
DEFAULT_BOOTSTRAP_REPLICATES = 20_000


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _counts(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    predicted = np.asarray(prediction, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError("Prediction and truth shapes differ")
    return np.asarray(
        [
            np.sum(predicted & target),
            np.sum(predicted & ~target),
            np.sum(~predicted & target),
            np.sum(~predicted & ~target),
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


def _metric_array(counts: np.ndarray, metric: str) -> np.ndarray:
    tp = counts[:, 0].astype(float)
    fp = counts[:, 1].astype(float)
    fn = counts[:, 2].astype(float)
    tn = counts[:, 3].astype(float)
    if metric == "iou":
        return tp / np.maximum(tp + fp + fn, 1.0)
    if metric == "f1":
        return 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1.0)
    if metric == "precision":
        return tp / np.maximum(tp + fp, 1.0)
    if metric == "recall":
        return tp / np.maximum(tp + fn, 1.0)
    if metric == "specificity":
        return tn / np.maximum(tn + fp, 1.0)
    raise ValueError(metric)


def _paired_bootstrap(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    metrics = ("iou", "f1", "precision", "recall")
    draws = {
        metric: np.empty(replicates, dtype=float) for metric in metrics
    }
    image_count = reference.shape[0]
    for start in range(0, replicates, 1_000):
        stop = min(start + 1_000, replicates)
        indices = rng.integers(
            0,
            image_count,
            size=(stop - start, image_count),
        )
        reference_sum = reference[indices].sum(axis=1)
        current_sum = current[indices].sum(axis=1)
        for metric in metrics:
            draws[metric][start:stop] = (
                _metric_array(current_sum, metric)
                - _metric_array(reference_sum, metric)
            )
    reference_metrics = _metrics(reference.sum(axis=0))
    current_metrics = _metrics(current.sum(axis=0))
    return {
        metric: {
            "reference": reference_metrics[metric],
            "current": current_metrics[metric],
            "difference": (
                current_metrics[metric] - reference_metrics[metric]
            ),
            "paired_whole_image_resampling_interval": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
            "resampling_probability_difference_gt_zero": float(
                np.mean(values > 0)
            ),
        }
        for metric, values in draws.items()
    }


def _distance_matched_expansion(
    historical: np.ndarray,
    ring: np.ndarray,
    count: int,
) -> np.ndarray:
    output = np.asarray(historical, dtype=bool).copy()
    if count <= 0:
        return output
    flat_ring = np.flatnonzero(ring)
    if count > flat_ring.size:
        raise ValueError("Requested expansion exceeds the available ring")
    distance = distance_transform_edt(~historical).ravel()[flat_ring]
    order = np.lexsort((flat_ring, distance))
    output.ravel()[flat_ring[order[:count]]] = True
    return output


def _load_truth(
    archive: ZipFile,
    record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    raw = archive.read(record["source_relative_path"])
    if _sha256_bytes(raw) != record["source_sha256"]:
        raise ValueError(f"Source checksum mismatch: {record['filename']}")
    image = _decode_grayscale_tiff(raw)
    if list(image.shape) != record["image_shape"]:
        raise ValueError(f"Source shape mismatch: {record['filename']}")
    if record["category"] == "defective":
        label_path = record["source_relative_path"].replace(
            "/images/", "/labels/"
        )
        truth = _decode_label(archive.read(label_path), image.shape) == 1
    else:
        truth = np.zeros(image.shape, dtype=bool)
    return image, truth


def analyze(
    *,
    random_replicates: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    if _sha256_file(DEFAULT_ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Dennler archive checksum changed")
    manifest_path = PREDICTION_ROOT / "prediction_manifest.json"
    seal_path = PREDICTION_ROOT / "prediction_seal.json"
    manifest = _read_json(manifest_path)
    seal = _read_json(seal_path)
    if _sha256_file(manifest_path) != seal["prediction_manifest_sha256"]:
        raise ValueError("Class-1 prediction manifest seal mismatch")
    if manifest.get("labels_read_during_prediction") is not False:
        raise ValueError("Class-1 prediction manifest is not label-separated")
    if manifest["counts"]["failed"]:
        raise ValueError("Class-1 prediction manifest contains failures")

    method_names = (
        "historical_h",
        "proxy_g",
        "stored_hybrid_p",
        "pure_dilation_h",
        "distance_matched_expansion",
    )
    counts: dict[str, list[np.ndarray]] = {
        name: [] for name in method_names
    }
    filenames: list[str] = []
    positive: list[bool] = []
    ambiguous: list[bool] = []
    ring_statistics: list[dict[str, int | float | str | bool]] = []
    random_parameters: list[tuple[int, int, int, int, int]] = []

    with ZipFile(DEFAULT_ARCHIVE) as archive:
        for record in manifest["images"]:
            if record["status"] != "success":
                raise ValueError(f"Unexpected failed case: {record['filename']}")
            prediction_path = PREDICTION_ROOT / record["prediction_path"]
            if _sha256_file(prediction_path) != record["prediction_sha256"]:
                raise ValueError(
                    f"Prediction checksum mismatch: {record['filename']}"
                )
            image, truth = _load_truth(archive, record)
            with np.load(prediction_path, allow_pickle=False) as values:
                historical = np.asarray(
                    values["historical_class1"], dtype=bool
                )
                proxy = np.asarray(
                    values["multigeometry_class1"], dtype=bool
                )
                hybrid = np.asarray(values["guided_expansion"], dtype=bool)
            if historical.shape != image.shape:
                raise ValueError(f"Prediction shape mismatch: {record['filename']}")
            radius = max(
                1,
                int(round(float(record["lattice_spacing_pixels"]))),
            )
            dilation = binary_dilation(historical, iterations=radius)
            ring = dilation & ~historical
            expected_hybrid = historical | (proxy & ring)
            if not np.array_equal(expected_hybrid, hybrid):
                raise ValueError(
                    f"Stored hybrid formula mismatch: {record['filename']}"
                )
            new_pixels = hybrid & ~historical
            new_count = int(new_pixels.sum())
            distance_matched = _distance_matched_expansion(
                historical,
                ring,
                new_count,
            )
            masks = {
                "historical_h": historical,
                "proxy_g": proxy,
                "stored_hybrid_p": hybrid,
                "pure_dilation_h": dilation,
                "distance_matched_expansion": distance_matched,
            }
            for name, mask in masks.items():
                counts[name].append(_counts(mask, truth))
            filenames.append(record["filename"])
            positive.append(bool(np.any(truth)))
            ambiguous.append(
                bool(record["reference_selection"]["ambiguous_identity"])
            )
            ring_true = int(np.sum(ring & truth))
            ring_size = int(ring.sum())
            historical_counts = counts["historical_h"][-1]
            random_parameters.append(
                (
                    int(historical_counts[0]),
                    int(historical_counts[1]),
                    int(truth.sum()),
                    ring_true,
                    ring_size,
                )
            )
            ring_statistics.append(
                {
                    "filename": record["filename"],
                    "radius_pixels": radius,
                    "ring_pixels": ring_size,
                    "source_truth_pixels_in_ring": ring_true,
                    "hybrid_new_pixels": new_count,
                    "hybrid_new_true_positive_pixels": int(
                        np.sum(new_pixels & truth)
                    ),
                    "ambiguous_identity": ambiguous[-1],
                }
            )

    arrays = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in counts.items()
    }
    metrics = {
        name: _metrics(values.sum(axis=0))
        for name, values in arrays.items()
    }
    positive_array = np.asarray(positive, dtype=bool)
    ambiguous_array = np.asarray(ambiguous, dtype=bool)

    rng = np.random.default_rng(seed)
    random_counts = np.zeros((random_replicates, 4), dtype=np.int64)
    total_pixels = int(sum(np.sum(row) for row in arrays["historical_h"]))
    total_truth = int(
        sum(tp + fn for tp, _, fn, _ in arrays["historical_h"])
    )
    for image_index, (h_tp, h_fp, _, ring_true, ring_size) in enumerate(
        random_parameters
    ):
        new_count = int(ring_statistics[image_index]["hybrid_new_pixels"])
        if new_count > ring_size:
            raise ValueError("Hybrid expansion exceeds dilation ring")
        sampled_true = rng.hypergeometric(
            ring_true,
            ring_size - ring_true,
            new_count,
            size=random_replicates,
        )
        sampled_false = new_count - sampled_true
        random_counts[:, 0] += h_tp + sampled_true
        random_counts[:, 1] += h_fp + sampled_false
    random_counts[:, 2] = total_truth - random_counts[:, 0]
    random_counts[:, 3] = (
        total_pixels - total_truth - random_counts[:, 1]
    )
    random_metric_values = {
        metric: _metric_array(random_counts, metric)
        for metric in ("iou", "f1", "precision", "recall")
    }

    def stratum_metrics(selector: np.ndarray) -> dict[str, Any]:
        return {
            name: {
                "image_count": int(selector.sum()),
                "metrics": _metrics(values[selector].sum(axis=0)),
            }
            for name, values in arrays.items()
        }

    positive_image_iou = {
        name: _metric_array(values[positive_array], "iou")
        for name, values in arrays.items()
    }
    return {
        "analysis_format": "phyevo-class1-local-null-models-v1",
        "scientific_status": (
            "retrospective_post_unblinding_same_archive_null_analysis"
        ),
        "external_generalization_claim_allowed": False,
        "training_or_fitting_used": False,
        "labels_read_by_analysis": True,
        "purpose": (
            "Test whether the stored H-seeded Class-1 increment exceeds "
            "local expansion nulls on the already consumed archive."
        ),
        "input_provenance": {
            "archive_path": DEFAULT_ARCHIVE.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "archive_sha256": _sha256_file(DEFAULT_ARCHIVE),
            "prediction_manifest_path": manifest_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "prediction_manifest_sha256": _sha256_file(manifest_path),
            "prediction_seal_path": seal_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "prediction_seal_sha256": _sha256_file(seal_path),
            "all_prediction_array_hashes_verified": True,
            "stored_hybrid_formula_verified_for_all_images": True,
        },
        "dataset": {
            "image_count": len(filenames),
            "class1_positive_image_count": int(positive_array.sum()),
            "nominal_normal_image_count": int((~positive_array).sum()),
            "ambiguous_identity_image_count": int(ambiguous_array.sum()),
        },
        "method_definitions": {
            "historical_h": "stored historical Class-1 seed H",
            "proxy_g": "stored multigeometry proxy G without H",
            "stored_hybrid_p": "H OR [G AND dilate(H, r)]",
            "pure_dilation_h": "dilate(H, r)",
            "distance_matched_expansion": (
                "H plus the same per-image number of new pixels as P, "
                "selected nearest to H within the identical dilation ring"
            ),
            "matched_random_ring": (
                "H plus the same per-image number of new pixels as P, "
                "sampled uniformly without replacement from the identical ring"
            ),
        },
        "micro_metrics": metrics,
        "paired_whole_image_descriptive_differences": {
            "p_minus_h": _paired_bootstrap(
                arrays["historical_h"],
                arrays["stored_hybrid_p"],
                replicates=bootstrap_replicates,
                seed=seed + 101,
            ),
            "p_minus_pure_dilation": _paired_bootstrap(
                arrays["pure_dilation_h"],
                arrays["stored_hybrid_p"],
                replicates=bootstrap_replicates,
                seed=seed + 102,
            ),
            "p_minus_distance_matched": _paired_bootstrap(
                arrays["distance_matched_expansion"],
                arrays["stored_hybrid_p"],
                replicates=bootstrap_replicates,
                seed=seed + 103,
            ),
            "g_minus_h": _paired_bootstrap(
                arrays["historical_h"],
                arrays["proxy_g"],
                replicates=bootstrap_replicates,
                seed=seed + 104,
            ),
        },
        "matched_random_ring_fixed_archive_randomization": {
            "replicates": random_replicates,
            "seed": seed,
            "metrics": {
                metric: {
                    "random_mean": float(values.mean()),
                    "random_95_interval": [
                        float(np.percentile(values, 2.5)),
                        float(np.percentile(values, 97.5)),
                    ],
                    "stored_hybrid_p": metrics["stored_hybrid_p"][metric],
                    "p_minus_random_mean": float(
                        (metrics["stored_hybrid_p"][metric] - values).mean()
                    ),
                    "p_minus_random_95_interval": [
                        float(
                            np.percentile(
                                metrics["stored_hybrid_p"][metric] - values,
                                2.5,
                            )
                        ),
                        float(
                            np.percentile(
                                metrics["stored_hybrid_p"][metric] - values,
                                97.5,
                            )
                        ),
                    ],
                    "randomization_probability_p_gt_random": float(
                        np.mean(metrics["stored_hybrid_p"][metric] > values)
                    ),
                }
                for metric, values in random_metric_values.items()
            },
            "interpretation": (
                "Randomization uncertainty holds the archive, H, ring and "
                "per-image added area fixed; it is not an external-sample CI."
            ),
        },
        "positive_image_pairwise_wins": {
            comparison: {
                "current_wins": int(np.sum(current > reference)),
                "ties": int(np.sum(current == reference)),
                "current_losses": int(np.sum(current < reference)),
            }
            for comparison, (reference, current) in {
                "p_vs_h": (
                    positive_image_iou["historical_h"],
                    positive_image_iou["stored_hybrid_p"],
                ),
                "p_vs_pure_dilation": (
                    positive_image_iou["pure_dilation_h"],
                    positive_image_iou["stored_hybrid_p"],
                ),
                "p_vs_distance_matched": (
                    positive_image_iou["distance_matched_expansion"],
                    positive_image_iou["stored_hybrid_p"],
                ),
                "g_vs_h": (
                    positive_image_iou["historical_h"],
                    positive_image_iou["proxy_g"],
                ),
            }.items()
        },
        "reference_ambiguity_strata": {
            "ambiguous": stratum_metrics(ambiguous_array),
            "not_ambiguous": stratum_metrics(~ambiguous_array),
        },
        "per_image_ring_statistics": ring_statistics,
        "resampling": {
            "bootstrap_replicates": bootstrap_replicates,
            "random_ring_replicates": random_replicates,
            "seed": seed,
            "inference_scope": (
                "Descriptive sensitivity within one consumed archive; images "
                "are not asserted to be independent acquisitions."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--random-replicates",
        type=int,
        default=DEFAULT_RANDOM_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    output = _inside_project(args.output)
    seal_path = output.with_name(output.stem + ".seal.json")
    if output.exists() or seal_path.exists():
        raise FileExistsError("Refusing to overwrite an existing analysis/seal")
    if args.random_replicates < 1 or args.bootstrap_replicates < 1:
        raise ValueError("Replicate counts must be positive")
    result = analyze(
        random_replicates=args.random_replicates,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    seal = {
        "seal_format": "phyevo-analysis-seal-v1",
        "analysis_path": output.relative_to(PROJECT_ROOT).as_posix(),
        "analysis_sha256": _sha256_file(output),
        "analysis_content_sha256": _json_hash(result),
        "analyzer_path": Path(__file__).resolve().relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "analyzer_sha256": _sha256_file(Path(__file__).resolve()),
        "scientific_status": result["scientific_status"],
    }
    _write_json(seal_path, seal)
    print(f"Wrote {output.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {seal_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
