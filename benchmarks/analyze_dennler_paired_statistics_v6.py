#!/usr/bin/env python
"""Paired image-level statistics for training-free Dennler comparisons.

The released Fourier/Bragg predictions are recomputed from the checksum-fixed
archive. Stored v0.4 and final Class-1 predictions are then evaluated against
the same pixel labels. Image-level bootstrap intervals quantify descriptive
within-benchmark differences; they are not external-generalization intervals.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any
from zipfile import ZipFile

import numpy as np
import scipy
import skimage
from scipy.ndimage import label as connected_components
from scipy.stats import wilcoxon

from run_dennler_reference_baseline import (
    DEFAULT_ARCHIVE,
    EXPECTED_ARCHIVE_SHA256,
    PROJECT_ROOT,
    _decode_grayscale_tiff,
    _iter_samples,
    _predict_timed,
    _sha256_file,
)
from tfree_detection.benchmark_provenance import (
    verify_prediction_source_manifest,
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "benchmarks/paper_baselines/dennler_paired_statistics_v6.json"
)
V4_ROOT = (
    PROJECT_ROOT / "outputs/dennler_posthoc_v4_diagnostic_v2/predictions"
)
CLASS1_ROOT = (
    PROJECT_ROOT / "outputs/class1_symmetry_final_v2/predictions"
)


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Output must remain inside the project")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _archive_source_index(archive_path: Path) -> dict[str, dict[str, Any]]:
    prefixes = (
        "hrstem_defects_dataset/defective/images/",
        "hrstem_defects_dataset/non_defective/images/",
    )
    records = {}
    with ZipFile(archive_path) as archive:
        image_paths = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith((".tif", ".tiff"))
            and name.startswith(prefixes)
        )
        if len(image_paths) != 61:
            raise ValueError("Fixed archive no longer contains 61 images")
        for source_path in image_paths:
            payload = archive.read(source_path)
            image = _decode_grayscale_tiff(payload)
            records[source_path] = {
                "filename": Path(source_path).name,
                "category": (
                    "defective"
                    if "/defective/images/" in source_path
                    else "normal"
                ),
                "source_sha256": hashlib.sha256(payload).hexdigest(),
                "image_shape": list(image.shape),
            }
    return records


def _prediction_index(
    root: Path,
    *,
    archive_entries: dict[str, dict[str, Any]],
) -> tuple[dict[str, Path], dict[str, Any]]:
    manifest_path = root / "prediction_manifest.json"
    seal_path = root / "prediction_seal.json"
    manifest = _read_json(manifest_path)
    seal = _read_json(seal_path)
    if manifest.get("labels_read_during_prediction") is not False:
        raise ValueError(f"Prediction manifest is not label-blind: {root}")
    if manifest["counts"]["failed"]:
        raise ValueError(f"Prediction manifest contains failures: {root}")
    if _sha256_file(manifest_path) != seal["prediction_manifest_sha256"]:
        raise ValueError(f"Prediction manifest seal mismatch: {root}")
    if "prediction_content_sha256" in seal:
        if _json_hash(manifest) != seal["prediction_content_sha256"]:
            raise ValueError(f"Prediction content seal mismatch: {root}")
    source_provenance = verify_prediction_source_manifest(
        manifest,
        expected_archive_sha256=EXPECTED_ARCHIVE_SHA256,
        archive_entries=archive_entries,
    )
    index = {
        record["filename"]: root / record["prediction_path"]
        for record in manifest["images"]
    }
    for record in manifest["images"]:
        prediction_path = root / record["prediction_path"]
        if _sha256_file(prediction_path) != record["prediction_sha256"]:
            raise ValueError(
                f"Prediction array checksum mismatch: {prediction_path}"
            )
    provenance = {
        "root": root.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "seal_sha256": _sha256_file(seal_path),
        "protocol_id": manifest["protocol_id"],
        "scientific_status": manifest["scientific_status"],
        "implementation_sha256": manifest["implementation_sha256"],
        "config_sha256": manifest["config_sha256"],
        "array_count": len(index),
        "all_array_sha256_verified": True,
        **source_provenance,
    }
    return index, provenance


def _counts(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    prediction = np.asarray(predicted, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    return np.asarray(
        [
            np.sum(prediction & target),
            np.sum(prediction & ~target),
            np.sum(~prediction & target),
            np.sum(~prediction & ~target),
        ],
        dtype=np.int64,
    )


def _metrics(counts: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = [float(value) for value in counts]
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def _metric_array(counts: np.ndarray, metric: str) -> np.ndarray:
    tp = counts[:, 0].astype(np.float64)
    fp = counts[:, 1].astype(np.float64)
    fn = counts[:, 2].astype(np.float64)
    tn = counts[:, 3].astype(np.float64)
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
    sample_count = reference.shape[0]
    differences = {
        metric: np.empty(replicates, dtype=np.float64)
        for metric in ("iou", "f1", "precision", "recall", "specificity")
    }
    batch_size = 1_000
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, sample_count, size=(stop - start, sample_count))
        ref_sum = reference[indices].sum(axis=1)
        cur_sum = current[indices].sum(axis=1)
        for metric in differences:
            differences[metric][start:stop] = (
                _metric_array(cur_sum, metric)
                - _metric_array(ref_sum, metric)
            )
    output: dict[str, Any] = {}
    ref_metrics = _metrics(reference.sum(axis=0))
    cur_metrics = _metrics(current.sum(axis=0))
    for metric, values in differences.items():
        output[metric] = {
            "reference": ref_metrics[metric],
            "current": cur_metrics[metric],
            "difference": cur_metrics[metric] - ref_metrics[metric],
            "paired_image_bootstrap_95_ci": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
            "bootstrap_probability_difference_gt_zero": float(
                np.mean(values > 0)
            ),
        }
    return output


def _composition_fixed_bootstrap(
    reference: np.ndarray,
    current: np.ndarray,
    positive: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Resample positive and normal images separately at fixed composition."""

    positive_indices = np.flatnonzero(positive)
    normal_indices = np.flatnonzero(~positive)
    rng = np.random.default_rng(seed)
    differences = {
        metric: np.empty(replicates, dtype=np.float64)
        for metric in ("iou", "f1", "precision", "recall")
    }
    batch_size = 1_000
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        size = stop - start
        sampled_positive = rng.choice(
            positive_indices,
            size=(size, len(positive_indices)),
            replace=True,
        )
        sampled_normal = rng.choice(
            normal_indices,
            size=(size, len(normal_indices)),
            replace=True,
        )
        ref_sum = (
            reference[sampled_positive].sum(axis=1)
            + reference[sampled_normal].sum(axis=1)
        )
        cur_sum = (
            current[sampled_positive].sum(axis=1)
            + current[sampled_normal].sum(axis=1)
        )
        for metric in differences:
            differences[metric][start:stop] = (
                _metric_array(cur_sum, metric)
                - _metric_array(ref_sum, metric)
            )
    return {
        "resampling_unit": "whole image within fixed truth-positive/normal strata",
        "positive_image_count": int(positive.sum()),
        "normal_image_count": int((~positive).sum()),
        "replicates": replicates,
        "seed": seed,
        "difference_95_interval": {
            metric: [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
            for metric, values in differences.items()
        },
    }


def _leave_one_image_out(
    reference: np.ndarray,
    current: np.ndarray,
    filenames: list[str],
) -> dict[str, Any]:
    differences = []
    for index, filename in enumerate(filenames):
        keep = np.arange(len(filenames)) != index
        difference = (
            _metrics(current[keep].sum(axis=0))["iou"]
            - _metrics(reference[keep].sum(axis=0))["iou"]
        )
        differences.append((difference, filename))
    ordered = sorted(differences)
    return {
        "metric": "micro_iou_difference_current_minus_reference",
        "minimum": float(ordered[0][0]),
        "minimum_when_omitting": ordered[0][1],
        "maximum": float(ordered[-1][0]),
        "maximum_when_omitting": ordered[-1][1],
        "interpretation": (
            "Finite-benchmark influence diagnostic; not an independence or "
            "external-generalization interval."
        ),
    }


def _paired_positive_image_summary(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, Any]:
    positive = (reference[:, 0] + reference[:, 2]) > 0
    ref_iou = _metric_array(reference[positive], "iou")
    cur_iou = _metric_array(current[positive], "iou")
    delta = cur_iou - ref_iou
    try:
        test = wilcoxon(cur_iou, ref_iou, alternative="greater")
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    except ValueError:
        statistic = float("nan")
        p_value = float("nan")
    return {
        "truth_positive_image_count": int(positive.sum()),
        "reference_mean_iou": float(ref_iou.mean()),
        "current_mean_iou": float(cur_iou.mean()),
        "mean_paired_iou_difference": float(delta.mean()),
        "reference_iou_median_iqr": [
            float(np.median(ref_iou)),
            float(np.percentile(ref_iou, 25)),
            float(np.percentile(ref_iou, 75)),
        ],
        "current_iou_median_iqr": [
            float(np.median(cur_iou)),
            float(np.percentile(cur_iou, 25)),
            float(np.percentile(cur_iou, 75)),
        ],
        "current_positive_images_below_iou_0.5": int(np.sum(cur_iou < 0.5)),
        "current_wins": int(np.sum(delta > 1e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
        "current_losses": int(np.sum(delta < -1e-12)),
        "wilcoxon_greater_statistic": statistic,
        "wilcoxon_greater_p_value_descriptive": p_value,
    }


def _filename_strata(filename: str) -> dict[str, str]:
    upper = filename.upper()
    modality = (
        "HAADF"
        if "HAADF" in upper
        else "ADF"
        if "ADF" in upper
        else "BF"
        if "BF" in upper
        else "unknown"
    )
    match = re.search(r"X(\d+)M", upper)
    magnification = f"x{match.group(1)}M" if match else "unknown"
    return {"modality": modality, "magnification": magnification}


def _stratified_micro_summary(
    reference: np.ndarray,
    current: np.ndarray,
    samples: list[Any],
) -> dict[str, Any]:
    strata: dict[str, dict[str, Any]] = {}
    for field in ("modality", "magnification"):
        groups: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            value = _filename_strata(sample.filename)[field]
            groups.setdefault(value, []).append(index)
        strata[field] = {}
        for value, indices in sorted(groups.items()):
            stratum_reference = reference[indices]
            stratum_current = current[indices]
            stratum_positive = (
                stratum_reference[:, 0] + stratum_reference[:, 2]
            ) > 0
            ref_metrics = _metrics(reference[indices].sum(axis=0))
            cur_metrics = _metrics(current[indices].sum(axis=0))
            positive_summary = _paired_positive_image_summary(
                stratum_reference,
                stratum_current,
            ) if np.any(stratum_positive) else None
            strata[field][value] = {
                "image_count": len(indices),
                "truth_positive_image_count": int(stratum_positive.sum()),
                "normal_image_count": int((~stratum_positive).sum()),
                "reference": ref_metrics,
                "current": cur_metrics,
                "current_minus_reference": {
                    key: cur_metrics[key] - ref_metrics[key]
                    for key in ref_metrics
                },
                "positive_image_summary": positive_summary,
            }
    return strata


def _component_areas(mask: np.ndarray) -> np.ndarray:
    labels, count = connected_components(np.asarray(mask, dtype=bool))
    if count == 0:
        return np.asarray([], dtype=np.int64)
    return np.bincount(labels.ravel())[1:].astype(np.int64)


def _normal_area_summary(
    reference: np.ndarray,
    current: np.ndarray,
    shapes: np.ndarray,
    normal: np.ndarray,
    reference_component_counts: np.ndarray,
    current_component_counts: np.ndarray,
    reference_component_areas: list[np.ndarray],
    current_component_areas: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    pixels = shapes[:, 0] * shapes[:, 1]
    ref_fraction = reference[:, 1] / pixels
    cur_fraction = current[:, 1] / pixels
    ref_normal = ref_fraction[normal]
    cur_normal = cur_fraction[normal]
    delta = cur_normal - ref_normal
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(replicates, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    ref_areas = np.concatenate(
        [reference_component_areas[index] for index in np.flatnonzero(normal)]
    )
    cur_areas = np.concatenate(
        [current_component_areas[index] for index in np.flatnonzero(normal)]
    )

    def area_summary(values: np.ndarray) -> dict[str, float | int | None]:
        if values.size == 0:
            return {"count": 0, "median": None, "p95": None, "maximum": None}
        return {
            "count": int(values.size),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "maximum": int(values.max()),
        }

    return {
        "normal_image_count": int(normal.sum()),
        "reference_mean_false_positive_area": float(ref_normal.mean()),
        "current_mean_false_positive_area": float(cur_normal.mean()),
        "mean_paired_difference_current_minus_reference": float(delta.mean()),
        "paired_image_bootstrap_95_ci": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "normal_images_with_lower_current_area": int(np.sum(delta < 0)),
        "reference_case_level_nonempty_rate": float(np.mean(ref_normal > 0)),
        "current_case_level_nonempty_rate": float(np.mean(cur_normal > 0)),
        "reference_component_count_mean_median_p95_max": [
            float(reference_component_counts[normal].mean()),
            float(np.median(reference_component_counts[normal])),
            float(np.percentile(reference_component_counts[normal], 95)),
            int(reference_component_counts[normal].max()),
        ],
        "current_component_count_mean_median_p95_max": [
            float(current_component_counts[normal].mean()),
            float(np.median(current_component_counts[normal])),
            float(np.percentile(current_component_counts[normal], 95)),
            int(current_component_counts[normal].max()),
        ],
        "connected_component_definition": (
            "scipy.ndimage.label default 2D cross structure (4-connectivity)"
        ),
        "reference_component_area_pixels": area_summary(ref_areas),
        "current_component_area_pixels": area_summary(cur_areas),
        "interpretation": (
            "Area burden is distinct from case-level false-positive rate; "
            "both are reported."
        ),
    }


def analyze(*, workers: int, replicates: int, seed: int) -> dict[str, Any]:
    if _sha256_file(DEFAULT_ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Dennler archive checksum changed")
    archive_entries = _archive_source_index(DEFAULT_ARCHIVE)
    v4, v4_provenance = _prediction_index(
        V4_ROOT,
        archive_entries=archive_entries,
    )
    class1, class1_provenance = _prediction_index(
        CLASS1_ROOT,
        archive_entries=archive_entries,
    )
    samples = list(_iter_samples(DEFAULT_ARCHIVE))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        predictions = list(executor.map(_predict_timed, samples))

    targets = {
        "class_2_coherence_loss": {
            "truth_classes": (2,),
            "reference_classes": (2,),
            "current_root": v4,
            "current_key": "coherence_loss_region",
            "current_run": "v0.4_coherence_loss",
        },
        "any_defect": {
            "truth_classes": (1, 2),
            "reference_classes": (1, 2),
            "current_root": v4,
            "current_key": "any_defect_region",
            "current_run": "v0.4_primary_union",
            "ablation_keys": (
                "historical_secondary_region",
                "coherence_loss_region",
                "any_defect_region",
            ),
            "historical_key": "historical_secondary_region",
        },
        "class_1_symmetry_relation_separate_run": {
            "truth_classes": (1,),
            "reference_classes": (1,),
            "current_root": class1,
            "current_key": "guided_expansion",
            "current_run": "class1_final_guided_expansion",
            "ablation_keys": (
                "historical_class1",
                "symmetry_domain_region",
                "multigeometry_class1",
                "guided_expansion",
                "conservative_union",
            ),
            "historical_key": "historical_class1",
        },
    }
    names = [sample.filename for sample in samples]
    if set(names) != set(v4) or set(names) != set(class1):
        raise ValueError("Prediction manifests do not match the 61-image archive")
    shapes = np.asarray([sample.image.shape for sample in samples], dtype=np.int64)
    normal = np.asarray([sample.category == "normal" for sample in samples])
    scored: dict[str, Any] = {}
    for target_index, (target_name, spec) in enumerate(targets.items()):
        reference_counts = []
        current_counts = []
        reference_component_counts = []
        current_component_counts = []
        reference_component_areas = []
        current_component_areas = []
        ablation_counts: dict[str, list[np.ndarray]] = {
            key: [] for key in spec.get("ablation_keys", ())
        }
        ablation_component_counts: dict[str, list[int]] = {
            key: [] for key in ablation_counts
        }
        ablation_component_areas: dict[str, list[np.ndarray]] = {
            key: [] for key in ablation_counts
        }
        composition_pixels: dict[str, list[int]] = {}
        per_image = []
        for sample, (prediction, _) in zip(samples, predictions):
            truth = np.isin(sample.truth, spec["truth_classes"])
            ref_mask = np.isin(prediction, spec["reference_classes"])
            with np.load(
                spec["current_root"][sample.filename], allow_pickle=False
            ) as archive:
                current_mask = np.asarray(archive[spec["current_key"]], dtype=bool)
                ablation_masks = {
                    key: np.asarray(archive[key], dtype=bool)
                    for key in ablation_counts
                }
            ref = _counts(ref_mask, truth)
            cur = _counts(current_mask, truth)
            reference_counts.append(ref)
            current_counts.append(cur)
            reference_component_counts.append(
                int(connected_components(ref_mask)[1])
            )
            current_component_counts.append(
                int(connected_components(current_mask)[1])
            )
            reference_component_areas.append(_component_areas(ref_mask))
            current_component_areas.append(_component_areas(current_mask))
            for key, mask in ablation_masks.items():
                ablation_counts[key].append(_counts(mask, truth))
                areas = _component_areas(mask)
                ablation_component_counts[key].append(int(areas.size))
                ablation_component_areas[key].append(areas)
            if target_name == "any_defect":
                historical_mask = ablation_masks[
                    "historical_secondary_region"
                ]
                coherence_mask = ablation_masks["coherence_loss_region"]
                zones = {
                    "historical_only": historical_mask & ~coherence_mask,
                    "coherence_only": coherence_mask & ~historical_mask,
                    "overlap": historical_mask & coherence_mask,
                }
            elif target_name == "class_1_symmetry_relation_separate_run":
                historical_mask = ablation_masks["historical_class1"]
                guided_mask = ablation_masks["guided_expansion"]
                zones = {
                    "historical_seed": historical_mask,
                    "guided_added_pixels": guided_mask & ~historical_mask,
                }
            else:
                zones = {}
            for zone_name, zone in zones.items():
                values = composition_pixels.setdefault(zone_name, [0, 0])
                values[0] += int(np.sum(zone & truth))
                values[1] += int(np.sum(zone & ~truth))
            per_image.append(
                {
                    "filename": sample.filename,
                    "category": sample.category,
                    "truth_positive_fraction": float(truth.mean()),
                    "reference_positive_fraction": float(ref_mask.mean()),
                    "current_positive_fraction": float(current_mask.mean()),
                    "reference_counts_tp_fp_fn_tn": ref.tolist(),
                    "current_counts_tp_fp_fn_tn": cur.tolist(),
                    "reference_iou": _metrics(ref)["iou"],
                    "current_iou": _metrics(cur)["iou"],
                }
            )
        reference_array = np.asarray(reference_counts)
        current_array = np.asarray(current_counts)
        reference_components = np.asarray(reference_component_counts)
        current_components = np.asarray(current_component_counts)
        positive = (reference_array[:, 0] + reference_array[:, 2]) > 0
        target_seed = seed + target_index
        normal_seed = seed + 100 + target_index
        composition_seed = seed + 200 + target_index
        ablation = {}
        historical_array = None
        if spec.get("historical_key") is not None:
            historical_array = np.asarray(
                ablation_counts[spec["historical_key"]]
            )
        for ablation_index, (key, counts) in enumerate(
            ablation_counts.items()
        ):
            count_array = np.asarray(counts)
            entry = {
                "micro_metrics": _metrics(count_array.sum(axis=0)),
                "versus_fourier_reference": _paired_bootstrap(
                    reference_array,
                    count_array,
                    replicates=replicates,
                    seed=seed + 300 + target_index * 20 + ablation_index,
                ),
                "positive_image_analysis": _paired_positive_image_summary(
                    reference_array,
                    count_array,
                ),
                "normal_control_analysis": _normal_area_summary(
                    reference_array,
                    count_array,
                    shapes,
                    normal,
                    reference_components,
                    np.asarray(ablation_component_counts[key]),
                    reference_component_areas,
                    ablation_component_areas[key],
                    replicates=replicates,
                    seed=seed + 500 + target_index * 20 + ablation_index,
                ),
            }
            if historical_array is not None and key != spec["historical_key"]:
                entry["increment_over_historical"] = _paired_bootstrap(
                    historical_array,
                    count_array,
                    replicates=replicates,
                    seed=seed + 400 + target_index * 20 + ablation_index,
                )
                entry["positive_image_increment_over_historical"] = (
                    _paired_positive_image_summary(
                        historical_array,
                        count_array,
                    )
                )
            ablation[key] = entry
        scored[target_name] = {
            "current_run": spec["current_run"],
            "current_prediction_key": spec["current_key"],
            "paired_micro_bootstrap": _paired_bootstrap(
                reference_array,
                current_array,
                replicates=replicates,
                seed=target_seed,
            ),
            "paired_bootstrap_seed": target_seed,
            "composition_fixed_bootstrap": _composition_fixed_bootstrap(
                reference_array,
                current_array,
                positive,
                replicates=replicates,
                seed=composition_seed,
            ),
            "leave_one_image_out_influence": _leave_one_image_out(
                reference_array,
                current_array,
                names,
            ),
            "positive_image_analysis": _paired_positive_image_summary(
                reference_array,
                current_array,
            ),
            "stratified_micro_analysis": _stratified_micro_summary(
                reference_array,
                current_array,
                samples,
            ),
            "normal_control_analysis": _normal_area_summary(
                reference_array,
                current_array,
                shapes,
                normal,
                reference_components,
                current_components,
                reference_component_areas,
                current_component_areas,
                replicates=replicates,
                seed=normal_seed,
            ),
            "normal_bootstrap_seed": normal_seed,
            "branch_ablation": ablation,
            "composition_pixel_decomposition": {
                key: {
                    "true_positive_pixels": values[0],
                    "false_positive_pixels": values[1],
                    "precision_within_zone": (
                        values[0] / max(values[0] + values[1], 1)
                    ),
                }
                for key, values in composition_pixels.items()
            },
            "per_image": per_image,
        }
    return {
        "analysis_format": "phyevo-dennler-paired-statistics-v6",
        "scientific_status": "same_benchmark_post_development_descriptive",
        "training_or_fitting_used": False,
        "labels_read_during_prediction": False,
        "labels_read_during_scoring": True,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "input_provenance": {
            "reference_runner": (
                "benchmarks/run_dennler_reference_baseline.py"
            ),
            "reference_runner_sha256": _sha256_file(
                PROJECT_ROOT / "benchmarks/run_dennler_reference_baseline.py"
            ),
            "reference_upstream_commit": (
                "72d37c3ca5a84cfa39337c7243a80e3ffcc0f648"
            ),
            "v0.4": v4_provenance,
            "class1_separate": class1_provenance,
        },
        "runtime_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
            "platform": platform.platform(),
        },
        "image_count": len(samples),
        "defective_image_count": int(np.sum(~normal)),
        "normal_image_count": int(np.sum(normal)),
        "bootstrap": {
            "resampling_unit": "whole image",
            "paired": True,
            "replicates": replicates,
            "random_seed": seed,
            "target_seed_policy": "base + target_index",
            "normal_seed_policy": "base + 100 + target_index",
            "composition_fixed_seed_policy": (
                "base + 200 + target_index"
            ),
        },
        "connected_component_definition": (
            "scipy.ndimage.label default 2D cross structure (4-connectivity)"
        ),
        "claim_boundary": (
            "Intervals quantify paired variation within the consumed Dennler "
            "benchmark. They do not estimate unseen-material generalization."
        ),
        "targets": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    output = _inside_project(args.output)
    seal_path = output.with_suffix(".seal.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if seal_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing seal: {seal_path}")
    if args.workers < 1 or args.bootstrap_replicates < 1_000:
        raise ValueError("workers must be positive and replicates >= 1000")
    result = analyze(
        workers=args.workers,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    seal_path.write_text(
        json.dumps(
            {
                "seal_format": "phyevo-dennler-paired-statistics-seal-v2",
                "analysis_path": output.relative_to(PROJECT_ROOT).as_posix(),
                "analysis_file_sha256": _sha256_file(output),
                "analysis_content_sha256": _json_hash(result),
                "analyzer_path": (
                    "benchmarks/analyze_dennler_paired_statistics_v6.py"
                ),
                "analyzer_sha256": _sha256_file(Path(__file__).resolve()),
                "input_provenance": result["input_provenance"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        name: {
            metric: values["paired_micro_bootstrap"][metric]
            for metric in ("iou", "f1")
        }
        for name, values in result["targets"].items()
    }, indent=2))
    print(f"Wrote {output.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {seal_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
    _decode_grayscale_tiff,
