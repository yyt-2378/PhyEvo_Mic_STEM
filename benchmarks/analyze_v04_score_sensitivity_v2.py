#!/usr/bin/env python
"""Audit v0.4 score composition and one-factor parameter sensitivity.

The analysis is explicitly retrospective and uses the already consumed
Dennler archive.  It first reconstructs the sealed score and mask without
labels, then evaluates fixed component ablations and one-factor perturbations
against the source-reference pixels.  It is a finite-archive diagnostic, not
an external-generalization estimate or a new blinded experiment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from zipfile import ZipFile

import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    gaussian_filter,
    label as connected_components,
)
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_dennler_reference_baseline import (  # noqa: E402
    DEFAULT_ARCHIVE,
    EXPECTED_ARCHIVE_SHA256,
    _decode_grayscale_tiff,
    _decode_label,
)
from tfree_detection.config import PreprocessConfig  # noqa: E402
from tfree_detection.diagnostic_v4 import (  # noqa: E402
    PosthocV4Config,
    _carrier_and_deformation_maps,
    _coherence_candidates,
    _edge_reliability,
    _lattice_model,
)
from tfree_detection.preprocessing import robust_normalize  # noqa: E402


PREDICTION_ROOT = (
    PROJECT_ROOT / "outputs/dennler_posthoc_v4_diagnostic_v2/predictions"
)
IMPLEMENTATION_PATH = PROJECT_ROOT / "src/tfree_detection/diagnostic_v4.py"
FOURIER_RESULT_PATH = (
    PROJECT_ROOT
    / "benchmarks/paper_baselines/dennler_fourier_reference_v1.json"
)
PAIRED_RESULT_PATH = (
    PROJECT_ROOT
    / "benchmarks/paper_baselines/dennler_paired_statistics_v6.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "benchmarks/paper_baselines/v04_score_sensitivity_v2.json"
)
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_SEED = 20260811


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    value: float | str
    description: str


THRESHOLDS = (0.40, 0.43, 0.46, 0.49, 0.52)
CARRIER_WEIGHTS = (0.50, 0.60, 0.68, 0.76, 0.84)
MORPHOLOGY_RADII = (0.25, 0.35, 0.45, 0.55, 0.65)
MINIMUM_REGION_CELLS = (2.5, 3.75, 5.0, 7.5, 10.0)
REFERENCE_QUANTILES = (0.70, 0.75, 0.80, 0.85, 0.90)


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Output must remain inside the project")
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
            "difference": current_metrics[metric] - reference_metrics[metric],
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


def _postprocess_score(
    score: np.ndarray,
    spacing: float,
    *,
    threshold: float,
    morphology_radius_cells: float,
    minimum_region_cells: float,
) -> np.ndarray:
    mask = np.asarray(score) >= threshold
    radius = max(1, int(round(morphology_radius_cells * spacing)))
    mask = binary_closing(mask, disk(radius))
    mask = binary_opening(mask, disk(max(1, radius // 2)))
    minimum_area = max(16, int(round(minimum_region_cells * spacing**2)))
    mask = remove_small_objects(mask, min_size=minimum_area, connectivity=2)
    return np.asarray(binary_fill_holes(mask), dtype=bool)


def _weighted_score(
    carrier_loss: np.ndarray,
    texture_loss: np.ndarray,
    spacing: float,
    carrier_weight: float,
    texture_weight: float,
) -> np.ndarray:
    return gaussian_filter(
        carrier_weight * carrier_loss + texture_weight * texture_loss,
        sigma=max(0.35 * spacing, 1.0),
        mode="reflect",
    ).astype(np.float32)


def _losses_at_quantile(
    carrier: np.ndarray,
    texture_energy: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    carrier_reference = float(np.quantile(carrier, quantile))
    texture_reference = float(np.quantile(texture_energy, quantile))
    carrier_loss = np.clip(
        1.0 - carrier / max(carrier_reference, 1e-12),
        0.0,
        1.0,
    )
    texture_loss = np.clip(
        1.0 - texture_energy / max(texture_reference, 1e-12),
        0.0,
        1.0,
    )
    return carrier_loss, texture_loss


def _variant_catalogue() -> list[Variant]:
    variants = [
        Variant("default", "default", 0.46, "sealed v0.4 score and mask"),
        Variant(
            "carrier_contribution_only",
            "component_ablation",
            "0.68*carrier_loss",
            "retain the original carrier coefficient and remove texture",
        ),
        Variant(
            "texture_contribution_only",
            "component_ablation",
            "0.32*texture_loss",
            "retain the original texture coefficient and remove carrier",
        ),
        Variant(
            "carrier_renormalized",
            "component_ablation",
            "1.0*carrier_loss",
            "carrier-only score renormalized to unit coefficient",
        ),
        Variant(
            "texture_renormalized",
            "component_ablation",
            "1.0*texture_loss",
            "texture-only score renormalized to unit coefficient",
        ),
        Variant(
            "equal_weights",
            "component_ablation",
            "0.50/0.50",
            "equal carrier and texture weights at the default threshold",
        ),
    ]
    variants.extend(
        Variant(
            f"threshold_{value:.2f}",
            "threshold",
            value,
            "one-factor score-threshold perturbation",
        )
        for value in THRESHOLDS
        if value != 0.46
    )
    variants.extend(
        Variant(
            f"carrier_weight_{value:.2f}",
            "carrier_weight",
            value,
            "one-factor carrier weight; texture weight is one minus carrier",
        )
        for value in CARRIER_WEIGHTS
        if value != 0.68
    )
    variants.extend(
        Variant(
            f"morphology_radius_{value:.2f}",
            "morphology_radius_cells",
            value,
            "one-factor morphology-radius perturbation in lattice cells",
        )
        for value in MORPHOLOGY_RADII
        if value != 0.45
    )
    variants.extend(
        Variant(
            f"minimum_region_{value:.2f}",
            "minimum_region_cells",
            value,
            "one-factor minimum-region perturbation in lattice-cell area",
        )
        for value in MINIMUM_REGION_CELLS
        if value != 5.0
    )
    variants.extend(
        Variant(
            f"reference_quantile_{value:.2f}",
            "reference_quantile",
            value,
            "one-factor shared carrier/texture reference-quantile perturbation",
        )
        for value in REFERENCE_QUANTILES
        if value != 0.80
    )
    return variants


def _masks_for_image(
    *,
    stored_score: np.ndarray,
    carrier: np.ndarray,
    texture_energy: np.ndarray,
    carrier_loss: np.ndarray,
    texture_loss: np.ndarray,
    spacing: float,
    config: PosthocV4Config,
) -> dict[str, np.ndarray]:
    defaults = {
        "threshold": config.coherence_threshold,
        "morphology_radius_cells": config.morphology_radius_cells,
        "minimum_region_cells": config.minimum_coherence_region_cells,
    }

    def mask(score: np.ndarray, **updates: float) -> np.ndarray:
        parameters = {**defaults, **updates}
        return _postprocess_score(score, spacing, **parameters)

    masks = {"default": mask(stored_score)}
    component_scores = {
        "carrier_contribution_only": _weighted_score(
            carrier_loss, texture_loss, spacing, 0.68, 0.0
        ),
        "texture_contribution_only": _weighted_score(
            carrier_loss, texture_loss, spacing, 0.0, 0.32
        ),
        "carrier_renormalized": _weighted_score(
            carrier_loss, texture_loss, spacing, 1.0, 0.0
        ),
        "texture_renormalized": _weighted_score(
            carrier_loss, texture_loss, spacing, 0.0, 1.0
        ),
        "equal_weights": _weighted_score(
            carrier_loss, texture_loss, spacing, 0.5, 0.5
        ),
    }
    masks.update({name: mask(score) for name, score in component_scores.items()})

    for threshold in THRESHOLDS:
        if threshold != config.coherence_threshold:
            masks[f"threshold_{threshold:.2f}"] = mask(
                stored_score,
                threshold=threshold,
            )
    for carrier_weight in CARRIER_WEIGHTS:
        if carrier_weight != 0.68:
            score = _weighted_score(
                carrier_loss,
                texture_loss,
                spacing,
                carrier_weight,
                1.0 - carrier_weight,
            )
            masks[f"carrier_weight_{carrier_weight:.2f}"] = mask(score)
    for radius in MORPHOLOGY_RADII:
        if radius != config.morphology_radius_cells:
            masks[f"morphology_radius_{radius:.2f}"] = mask(
                stored_score,
                morphology_radius_cells=radius,
            )
    for cells in MINIMUM_REGION_CELLS:
        if cells != config.minimum_coherence_region_cells:
            masks[f"minimum_region_{cells:.2f}"] = mask(
                stored_score,
                minimum_region_cells=cells,
            )
    for quantile in REFERENCE_QUANTILES:
        if quantile != config.carrier_reference_quantile:
            carrier_q, texture_q = _losses_at_quantile(
                carrier,
                texture_energy,
                quantile,
            )
            score = _weighted_score(
                carrier_q,
                texture_q,
                spacing,
                0.68,
                0.32,
            )
            masks[f"reference_quantile_{quantile:.2f}"] = mask(score)
    return masks


def _analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    config = PosthocV4Config(**record["config"])
    with ZipFile(DEFAULT_ARCHIVE) as archive:
        raw = archive.read(record["source_relative_path"])
        if _sha256_bytes(raw) != record["source_sha256"]:
            raise ValueError(f"Source checksum mismatch: {record['filename']}")
        image = _decode_grayscale_tiff(raw)
        expected_shape = tuple(int(value) for value in record["image_shape"])
        if image.shape != expected_shape:
            raise ValueError(
                "Source shape mismatch: "
                f"{record['filename']} decoded={image.shape} manifest={expected_shape}"
            )
        if record["category"] == "defective":
            label_path = record["source_relative_path"].replace(
                "/images/", "/labels/"
            )
            truth = _decode_label(archive.read(label_path), image.shape)
        else:
            truth = np.zeros(image.shape, dtype=np.uint8)

    prediction_path = PREDICTION_ROOT / record["prediction_path"]
    if _sha256_file(prediction_path) != record["prediction_sha256"]:
        raise ValueError(f"Prediction checksum mismatch: {record['filename']}")
    with np.load(prediction_path, allow_pickle=False) as values:
        stored_mask = np.asarray(values["coherence_loss_region"], dtype=bool)
        stored_score = np.asarray(values["coherence_loss_score"], dtype=np.float32)
        stored_reliability = np.asarray(values["reliability"], dtype=np.float32)
    for name, value in (
        ("coherence_loss_region", stored_mask),
        ("coherence_loss_score", stored_score),
        ("reliability", stored_reliability),
    ):
        if value.shape != expected_shape:
            raise ValueError(
                f"Stored {name} shape mismatch: {record['filename']} "
                f"array={value.shape} manifest={expected_shape}"
            )

    normalized, _ = robust_normalize(
        image,
        PreprocessConfig(lower_percentile=0.5, upper_percentile=99.5),
    )
    basis, direct_basis, spacing = _lattice_model(normalized, config)
    reliability = _edge_reliability(normalized.shape, spacing, config)
    carrier, _ = _carrier_and_deformation_maps(
        normalized,
        basis,
        spacing,
    )
    recomputed_mask, maps, _ = _coherence_candidates(
        normalized,
        carrier,
        spacing,
        reliability,
        config,
    )
    if spacing != float(record["lattice_spacing_pixels"]):
        raise ValueError(f"Lattice spacing mismatch: {record['filename']}")
    if not np.array_equal(recomputed_mask, stored_mask):
        raise ValueError(f"Stored mask reconstruction failed: {record['filename']}")
    if not np.array_equal(maps["coherence_loss_score"], stored_score):
        raise ValueError(f"Stored score reconstruction failed: {record['filename']}")
    if not np.array_equal(reliability, stored_reliability):
        raise ValueError(f"Stored support-prior reconstruction failed: {record['filename']}")

    masks = _masks_for_image(
        stored_score=stored_score,
        carrier=maps["carrier_amplitude"],
        texture_energy=maps["highpass_energy"],
        carrier_loss=maps["carrier_loss"],
        texture_loss=maps["highpass_loss"],
        spacing=spacing,
        config=config,
    )
    if not np.array_equal(masks["default"], stored_mask):
        raise ValueError(f"Stored score postprocessing failed: {record['filename']}")

    target_masks = {
        "class_2": truth == 2,
        "any_defect": truth > 0,
    }
    variant_counts = {
        name: {
            target: _counts(mask_value, target_value).tolist()
            for target, target_value in target_masks.items()
        }
        for name, mask_value in masks.items()
    }
    normal_controls = {
        name: {
            "positive_area_fraction": float(mask_value.mean()),
            "nonempty": bool(np.any(mask_value)),
            "raw_4_connected_components": int(
                connected_components(mask_value)[1]
            ),
        }
        for name, mask_value in masks.items()
        if record["category"] == "normal"
    }
    return {
        "filename": record["filename"],
        "category": record["category"],
        "truth_positive": {
            target: bool(np.any(target_mask))
            for target, target_mask in target_masks.items()
        },
        "variant_counts": variant_counts,
        "normal_controls": normal_controls,
        "verification": {
            "source_shape_exact": True,
            "prediction_array_shapes_exact": True,
            "stored_mask_exact": True,
            "stored_score_exact": True,
            "stored_support_prior_exact": True,
            "lattice_spacing_pixels": spacing,
            "reciprocal_basis_fy_fx": [list(value) for value in basis],
            "direct_basis_xy_columns": direct_basis.tolist(),
        },
    }


def _normal_summary(
    per_image: list[dict[str, Any]],
    variant_name: str,
) -> dict[str, Any]:
    rows = [
        record["normal_controls"][variant_name]
        for record in per_image
        if record["category"] == "normal"
    ]
    areas = np.asarray([row["positive_area_fraction"] for row in rows])
    components = np.asarray(
        [row["raw_4_connected_components"] for row in rows]
    )
    return {
        "image_count": len(rows),
        "mean_positive_area_fraction": float(areas.mean()),
        "p95_positive_area_fraction": float(np.percentile(areas, 95)),
        "maximum_positive_area_fraction": float(areas.max()),
        "nonempty_image_count": int(sum(row["nonempty"] for row in rows)),
        "nonempty_rate": float(np.mean([row["nonempty"] for row in rows])),
        "raw_4_connected_components_mean": float(components.mean()),
        "raw_4_connected_components_median": float(np.median(components)),
        "raw_4_connected_components_p95": float(
            np.percentile(components, 95)
        ),
        "raw_4_connected_components_maximum": int(components.max()),
    }


def _aggregate(
    per_image: list[dict[str, Any]],
    variants: list[Variant],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    scored: dict[str, Any] = {"class_2": {}, "any_defect": {}}
    for variant in variants:
        arrays[variant.name] = {}
        normal = _normal_summary(per_image, variant.name)
        for target in scored:
            values = np.asarray(
                [
                    record["variant_counts"][variant.name][target]
                    for record in per_image
                ],
                dtype=np.int64,
            )
            arrays[variant.name][target] = values
            positive = np.asarray(
                [record["truth_positive"][target] for record in per_image],
                dtype=bool,
            )
            per_positive_iou = _metric_array(values[positive], "iou")
            scored[target][variant.name] = {
                "micro_metrics": _metrics(values.sum(axis=0)),
                "truth_positive_image_count": int(positive.sum()),
                "positive_image_iou_mean": float(per_positive_iou.mean()),
                "positive_image_iou_median": float(
                    np.median(per_positive_iou)
                ),
                "normal_controls": normal,
            }
    return arrays, scored


def _sensitivity_summary(
    scored: dict[str, Any],
    variants: list[Variant],
    baseline: dict[str, float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    families = sorted(
        {
            variant.family
            for variant in variants
            if variant.family not in {"default", "component_ablation"}
        }
    )
    default = scored["default"]
    for family in families:
        names = ["default"] + [
            variant.name for variant in variants if variant.family == family
        ]
        iou = {
            name: scored[name]["micro_metrics"]["iou"] for name in names
        }
        area = {
            name: scored[name]["normal_controls"][
                "mean_positive_area_fraction"
            ]
            for name in names
        }
        output[family] = {
            "variant_names": names,
            "default_iou": default["micro_metrics"]["iou"],
            "iou_minimum": min(iou.values()),
            "iou_maximum": max(iou.values()),
            "iou_range": max(iou.values()) - min(iou.values()),
            "normal_mean_area_minimum": min(area.values()),
            "normal_mean_area_maximum": max(area.values()),
            "all_variants_above_fourier_reference_iou": all(
                value > baseline["iou"] for value in iou.values()
            ),
        }
    return output


def analyze(
    *,
    workers: int,
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
        raise ValueError("v0.4 prediction manifest seal mismatch")
    if _json_hash(manifest) != seal["prediction_content_sha256"]:
        raise ValueError("v0.4 prediction content seal mismatch")
    if manifest.get("labels_read_during_prediction") is not False:
        raise ValueError("v0.4 prediction manifest is not label-separated")
    if manifest["counts"]["failed"]:
        raise ValueError("v0.4 prediction manifest contains failures")
    if _sha256_file(IMPLEMENTATION_PATH) != manifest["implementation_sha256"]:
        raise ValueError("Current v0.4 implementation differs from sealed run")
    if PosthocV4Config().to_dict() != manifest["config"]:
        raise ValueError("Current v0.4 defaults differ from sealed run")

    records = [
        {**record, "config": manifest["config"]}
        for record in manifest["images"]
    ]
    if workers == 1:
        iterator = map(_analyze_record, records)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_analyze_record, records)
    per_image = []
    try:
        for index, result in enumerate(iterator, start=1):
            per_image.append(result)
            print(
                f"[{index:02d}/{len(records)}] verified {result['filename']}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    variants = _variant_catalogue()
    expected_names = {variant.name for variant in variants}
    for record in per_image:
        if set(record["variant_counts"]) != expected_names:
            raise ValueError("Variant catalogue and worker output differ")
    arrays, targets = _aggregate(per_image, variants)

    fourier = _read_json(FOURIER_RESULT_PATH)
    fourier_metrics = {
        "class_2": fourier["targets"]["full_field"]
        ["class_2_blurred_or_amorphous"]["micro_pixel_metrics"],
        "any_defect": fourier["targets"]["full_field"]
        ["any_defect"]["micro_pixel_metrics"],
    }
    paired = _read_json(PAIRED_RESULT_PATH)
    expected_default = {
        "class_2": paired["targets"]["class_2_coherence_loss"]
        ["paired_micro_bootstrap"]["iou"]["current"],
        "any_defect": paired["targets"]["any_defect"]
        ["branch_ablation"]["coherence_loss_region"]
        ["micro_metrics"]["iou"],
    }
    for target, value in expected_default.items():
        observed = targets[target]["default"]["micro_metrics"]["iou"]
        if observed != value:
            raise ValueError(
                f"Default {target} IoU differs from sealed v6 statistics"
            )

    component_names = [
        variant.name
        for variant in variants
        if variant.family == "component_ablation"
    ]
    component_differences = {
        target: {
            name: _paired_bootstrap(
                arrays["default"][target],
                arrays[name][target],
                replicates=bootstrap_replicates,
                seed=seed + 100 * target_index + name_index,
            )
            for name_index, name in enumerate(component_names, start=1)
        }
        for target_index, target in enumerate(
            ("class_2", "any_defect"), start=1
        )
    }
    sensitivity = {
        target: _sensitivity_summary(
            targets[target],
            variants,
            fourier_metrics[target],
        )
        for target in targets
    }

    return {
        "analysis_format": "phyevo-v04-score-sensitivity-v2",
        "scientific_status": (
            "retrospective_post_unblinding_same_archive_sensitivity"
        ),
        "external_generalization_claim_allowed": False,
        "training_or_fitting_used": False,
        "labels_read_by_analysis": True,
        "purpose": (
            "Reconstruct the sealed score, expose its component asymmetry, "
            "and measure one-factor sensitivity on the consumed archive."
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
            "implementation_path": IMPLEMENTATION_PATH.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "implementation_sha256": _sha256_file(IMPLEMENTATION_PATH),
            "fourier_result_path": FOURIER_RESULT_PATH.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "fourier_result_sha256": _sha256_file(FOURIER_RESULT_PATH),
            "paired_result_path": PAIRED_RESULT_PATH.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "paired_result_sha256": _sha256_file(PAIRED_RESULT_PATH),
            "all_61_source_hashes_and_shapes_verified": True,
            "all_61_prediction_array_hashes_verified": True,
            "all_61_stored_masks_scores_and_support_priors_exact": True,
        },
        "dataset": {
            "image_count": len(per_image),
            "defective_image_count": sum(
                record["category"] == "defective" for record in per_image
            ),
            "normal_image_count": sum(
                record["category"] == "normal" for record in per_image
            ),
            "class_2_positive_image_count": sum(
                record["truth_positive"]["class_2"] for record in per_image
            ),
            "any_defect_positive_image_count": sum(
                record["truth_positive"]["any_defect"] for record in per_image
            ),
        },
        "analytical_score_audit": {
            "formula": (
                "gaussian_filter(0.68*carrier_loss + "
                "0.32*texture_loss, sigma=max(0.35*spacing,1))"
            ),
            "default_threshold": 0.46,
            "carrier_loss_range": [0.0, 1.0],
            "texture_loss_range": [0.0, 1.0],
            "carrier_loss_needed_to_reach_threshold_without_texture_before_smoothing": (
                0.46 / 0.68
            ),
            "maximum_texture_contribution_before_smoothing": 0.32,
            "texture_contribution_alone_can_reach_default_threshold": False,
            "support_prior_multiplies_or_thresholds_score": False,
            "interpretation": (
                "At the default threshold, the texture term can modulate a "
                "carrier-supported candidate but cannot trigger one alone."
            ),
        },
        "variant_definitions": [asdict(variant) for variant in variants],
        "fourier_reference_metrics": fourier_metrics,
        "targets": targets,
        "component_ablation_paired_differences_from_default": (
            component_differences
        ),
        "one_factor_sensitivity_summary": sensitivity,
        "per_image_reconstruction": [
            {
                "filename": record["filename"],
                "category": record["category"],
                **record["verification"],
            }
            for record in per_image
        ],
        "resampling": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "unit": "whole image",
            "inference_scope": (
                "Descriptive sensitivity within one consumed archive; images "
                "are not asserted to be independent acquisitions."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
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
    if args.workers < 1 or args.bootstrap_replicates < 1:
        raise ValueError("Workers and bootstrap replicate count must be positive")
    result = analyze(
        workers=args.workers,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    seal = {
        "seal_format": "phyevo-analysis-seal-v2",
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
