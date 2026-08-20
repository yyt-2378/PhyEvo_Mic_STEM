#!/usr/bin/env python3
"""Audit per-image and severity-ladder behaviour of the sealed v0.4 stresses.

This analysis does not rerun the detector and does not introduce a new
operating point. It reads the already sealed counterfactual prediction and
score records, separates empty-empty mask pairs from evaluable per-image
stability, reconstructs the common 13-image severity ladder, and records
carrier-scale switches. Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "benchmarks/paper_baselines/v04_physical_counterfactual_score_v1.json"
SCORE_SEAL = ROOT / "benchmarks/paper_baselines/v04_physical_counterfactual_score_v1.seal.json"
PREDICTION_ROOT = ROOT / "benchmarks/paper_baselines/v04_physical_counterfactual_predictions_v1"
DEFAULT_OUTPUT = ROOT / "benchmarks/paper_baselines/v04_counterfactual_per_image_audit_v1.json"
SPACING_MAX = 28.0
FAMILIES = (
    "transfer_loss_proxy",
    "count_noise_proxy",
    "linear_drift_proxy",
    "scan_jitter_proxy",
    "field_edge_truncation",
)
LEVELS = ("low", "nominal", "high")


def _inside_root(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _metrics(confusion: Iterable[int]) -> dict[str, float]:
    tp, fp, fn, tn = (int(value) for value in confusion)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "iou": float(tp / max(tp + fp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
    }


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "minimum": None,
            "q1": None,
            "median": None,
            "q3": None,
            "maximum": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "minimum": float(np.min(array)),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q3": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _iou_or_none(confusion: Iterable[int]) -> float | None:
    tp, fp, fn, _ = (int(value) for value in confusion)
    union = tp + fp + fn
    return None if union == 0 else float(tp / union)


def _target_iou(confusion: Iterable[int]) -> float:
    tp, fp, fn, _ = (int(value) for value in confusion)
    return float(tp / max(tp + fp + fn, 1))


def _sum_confusions(rows: list[dict[str, Any]], key: str) -> list[int]:
    return (
        np.sum(
            np.asarray([row[key] for row in rows], dtype=np.int64),
            axis=0,
            dtype=np.int64,
        )
        .astype(int)
        .tolist()
    )


def _validate_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    score = json.loads(SCORE.read_text(encoding="utf-8"))
    score_seal = json.loads(SCORE_SEAL.read_text(encoding="utf-8"))
    if score_seal["score_sha256"] != _sha256(SCORE):
        raise RuntimeError("Counterfactual score file hash mismatch")
    if score_seal["score_content_sha256"] != _canonical_hash(score):
        raise RuntimeError("Counterfactual score content mismatch")

    manifest_path = PREDICTION_ROOT / "prediction_manifest.json"
    prediction_seal_path = PREDICTION_ROOT / "prediction_seal.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_seal = json.loads(prediction_seal_path.read_text(encoding="utf-8"))
    if prediction_seal["prediction_manifest_sha256"] != _sha256(manifest_path):
        raise RuntimeError("Counterfactual prediction manifest hash mismatch")
    if prediction_seal["prediction_content_sha256"] != _canonical_hash(manifest):
        raise RuntimeError("Counterfactual prediction content mismatch")
    if manifest["counts"] != {
        "images": 61,
        "variant_attempts": 435,
        "successful_variants": 435,
        "failed_variants": 0,
    }:
        raise RuntimeError("Counterfactual prediction set is incomplete")
    for record in manifest["images"]:
        array_path = PREDICTION_ROOT / record["array_path"]
        if _sha256(array_path) != record["array_sha256"]:
            raise RuntimeError(f"Prediction array hash mismatch: {array_path}")
    return score, manifest


def _per_image_nominal(variant: dict[str, Any]) -> dict[str, Any]:
    rows = variant["per_image"]
    evaluable: list[tuple[dict[str, Any], float]] = []
    legacy_values: list[float] = []
    both_empty = 0
    one_empty = 0
    baseline_nonempty = 0
    prediction_nonempty = 0

    for row in rows:
        confusion = row["stability_confusion_tp_fp_fn_tn"]
        tp, fp, fn, _ = (int(value) for value in confusion)
        baseline_count = tp + fn
        prediction_count = tp + fp
        baseline_nonempty += baseline_count > 0
        prediction_nonempty += prediction_count > 0
        if baseline_count == 0 and prediction_count == 0:
            both_empty += 1
        if (baseline_count == 0) != (prediction_count == 0):
            one_empty += 1
        legacy_values.append(float(tp / max(tp + fp + fn, 1)))
        iou = _iou_or_none(confusion)
        if iou is not None:
            evaluable.append((row, iou))

    evaluable.sort(key=lambda item: (item[1], item[0]["filename"]))
    target_strata: dict[str, Any] = {}
    for target, current_key, baseline_key in (
        ("any_defect", "any_confusion_tp_fp_fn_tn", "baseline_any_confusion_tp_fp_fn_tn"),
        ("class_2", "class2_confusion_tp_fp_fn_tn", "baseline_class2_confusion_tp_fp_fn_tn"),
    ):
        positive_rows = [
            row
            for row in rows
            if int(row[baseline_key][0]) + int(row[baseline_key][2]) > 0
        ]
        deltas = [
            _target_iou(row[current_key]) - _target_iou(row[baseline_key])
            for row in positive_rows
        ]
        target_strata[target] = {
            "target_positive_images": len(positive_rows),
            "per_image_delta_iou": _summarize(deltas),
        }

    defective_iou = [iou for row, iou in evaluable if row["category"] == "defective"]
    normal_iou = [iou for row, iou in evaluable if row["category"] == "normal"]
    return {
        "images": len(rows),
        "legacy_zero_when_both_empty_summary": _summarize(legacy_values),
        "empty_pair_accounting": {
            "both_masks_empty": both_empty,
            "exactly_one_mask_empty": one_empty,
            "baseline_mask_nonempty": baseline_nonempty,
            "perturbed_mask_nonempty": prediction_nonempty,
            "evaluable_nonempty_union": len(evaluable),
        },
        "evaluable_per_image_stability_iou": _summarize([iou for _, iou in evaluable]),
        "evaluable_below_threshold_counts": {
            "below_0.90": sum(iou < 0.90 for _, iou in evaluable),
            "below_0.75": sum(iou < 0.75 for _, iou in evaluable),
            "below_0.50": sum(iou < 0.50 for _, iou in evaluable),
        },
        "category_strata_evaluable_stability": {
            "defective": _summarize(defective_iou),
            "normal": _summarize(normal_iou),
        },
        "target_positive_per_image_deltas": target_strata,
        "lowest_evaluable_images": [
            {
                "image_id": row["image_id"],
                "filename": row["filename"],
                "category": row["category"],
                "stability_iou": iou,
                "baseline_positive_pixels": int(row["stability_confusion_tp_fp_fn_tn"][0])
                + int(row["stability_confusion_tp_fp_fn_tn"][2]),
                "perturbed_positive_pixels": int(row["stability_confusion_tp_fp_fn_tn"][0])
                + int(row["stability_confusion_tp_fp_fn_tn"][1]),
            }
            for row, iou in evaluable[:5]
        ],
    }


def _aggregate_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stability = _sum_confusions(rows, "stability_confusion_tp_fp_fn_tn")
    result: dict[str, Any] = {
        "images": len(rows),
        "mask_self_consistency": _metrics(stability),
    }
    for target, current_key, baseline_key in (
        ("any_defect", "any_confusion_tp_fp_fn_tn", "baseline_any_confusion_tp_fp_fn_tn"),
        ("class_2", "class2_confusion_tp_fp_fn_tn", "baseline_class2_confusion_tp_fp_fn_tn"),
    ):
        current = _metrics(_sum_confusions(rows, current_key))
        baseline = _metrics(_sum_confusions(rows, baseline_key))
        result[target] = {
            "current": current,
            "aligned_baseline": baseline,
            "delta_iou": float(current["iou"] - baseline["iou"]),
        }
    return result


def _common_ladder(score: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in FAMILIES:
        low_rows = score["variants"][f"{family}__low"]["per_image"]
        common_ids = {row["image_id"] for row in low_rows}
        if len(common_ids) != 13:
            raise RuntimeError(f"Unexpected ladder subset for {family}")
        levels: dict[str, Any] = {}
        for level in LEVELS:
            variant = score["variants"][f"{family}__{level}"]
            rows = [row for row in variant["per_image"] if row["image_id"] in common_ids]
            if len(rows) != 13:
                raise RuntimeError(f"Incomplete common ladder: {family} {level}")
            levels[level] = {
                "operator_value": variant["value"],
                **_aggregate_variant_rows(rows),
            }
        output[family] = {
            "common_subset_images": 13,
            "selection": "zero-based stored-manifest index divisible by five; label independent",
            "levels": levels,
        }
    return output


def _spacing_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    for image in manifest["images"]:
        baseline = float(image["baseline_lattice_spacing_pixels"])
        for variant in image["variants"]:
            if variant["status"] != "success":
                continue
            reestimated = float(variant["reestimated_lattice_spacing_pixels"])
            relative = float(abs(reestimated - baseline) / max(baseline, 1e-12))
            record = {
                "variant": variant["variant"],
                "family": variant["family"],
                "level": variant["level"],
                "image_id": image["image_id"],
                "filename": image["filename"],
                "baseline_spacing_pixels": baseline,
                "reestimated_spacing_pixels": reestimated,
                "absolute_relative_change": relative,
                "reestimated_at_upper_clip": bool(reestimated >= SPACING_MAX - 1e-9),
            }
            by_variant.setdefault(variant["variant"], []).append(record)
            all_records.append(record)

    summaries: dict[str, Any] = {}
    for variant, records in sorted(by_variant.items()):
        changes = [record["absolute_relative_change"] for record in records]
        summaries[variant] = {
            "images": len(records),
            "absolute_relative_change": _summarize(changes),
            "change_over_20_percent": sum(value > 0.20 for value in changes),
            "reestimated_at_upper_clip": sum(record["reestimated_at_upper_clip"] for record in records),
        }
    worst = sorted(
        all_records,
        key=lambda record: (-record["absolute_relative_change"], record["variant"], record["filename"]),
    )[:12]
    return {
        "configured_upper_clip_pixels": SPACING_MAX,
        "variant_summaries": summaries,
        "largest_spacing_switches": worst,
    }


def analyze(output: str | Path) -> None:
    destination = _inside_root(Path(output))
    seal_path = destination.with_suffix(".seal.json")
    if destination.exists() or seal_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination} or {seal_path}")
    score, manifest = _validate_inputs()

    nominal = {
        family: _per_image_nominal(score["variants"][f"{family}__nominal"])
        for family in FAMILIES
    }
    payload = {
        "result_format": "phyevo-mic-v04-counterfactual-per-image-audit-v1",
        "scientific_status": "retrospective_consumed_archive_image_operator_audit",
        "external_generalization_claimed": False,
        "physical_microscope_tolerance_claimed": False,
        "inputs": {
            "score_path": str(SCORE.relative_to(ROOT)),
            "score_sha256": _sha256(SCORE),
            "prediction_manifest_path": str(
                (PREDICTION_ROOT / "prediction_manifest.json").relative_to(ROOT)
            ),
            "prediction_manifest_sha256": _sha256(PREDICTION_ROOT / "prediction_manifest.json"),
        },
        "interpretation_rules": {
            "mask_stability_term": "mask self-consistency with the aligned stored unperturbed v0.4 mask",
            "empty_empty_rule": "per-image stability is undefined when both masks are empty and is excluded from evaluable distribution summaries",
            "legacy_micro_rule": "micro summaries remain pixel weighted and must be shown with per-image distributions",
            "multiplicity": "15 image-operator variants; two source-reference targets plus mask self-consistency; intervals are exploratory and unadjusted",
        },
        "nominal_per_image": nominal,
        "common_13_image_severity_ladder": _common_ladder(score),
        "spacing_reestimation": _spacing_audit(manifest),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, payload)
    seal = {
        "seal_format": "phyevo-mic-v04-counterfactual-per-image-audit-seal-v1",
        "analysis_path": str(destination.relative_to(ROOT)),
        "analysis_sha256": _sha256(destination),
        "analysis_content_sha256": _canonical_hash(payload),
        "analyzer_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "analyzer_sha256": _sha256(Path(__file__).resolve()),
        "source_score_sha256": _sha256(SCORE),
        "prediction_manifest_sha256": _sha256(PREDICTION_ROOT / "prediction_manifest.json"),
    }
    _write_json(seal_path, seal)
    print(json.dumps({"analysis": str(destination), "seal": str(seal_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    analyze(args.output)


if __name__ == "__main__":
    main()
