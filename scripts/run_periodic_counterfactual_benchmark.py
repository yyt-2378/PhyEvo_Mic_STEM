#!/usr/bin/env python
"""Seal predictions, then score periodic motif evidence on admitted train data."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from skimage.io import imread

from tfree_detection.config import load_config
from tfree_detection.data_intake import (
    DatasetIntakePlan,
    FileRole,
    sha256_file,
)
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.periodic_counterfactual import (
    PeriodicCounterfactualConfig,
    detect_periodic_counterfactual,
)
from tfree_detection.site_consensus import fuse_site_consensus
from tfree_detection.site_matching import match_sites_lexicographic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "datasets"
    / "intake"
    / "emseek_wse2_train_v1.sealed.yaml"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "atomic_point_defect.yaml"


def _project_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Path must remain inside the project: {path}"
        ) from error
    return resolved


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _image_files(
    plan: DatasetIntakePlan,
) -> list[dict[str, str]]:
    images = []
    for group in plan.groups:
        for item in group.files:
            if item.role != FileRole.IMAGE:
                continue
            images.append(
                {
                    "file_id": item.file_id,
                    "pair_id": str(item.pair_id),
                    "relative_path": item.relative_path,
                    "source_sha256": item.sha256,
                    "group_id": group.group_id,
                }
            )
    images.sort(key=lambda item: item["file_id"])
    return images


def _predict_one(
    item: dict[str, str],
    config_path: str,
    motif_values: dict[str, Any],
    detector: str,
) -> dict[str, Any]:
    try:
        config = load_config(config_path)
        if detector == "periodic_counterfactual":
            motif = PeriodicCounterfactualConfig(**motif_values)
            result = detect_periodic_counterfactual(
                PROJECT_ROOT / item["relative_path"],
                config,
                motif,
            )
            spacing = float(
                result.metadata["lattice_spacing_pixels"]
            )
            translations = result.metadata["translation_vectors_yx"]
            valid_border = int(
                result.metadata["valid_border_margin_pixels"]
            )
            annotations_read = bool(
                result.metadata["annotations_read"]
            )
        elif detector == "legacy_atomic":
            result = TrainingFreeDetectionFramework(config).detect(
                PROJECT_ROOT / item["relative_path"]
            )
            spacing = float(
                np.min(
                    result.metadata["direct_basis_lengths_pixels"]
                )
            )
            translations = []
            valid_border = 0
            annotations_read = bool(
                result.metadata["supervised_labels_used"]
            )
        else:
            raise ValueError(f"Unsupported detector branch: {detector}")
        return {
            **item,
            "status": "success",
            "failure": None,
            "predicted_xy": [
                [float(value) for value in region.centroid_xy]
                for region in result.regions
            ],
            "candidate_score": [
                float(region.mean_score) for region in result.regions
            ],
            "lattice_spacing_pixels": spacing,
            "translation_vectors_yx": translations,
            "valid_border_margin_pixels": valid_border,
            "annotations_read": annotations_read,
        }
    except Exception as error:
        return {
            **item,
            "status": "failure",
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "predicted_xy": [],
            "candidate_score": [],
            "lattice_spacing_pixels": None,
            "translation_vectors_yx": [],
            "valid_border_margin_pixels": None,
            "annotations_read": False,
        }


def predict(arguments: argparse.Namespace) -> None:
    plan_path = _project_path(arguments.plan)
    config_path = _project_path(arguments.config)
    output = _project_path(arguments.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite prediction run: {output}"
        )
    plan = DatasetIntakePlan.from_yaml(plan_path)
    plan.validate(
        project_root=PROJECT_ROOT,
        verify_files=True,
        require_seal=True,
    )
    images = _image_files(plan)
    if arguments.limit is not None:
        images = images[: arguments.limit]
    if not images:
        raise ValueError("No admitted images are available")
    motif = PeriodicCounterfactualConfig()
    motif.validate()
    motif_values = {
        key: value for key, value in motif.__dict__.items()
    }
    output.mkdir(parents=True)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir()
    records = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=arguments.workers
    ) as executor:
        futures = [
            executor.submit(
                _predict_one,
                item,
                str(config_path),
                motif_values,
                arguments.detector,
            )
            for item in images
        ]
        for index, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            target = predictions_dir / f"{record['file_id']}.json"
            target.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            record["prediction_file_sha256"] = sha256_file(target)
            records.append(record)
            print(
                f"[{index}/{len(images)}] {record['status']}: "
                f"{record['file_id']}",
                flush=True,
            )
    records.sort(key=lambda item: item["file_id"])
    prediction_content = [
        {
            "file_id": item["file_id"],
            "source_sha256": item["source_sha256"],
            "status": item["status"],
            "prediction_file_sha256": item[
                "prediction_file_sha256"
            ],
        }
        for item in records
    ]
    manifest = {
        "manifest_format": "phyevo-mic-development-predictions-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": plan.dataset_id,
        "plan_id": plan.plan_id,
        "plan_content_sha256": plan.content_hash(),
        "plan_file_sha256": sha256_file(plan_path),
        "config_file": str(config_path.relative_to(PROJECT_ROOT)),
        "config_file_sha256": sha256_file(config_path),
        "detector": (
            "periodic_counterfactual_motif_v1"
            if arguments.detector == "periodic_counterfactual"
            else "legacy_atomic_lattice_registration_v0_2"
        ),
        "detector_config": (
            motif_values
            if arguments.detector == "periodic_counterfactual"
            else {}
        ),
        "detector_config_sha256": (
            motif.content_hash()
            if arguments.detector == "periodic_counterfactual"
            else sha256_file(config_path)
        ),
        "partition": "train",
        "development_only": True,
        "labels_enumerated_during_prediction": False,
        "label_pixels_read_during_prediction": False,
        "image_count": len(records),
        "success_count": sum(
            item["status"] == "success" for item in records
        ),
        "failure_count": sum(
            item["status"] == "failure" for item in records
        ),
        "prediction_files": prediction_content,
    }
    manifest["prediction_set_sha256"] = _canonical_hash(
        prediction_content
    )
    manifest_path = output / "prediction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "prediction_set_sha256": manifest[
                    "prediction_set_sha256"
                ],
                "success_count": manifest["success_count"],
                "failure_count": manifest["failure_count"],
                "manifest": str(
                    manifest_path.relative_to(PROJECT_ROOT)
                ),
            },
            indent=2,
        )
    )


def _verify_predictions(
    prediction_root: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    content = []
    records = []
    for item in manifest["prediction_files"]:
        target = (
            prediction_root
            / "predictions"
            / f"{item['file_id']}.json"
        )
        if sha256_file(target) != item["prediction_file_sha256"]:
            raise ValueError(
                f"Prediction checksum mismatch: {target}"
            )
        record = json.loads(target.read_text(encoding="utf-8"))
        if record["source_sha256"] != item["source_sha256"]:
            raise ValueError("Prediction source checksum mismatch")
        records.append(record)
        content.append(item)
    if _canonical_hash(content) != manifest["prediction_set_sha256"]:
        raise ValueError("Prediction-set seal is invalid")
    return records


def _truth_xy(label_path: Path) -> np.ndarray:
    label_image = np.asarray(imread(label_path))
    if label_image.ndim == 2:
        mask = label_image != 0
    elif label_image.ndim == 3:
        mask = np.any(label_image[..., :3] != 0, axis=-1)
    else:
        raise ValueError(
            f"Unsupported annotation shape: {label_image.shape}"
        )
    components, count = ndimage.label(mask)
    if count == 0:
        return np.empty((0, 2), dtype=float)
    centers_yx = np.asarray(
        ndimage.center_of_mass(
            mask, components, range(1, count + 1)
        ),
        dtype=float,
    )
    return centers_yx[:, ::-1]


def _match_sites(
    prediction: np.ndarray,
    truth: np.ndarray,
    tolerance: float,
) -> tuple[int, np.ndarray]:
    result = match_sites_lexicographic(
        prediction,
        truth,
        tolerance_pixels=tolerance,
    )
    return result.matched_count, result.distances


def score(arguments: argparse.Namespace) -> None:
    plan_path = _project_path(arguments.plan)
    prediction_root = _project_path(arguments.predictions)
    output = _project_path(arguments.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite development score: {output}"
        )
    plan = DatasetIntakePlan.from_yaml(plan_path)
    plan.validate(
        project_root=PROJECT_ROOT,
        verify_files=True,
        require_seal=True,
    )
    manifest_path = prediction_root / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["plan_content_sha256"] != plan.content_hash():
        raise ValueError("Prediction plan differs from scoring plan")
    records = _verify_predictions(prediction_root, manifest)

    labels_by_pair = {}
    for group in plan.groups:
        for item in group.files:
            if item.role == FileRole.ANNOTATION:
                labels_by_pair[str(item.pair_id)] = item
    per_image = []
    total_predictions = 0
    total_truth = 0
    total_matched = 0
    all_distances = []
    for record in records:
        annotation = labels_by_pair.get(record["pair_id"])
        if annotation is None:
            raise ValueError(
                f"No annotation pair for {record['pair_id']}"
            )
        truth = _truth_xy(PROJECT_ROOT / annotation.relative_path)
        predicted = np.asarray(
            record["predicted_xy"], dtype=float
        ).reshape(-1, 2)
        spacing = record["lattice_spacing_pixels"]
        tolerance = (
            0.35 * float(spacing)
            if spacing is not None
            else 1.0
        )
        matched, distances = _match_sites(
            predicted, truth, tolerance
        )
        total_predictions += len(predicted)
        total_truth += len(truth)
        total_matched += matched
        all_distances.extend(distances.tolist())
        precision = matched / max(len(predicted), 1)
        recall = matched / max(len(truth), 1)
        per_image.append(
            {
                "file_id": record["file_id"],
                "status": record["status"],
                "predicted_sites": len(predicted),
                "truth_sites": len(truth),
                "matched_sites": matched,
                "site_precision": precision,
                "site_recall": recall,
                "site_f1": (
                    2 * precision * recall
                    / max(precision + recall, 1e-12)
                ),
                "matching_tolerance_pixels": tolerance,
                "mean_localization_error_pixels": (
                    float(np.mean(distances))
                    if len(distances)
                    else None
                ),
            }
        )
    precision = total_matched / max(total_predictions, 1)
    recall = total_matched / max(total_truth, 1)
    score_payload = {
        "score_format": "phyevo-mic-development-score-v1",
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": plan.dataset_id,
        "partition": "train",
        "development_only": True,
        "independent_generalization_claim_allowed": False,
        "reason": (
            "The source group is admitted for training and the candidate "
            "quantile was developed on this source; these are not Gold "
            "calibration or locked external results."
        ),
        "target": "union_of_all_nonbackground_point_annotations",
        "palette_subclass_semantics": "not_used",
        "annotation_qualification": "author_curated_consensus_claim_without_reviewer_count",
        "prediction_set_sha256": manifest["prediction_set_sha256"],
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "failure_policy": "failed image contributes zero predicted sites",
        "matching_rule": (
            "lexicographic_max_cardinality_then_min_distance_with_"
            "0.35_lattice_spacing_tolerance"
        ),
        "matching_version": "site_matching_v2",
        "micro_site_metrics": {
            "predicted_sites": total_predictions,
            "truth_sites": total_truth,
            "matched_sites": total_matched,
            "site_precision": precision,
            "site_recall": recall,
            "site_f1": (
                2
                * precision
                * recall
                / max(precision + recall, 1e-12)
            ),
            "mean_localization_error_pixels": (
                float(np.mean(all_distances))
                if all_distances
                else None
            ),
            "p95_localization_error_pixels": (
                float(np.percentile(all_distances, 95))
                if all_distances
                else None
            ),
        },
        "macro_site_metrics": {
            "site_precision": float(
                np.mean([item["site_precision"] for item in per_image])
            ),
            "site_recall": float(
                np.mean([item["site_recall"] for item in per_image])
            ),
            "site_f1": float(
                np.mean([item["site_f1"] for item in per_image])
            ),
        },
        "image_count": len(per_image),
        "prediction_failure_count": manifest["failure_count"],
        "per_image": per_image,
    }
    score_payload["score_content_sha256"] = _canonical_hash(
        score_payload
    )
    output.mkdir(parents=True)
    target = output / "development_metrics.json"
    target.write_text(
        json.dumps(score_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "image_count": len(per_image),
                "prediction_failure_count": manifest["failure_count"],
                "micro_site_metrics": score_payload[
                    "micro_site_metrics"
                ],
                "score_content_sha256": score_payload[
                    "score_content_sha256"
                ],
                "output": str(target.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )


def fuse(arguments: argparse.Namespace) -> None:
    plan_path = _project_path(arguments.plan)
    primary_root = _project_path(arguments.primary)
    secondary_root = _project_path(arguments.secondary)
    output = _project_path(arguments.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite consensus predictions: {output}"
        )
    plan = DatasetIntakePlan.from_yaml(plan_path)
    plan.validate(
        project_root=PROJECT_ROOT,
        verify_files=True,
        require_seal=True,
    )
    primary_manifest_path = primary_root / "prediction_manifest.json"
    secondary_manifest_path = (
        secondary_root / "prediction_manifest.json"
    )
    primary_manifest = json.loads(
        primary_manifest_path.read_text(encoding="utf-8")
    )
    secondary_manifest = json.loads(
        secondary_manifest_path.read_text(encoding="utf-8")
    )
    for manifest in (primary_manifest, secondary_manifest):
        if manifest["plan_content_sha256"] != plan.content_hash():
            raise ValueError(
                "Consensus parent was produced from a different plan"
            )
    primary_records = {
        item["file_id"]: item
        for item in _verify_predictions(
            primary_root, primary_manifest
        )
    }
    secondary_records = {
        item["file_id"]: item
        for item in _verify_predictions(
            secondary_root, secondary_manifest
        )
    }
    if set(primary_records) != set(secondary_records):
        raise ValueError(
            "Consensus parents contain different image sets"
        )
    output.mkdir(parents=True)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir()
    records = []
    for index, file_id in enumerate(sorted(primary_records), start=1):
        primary = primary_records[file_id]
        secondary = secondary_records[file_id]
        if primary["source_sha256"] != secondary["source_sha256"]:
            raise ValueError(
                f"Consensus source mismatch for {file_id}"
            )
        spacing = float(primary["lattice_spacing_pixels"])
        tolerance = arguments.tolerance_fraction * spacing
        fused = fuse_site_consensus(
            np.asarray(primary["predicted_xy"], dtype=float),
            np.asarray(secondary["predicted_xy"], dtype=float),
            tolerance_pixels=tolerance,
        )
        primary_scores = np.asarray(
            primary["candidate_score"], dtype=float
        )
        secondary_scores = np.asarray(
            secondary["candidate_score"], dtype=float
        )
        consensus_scores = (
            0.5
            * (
                primary_scores[fused.primary_indices]
                + secondary_scores[fused.secondary_indices]
            )
        )
        record = {
            "file_id": file_id,
            "pair_id": primary["pair_id"],
            "relative_path": primary["relative_path"],
            "source_sha256": primary["source_sha256"],
            "group_id": primary["group_id"],
            "status": (
                "success"
                if (
                    primary["status"] == "success"
                    and secondary["status"] == "success"
                )
                else "failure"
            ),
            "failure": (
                None
                if (
                    primary["status"] == "success"
                    and secondary["status"] == "success"
                )
                else {
                    "type": "ParentPredictionFailure",
                    "message": (
                        "At least one parent detector failed; no consensus "
                        "candidate is trusted for this image."
                    ),
                }
            ),
            "predicted_xy": fused.consensus_xy.tolist(),
            "candidate_score": consensus_scores.tolist(),
            "confirmed_candidate_xy": fused.consensus_xy.tolist(),
            "primary_only_uncertain_xy": (
                fused.primary_only_xy.tolist()
            ),
            "secondary_only_uncertain_xy": (
                fused.secondary_only_xy.tolist()
            ),
            "agreement_distances_pixels": (
                fused.agreement_distances.tolist()
            ),
            "consensus_summary": fused.to_summary(),
            "lattice_spacing_pixels": spacing,
            "translation_vectors_yx": [],
            "valid_border_margin_pixels": max(
                int(primary["valid_border_margin_pixels"] or 0),
                int(secondary["valid_border_margin_pixels"] or 0),
            ),
            "annotations_read": False,
        }
        if record["status"] == "failure":
            record["predicted_xy"] = []
            record["candidate_score"] = []
            record["confirmed_candidate_xy"] = []
        target = predictions_dir / f"{file_id}.json"
        target.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record["prediction_file_sha256"] = sha256_file(target)
        records.append(record)
        print(
            f"[{index}/{len(primary_records)}] "
            f"{record['status']}: {file_id}",
            flush=True,
        )
    prediction_content = [
        {
            "file_id": item["file_id"],
            "source_sha256": item["source_sha256"],
            "status": item["status"],
            "prediction_file_sha256": item[
                "prediction_file_sha256"
            ],
        }
        for item in records
    ]
    fusion_rule = {
        "matching": (
            "lexicographic_max_cardinality_then_min_distance_"
            "within_tolerance"
        ),
        "matching_version": "site_matching_v2",
        "tolerance_fraction_of_primary_lattice_spacing": (
            arguments.tolerance_fraction
        ),
        "consensus_coordinate": "arithmetic_mean",
        "consensus_score": "arithmetic_mean",
        "unmatched_policy": "preserve_as_uncertain_not_scored_as_confirmed",
        "label_tier": "silver",
        "gold_created": False,
    }
    manifest = {
        "manifest_format": "phyevo-mic-development-predictions-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": plan.dataset_id,
        "plan_id": plan.plan_id,
        "plan_content_sha256": plan.content_hash(),
        "plan_file_sha256": sha256_file(plan_path),
        "config_file": None,
        "config_file_sha256": None,
        "detector": "two_physics_teacher_site_consensus_v2",
        "detector_config": fusion_rule,
        "detector_config_sha256": _canonical_hash(fusion_rule),
        "partition": "train",
        "development_only": True,
        "labels_enumerated_during_prediction": False,
        "label_pixels_read_during_prediction": False,
        "parent_prediction_sets": [
            {
                "role": "primary",
                "detector": primary_manifest["detector"],
                "prediction_set_sha256": primary_manifest[
                    "prediction_set_sha256"
                ],
                "manifest_sha256": sha256_file(
                    primary_manifest_path
                ),
            },
            {
                "role": "secondary",
                "detector": secondary_manifest["detector"],
                "prediction_set_sha256": secondary_manifest[
                    "prediction_set_sha256"
                ],
                "manifest_sha256": sha256_file(
                    secondary_manifest_path
                ),
            },
        ],
        "image_count": len(records),
        "success_count": sum(
            item["status"] == "success" for item in records
        ),
        "failure_count": sum(
            item["status"] == "failure" for item in records
        ),
        "prediction_files": prediction_content,
    }
    manifest["prediction_set_sha256"] = _canonical_hash(
        prediction_content
    )
    manifest_path = output / "prediction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "prediction_set_sha256": manifest[
                    "prediction_set_sha256"
                ],
                "success_count": manifest["success_count"],
                "failure_count": manifest["failure_count"],
                "manifest": str(
                    manifest_path.relative_to(PROJECT_ROOT)
                ),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    predict_parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG
    )
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--workers", type=int, default=4)
    predict_parser.add_argument("--limit", type=int, default=None)
    predict_parser.add_argument(
        "--detector",
        choices=("periodic_counterfactual", "legacy_atomic"),
        default="periodic_counterfactual",
    )

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    score_parser.add_argument(
        "--predictions", type=Path, required=True
    )
    score_parser.add_argument("--output", type=Path, required=True)
    fuse_parser = subparsers.add_parser("fuse")
    fuse_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    fuse_parser.add_argument("--primary", type=Path, required=True)
    fuse_parser.add_argument("--secondary", type=Path, required=True)
    fuse_parser.add_argument("--output", type=Path, required=True)
    fuse_parser.add_argument(
        "--tolerance-fraction", type=float, default=0.35
    )
    arguments = parser.parse_args()
    if getattr(arguments, "workers", 1) < 1:
        raise ValueError("--workers must be positive")
    if arguments.command == "predict":
        predict(arguments)
    elif arguments.command == "score":
        score(arguments)
    else:
        if arguments.tolerance_fraction <= 0:
            raise ValueError("--tolerance-fraction must be positive")
        fuse(arguments)


if __name__ == "__main__":
    main()
