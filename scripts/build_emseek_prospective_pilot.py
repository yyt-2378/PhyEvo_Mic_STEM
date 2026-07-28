#!/usr/bin/env python
"""Build a label-blind, multi-image 100-candidate expert pilot package."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from skimage.draw import disk
from skimage.io import imread

from tfree_detection.data_intake import sha256_file
from tfree_detection.governance import CandidateRecord, DataPartition
from tfree_detection.review_ui import (
    ReviewBundleItem,
    build_multisource_review_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "benchmarks/development_runs"
    / "emseek_wse2_two_teacher_consensus_v2_predictions"
)
BRANCH_KEYS = {
    "confirmed_consensus": "confirmed_candidate_xy",
    "periodic_only_uncertain": "primary_only_uncertain_xy",
    "legacy_only_uncertain": "secondary_only_uncertain_xy",
}


def _inside_project(path: Path) -> Path:
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


def _load_predictions(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["detector"] != "two_physics_teacher_site_consensus_v2":
        raise ValueError("Pilot requires the v2 two-teacher prediction set")
    if manifest["partition"] != DataPartition.TRAIN.value:
        raise ValueError("Pilot candidates must come from train")
    if manifest["label_pixels_read_during_prediction"]:
        raise ValueError("Prediction set is not label-blind")
    records = []
    content = []
    for item in manifest["prediction_files"]:
        path = root / "predictions" / f"{item['file_id']}.json"
        if sha256_file(path) != item["prediction_file_sha256"]:
            raise ValueError(f"Prediction checksum mismatch: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        records.append(record)
        content.append(item)
    if _canonical_hash(content) != manifest["prediction_set_sha256"]:
        raise ValueError("Prediction-set seal is invalid")
    return manifest, records


def _branch_candidates(
    records: list[dict[str, Any]], branch: str
) -> list[dict[str, Any]]:
    key = BRANCH_KEYS[branch]
    candidates = []
    for record in records:
        for index, point in enumerate(record[key]):
            candidates.append(
                {
                    "file_id": record["file_id"],
                    "relative_path": record["relative_path"],
                    "source_sha256": record["source_sha256"],
                    "branch": branch,
                    "branch_index": index,
                    "xy": [float(point[0]), float(point[1])],
                }
            )
    return candidates


def _spread_selection(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    generator: np.random.Generator,
) -> list[dict[str, Any]]:
    order = generator.permutation(len(candidates))
    shuffled = [candidates[int(index)] for index in order]
    selected = []
    selected_keys: set[tuple[str, int]] = set()
    per_source: dict[str, int] = {}
    cap = 1
    while len(selected) < count:
        changed = False
        for candidate in shuffled:
            key = (
                candidate["file_id"],
                int(candidate["branch_index"]),
            )
            if key in selected_keys:
                continue
            source = candidate["file_id"]
            if per_source.get(source, 0) >= cap:
                continue
            selected.append(candidate)
            selected_keys.add(key)
            per_source[source] = per_source.get(source, 0) + 1
            changed = True
            if len(selected) == count:
                break
        if not changed:
            raise ValueError("Not enough candidates for requested pilot tier")
        cap += 1
    return selected


def _normalize(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 3:
        values = np.mean(values[..., :3], axis=-1)
    if values.ndim != 2:
        raise ValueError(f"Unsupported source image shape: {values.shape}")
    values = values.astype(np.float32)
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low:
        raise ValueError("Source image has no usable intensity range")
    return np.clip((values - low) / (high - low), 0, 1)


def _random_controls(
    records: list[dict[str, Any]],
    *,
    count: int,
    generator: np.random.Generator,
    image_cache: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    if count > len(records):
        raise ValueError("Random controls require distinct source images")
    selected_indices = generator.choice(
        len(records), size=count, replace=False
    )
    controls = []
    for index in selected_indices:
        record = records[int(index)]
        image = image_cache.setdefault(
            record["file_id"],
            _normalize(
                imread(PROJECT_ROOT / record["relative_path"])
            ),
        )
        margin = min(64, max(8, min(image.shape) // 8))
        x = float(
            generator.integers(margin, image.shape[1] - margin)
        )
        y = float(
            generator.integers(margin, image.shape[0] - margin)
        )
        controls.append(
            {
                "file_id": record["file_id"],
                "relative_path": record["relative_path"],
                "source_sha256": record["source_sha256"],
                "branch": "random_sham_prompt_control",
                "branch_index": int(index),
                "xy": [x, y],
            }
        )
    return controls


def _review_items(
    selections: list[dict[str, Any]],
    *,
    pilot_id: str,
    seed: int,
    prediction_set_sha256: str,
    image_cache: dict[str, np.ndarray],
) -> tuple[list[ReviewBundleItem], list[dict[str, Any]]]:
    items = []
    private_records = []
    branch_score = {
        "confirmed_consensus": 1.0,
        "periodic_only_uncertain": 0.75,
        "legacy_only_uncertain": 0.5,
        "random_sham_prompt_control": 0.0,
    }
    for position, selection in enumerate(selections, start=1):
        source_path = PROJECT_ROOT / selection["relative_path"]
        if sha256_file(source_path) != selection["source_sha256"]:
            raise ValueError(f"Source checksum mismatch: {source_path}")
        image = image_cache.setdefault(
            selection["file_id"],
            _normalize(imread(source_path)),
        )
        x, y = selection["xy"]
        radius = 4
        center_x = int(round(x))
        center_y = int(round(y))
        bbox = (
            max(0, center_x - radius),
            max(0, center_y - radius),
            min(image.shape[1], center_x + radius + 1),
            min(image.shape[0], center_y + radius + 1),
        )
        mask = np.zeros(image.shape, dtype=bool)
        rows, columns = disk(
            (center_y, center_x), radius, shape=image.shape
        )
        mask[rows, columns] = True
        digest = hashlib.sha256(
            (
                f"{pilot_id}|{seed}|{position}|"
                f"{selection['source_sha256']}|{x:.6f}|{y:.6f}"
            ).encode("utf-8")
        ).hexdigest()[:20]
        candidate_id = f"pilot:{pilot_id}:{digest}"
        branch = selection["branch"]
        candidate = CandidateRecord(
            candidate_id=candidate_id,
            task="atomic_point_defect",
            source_id=selection["file_id"],
            dataset_id="emseek_wse2_atomic_defects",
            partition=DataPartition.TRAIN.value,
            geometry="point",
            violation_types=(branch,),
            defect_type=branch,
            bbox_xyxy=bbox,
            centroid_xy=(x, y),
            area_pixels=int(len(rows)),
            teacher_score=branch_score[branch],
            legacy_confidence=branch_score[branch],
            reliability=(
                1.0 if branch == "confirmed_consensus" else 0.5
            ),
            decision_state=(
                "confirmed"
                if branch == "confirmed_consensus"
                else "uncertain"
            ),
            label_tier=(
                "silver"
                if branch == "confirmed_consensus"
                else "unresolved"
            ),
            evidence_vector={
                "selection_branch": {
                    "value": branch_score[branch],
                    "reliability": 1.0,
                }
            },
            quality_flags=(
                ("sham_prompt_control",)
                if branch == "random_sham_prompt_control"
                else ()
            ),
            provenance={
                "selection_branch": branch,
                "parent_prediction_set_sha256": prediction_set_sha256,
                "author_labels_read": False,
            },
        )
        items.append(
            ReviewBundleItem(
                candidate=candidate,
                normalized_image=image,
                mask=mask,
                source_provenance={
                    "file_id": selection["file_id"],
                    "relative_path": selection["relative_path"],
                    "source_file_sha256": selection["source_sha256"],
                    "normalization": "whole_image_percentile_1_99",
                    "author_labels_read": False,
                },
            )
        )
        private_records.append(
            {
                "presentation_position": position,
                "candidate_id": candidate_id,
                **selection,
            }
        )
    return items, private_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", type=Path, default=DEFAULT_PREDICTIONS
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pilot-id", default="emseek-wse2-disagreement-pilot-v1"
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--per-tier", type=int, default=25)
    arguments = parser.parse_args()
    predictions = _inside_project(arguments.predictions)
    output = _inside_project(arguments.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite pilot package: {output}"
        )
    if arguments.per_tier < 1:
        raise ValueError("--per-tier must be positive")
    manifest, records = _load_predictions(predictions)
    generator = np.random.default_rng(arguments.seed)
    selected = []
    for branch in BRANCH_KEYS:
        selected.extend(
            _spread_selection(
                _branch_candidates(records, branch),
                count=arguments.per_tier,
                generator=generator,
            )
        )
    image_cache: dict[str, np.ndarray] = {}
    selected.extend(
        _random_controls(
            records,
            count=arguments.per_tier,
            generator=generator,
            image_cache=image_cache,
        )
    )
    shuffle_order = generator.permutation(len(selected))
    selected = [selected[int(index)] for index in shuffle_order]
    items, private_records = _review_items(
        selected,
        pilot_id=arguments.pilot_id,
        seed=arguments.seed,
        prediction_set_sha256=manifest["prediction_set_sha256"],
        image_cache=image_cache,
    )
    output.mkdir(parents=True)
    blinded = build_multisource_review_bundle(
        items,
        output / "blinded",
        show_model_assistance=False,
        title="PhyEvo-Mic WSe2 pilot — blinded",
    )
    assisted = build_multisource_review_bundle(
        items,
        output / "assisted",
        show_model_assistance=True,
        title="PhyEvo-Mic WSe2 pilot — assisted",
    )
    branch_counts = {
        branch: sum(
            item["branch"] == branch for item in private_records
        )
        for branch in (*BRANCH_KEYS, "random_sham_prompt_control")
    }
    private_key = {
        "pilot_key_format": "phyevo-mic-private-pilot-key-v1",
        "pilot_id": arguments.pilot_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": arguments.seed,
        "selection_policy": (
            "source-spread deterministic sampling of three teacher states "
            "plus uniform sham prompts on distinct source images"
        ),
        "prediction_set_sha256": manifest["prediction_set_sha256"],
        "label_pixels_read": False,
        "author_annotation_files_enumerated": False,
        "branch_counts": branch_counts,
        "candidate_count": len(private_records),
        "distribution_warning": (
            "Do not distribute this private key to reviewers before all "
            "responses are sealed."
        ),
        "reviewer_allocation_required": (
            "Each reviewer-candidate pair must see exactly one condition; "
            "balance assisted/blinded order across independent reviewers."
        ),
        "candidates": private_records,
    }
    private_key["content_sha256"] = _canonical_hash(private_key)
    key_path = output / "private_selection_key.json"
    key_path.write_text(
        json.dumps(private_key, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    seal = {
        "pilot_seal_format": "phyevo-mic-prospective-pilot-v1",
        "pilot_id": arguments.pilot_id,
        "candidate_count": len(items),
        "source_count": blinded["source_count"],
        "prediction_set_sha256": manifest["prediction_set_sha256"],
        "private_selection_key_sha256": sha256_file(key_path),
        "private_selection_content_sha256": private_key["content_sha256"],
        "blinded_bundle_content_sha256": blinded[
            "bundle_content_sha256"
        ],
        "assisted_bundle_content_sha256": assisted[
            "bundle_content_sha256"
        ],
        "labels_read_during_selection": False,
        "scientific_gate_complete": False,
        "pending": (
            "two independent STEM reviewers, frozen allocation, response "
            "validation, consensus compilation, and time/bias analysis"
        ),
    }
    seal["seal_content_sha256"] = _canonical_hash(seal)
    seal_path = output / "pilot_seal.json"
    seal_path.write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "pilot_id": arguments.pilot_id,
                "candidate_count": len(items),
                "source_count": blinded["source_count"],
                "branch_counts": branch_counts,
                "pilot_seal": str(
                    seal_path.relative_to(PROJECT_ROOT)
                ),
                "seal_content_sha256": seal["seal_content_sha256"],
                "labels_read_during_selection": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
