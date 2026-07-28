#!/usr/bin/env python
"""Score confirmed and uncertain site tiers as a selective prediction curve."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.io import imread

from tfree_detection.data_intake import (
    DatasetIntakePlan,
    FileRole,
    sha256_file,
)
from tfree_detection.site_matching import match_sites_lexicographic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "datasets/intake/emseek_wse2_train_v1.sealed.yaml"
)


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Path must remain inside the project: {path}"
        ) from error
    return resolved


def _truth(path: Path) -> np.ndarray:
    values = np.asarray(imread(path))
    mask = (
        values != 0
        if values.ndim == 2
        else np.any(values[..., :3] != 0, axis=-1)
    )
    components, count = ndimage.label(mask)
    if not count:
        return np.empty((0, 2), dtype=float)
    centers_yx = np.asarray(
        ndimage.center_of_mass(
            mask, components, range(1, count + 1)
        ),
        dtype=float,
    )
    return centers_yx[:, ::-1]


def _matched(
    predicted: np.ndarray, truth: np.ndarray, tolerance: float
) -> tuple[int, list[float]]:
    result = match_sites_lexicographic(
        predicted,
        truth,
        tolerance_pixels=tolerance,
    )
    return result.matched_count, result.distances.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    plan_path = _inside(arguments.plan)
    prediction_root = _inside(arguments.predictions)
    output = _inside(arguments.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite triage score: {output}"
        )
    plan = DatasetIntakePlan.from_yaml(plan_path)
    plan.validate(
        project_root=PROJECT_ROOT,
        verify_files=True,
        require_seal=True,
    )
    labels = {
        str(item.pair_id): item
        for group in plan.groups
        for item in group.files
        if item.role == FileRole.ANNOTATION
    }
    manifest_path = prediction_root / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["detector"] != "two_physics_teacher_site_consensus_v2":
        raise ValueError("Triage score requires consensus predictions")
    if manifest["plan_content_sha256"] != plan.content_hash():
        raise ValueError("Prediction plan differs from score plan")

    tiers = {
        "confirmed_consensus": {
            "description": (
                "Two independent physics teachers agree; automatic Silver "
                "candidate, not Gold."
            ),
            "predicted": 0,
            "matched": 0,
            "truth": 0,
            "distances": [],
        },
        "confirmed_plus_periodic_only": {
            "description": (
                "Add high-precision periodic-only disagreements as uncertain "
                "candidates."
            ),
            "predicted": 0,
            "matched": 0,
            "truth": 0,
            "distances": [],
        },
        "all_candidates": {
            "description": (
                "Add legacy-only high-recall disagreements; every unmatched "
                "candidate remains uncertain."
            ),
            "predicted": 0,
            "matched": 0,
            "truth": 0,
            "distances": [],
        },
    }
    branch_counts = {
        "confirmed_consensus": 0,
        "periodic_only_uncertain": 0,
        "legacy_only_uncertain": 0,
    }
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
        truth = _truth(
            PROJECT_ROOT / labels[record["pair_id"]].relative_path
        )
        confirmed = np.asarray(
            record["confirmed_candidate_xy"], dtype=float
        ).reshape(-1, 2)
        periodic_only = np.asarray(
            record["primary_only_uncertain_xy"], dtype=float
        ).reshape(-1, 2)
        legacy_only = np.asarray(
            record["secondary_only_uncertain_xy"], dtype=float
        ).reshape(-1, 2)
        branch_counts["confirmed_consensus"] += len(confirmed)
        branch_counts["periodic_only_uncertain"] += len(periodic_only)
        branch_counts["legacy_only_uncertain"] += len(legacy_only)
        predictions_by_tier = {
            "confirmed_consensus": confirmed,
            "confirmed_plus_periodic_only": np.vstack(
                (confirmed, periodic_only)
            ),
            "all_candidates": np.vstack(
                (confirmed, periodic_only, legacy_only)
            ),
        }
        tolerance = float(
            record["consensus_summary"]["tolerance_pixels"]
        )
        for name, predicted in predictions_by_tier.items():
            matched, distances = _matched(
                predicted, truth, tolerance
            )
            tiers[name]["predicted"] += len(predicted)
            tiers[name]["matched"] += matched
            tiers[name]["truth"] += len(truth)
            tiers[name]["distances"].extend(distances)

    total_pool = sum(branch_counts.values())
    result_tiers = {}
    for name, values in tiers.items():
        precision = values["matched"] / max(values["predicted"], 1)
        recall = values["matched"] / max(values["truth"], 1)
        result_tiers[name] = {
            "description": values["description"],
            "predicted_sites": values["predicted"],
            "truth_sites": values["truth"],
            "matched_sites": values["matched"],
            "site_precision": precision,
            "site_recall": recall,
            "site_f1": (
                2
                * precision
                * recall
                / max(precision + recall, 1e-12)
            ),
            "selective_risk_one_minus_precision": 1.0 - precision,
            "candidate_pool_coverage": (
                values["predicted"] / max(total_pool, 1)
            ),
            "mean_localization_error_pixels": (
                float(np.mean(values["distances"]))
                if values["distances"]
                else None
            ),
        }
    payload = {
        "score_format": "phyevo-mic-consensus-triage-v1",
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": plan.dataset_id,
        "partition": "train",
        "development_only": True,
        "independent_generalization_claim_allowed": False,
        "prediction_set_sha256": manifest["prediction_set_sha256"],
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "matching_version": "site_matching_v2",
        "branch_candidate_counts": branch_counts,
        "active_review_pool_size": (
            branch_counts["periodic_only_uncertain"]
            + branch_counts["legacy_only_uncertain"]
        ),
        "tiers": result_tiers,
    }
    payload["score_content_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
