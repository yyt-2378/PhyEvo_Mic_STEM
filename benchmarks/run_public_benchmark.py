#!/usr/bin/env python
"""Reproduce qualitative checks on checksum-pinned public microscopy data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from tfree_detection.config import load_config
from tfree_detection.framework import TrainingFreeDetectionFramework


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "public"
    / "guo_2021_atomic_defects"
)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def checked_path(filename: str, expected_md5: str) -> Path:
    path = DATA_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run scripts/fetch_public_dataset.py first"
        )
    actual = md5(path)
    if actual != expected_md5:
        raise ValueError(
            f"Checksum mismatch for {path}: {actual} != {expected_md5}"
        )
    return path


def main() -> None:
    atomic_path = checked_path(
        "2H_MoWTe2_large_field_of_view_acquisition.ndata1",
        "5ba0698e02efa481ad425b9d3b1165b3",
    )
    stacking_path = checked_path(
        "relaxed_single_Al_partial_dislocations_verified.tif",
        "7f67ed89f081eab4e946c5916977f2cf",
    )
    interface_path = checked_path(
        "ZrO2_monoclinic_twin_boundary.tif",
        "06fd28497b94c5a60b2afb6f1fc9deee",
    )

    atomic = TrainingFreeDetectionFramework(
        load_config(PROJECT_ROOT / "configs" / "atomic_point_defect.yaml")
    ).detect(atomic_path)
    stacking = TrainingFreeDetectionFramework(
        load_config(
            PROJECT_ROOT / "configs" / "public_al_stacking_fault.yaml"
        )
    ).detect(stacking_path)
    interface = TrainingFreeDetectionFramework(
        load_config(PROJECT_ROOT / "configs" / "phase_boundary.yaml")
    ).detect(interface_path)

    atomic_types = Counter(region.defect_type for region in atomic.regions)
    checks = {
        "atomic_candidate_fraction_below_10_percent": (
            len(atomic.regions)
            / max(atomic.metadata["expected_site_count"], 1)
            < 0.10
        ),
        "atomic_auto_detected_multiple_sublattices": (
            atomic.metadata["inferred_sublattice_count"] >= 2
        ),
        "stacking_has_compact_candidate_set": (
            1 <= len(stacking.regions) <= 10
        ),
        "interface_has_two_crystalline_domains": (
            interface.metadata["genuine_two_phase"]
            and len(interface.regions) >= 1
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Public benchmark sanity check failed: {checks}")

    payload = {
        "dataset": {
            "doi": "10.5281/zenodo.5520169",
            "license": "CC-BY-4.0",
            "checksums_verified": True,
            "pixel_ground_truth_available": False,
        },
        "metric_scope": (
            "reproducibility and qualitative sanity only; no public-data "
            "accuracy is claimed without exhaustive machine-readable labels"
        ),
        "neural_training_used": False,
        "supervised_labels_used": False,
        "per_image_statistical_adaptation": True,
        "checks": checks,
        "tasks": {
            "atomic_point_defect": {
                "image_shape": list(atomic.image_shape),
                "expected_sites": atomic.metadata["expected_site_count"],
                "inferred_sublattices": atomic.metadata[
                    "inferred_sublattice_count"
                ],
                "reliable_image_fraction": atomic.metadata[
                    "nonrigid_registration"
                ]["reliable_image_fraction"],
                "candidate_count": len(atomic.regions),
                "candidate_fraction_of_sites": (
                    len(atomic.regions)
                    / atomic.metadata["expected_site_count"]
                ),
                "candidate_types": dict(sorted(atomic_types.items())),
            },
            "stacking_fault": {
                "image_shape": list(stacking.image_shape),
                "candidate_count": len(stacking.regions),
                "valid_border_margin_pixels": stacking.metadata[
                    "valid_border_margin_pixels"
                ],
                "candidate_boxes": [
                    list(region.bbox_xyxy) for region in stacking.regions
                ],
            },
            "phase_boundary": {
                "image_shape": list(interface.image_shape),
                "candidate_count": len(interface.regions),
                "domain_fractions": interface.metadata["domain_fractions"],
                "specimen_mask_applied": interface.metadata[
                    "specimen_mask_applied"
                ],
                "candidate_boxes": [
                    list(region.bbox_xyxy) for region in interface.regions
                ],
            },
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
