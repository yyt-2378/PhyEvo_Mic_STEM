#!/usr/bin/env python
"""Resolve the historical candidate catalog against current local state."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from tfree_detection.data_intake import DatasetIntakePlan
from tfree_detection.governance import DataRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_BINDINGS = {
    "emseek_huggingface": {
        "dataset_ids": [
            "emseek_wse2_atomic_defects",
            "emseek_remaining_v1",
        ],
        "plans": [
            "datasets/intake/emseek_wse2_train_v1.sealed.yaml",
            "datasets/intake/emseek_balanced_subset_train_v1.sealed.yaml",
            "datasets/intake/emseek_remaining_train_v1.sealed.yaml",
        ],
    },
    "guo_2021_translational_invariance": {
        "dataset_ids": [
            "guo_2021_atomic_defects",
            "guo_2021_superscan",
        ],
        "plans": [
            "datasets/intake/guo_superscan_development_v1.sealed.yaml"
        ],
    },
    "dennler_2021_hrstem_defects": {
        "dataset_ids": ["dennler_2021_hrstem_defects"],
        "plans": [],
    },
    "al_sigma3_twin_boundaries": {
        "dataset_ids": ["al_sigma3_twin_boundaries"],
        "plans": [
            "datasets/intake/al_sigma3_development_v1.sealed.yaml"
        ],
    },
}


def _inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())
    return resolved


def _plan_status(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    plan = DatasetIntakePlan.from_yaml(path)
    plan.validate(require_seal=True)
    files = [item for group in plan.groups for item in group.files]
    present = [
        item
        for item in files
        if (PROJECT_ROOT / item.relative_path).is_file()
    ]
    files_verified = len(present) == len(files)
    if files_verified:
        plan.validate(project_root=PROJECT_ROOT, verify_files=True)
    return {
        "path": relative_path,
        "plan_id": plan.plan_id,
        "dataset_id": plan.dataset_id,
        "admission_state": plan.admission_state.value,
        "group_count": len(plan.groups),
        "file_count": len(files),
        "local_file_count": len(present),
        "all_files_local": files_verified,
        "all_local_files_verified": files_verified,
        "content_sha256": plan.content_hash(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    catalog_path = _inside(arguments.catalog)
    registry_path = _inside(arguments.registry)
    output_path = _inside(arguments.output)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite operational status: {output_path}"
        )

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    registry = DataRegistry.from_yaml(registry_path)
    asset_counts = Counter(asset.dataset_id for asset in registry.assets)
    sources = []
    for source in catalog["sources"]:
        source_id = str(source["source_id"])
        binding = SOURCE_BINDINGS.get(
            source_id, {"dataset_ids": [], "plans": []}
        )
        plans = [_plan_status(path) for path in binding["plans"]]
        registered = {
            dataset_id: int(asset_counts.get(dataset_id, 0))
            for dataset_id in binding["dataset_ids"]
        }
        local_assets = sum(registered.values())
        if source_id == "dennler_2021_hrstem_defects":
            effective_state = "consumed_locked_external_test"
        elif source_id == "emseek_huggingface":
            effective_state = "partially_admitted_remote_scope_frozen"
        elif local_assets and all(
            plan["all_files_local"] for plan in plans
        ):
            effective_state = "admitted"
        elif source["status"] == "quarantine_license":
            effective_state = "quarantine_license"
        else:
            effective_state = str(source["status"])
        sources.append(
            {
                "source_id": source_id,
                "title": source["title"],
                "catalog_status_v1": source["status"],
                "effective_state": effective_state,
                "license": source["license"],
                "registry_dataset_asset_counts": registered,
                "registered_asset_count": local_assets,
                "sealed_plans": plans,
                "stage": {
                    "catalogued": True,
                    "license_verified": bool(
                        source["license"].get("verified", False)
                    ),
                    "sealed": bool(plans)
                    and all(plan["content_sha256"] for plan in plans),
                    "all_planned_files_local": bool(plans)
                    and all(plan["all_files_local"] for plan in plans),
                    "admitted": local_assets > 0,
                },
                "human_or_external_blockers": list(
                    source.get("limitations", [])
                ),
            }
        )
    state_counts = Counter(item["effective_state"] for item in sources)
    payload = {
        "status_format": "tfree-public-dataset-operational-status-v1",
        "catalog_version": catalog["catalog_version"],
        "registry_version": registry.registry_version,
        "registry_sha256": registry.content_hash(),
        "source_count": len(sources),
        "effective_state_counts": dict(sorted(state_counts.items())),
        "sources": sources,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path.relative_to(PROJECT_ROOT)),
                "source_count": len(sources),
                "effective_state_counts": dict(
                    sorted(state_counts.items())
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
