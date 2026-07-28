#!/usr/bin/env python
"""Seal, verify, and admit prospectively grouped external datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfree_detection.data_intake import (
    DatasetIntakePlan,
    merge_plan_into_registry,
    merge_plans_into_registry,
    seal_plan,
    write_plan,
    write_registry,
)
from tfree_detection.governance import DataRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Path must remain inside the project: {path}"
        ) from error
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--draft", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--plan", type=Path, required=True)
    verify_parser.add_argument("--verify-files", action="store_true")

    admit_parser = subparsers.add_parser("admit")
    admit_parser.add_argument("--plan", type=Path, required=True)
    admit_parser.add_argument("--base-registry", type=Path, required=True)
    admit_parser.add_argument("--output-registry", type=Path, required=True)
    admit_parser.add_argument("--registry-version", required=True)
    admit_parser.add_argument("--allow-exact-existing", action="store_true")
    admit_many_parser = subparsers.add_parser("admit-many")
    admit_many_parser.add_argument(
        "--plan",
        type=Path,
        action="append",
        required=True,
        help="sealed plan; repeat in the intended admission order",
    )
    admit_many_parser.add_argument(
        "--base-registry", type=Path, required=True
    )
    admit_many_parser.add_argument(
        "--output-registry", type=Path, required=True
    )
    admit_many_parser.add_argument("--registry-version", required=True)
    admit_many_parser.add_argument(
        "--allow-exact-existing", action="store_true"
    )
    arguments = parser.parse_args()

    if arguments.command == "seal":
        draft_path = _inside_project(arguments.draft)
        output_path = _inside_project(arguments.output)
        draft = DatasetIntakePlan.from_yaml(draft_path)
        sealed = seal_plan(draft)
        write_plan(sealed, output_path)
        print(
            json.dumps(
                {
                    "sealed": True,
                    "plan_id": sealed.plan_id,
                    "content_sha256": sealed.content_hash(),
                    "output": str(output_path.relative_to(PROJECT_ROOT)),
                },
                indent=2,
            )
        )
        return

    if arguments.command == "verify":
        plan_path = _inside_project(arguments.plan)
        plan = DatasetIntakePlan.from_yaml(plan_path)
        plan.validate(
            project_root=PROJECT_ROOT,
            verify_files=arguments.verify_files,
            require_seal=True,
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "plan_id": plan.plan_id,
                    "group_count": len(plan.groups),
                    "file_count": sum(
                        len(group.files) for group in plan.groups
                    ),
                    "files_verified": arguments.verify_files,
                    "content_sha256": plan.content_hash(),
                },
                indent=2,
            )
        )
        return

    if arguments.command == "admit-many":
        plan_paths = [_inside_project(path) for path in arguments.plan]
        base_path = _inside_project(arguments.base_registry)
        output_path = _inside_project(arguments.output_registry)
        plans = [DatasetIntakePlan.from_yaml(path) for path in plan_paths]
        for plan in plans:
            plan.validate(
                project_root=PROJECT_ROOT,
                verify_files=True,
                require_seal=True,
            )
        base = DataRegistry.from_yaml(base_path)
        merged = merge_plans_into_registry(
            base=base,
            plans=plans,
            registry_version=arguments.registry_version,
            allow_exact_existing=arguments.allow_exact_existing,
        )
        write_registry(merged, output_path)
        print(
            json.dumps(
                {
                    "admitted": True,
                    "plan_ids": [plan.plan_id for plan in plans],
                    "registry_version": merged.registry_version,
                    "asset_count": len(merged.assets),
                    "registry_sha256": merged.content_hash(),
                    "output": str(output_path.relative_to(PROJECT_ROOT)),
                },
                indent=2,
            )
        )
        return

    plan_path = _inside_project(arguments.plan)
    base_path = _inside_project(arguments.base_registry)
    output_path = _inside_project(arguments.output_registry)
    plan = DatasetIntakePlan.from_yaml(plan_path)
    plan.validate(
        project_root=PROJECT_ROOT,
        verify_files=True,
        require_seal=True,
    )
    base = DataRegistry.from_yaml(base_path)
    merged = merge_plan_into_registry(
        base=base,
        plan=plan,
        registry_version=arguments.registry_version,
        allow_exact_existing=arguments.allow_exact_existing,
    )
    write_registry(merged, output_path)
    print(
        json.dumps(
            {
                "admitted": True,
                "plan_id": plan.plan_id,
                "registry_version": merged.registry_version,
                "asset_count": len(merged.assets),
                "registry_sha256": merged.content_hash(),
                "output": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
