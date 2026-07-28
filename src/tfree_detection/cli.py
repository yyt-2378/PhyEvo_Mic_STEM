"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .atomic_consensus import detect_atomic_consensus
from .dataset_label_adapter import (
    ExternalLabel,
    adapter_manifest,
    default_label_adapter_registry,
)
from .evidence import ontology_manifest
from .framework import TrainingFreeDetectionFramework
from .governance import DataPartition, DataRegistry
from .preprocessing import list_emd_datasets
from .reporting import save_result
from .temporal_support import (
    TemporalSupportConfig,
    analyze_temporal_support,
    save_temporal_support,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tfree-detect",
        description="Training-free microscopy structural anomaly detection",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    detect = subparsers.add_parser(
        "detect", help="run a registered detection task"
    )
    detect.add_argument("--input", required=True, help="input microscopy image")
    detect.add_argument(
        "--config", default=None, help="YAML configuration (defaults are built in)"
    )
    detect.add_argument(
        "--output", required=True, help="directory for result artifacts"
    )
    detect.add_argument(
        "--skeleton",
        default=None,
        help="optional ideal lattice skeleton JSON",
    )
    detect.add_argument(
        "--exemplar",
        default=None,
        help="optional defect exemplar image for tasks that use it",
    )
    stack = detect.add_mutually_exclusive_group()
    stack.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="explicit frame from an NData/EMD stack",
    )
    stack.add_argument(
        "--projection",
        choices=("mean", "median", "max"),
        default=None,
        help="explicit projection over the stack frame axis",
    )
    detect.add_argument(
        "--dataset-path",
        default=None,
        help="explicit Data/Image/.../Data path for a multi-channel EMD",
    )
    detect.add_argument(
        "--line-topology",
        action="store_true",
        help="add skeleton/end-point/junction descriptors for line/interface tasks",
    )
    consensus = subparsers.add_parser(
        "detect-atomic-consensus",
        help="run lattice-residual and periodic-counterfactual teachers",
    )
    consensus.add_argument("--input", required=True)
    consensus.add_argument("--config", required=True)
    consensus.add_argument("--output", required=True)
    consensus_stack = consensus.add_mutually_exclusive_group()
    consensus_stack.add_argument("--frame-index", type=int, default=None)
    consensus_stack.add_argument(
        "--projection",
        choices=("mean", "median", "max"),
        default=None,
    )
    consensus.add_argument("--dataset-path", default=None)
    registry = subparsers.add_parser(
        "validate-registry",
        help="validate partition leakage and active-learning eligibility",
    )
    registry.add_argument(
        "--registry",
        default="datasets/partition_registry.yaml",
        help="versioned data partition registry",
    )
    registry.add_argument(
        "--project-root",
        default=".",
        help="root used to resolve and checksum registered relative paths",
    )
    subparsers.add_parser(
        "ontology",
        help="print the machine-readable defect/evidence ontology",
    )
    subparsers.add_parser(
        "label-adapters",
        help="print registered external-dataset label semantics",
    )
    adapt = subparsers.add_parser(
        "adapt-labels",
        help="translate external JSON/JSONL labels without losing provenance",
    )
    adapt.add_argument("--input", required=True, help="input JSON or JSONL")
    adapt.add_argument(
        "--output",
        required=True,
        help="new JSONL output; existing files are never overwritten",
    )
    inspect_emd = subparsers.add_parser(
        "inspect-emd",
        help="list image datasets and stack shapes in a Velox EMD file",
    )
    inspect_emd.add_argument("--input", required=True, help="input .emd file")
    temporal = subparsers.add_parser(
        "assess-temporal-support",
        help="register a stack and add non-destructive persistence evidence",
    )
    temporal.add_argument("--input", required=True)
    temporal.add_argument("--config", required=True)
    temporal.add_argument("--output", required=True)
    temporal.add_argument("--reference-frame-index", type=int, required=True)
    temporal.add_argument("--dataset-path", default=None)
    temporal.add_argument("--max-frames", type=int, default=9)
    temporal.add_argument("--atomic-consensus", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "detect":
        config = load_config(args.config)
        framework = TrainingFreeDetectionFramework(config)
        result = framework.detect(
            Path(args.input),
            skeleton=args.skeleton,
            exemplar=args.exemplar,
            frame_index=args.frame_index,
            projection=args.projection,
            dataset_path=args.dataset_path,
            line_topology=args.line_topology,
        )
        artifacts = save_result(result, config, args.output)
        payload = result.summary()
        payload["artifacts"] = artifacts
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "validate-registry":
        registry = DataRegistry.from_yaml(args.registry)
        file_verification = registry.verify_files(args.project_root)
        counts = {
            partition.value: len(registry.assets_for_partition(partition))
            for partition in DataPartition
        }
        print(
            json.dumps(
                {
                    "registry_version": registry.registry_version,
                    "registry_hash": registry.content_hash(),
                    "split_unit": registry.split_unit,
                    "partition_counts": counts,
                    "active_learning_asset_count": len(
                        registry.active_learning_assets()
                    ),
                    "file_verification": file_verification,
                    "valid": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "detect-atomic-consensus":
        config = load_config(args.config)
        result = detect_atomic_consensus(
            Path(args.input),
            config,
            frame_index=args.frame_index,
            projection=args.projection,
            dataset_path=args.dataset_path,
        )
        artifacts = save_result(result, config, args.output)
        payload = result.summary()
        payload["artifacts"] = artifacts
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "ontology":
        print(
            json.dumps(
                ontology_manifest(), ensure_ascii=False, indent=2
            )
        )
    elif args.command == "label-adapters":
        print(
            json.dumps(
                adapter_manifest(), ensure_ascii=False, indent=2
            )
        )
    elif args.command == "adapt-labels":
        source = Path(args.input)
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite adapted labels: {output}"
            )
        text = source.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if stripped.startswith("["):
            records = json.loads(text)
        else:
            records = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        if not isinstance(records, list):
            raise ValueError("Adapt-label input must be a JSON list or JSONL")
        registry = default_label_adapter_registry()
        adapted = [
            registry.adapt(ExternalLabel.from_dict(record)).to_dict()
            for record in records
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            for record in adapted:
                stream.write(
                    json.dumps(
                        record, ensure_ascii=False, sort_keys=True
                    )
                )
                stream.write("\n")
        print(
            json.dumps(
                {
                    "adapted": len(adapted),
                    "output": str(output.resolve()),
                    "adapter_protocol_version": (
                        adapter_manifest()["adapter_protocol_version"]
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "inspect-emd":
        datasets = list_emd_datasets(args.input)
        print(
            json.dumps(
                {
                    "input": str(Path(args.input).resolve()),
                    "dataset_count": len(datasets),
                    "datasets": datasets,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "assess-temporal-support":
        config = load_config(args.config)
        if args.atomic_consensus:
            candidate_result = detect_atomic_consensus(
                Path(args.input),
                config,
                frame_index=args.reference_frame_index,
                dataset_path=args.dataset_path,
            )
        else:
            candidate_result = TrainingFreeDetectionFramework(config).detect(
                Path(args.input),
                frame_index=args.reference_frame_index,
                dataset_path=args.dataset_path,
            )
        report = analyze_temporal_support(
            Path(args.input),
            candidate_result,
            config.preprocess,
            reference_frame_index=args.reference_frame_index,
            dataset_path=args.dataset_path,
            config=TemporalSupportConfig(
                maximum_sampled_frames=args.max_frames
            ),
        )
        artifacts = save_temporal_support(
            report, candidate_result, args.output
        )
        print(
            json.dumps(
                {
                    **report.summary(),
                    "artifacts": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
