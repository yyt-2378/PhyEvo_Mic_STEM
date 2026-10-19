"""Command-line interface."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from skimage.io import imread

from .config import load_config
from .atomic_consensus import detect_atomic_consensus
from .atomic_site_inventory import (
    analyze_atomic_site_inventory,
    save_atomic_site_inventory,
)
from .dataset_label_adapter import (
    ExternalLabel,
    adapter_manifest,
    default_label_adapter_registry,
)
from .evidence import ontology_manifest
from .evaluation import boundary_refinement_comparison
from .fau_stack_analysis import analyze_fau_stacks, save_fau_stack_analysis
from .framework import TrainingFreeDetectionFramework
from .governance import DataPartition, DataRegistry
from .preprocessing import list_emd_datasets
from .reporting import save_result
from .reference_deformation import (
    compare_to_reference,
    save_reference_comparison,
)
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
    detect.add_argument("--student-checkpoint", default=None, help="Optional expert student checkpoint directory; shadow output only")
    detect.add_argument("--allow-experimental-student", action="store_true", help="Explicitly allow uncalibrated expert-student shadow inference")
    detect.add_argument("--student-device", default="cpu")
    detect.add_argument("--project-root", default=None, help="Project boundary for opt-in expert student artifacts")
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
    boundary = subparsers.add_parser(
        "evaluate-boundary-refinement",
        help="compare coarse and final geometry against one Gold boundary mask",
    )
    boundary.add_argument("--coarse", required=True)
    boundary.add_argument("--final", required=True)
    boundary.add_argument("--truth", required=True)
    boundary.add_argument("--physics-core", default=None)
    boundary.add_argument("--negative-reference", default=None)
    boundary.add_argument("--tolerance-pixels", type=float, default=2.0)
    boundary.add_argument("--fallback-used", action="store_true")
    boundary.add_argument(
        "--output",
        default=None,
        help="optional new JSON file; existing files are never overwritten",
    )
    reference = subparsers.add_parser(
        "compare-reference",
        help="compare moving-image lattice deformation against one reference",
    )
    reference.add_argument(
        "--reference", required=True, help="fixed reference/template image"
    )
    reference.add_argument(
        "--moving",
        required=True,
        nargs="+",
        help="one or more images to compare against the reference",
    )
    reference.add_argument(
        "--config", required=True, help="GPA-enabled detector configuration"
    )
    reference.add_argument(
        "--output", required=True, help="new output directory"
    )
    atomic_sites = subparsers.add_parser(
        "map-atomic-sites",
        help=(
            "align images to one reference lattice and map O/Si/rare-earth "
            "sites by declared 4D-STEM brightness order"
        ),
    )
    atomic_sites.add_argument(
        "--reference", required=True, help="reference/template image"
    )
    atomic_sites.add_argument(
        "--moving",
        required=True,
        nargs="+",
        help="one or more images to align and map",
    )
    atomic_sites.add_argument(
        "--config",
        required=True,
        help="detector YAML supplying preprocessing and spectral policy",
    )
    atomic_sites.add_argument(
        "--output", required=True, help="new output directory"
    )
    fau_stacks = subparsers.add_parser(
        "analyze-fau-zstacks",
        help=(
            "retain fixed-delta-BIC site-I structural candidates and compare "
            "the Si/O FAU framework against a guest-free CIF"
        ),
    )
    fau_stacks.add_argument(
        "--cif", required=True, help="guest-free Si/O framework CIF"
    )
    fau_stacks.add_argument(
        "--reference-stack",
        required=True,
        help="Y reference QYX TIFF stack",
    )
    fau_stacks.add_argument(
        "--reference-is-rare-earth-free",
        action="store_true",
        help=(
            "declare the reference as a normalized false-positive control with "
            "no rare-earth atoms"
        ),
    )
    fau_stacks.add_argument(
        "--moving-stack",
        required=True,
        nargs="+",
        help="one or more moving QYX TIFF stacks (labels use parent names)",
    )
    fau_stacks.add_argument(
        "--zone-axis",
        required=True,
        nargs=3,
        type=int,
        metavar=("H", "K", "L"),
        help="known crystallographic projection direction, e.g. 1 0 1",
    )
    fau_stacks.add_argument(
        "--output", required=True, help="new output directory"
    )
    return parser


def _load_binary_mask(path: str | Path) -> np.ndarray:
    source = Path(path)
    values = (
        np.load(source, allow_pickle=False)
        if source.suffix.lower() == ".npy"
        else imread(source)
    )
    array = np.asarray(values)
    if array.ndim == 3:
        array = np.max(array, axis=-1)
    if array.ndim != 2:
        raise ValueError(f"Boundary mask must be 2-D: {source}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Boundary mask contains non-finite values: {source}")
    return np.asarray(array > 0, dtype=bool)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "detect":
        config = load_config(args.config)
        framework = TrainingFreeDetectionFramework(config)
        detect_options = dict(
            skeleton=args.skeleton,
            exemplar=args.exemplar,
            frame_index=args.frame_index,
            projection=args.projection,
            dataset_path=args.dataset_path,
            line_topology=args.line_topology,
            sam3_exchange_directory=(
                Path(args.output) / "sam3_exchange"
                if config.stage2.sam3.exchange_directory is None
                else None
            ),
        )
        assisted = None
        if args.student_checkpoint:
            from .expert_runtime import detect_with_expert_student
            from .expert_supervision import contained_path
            bounded_output = contained_path(Path(args.project_root or Path.cwd()), args.output)
            if bounded_output.exists():
                raise FileExistsError("Expert-student inference requires a new output directory")
            assisted = detect_with_expert_student(
                framework, Path(args.input), checkpoint=args.student_checkpoint,
                project_root=args.project_root or Path.cwd(),
                allow_experimental=args.allow_experimental_student,
                student_device=args.student_device, **detect_options,
            )
            result = assisted.teacher
        else:
            result = framework.detect(Path(args.input), **detect_options)
        artifacts = save_result(result, config, args.output)
        if assisted is not None and assisted.student is not None:
            from .expert_runtime import save_shadow_prediction
            artifacts.update(save_shadow_prediction(
                assisted.student, Path(args.output) / "expert_student_shadow",
                project_root=Path(args.project_root or Path.cwd()),
            ))
        payload = result.summary()
        payload["artifacts"] = artifacts
        if assisted is not None:
            payload["expert_student_status"] = assisted.student_status
            payload["neural_training_used_in_student_branch"] = assisted.student is not None
            from .expert_supervision import write_new_json
            combined_path = bounded_output / "expert_assisted_summary.json"
            artifacts["expert_assisted_summary"] = str(combined_path)
            write_new_json(combined_path, payload)
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
    elif args.command == "evaluate-boundary-refinement":
        report = boundary_refinement_comparison(
            _load_binary_mask(args.coarse),
            _load_binary_mask(args.final),
            _load_binary_mask(args.truth),
            tolerance_pixels=args.tolerance_pixels,
            physics_core=(
                _load_binary_mask(args.physics_core)
                if args.physics_core
                else None
            ),
            negative_reference=(
                _load_binary_mask(args.negative_reference)
                if args.negative_reference
                else None
            ),
            fallback_used=args.fallback_used,
        )
        if args.output is not None:
            target = Path(args.output)
            if target.exists():
                raise FileExistsError(
                    f"Refusing to overwrite boundary report: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
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
    elif args.command == "compare-reference":
        config = load_config(args.config)
        if not config.stage2.enabled or not config.stage2.phase.enabled:
            raise ValueError(
                "compare-reference requires stage2.phase.enabled in the config"
            )
        # Frozen-DINO verification does not contribute to the cross-image GPA
        # tensor.  Disable it for this command to avoid redundant inference.
        comparison_config = copy.deepcopy(config)
        comparison_config.stage2.dino.enabled = False
        comparison_config.stage2.sam3.enabled = False
        framework = TrainingFreeDetectionFramework(comparison_config)
        reference_path = Path(args.reference)
        reference_result = framework.detect(reference_path)
        comparisons = []
        for moving_name in args.moving:
            moving_path = Path(moving_name)
            moving_result = framework.detect(moving_path)
            comparisons.append(
                compare_to_reference(
                    reference_result,
                    moving_result,
                    reference_label=reference_path.stem,
                    moving_label=moving_path.stem,
                )
            )
        artifacts = save_reference_comparison(
            reference_result,
            comparisons,
            reference_label=reference_path.stem,
            output_directory=args.output,
            dpi=config.output.figure_dpi,
        )
        print(
            json.dumps(
                {
                    "reference": str(reference_path.resolve()),
                    "moving": [str(Path(item).resolve()) for item in args.moving],
                    "dino_used": False,
                    "comparisons": [item.summary for item in comparisons],
                    "artifacts": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "map-atomic-sites":
        config = load_config(args.config)
        inventory = analyze_atomic_site_inventory(
            Path(args.reference),
            [Path(item) for item in args.moving],
            config,
        )
        artifacts = save_atomic_site_inventory(
            inventory,
            args.output,
            dpi=config.output.figure_dpi,
        )
        print(
            json.dumps(
                {
                    "reference": inventory.reference_label,
                    "common_canvas_shape": list(
                        inventory.common_canvas_shape
                    ),
                    "site_counts": {
                        image.label: int(len(image.species))
                        for image in inventory.images
                    },
                    "artifacts": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "analyze-fau-zstacks":
        reference_path = Path(args.reference_stack)
        moving_paths = [Path(item) for item in args.moving_stack]
        reference_label = reference_path.parent.name
        stacks = {reference_label: reference_path}
        for path in moving_paths:
            label = path.parent.name
            if label in stacks:
                raise ValueError(
                    "Stack parent directories must provide unique labels"
                )
            stacks[label] = path
        analysis = analyze_fau_stacks(
            args.cif,
            stacks,
            reference_label=reference_label,
            reference_is_rare_earth_free=args.reference_is_rare_earth_free,
            zone_axis=args.zone_axis,
        )
        artifacts = save_fau_stack_analysis(
            analysis,
            args.output,
        )
        print(
            json.dumps(
                {
                    "reference": reference_label,
                    "images": list(stacks),
                    "zone_axes": {
                        item.label: list(item.template_fit.zone_axis)
                        for item in analysis.images
                    },
                    "artifacts": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
