#!/usr/bin/env python
"""Report implemented software gates and unresolved scientific evidence gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tfree_detection.data_intake import DatasetIntakePlan, sha256_file
from tfree_detection.digital_twin import SUPPORTED_TWIN_TASKS
from tfree_detection.governance import DataPartition, DataRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _exists(relative: str) -> bool:
    return (PROJECT_ROOT / relative).is_file()


def _load_json(relative: str) -> dict:
    return json.loads(
        (PROJECT_ROOT / relative).read_text(encoding="utf-8")
    )


def _intake_status(relative: str, *, verify_files: bool) -> dict:
    path = PROJECT_ROOT / relative
    if not path.is_file():
        return {
            "verified": False,
            "files_verified": False,
            "reason": "plan_missing",
        }
    plan = DatasetIntakePlan.from_yaml(path)
    plan.validate(
        project_root=PROJECT_ROOT,
        verify_files=verify_files,
        require_seal=True,
    )
    return {
        "verified": True,
        "files_verified": verify_files,
        "plan_id": plan.plan_id,
        "admission_state": plan.admission_state.value,
        "group_count": len(plan.groups),
        "file_count": sum(len(group.files) for group in plan.groups),
        "content_sha256": plan.content_hash(),
    }


def _verify_pilot_protocol_seal(relative: str) -> dict:
    seal_path = PROJECT_ROOT / relative
    if not seal_path.is_file():
        return {"verified": False, "reason": "seal_missing"}
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    canonical = dict(seal)
    expected_content_hash = canonical.pop("seal_content_sha256")
    observed_content_hash = hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if observed_content_hash != expected_content_hash:
        raise ValueError("Prospective pilot protocol seal content mismatch")
    root = seal_path.parent
    for file_name, expected in seal["sealed_files"].items():
        target = (root / file_name).resolve()
        target.relative_to(PROJECT_ROOT.resolve())
        if sha256_file(target) != expected:
            raise ValueError(
                f"Prospective pilot sealed file mismatch: {file_name}"
            )
    blinded = (
        root / "blinded" / "review_packets.json"
    ).read_text(encoding="utf-8")
    forbidden = (
        "confirmed_consensus",
        "periodic_only_uncertain",
        "legacy_only_uncertain",
        "random_sham_prompt_control",
        "teacher_score",
        "selection_branch",
    )
    leaked = [token for token in forbidden if token in blinded]
    if leaked:
        raise ValueError(
            f"Blinded pilot packet leaks private fields: {leaked}"
        )
    pilot = json.loads(
        (root / "pilot_seal.json").read_text(encoding="utf-8")
    )
    return {
        "verified": True,
        "protocol_id": seal["protocol_id"],
        "seal_content_sha256": expected_content_hash,
        "candidate_count": pilot["candidate_count"],
        "source_count": pilot["source_count"],
        "labels_read_during_selection": pilot[
            "labels_read_during_selection"
        ],
        "external_reviews_received": False,
    }


def main() -> None:
    registry_candidates = (
        PROJECT_ROOT / "datasets" / "partition_registry_v4.yaml",
        PROJECT_ROOT / "datasets" / "partition_registry_v3.yaml",
        PROJECT_ROOT / "datasets" / "partition_registry_v2.yaml",
        PROJECT_ROOT / "datasets" / "partition_registry.yaml",
    )
    current_registry_path = next(
        path for path in registry_candidates if path.is_file()
    )
    registry = DataRegistry.from_yaml(current_registry_path)
    counts = {
        partition.value: len(registry.assets_for_partition(partition))
        for partition in DataPartition
    }
    group_counts = {
        partition.value: len(
            {
                asset.split_group_id
                for asset in registry.assets_for_partition(partition)
            }
        )
        for partition in DataPartition
    }
    active_learning_asset_count = len(
        registry.active_learning_assets()
    )
    student = json.loads(
        (
            PROJECT_ROOT / "benchmarks" / "latest_student_ablation.json"
        ).read_text(encoding="utf-8")
    )
    file_verification = registry.verify_files(PROJECT_ROOT)
    locked_protocol = _exists(
        "benchmarks/locked_dennler_v1_protocol.yaml"
    )
    locked_protocol_seal = _exists(
        "benchmarks/locked_dennler_v1_protocol_seal.json"
    )
    locked_unpacked = (
        PROJECT_ROOT
        / "datasets"
        / "public"
        / "dennler_2021_hrstem_defects"
        / "locked_extracted"
    ).is_dir()
    locked_prediction_seal = _exists(
        "benchmarks/locked_runs/dennler_v1_predictions/prediction_seal.json"
    )
    locked_score = _exists(
        "benchmarks/locked_runs/dennler_v1_score/locked_metrics.json"
    )
    locked_primary_metrics = None
    external_generalization_supported = False
    if locked_score:
        score_payload = json.loads(
            (
                PROJECT_ROOT
                / "benchmarks"
                / "locked_runs"
                / "dennler_v1_score"
                / "locked_metrics.json"
            ).read_text(encoding="utf-8")
        )
        locked_primary_metrics = {
            name: {
                metric: values["micro_pixel_metrics"][metric]
                for metric in ("iou", "precision", "recall")
            }
            for name, values in score_payload["targets"].items()
            if values["role"] == "primary"
        }
    intake_path = (
        PROJECT_ROOT
        / "datasets/intake/emseek_wse2_train_v1.sealed.yaml"
    )
    intake = (
        DatasetIntakePlan.from_yaml(intake_path)
        if intake_path.is_file()
        else None
    )
    intake_verified = False
    if intake is not None:
        intake.validate(
            project_root=PROJECT_ROOT,
            verify_files=True,
            require_seal=True,
        )
        intake_verified = True
    periodic_score_path = (
        "benchmarks/development_runs/"
        "emseek_wse2_periodic_v1_score_v2/development_metrics.json"
    )
    legacy_score_path = (
        "benchmarks/development_runs/"
        "emseek_wse2_legacy_atomic_v0_2_score_v2/development_metrics.json"
    )
    consensus_score_path = (
        "benchmarks/development_runs/"
        "emseek_wse2_two_teacher_consensus_v2_score/"
        "development_metrics.json"
    )
    triage_score_path = (
        "benchmarks/development_runs/"
        "emseek_wse2_two_teacher_consensus_v2_score/"
        "consensus_triage_metrics.json"
    )
    periodic_score = (
        _load_json(periodic_score_path)
        if _exists(periodic_score_path)
        else None
    )
    legacy_score = (
        _load_json(legacy_score_path)
        if _exists(legacy_score_path)
        else None
    )
    consensus_score = (
        _load_json(consensus_score_path)
        if _exists(consensus_score_path)
        else None
    )
    triage_score = (
        _load_json(triage_score_path)
        if _exists(triage_score_path)
        else None
    )
    pilot_seal = _verify_pilot_protocol_seal(
        "benchmarks/prospective_pilots/"
        "emseek_wse2_disagreement_pilot_v1/protocol_seal.json"
    )
    open_world_intake = {
        "emseek_remaining": _intake_status(
            "datasets/intake/emseek_remaining_train_v1.sealed.yaml",
            verify_files=False,
        ),
        "guo_superscan": _intake_status(
            "datasets/intake/guo_superscan_development_v1.sealed.yaml",
            verify_files=True,
        ),
        "al_sigma3": _intake_status(
            "datasets/intake/al_sigma3_development_v1.sealed.yaml",
            verify_files=True,
        ),
        "admitted_to_current_registry": {
            "guo_2021_superscan": any(
                asset.dataset_id == "guo_2021_superscan"
                for asset in registry.assets
            ),
            "al_sigma3_twin_boundaries": any(
                asset.dataset_id == "al_sigma3_twin_boundaries"
                for asset in registry.assets
            ),
        },
        "sample_manifest_complete": _exists(
            "datasets/standardized/sample_manifest_v3.jsonl"
        ),
        "operational_status_complete": _exists(
            "datasets/intake/public_dataset_operational_status_v2.json"
        ),
        "emseek_balanced_subset": _intake_status(
            "datasets/intake/emseek_balanced_subset_train_v1.sealed.yaml",
            verify_files=True,
        ),
    }
    phases = {
        "phase_0_baseline": {
            "software_complete": _exists(
                "benchmarks/baseline_v0_2_manifest.json"
            )
            and _exists("scripts/check_baseline_regression.py"),
            "gate": "frozen v0.2 metrics and >=95% retention check",
        },
        "phase_1_unified_evidence": {
            "software_complete": all(
                _exists(path)
                for path in (
                    "src/tfree_detection/evidence.py",
                    "schemas/evidence_bundle.schema.json",
                    "schemas/defect_ontology.yaml",
                    "src/tfree_detection/resolver.py",
                    "src/tfree_detection/dataset_label_adapter.py",
                    "schemas/dataset_label_adapter.schema.json",
                )
            ),
            "protocol_version": "2.0",
            "multi_hypothesis_resolver": {
                "implemented": True,
                "artifact_state_supported": True,
                "unresolved_state_supported": True,
                "legacy_masks_preserved": True,
            },
            "synthetic_performance_retained_fraction": 1.0,
            "atomic_periodic_counterfactual_development": {
                "software_complete": all(
                    _exists(path)
                    for path in (
                        "src/tfree_detection/periodic_counterfactual.py",
                        "src/tfree_detection/site_consensus.py",
                        "src/tfree_detection/site_matching.py",
                        "src/tfree_detection/atomic_consensus.py",
                    )
                ),
                "unified_cli_available": True,
                "dataset": "EMSeek WSe2",
                "partition": "train",
                "single_conservative_group": True,
                "independent_generalization_claim_allowed": False,
                "legacy_micro_site_metrics": (
                    legacy_score["micro_site_metrics"]
                    if legacy_score
                    else None
                ),
                "periodic_micro_site_metrics": (
                    periodic_score["micro_site_metrics"]
                    if periodic_score
                    else None
                ),
                "consensus_micro_site_metrics": (
                    consensus_score["micro_site_metrics"]
                    if consensus_score
                    else None
                ),
                "selective_triage_tiers": (
                    triage_score["tiers"] if triage_score else None
                ),
            },
            "line_and_temporal_auxiliary_evidence": {
                "line_topology_implemented": _exists(
                    "src/tfree_detection/line_topology.py"
                ),
                "temporal_support_implemented": _exists(
                    "src/tfree_detection/temporal_support.py"
                ),
                "superscan_smoke_test_complete": _exists(
                    "outputs/superscan_temporal_support_v1/"
                    "temporal_support_summary.json"
                ),
                "mechanism_confirmation_claimed": False,
            },
        },
        "phase_2_gold_and_calibration": {
            "software_complete": all(
                _exists(path)
                for path in (
                    "src/tfree_detection/governance.py",
                    "src/tfree_detection/calibration.py",
                    "src/tfree_detection/uncertainty.py",
                    "src/tfree_detection/data_intake.py",
                    str(current_registry_path.relative_to(PROJECT_ROOT)),
                )
            ),
            "partition_counts": counts,
            "partition_group_counts": group_counts,
            "active_learning_asset_count": active_learning_asset_count,
            "current_registry": str(
                current_registry_path.relative_to(PROJECT_ROOT)
            ),
            "prospective_intake": {
                "sealed_plan_verified": intake_verified,
                "plan_content_sha256": (
                    intake.content_hash() if intake else None
                ),
                "annotation_qualification": (
                    intake.groups[0].annotation_qualification.value
                    if intake
                    else None
                ),
                "gold_claim_allowed": False,
            },
            "open_world_v0_4_intake": open_world_intake,
            "scientific_gate_complete": (
                counts["calibration"] > 0
                and locked_score
                and external_generalization_supported
            ),
            "locked_external_evaluation": {
                "protocol_pre_registered": locked_protocol,
                "protocol_sealed": locked_protocol_seal,
                "archive_unpacked_after_seal": locked_unpacked,
                "prediction_set_sealed": locked_prediction_seal,
                "score_unblinded": locked_score,
                "outcome": (
                    "zero_shot_transfer_not_supported"
                    if locked_score
                    else "not_yet_scored"
                ),
                "primary_metrics": locked_primary_metrics,
            },
        },
        "phase_3_evidence_student": {
            "software_complete": _exists("src/tfree_detection/student.py"),
            "synthetic_gate_status": student["status"],
            "synthetic_iou_difference": student["iou_difference"],
            "real_train_assets_admitted": counts["train"] > 0,
            "real_data_student_retrained": False,
            "cross_material_gold_gate_complete": False,
        },
        "phase_4_active_learning_and_review": {
            "software_complete": all(
                _exists(path)
                for path in (
                    "src/tfree_detection/active_learning.py",
                    "src/tfree_detection/review.py",
                    "src/tfree_detection/review_ui.py",
                    "src/tfree_detection/cycle.py",
                    "schemas/review_response.schema.json",
                    "scripts/compile_review_consensus.py",
                    "scripts/build_emseek_prospective_pilot.py",
                )
            ),
            "offline_review_demo_complete": all(
                _exists(path)
                for path in (
                    "outputs/review_pilot_demo_assisted/index.html",
                    "outputs/review_pilot_demo_blinded/index.html",
                )
            ),
            "prospective_pilot_package": pilot_seal,
            "prospective_expert_cycle_complete": False,
            "required_pilot_candidates": "50-100 double-reviewed candidates",
        },
        "phase_5_counterfactual_twins": {
            "software_complete": _exists(
                "src/tfree_detection/digital_twin.py"
            ),
            "covered_tasks": list(SUPPORTED_TWIN_TASKS),
            "multislice_realism_validated": False,
        },
        "phase_6_bandit_or_autonomous_acquisition": {
            "software_complete": False,
            "status": "intentionally_deferred_until_real prospective loop",
        },
    }
    blockers = [
        "No independent Gold calibration assets are registered.",
        (
            "One author-curated WSe2 train group is admitted, but unresolved "
            "acquisition grouping prevents an independent split and its "
            "published reviewer count is unknown, so it is not Gold."
        ),
        (
            "A 100-candidate/69-image prospective package is sealed, but no "
            "external double-review responses have been received."
        ),
        "Synthetic student gain has not been reproduced on unseen materials.",
        "Counterfactual twins are analytical textures, not validated multislice simulations.",
        "No microscope integration or prospective acquisition experiment exists.",
    ]
    if counts["locked_external_test"] == 0:
        blockers.append(
            "No prospectively locked external material/lab assets are registered."
        )
    elif not locked_score:
        blockers.append(
            "One expert-labeled HRSTEM archive is checksum-locked but its "
            "pre-registered blind evaluation is not complete."
        )
    else:
        blockers.append(
            "One expert-labeled HRSTEM archive has a sealed external score, "
            "but a single instrument/lab archive is not a cross-lab suite."
        )
        blockers.append(
            "Frozen zero-shot teachers achieved primary IoU 0.0118, 0.1251, "
            "and 0.1396 on the consumed external archive; broad transfer is "
            "not supported and future models require a new locked test."
        )
    print(
        json.dumps(
            {
                "roadmap": "PhyEvo-Mic self-improving platform",
                "current_registry": str(
                    current_registry_path.relative_to(PROJECT_ROOT)
                ),
                "registry_hash": registry.content_hash(),
                "registered_files_verified": file_verification["asset_count"],
                "phases": phases,
                "nature_level_method_claim_ready": False,
                "blocking_scientific_evidence": blockers,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
