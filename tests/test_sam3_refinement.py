import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tfree_detection.config import DetectorConfig, SAM3RefinementConfig
from tfree_detection.governance import candidate_records_from_result
from tfree_detection.models import DetectionRegion, DetectionResult
from tfree_detection.stage2.sam3_refine import (
    SAM3Proposal,
    build_interface_prompt_plan,
    build_physics_prompt_plan,
    evaluate_candidate_sam3_proposals,
    evaluate_refinement_proposals,
    evaluate_sam3_proposals,
    load_sam3_response,
)


def _proposal(mask: np.ndarray, index: int) -> SAM3Proposal:
    return SAM3Proposal(
        proposal_id=f"proposal-{index}",
        mask=mask,
        score=0.9,
        ensemble_index=index,
    )


def _gate_fixture():
    coarse = np.zeros((48, 48), dtype=bool)
    coarse[10:34, 10:34] = True
    core = np.zeros_like(coarse)
    core[16:26, 16:26] = True
    negative = np.zeros_like(coarse)
    negative[:5, :] = True
    score = np.zeros(coarse.shape, dtype=np.float32)
    score[coarse] = 1.0
    return coarse, core, negative, score


def test_policy_a_rejects_proposal_that_deletes_more_than_five_percent_core():
    coarse, core, negative, score = _gate_fixture()
    proposal_mask = coarse.copy()
    proposal_mask[16, 16:22] = False
    proposals = [_proposal(proposal_mask, index) for index in range(8)]

    result = evaluate_sam3_proposals(
        coarse,
        score,
        core,
        negative,
        proposals,
        [],
        SAM3RefinementConfig(),
    )

    assert result.metrics["seed_core_recall"] == pytest.approx(0.94)
    assert not result.accepted
    assert result.status == "rejected_fallback"
    assert "seed_core_recall_below_0.950" in result.rejection_reasons
    assert len(result.proposals) == 8
    assert np.array_equal(result.refined_mask, proposal_mask)
    assert np.array_equal(result.final_mask, coarse)


def test_policy_a_accepts_stable_geometry_that_preserves_physics_core():
    coarse, core, negative, score = _gate_fixture()
    proposals = [_proposal(coarse.copy(), index) for index in range(8)]

    result = evaluate_sam3_proposals(
        coarse,
        score,
        core,
        negative,
        proposals,
        [],
        SAM3RefinementConfig(),
    )

    assert result.accepted
    assert result.status == "accepted"
    assert result.metrics["seed_core_recall"] == 1.0
    assert np.array_equal(result.final_mask, coarse)
    assert np.array_equal(result.stable_core, coarse)


def test_consensus_exposes_possible_extent_and_boundary_uncertainty():
    coarse, core, negative, score = _gate_fixture()
    masks = []
    for index in range(4):
        mask = coarse.copy()
        if index < 2:
            mask[34, 20] = True
        masks.append(_proposal(mask, index))

    result = evaluate_sam3_proposals(
        coarse,
        score,
        core,
        negative,
        masks,
        [],
        SAM3RefinementConfig(),
    )

    assert result.inclusion_probability[34, 20] == pytest.approx(0.5)
    assert result.refined_mask[34, 20]
    assert result.possible_extent[34, 20]
    assert not result.stable_core[34, 20]
    assert result.boundary_uncertainty[34, 20] == pytest.approx(1.0)


def test_negative_reference_leakage_rejects_visual_overexpansion():
    coarse, core, negative, score = _gate_fixture()
    overexpanded = coarse.copy()
    overexpanded[2, :] = True
    proposals = [_proposal(overexpanded, index) for index in range(8)]

    result = evaluate_sam3_proposals(
        coarse,
        score,
        core,
        negative,
        proposals,
        [],
        SAM3RefinementConfig(),
    )

    assert result.metrics["negative_leakage"] > 0.05
    assert "negative_leakage_above_0.050" in result.rejection_reasons
    assert np.array_equal(result.final_mask, coarse)


def test_unjustified_merge_between_physics_candidates_is_rejected():
    coarse = np.zeros((48, 48), dtype=bool)
    coarse[12:24, 6:18] = True
    coarse[12:24, 30:42] = True
    core = coarse.copy()
    negative = np.zeros_like(coarse)
    score = coarse.astype(np.float32)
    merged = coarse.copy()
    merged[16:20, 18:30] = True
    proposals = [_proposal(merged, index) for index in range(8)]

    result = evaluate_sam3_proposals(
        coarse,
        score,
        core,
        negative,
        proposals,
        [],
        SAM3RefinementConfig(),
    )

    assert result.metrics["component_merge_detected"] is True
    assert "unjustified_component_merge" in result.rejection_reasons
    assert np.array_equal(result.final_mask, coarse)


def test_physics_prompt_generation_is_deterministic_and_semantic():
    coarse = np.zeros((64, 64), dtype=bool)
    coarse[12:28, 10:26] = True
    coarse[38:54, 40:56] = True
    score = np.zeros(coarse.shape, dtype=np.float32)
    score[coarse] = 1.0
    score[18:23, 16:21] = 2.0
    score[44:49, 46:51] = 2.0
    config = SAM3RefinementConfig(
        ensemble_size=4,
        box_padding_pixels=8,
    )

    first = build_physics_prompt_plan(coarse, score, {}, config)
    second = build_physics_prompt_plan(coarse, score, {}, config)

    assert [item.to_dict() for item in first.prompts] == [
        item.to_dict() for item in second.prompts
    ]
    assert len(first.prompts) == 8
    assert np.all(first.physics_core <= coarse)
    assert not np.any(first.negative_reference & coarse)
    for prompt in first.prompts:
        for x, y in prompt.positive_points_xy:
            assert first.physics_core[y, x]
        for x, y in prompt.negative_points_xy:
            assert first.negative_reference[y, x]


def test_configuration_enforces_policy_a_fallback():
    config = DetectorConfig()
    config.stage2.sam3.enabled = True
    config.stage2.sam3.fallback_to_coarse = False

    with pytest.raises(ValueError, match="Policy A"):
        config.validate()


def test_configuration_cannot_weaken_policy_a_core_recall():
    config = DetectorConfig()
    config.stage2.sam3.enabled = True
    config.stage2.sam3.minimum_seed_core_recall = 0.94

    with pytest.raises(ValueError, match="Policy A"):
        config.validate()


def test_candidate_local_failure_does_not_erase_accepted_neighbor():
    coarse = np.zeros((56, 72), dtype=bool)
    coarse[12:36, 8:28] = True
    coarse[12:36, 42:62] = True
    score = coarse.astype(np.float32)
    plan = build_physics_prompt_plan(
        coarse,
        score,
        {},
        SAM3RefinementConfig(ensemble_size=4),
    )

    accepted_geometry = coarse & (plan.component_labels == 1)
    accepted_geometry[:, 8] = False
    accepted_geometry[:, 28] = True
    rejected_geometry = coarse & (plan.component_labels == 2)
    rejected_core = plan.physics_core & (plan.component_labels == 2)
    removed_count = max(1, int(np.ceil(rejected_core.sum() * 0.06)))
    rejected_geometry.flat[
        np.flatnonzero(rejected_core)[:removed_count]
    ] = False

    proposals = []
    for candidate_id, component_label, geometry in (
        ("candidate-0001", 1, accepted_geometry),
        ("candidate-0002", 2, rejected_geometry),
    ):
        proposals.extend(
            SAM3Proposal(
                proposal_id=f"{candidate_id}-{index}",
                candidate_id=candidate_id,
                component_label=component_label,
                mask=geometry.copy(),
                score=0.9,
                ensemble_index=index,
            )
            for index in range(4)
        )

    result = evaluate_candidate_sam3_proposals(
        coarse,
        score,
        plan,
        proposals,
        SAM3RefinementConfig(ensemble_size=4),
    )

    by_id = {item.candidate_id: item for item in result.candidate_results}
    summary = result.to_summary()
    assert by_id["candidate-0001"].accepted
    assert not by_id["candidate-0002"].accepted
    assert result.status == "partially_accepted"
    assert summary["accepted_candidate_count"] == 1
    assert summary["runtime_and_memory"]["worker_runtime_seconds"] is None
    assert len(summary["candidate_results"]) == 2
    assert np.array_equal(
        result.final_mask & (plan.component_labels == 2),
        coarse & (plan.component_labels == 2),
    )
    assert np.array_equal(
        result.final_mask & ~(
            coarse & (plan.component_labels == 2)
        ),
        accepted_geometry,
    )

    regions = [
        DetectionRegion(
            region_id=index,
            bbox_xyxy=bbox,
            centroid_xy=centroid,
            area_pixels=int((plan.component_labels == index).sum()),
            mean_score=1.0,
            max_score=1.0,
            confidence=0.8,
            confidence_level="high",
            evidence_groups=["physics"],
            candidate_id=f"external-candidate-{index}",
            spatial_support="region",
        )
        for index, bbox, centroid in (
            (1, (8, 12, 28, 36), (18.0, 24.0)),
            (2, (42, 12, 62, 36), (52.0, 24.0)),
        )
    ]
    detection = DetectionResult(
        task="amorphous_region",
        image_shape=coarse.shape,
        normalized_image=np.zeros(coarse.shape, dtype=np.float32),
        score_map=score,
        probability_map=score,
        mask=coarse,
        regions=regions,
        feature_maps={},
        evidence_maps={},
        evidence_bundle=SimpleNamespace(
            provenance={"source_id": "synthetic"}
        ),
        boundary_refinement=result,
    )
    records = candidate_records_from_result(detection)
    assert records[0].geometry_source == "sam3_consensus"
    assert records[0].boundary_refinement_status == "accepted"
    assert records[0].geometry_area_pixels == int(
        accepted_geometry.sum()
    )
    assert records[1].geometry_source == "physics_coarse_mask"
    assert records[1].boundary_refinement_status == "rejected_fallback"
    assert records[1].geometry_area_pixels == regions[1].area_pixels


def test_phase_interface_is_derived_from_two_domains_not_a_filled_region():
    coarse = np.zeros((64, 64), dtype=bool)
    coarse[:, 29:35] = True
    score = coarse.astype(np.float32)
    domains = np.zeros(coarse.shape, dtype=np.int8)
    domains[:, 32:] = 1
    valid = np.ones_like(coarse)
    config = SAM3RefinementConfig(
        ensemble_size=4,
        box_padding_pixels=8,
        box_padding_lattice_cells=0,
        interface_max_gap_pixels=4,
        interface_half_width_pixels=2,
    )
    plan = build_interface_prompt_plan(
        coarse,
        score,
        domains,
        valid,
        {},
        config,
    )

    assert plan.geometry_kind == "paired_domains"
    assert {prompt.prompt_role for prompt in plan.prompts} == {
        "domain_a",
        "domain_b",
    }
    assert len(plan.prompts) == 2 * config.ensemble_size

    domain_a = domains == 0
    domain_b = domains == 1
    proposals = [
        SAM3Proposal(
            proposal_id=f"{prompt.prompt_role}-{prompt.ensemble_index}",
            candidate_id=prompt.candidate_id,
            component_label=prompt.component_label,
            prompt_role=prompt.prompt_role,
            mask=(
                domain_a.copy()
                if prompt.prompt_role == "domain_a"
                else domain_b.copy()
            ),
            score=0.95,
            ensemble_index=prompt.ensemble_index,
        )
        for prompt in plan.prompts
    ]
    result = evaluate_refinement_proposals(
        coarse,
        score,
        plan,
        proposals,
        config,
    )

    assert result.accepted
    assert result.provenance["geometry_kind"] == "paired_domains"
    assert len(result.proposals) == config.ensemble_size
    assert result.final_mask.sum() < 0.20 * result.final_mask.size
    outside_interface = result.final_mask.copy()
    outside_interface[:, 24:40] = False
    assert not np.any(outside_interface)


def test_protocol_v2_requires_every_candidate_role_ensemble_pair(
    tmp_path: Path,
):
    prompts = [
        {
            "candidate_id": "candidate-0001",
            "component_label": 1,
            "ensemble_index": ensemble_index,
            "prompt_role": role,
        }
        for ensemble_index in range(2)
        for role in ("domain_a", "domain_b")
    ]
    request = {
        "protocol_version": "tfree-sam3-boundary-v2",
        "request_id": "request-v2",
        "response_file": "sam_response.json",
        "arrays": {"coarse_mask": {"shape": [8, 8]}},
        "prompts": prompts,
    }
    records = []
    for index, prompt in enumerate(prompts):
        mask_path = tmp_path / f"proposal-{index}.npy"
        np.save(mask_path, np.eye(8, dtype=bool), allow_pickle=False)
        records.append(
            {
                "proposal_id": f"proposal-{index}",
                **prompt,
                "mask_file": mask_path.name,
                "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                "score": 0.9,
            }
        )
    response_path = tmp_path / "sam_response.json"
    response_path.write_text(
        json.dumps(
            {
                "protocol_version": "tfree-sam3-boundary-v2",
                "request_id": "request-v2",
                "proposals": records,
                "worker": {"backend": "synthetic"},
            }
        ),
        encoding="utf-8",
    )
    request_path = tmp_path / "sam_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    proposals, _ = load_sam3_response(request_path, request)
    assert len(proposals) == 4
    assert {
        (item.candidate_id, item.prompt_role, item.ensemble_index)
        for item in proposals
    } == {
        ("candidate-0001", "domain_a", 0),
        ("candidate-0001", "domain_b", 0),
        ("candidate-0001", "domain_a", 1),
        ("candidate-0001", "domain_b", 1),
    }

    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["proposals"] = response["proposals"][:-1]
    response_path.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 4"):
        load_sam3_response(request_path, request)
