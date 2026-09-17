from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tfree_detection.config import DetectorConfig
from tfree_detection.context import IdealSkeleton, SkeletonEdge, SkeletonNode
from tfree_detection.fau_stack_analysis import _fit_template_empty_center_profile
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.structure_guided import (
    ProjectedSiteProbe,
    SampleRole,
    StructureGuidedPolicy,
    analyze_projected_site_probes,
    projected_probes_from_skeleton,
)
from tfree_detection.structure_guided.adapters.fau import (
    analyze_fau_site_i_with_generic_engine,
    projected_probes_from_fau_site_i,
)
from tfree_detection.structure_guided.model_selection import (
    compare_optional_center_profile,
)


def _gaussian_image(
    *, center_amplitude: float, shape: tuple[int, int] = (64, 64)
) -> np.ndarray:
    y, x = np.mgrid[: shape[0], : shape[1]]

    def peak(cx: float, cy: float, amplitude: float) -> np.ndarray:
        return amplitude * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * 1.25**2)
        )

    return (
        peak(24.0, 32.0, 1.0)
        + peak(40.0, 32.0, 1.0)
        + peak(32.0, 32.0, center_amplitude)
        + 0.01 * np.sin(x / 5.0)
    )


def _probe(family: str = "generic_optional_site") -> ProjectedSiteProbe:
    return ProjectedSiteProbe(
        probe_id="site@0,0",
        family=family,
        center_xy=(32.0, 32.0),
        endpoint_xy=((24.0, 32.0), (40.0, 32.0)),
    )


def _periodic_skeleton(*, site_label: str = "optional") -> IdealSkeleton:
    return IdealSkeleton(
        basis=np.asarray(((32.0, 0.0), (0.0, 32.0))),
        nodes=[
            SkeletonNode("left", (0.25, 0.5), "framework"),
            SkeletonNode("right", (0.75, 0.5), "framework"),
            SkeletonNode(site_label, (0.5, 0.5), "optional", optional=True),
        ],
        edges=[
            SkeletonEdge("left", site_label),
            SkeletonEdge(site_label, "right"),
        ],
        metadata={"origin_xy": [16.0, 16.0]},
    )


def test_optional_center_is_selected_but_empty_center_is_not() -> None:
    occupied = analyze_projected_site_probes(
        _gaussian_image(center_amplitude=0.7), [_probe()]
    )
    empty = analyze_projected_site_probes(
        _gaussian_image(center_amplitude=0.0), [_probe()]
    )

    assert occupied.delta_bic[0] >= 6.0
    assert occupied.candidate.tolist() == [True]
    assert empty.candidate.tolist() == [False]


def test_target_free_reference_discards_structural_positives_and_forces_zero_candidates() -> None:
    analysis = analyze_projected_site_probes(
        _gaussian_image(center_amplitude=0.7),
        [_probe()],
        sample_role=SampleRole.TARGET_FREE_REFERENCE,
    )

    assert analysis.policy_suppressed is True
    assert analysis.model_valid.tolist() == [False]
    assert np.isneginf(analysis.delta_bic).all()
    assert analysis.structural_positive.tolist() == [False]
    assert analysis.candidate.tolist() == [False]
    assert analysis.summary()["candidate_count"] == 0
    assert "false_positive" not in repr(analysis.summary()).lower()


def test_profile_model_is_numerically_equivalent_to_frozen_fau_v20() -> None:
    image = _gaussian_image(center_amplitude=0.7)
    probe = _probe()
    legacy = _fit_template_empty_center_profile(
        image,
        np.asarray(probe.center_xy),
        np.asarray(probe.endpoint_xy),
        1.25,
    )
    generic = compare_optional_center_profile(
        image,
        probe.center_xy,
        probe.endpoint_xy,
        sigma_pixels=1.25,
        minimum_endpoint_separation_pixels=2.0,
    )

    assert generic.valid is True
    assert generic.delta_bic == pytest.approx(legacy[2], abs=1e-10)
    assert generic.center_amplitude == pytest.approx(legacy[3], abs=1e-10)


def test_material_and_site_names_do_not_change_numeric_decision() -> None:
    image = _gaussian_image(center_amplitude=0.7)
    first = analyze_projected_site_probes(image, [_probe("FAU_site_I")])
    second = analyze_projected_site_probes(image, [_probe("unrelated_material_X")])

    np.testing.assert_allclose(first.delta_bic, second.delta_bic)
    np.testing.assert_array_equal(first.candidate, second.candidate)


def test_skeleton_requires_two_non_optional_endpoint_nodes() -> None:
    skeleton = _periodic_skeleton()
    skeleton.edges.pop()
    with pytest.raises(ValueError, match="exactly two non-optional"):
        projected_probes_from_skeleton(skeleton, (64, 64))


def test_checked_in_structure_guided_skeleton_example_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    skeleton = IdealSkeleton.from_json(
        root / "schemas/structure_guided_skeleton.example.json"
    )
    probes = projected_probes_from_skeleton(skeleton, (64, 64))
    assert probes
    assert all(probe.family == "optional_site" for probe in probes)


def test_main_framework_runs_structure_guided_task_and_evidence_protocol() -> None:
    config = DetectorConfig(task="structure_guided_site")
    config.structure_guided.minimum_delta_bic = 6.0
    framework = TrainingFreeDetectionFramework(config)
    skeleton = _periodic_skeleton()
    image = np.zeros((64, 64), dtype=float)
    y, x = np.mgrid[:64, :64]
    probes = projected_probes_from_skeleton(skeleton, image.shape)
    for probe in probes:
        for px, py in probe.endpoint_xy:
            image += np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * 1.25**2))
        px, py = probe.center_xy
        image += 0.7 * np.exp(
            -((x - px) ** 2 + (y - py) ** 2) / (2 * 1.25**2)
        )

    result = framework.detect(image, skeleton=skeleton)

    assert "structure_guided_site" in framework.available_tasks
    assert result.task == "structure_guided_site"
    assert len(result.regions) >= 1
    assert result.evidence_bundle is not None
    assert "node.template_model_gain" in result.evidence_bundle.fields


def test_main_framework_reference_role_exposes_no_scores_or_candidates() -> None:
    config = DetectorConfig(task="structure_guided_site")
    config.structure_guided.sample_role = "target_free_reference"
    framework = TrainingFreeDetectionFramework(config)
    result = framework.detect(
        _gaussian_image(center_amplitude=0.7), skeleton=_periodic_skeleton()
    )

    assert result.regions == []
    assert not result.mask.any()
    assert not result.score_map.any()
    assert not result.evidence_maps["group_structural_model_gain"].any()
    assert not result.evidence_maps["group_center_component"].any()
    assert result.metadata["structure_guided"]["candidate_count"] == 0


def test_fau_adapter_translates_geometry_and_enforces_reference_role() -> None:
    legacy = SimpleNamespace(
        expected_xy=np.asarray(((32.0, 32.0),)),
        oxygen_endpoint_xy=np.asarray((((24.0, 32.0), (40.0, 32.0)),)),
        canonical_site_id=np.asarray(("I-004",)),
        model_valid=np.asarray((True,)),
    )
    probes = projected_probes_from_fau_site_i(legacy)
    assert probes[0].probe_id == "I-004"

    analysis = analyze_fau_site_i_with_generic_engine(
        _gaussian_image(center_amplitude=0.7),
        legacy,
        label="Y",
        reference_label="Y",
        reference_is_target_free=True,
        policy=StructureGuidedPolicy(minimum_delta_bic=6.0),
    )
    assert analysis.sample_role is SampleRole.TARGET_FREE_REFERENCE
    assert np.isneginf(analysis.delta_bic).all()
    assert analysis.candidate.tolist() == [False]
