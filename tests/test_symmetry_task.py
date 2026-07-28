"""End-to-end tests for the group-aware Class1 task."""

from __future__ import annotations

import numpy as np

from tfree_detection.config import DetectorConfig
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.synthetic import (
    make_clean_lattice,
    make_phase_shifted_symmetry_domain,
)


def _framework() -> TrainingFreeDetectionFramework:
    return TrainingFreeDetectionFramework(
        DetectorConfig(task="symmetry_relation")
    )


def test_clean_lattice_can_pass_the_no_defect_gate() -> None:
    result = _framework().detect(make_clean_lattice())

    assert not np.any(result.mask)
    assert result.metadata["no_defect_gate_passed"]
    assert result.metadata["geometry_counts"] == {
        "symmetry_domain_region": 0,
        "interface_line": 0,
        "phase_slip_line": 0,
        "dislocation_core_point": 0,
    }


def test_phase_shift_outputs_region_interface_and_five_teacher_evidence() -> None:
    image, domain_truth, interface_truth = (
        make_phase_shifted_symmetry_domain()
    )
    result = _framework().detect(image)
    domain = result.feature_maps["symmetry_domain_region"] >= 0.5
    interface = result.feature_maps["interface_line"] >= 0.5

    assert float(np.mean(domain[domain_truth])) >= 0.75
    assert float(np.mean(domain[~domain_truth])) <= 0.02
    assert float(np.mean(interface[interface_truth])) >= 0.25
    assert {
        "group_phase_motif_residual",
        "group_orientation_strain",
        "group_displacement_discontinuity",
        "group_burgers_loop_inconsistency",
        "group_scan_time_artifact",
    }.issubset(result.evidence_maps)
    assert {
        "region",
        "interface",
        "line",
        "point",
    }.issuperset(region.spatial_support for region in result.regions)
    assert result.metadata["topology_counts"]["supported_endpoints"] == 0
    assert (
        result.metadata["geometry_counts"]["dislocation_core_point"]
        == 0
    )
    assert result.evidence_bundle is not None
    assert "translation_or_sublattice_domain" in (
        result.evidence_bundle.hypotheses
    )
