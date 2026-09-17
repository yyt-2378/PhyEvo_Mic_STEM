import json

import numpy as np

from tfree_detection.models import DetectionResult
from tfree_detection.reference_deformation import (
    ReferenceDeformationResult,
    compare_to_reference,
    match_lattice_bases,
    save_candidate_hotspot_overlap,
    summarize_candidate_hotspot_overlap,
)


def _result(basis: np.ndarray, shape: tuple[int, int] = (96, 112)) -> DetectionResult:
    vectors = [basis[:, 0].tolist(), basis[:, 1].tolist()]
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    return DetectionResult(
        task="amorphous_region",
        image_shape=shape,
        normalized_image=np.tile(
            np.linspace(0.0, 1.0, shape[1], dtype=np.float32),
            (shape[0], 1),
        ),
        score_map=zeros,
        probability_map=zeros,
        mask=np.zeros(shape, dtype=bool),
        regions=[],
        feature_maps={},
        evidence_maps={
            "gpa_exx": zeros.copy(),
            "gpa_eyy": zeros.copy(),
            "gpa_exy": zeros.copy(),
            "gpa_rotation": zeros.copy(),
            "gpa_validity": ones.copy(),
            "gpa_lattice_coherence": ones.copy(),
        },
        metadata={
            "stage2": {
                "phase": {
                    "direct_basis_vectors_a1_a2_xy_pixels": vectors,
                    "minimum_reliable_coherence": 0.65,
                }
            }
        },
    )


def test_lattice_match_separates_similarity_from_shape_distortion() -> None:
    reference = np.asarray([[20.0, 0.0], [0.0, 20.0]])
    angle = np.deg2rad(12.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    target_to_reference = 1.25 * rotation
    residual_shape = np.diag([np.exp(0.02), np.exp(-0.02)])
    moving = (
        np.linalg.inv(target_to_reference) @ residual_shape @ reference
    )

    match = match_lattice_bases(reference, moving)

    assert np.isclose(match["scale"], 1.25, rtol=2e-3)
    assert np.isclose(match["rotation_degrees"], 12.0, atol=0.2)
    assert match["relative_residual"] > 0.01


def test_reference_comparison_reports_global_principal_distortion() -> None:
    reference_basis = np.asarray([[18.0, 0.0], [0.0, 18.0]])
    residual_shape = np.diag([np.exp(0.03), np.exp(-0.03)])
    moving_basis = residual_shape @ reference_basis / 1.2

    comparison = compare_to_reference(
        _result(reference_basis),
        _result(moving_basis, shape=(88, 104)),
        reference_label="Y",
        moving_label="sample",
    )

    principal = comparison.summary["global_scale_independent_lattice_shape"][
        "principal_log_strain_percent"
    ]
    assert np.allclose(principal, [-3.0, 3.0], atol=0.05)
    assert comparison.aligned_image.shape == (96, 112)
    assert comparison.maps_aligned["relative_exx"].shape == (96, 112)
    assert comparison.summary["stress_status"].startswith(
        "physical stress not computed"
    )


def test_pure_zoom_and_rotation_do_not_create_shape_strain() -> None:
    reference_basis = np.asarray([[20.0, 0.0], [0.0, 20.0]])
    angle = np.deg2rad(5.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    target_to_reference = 1.3 * rotation
    moving_basis = np.linalg.inv(target_to_reference) @ reference_basis

    comparison = compare_to_reference(
        _result(reference_basis),
        _result(moving_basis),
        reference_label="Y",
        moving_label="pure_registration_nuisance",
    )

    registration = comparison.summary["registration_nuisance"]
    principal = comparison.summary["global_scale_independent_lattice_shape"][
        "principal_log_strain_percent"
    ]
    assert np.isclose(registration["moving_to_reference_isotropic_pixel_scale"], 1.3)
    assert np.isclose(
        registration[
            "moving_to_reference_rotation_degrees_modulo_lattice_symmetry"
        ],
        5.0,
    )
    assert np.allclose(principal, [0.0, 0.0], atol=1e-10)


def _hotspot_comparison() -> ReferenceDeformationResult:
    shape = (40, 40)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    distortion = (xx / (shape[1] - 1)).astype(np.float32)
    stress = (yy / (shape[0] - 1)).astype(np.float32)
    reliable = np.ones(shape, dtype=np.float32)
    return ReferenceDeformationResult(
        label="sample",
        summary={},
        maps_native={
            "relative_distortion_magnitude": distortion,
            "stress_hotspot_proxy": stress,
            "reliability": reliable,
        },
        maps_aligned={},
        aligned_image=np.zeros(shape, dtype=np.float32),
        aligned_reliability=np.ones(shape, dtype=bool),
    )


def test_candidate_overlap_is_posthoc_and_uses_native_coordinates() -> None:
    audit = summarize_candidate_hotspot_overlap(
        _hotspot_comparison(),
        np.asarray([[38.0, 38.0], [2.0, 2.0], [50.0, 50.0]]),
        candidate_ids=["hot", "cold", "outside"],
        support_radius_pixels=1.0,
    )

    assert audit["candidate_count"] == 3
    assert audit["candidate_count_with_dual_hotspot_support"] == 1
    records = {item["candidate_id"]: item for item in audit["candidates"]}
    assert records["hot"]["dual_hotspot_support"] is True
    assert records["cold"]["dual_hotspot_support"] is False
    assert records["outside"]["inside_image"] is False


def test_candidate_overlap_bundle_is_auditable(tmp_path) -> None:
    comparison = _hotspot_comparison()
    outputs = save_candidate_hotspot_overlap(
        [comparison],
        {"sample": np.asarray([[38.0, 38.0]])},
        candidate_ids_by_label={"sample": ["site_I_1"]},
        control_points_by_label={
            "sample": np.asarray([[38.0, 38.0], [2.0, 2.0]])
        },
        control_ids_by_label={"sample": ["site_I_1", "site_I_2"]},
        native_images_by_label={"sample": np.zeros((40, 40), dtype=np.float32)},
        output_directory=tmp_path,
        support_radius_pixels=1.0,
        dpi=72,
    )

    assert set(outputs) == {
        "candidate_hotspot_summary_json",
        "candidate_hotspot_table_csv",
        "candidate_hotspot_figure",
        "site_I_control_hotspot_table_csv",
    }
    assert all((tmp_path / path.split("/")[-1]).is_file() for path in outputs.values())
    payload = json.loads(
        (tmp_path / "candidate_gpa_hotspot_overlap.json").read_text()
    )
    baseline = payload["comparisons"][0]["site_I_control_baseline"]
    assert baseline["control_site_count"] == 2
    assert baseline["control_count_with_dual_hotspot_support"] == 1
    assert baseline["one_sided_hypergeometric_enrichment_p_value"] == 0.5
