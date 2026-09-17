from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from tfree_detection.fau_stack_analysis import (
    CIFStructure,
    FAUStackConfig,
    LocalDeformation,
    RareEarthEvidence,
    SkeletonFirstSiteEvidence,
    StackPeakResult,
    TemplateFit,
    _generate_expected_points,
    _global_comparison,
    _associate_all_peaks_with_aligned_sites,
    _aligned_site_interior_mask,
    _aligned_site_markers_allowed,
    _cluster_expert_review_members,
    _expert_review_motif_threshold,
    _normalize_repeated_site_contrast,
    _normalize_local_same_type_contrast,
    _fit_template_empty_center_profile,
    _periodic_segment_fragments,
    apply_rare_earth_free_reference_audit,
    apply_skeleton_first_reference_audit,
    detect_persistent_stack_peaks,
    estimate_local_deformation,
    fit_constrained_brightness_mixture,
    guest_coordinates_in_cif_cell,
    identify_distortion_gated_rare_earth,
    match_framework_sites,
    parse_and_expand_cif,
    project_fau_framework_skeleton,
    project_fau_site_i_template,
    project_fau_101_rectangular_framework,
    project_fau_101_rectangular_site_i,
    project_structure,
    score_zone_axes,
)


def _minimal_evidence(local_z: np.ndarray) -> RareEarthEvidence:
    count = len(local_z)
    positive = np.ones(count, dtype=bool)
    return RareEarthEvidence(
        screen_positive=positive.copy(),
        candidate=positive.copy(),
        inside_hotspot_region=positive.copy(),
        extreme_brightness=positive.copy(),
        local_brightness_z=np.asarray(local_z, dtype=float),
        global_brightness_z=np.asarray(local_z, dtype=float),
        nearest_hotspot_distance_pixels=np.zeros(count),
        hotspot_framework_site=positive.copy(),
        distortion_threshold=0.01,
        maximum_shear_threshold=0.005,
        framework_brightness_threshold=4.0,
        evidence_score=np.ones(count),
        negative_control_tail_probability=np.full(count, np.nan),
    )


def _template_fit_with_basis(basis: np.ndarray) -> TemplateFit:
    return TemplateFit(
        direct_basis_pixels=np.asarray(basis, dtype=float),
        zone_axis=(1, 0, 1),
        zone_scores={"[101]": 1.0},
        zone_score_margin=1.0,
        unit_cell_transform=np.eye(2, dtype=int),
        phase_shift_uv=np.zeros(2),
        expected_points_xy=np.empty((0, 2)),
        expected_species=np.empty(0, dtype=object),
        matched_observed_indices=np.empty(0, dtype=int),
        matched_expected_indices=np.empty(0, dtype=int),
        match_distances_pixels=np.empty(0),
        framework_observed_count=0,
        template_ncc=1.0,
        angstrom_per_pixel=0.2,
    )


def test_safe_cif_parser_expands_explicit_operations(tmp_path: Path) -> None:
    source = tmp_path / "tiny.cif"
    source.write_text(
        """data_tiny
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_Int_Tables_number 1
loop_
_symmetry_equiv_pos_as_xyz
'+x,+y,+z'
'1/2+x,1/2+y,+z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
O1 O 0.1 0.2 0.3
T1 Si 0.4 0.2 0.3
""",
        encoding="utf-8",
    )

    structure = parse_and_expand_cif(source)

    assert structure.symmetry_operation_count == 2
    assert len(structure.species) == 4
    assert int(np.sum(structure.species == "O")) == 2
    assert int(np.sum(structure.species == "Si")) == 2


def test_cubic_101_projection_has_expected_fau_equivalent_phase_count() -> None:
    cif = Path("real_data_test/4D数据计算晶格变化_0901/FAU.cif")
    if not cif.is_file():
        pytest.skip("local FAU CIF is not distributed with the code")
    template = project_structure(parse_and_expand_cif(cif), (1, 0, 1))

    assert len(template.species) == 78
    assert int(np.sum(template.species == "O")) == 54

    site_i = project_fau_site_i_template(
        parse_and_expand_cif(cif), (1, 0, 1)
    )
    assert len(site_i.phase_uv) == 3
    assert set(site_i.species.tolist()) == {"FAU_site_I"}
    assert int(np.sum(template.species == "Si")) == 24


def test_projected_framework_splits_periodic_bond_at_cell_boundary() -> None:
    fragments = _periodic_segment_fragments(
        np.asarray([0.98, 0.50]),
        np.asarray([0.02, 0.50]),
    )

    ordered = sorted(fragments.tolist(), key=lambda item: item[0][0])
    assert np.allclose(ordered[0], [[0.0, 0.5], [0.02, 0.5]])
    assert np.allclose(ordered[1], [[0.98, 0.5], [1.0, 0.5]])
    assert np.isclose(
        sum(np.linalg.norm(np.diff(fragment, axis=0)) for fragment in fragments),
        0.04,
    )


def test_fau_101_skeleton_recovers_tetrahedral_si_o_connectivity() -> None:
    cif = Path("real_data_test/4D数据计算晶格变化_0901/FAU.cif")
    if not cif.is_file():
        pytest.skip("local FAU CIF is not distributed with the code")

    skeleton = project_fau_framework_skeleton(
        parse_and_expand_cif(cif), (1, 0, 1)
    )

    assert skeleton.three_dimensional_bond_count == 768
    assert skeleton.bond_cutoff_angstrom == 2.1
    assert len(skeleton.bond_segments_uv) > 0
    assert np.all((skeleton.bond_segments_uv >= 0) & (skeleton.bond_segments_uv <= 1))


def test_rectangular_101_framework_preserves_physical_metric() -> None:
    cif = Path("real_data_test/4D数据计算晶格变化_0901/FAU.cif")
    if not cif.is_file():
        pytest.skip("local FAU CIF is not distributed with the code")

    framework = project_fau_101_rectangular_framework(parse_and_expand_cif(cif))

    assert np.isclose(framework.width_angstrom / framework.height_angstrom, np.sqrt(2))
    assert len(framework.atom_fractional_xy) == 4 * 78
    assert framework.three_dimensional_bond_count == 2 * 768
    assert np.all(
        (framework.bond_segments_fractional_xy >= 0)
        & (framework.bond_segments_fractional_xy <= 1)
    )


def test_rectangular_101_framework_retains_periodic_central_oxygen_columns() -> None:
    cif = Path("real_data_test/4D数据计算晶格变化_0901/FAU.cif")
    if not cif.is_file():
        pytest.skip("local FAU CIF is not distributed with the code")
    framework = project_fau_101_rectangular_framework(parse_and_expand_cif(cif))
    on_central_axis = np.isclose(framework.atom_fractional_xy[:, 0], 0.5)
    oxygen_y = np.sort(
        framework.atom_fractional_xy[
            on_central_axis & (framework.species == "O"), 1
        ]
    )

    assert np.allclose(
        oxygen_y,
        [0.0331, 0.1429, 0.3217, 0.6783, 0.8571, 0.9669],
    )
    assert not np.any(on_central_axis & (framework.species == "Si"))


def test_rectangular_101_site_i_template_has_twelve_periodic_probes() -> None:
    cif = Path("real_data_test/4D数据计算晶格变化_0901/FAU.cif")
    if not cif.is_file():
        pytest.skip("local FAU CIF is not distributed with the code")

    probes = project_fau_101_rectangular_site_i(parse_and_expand_cif(cif))

    assert probes.shape == (12, 2)
    assert np.all((probes >= 0) & (probes < 1))
    assert np.sum(np.isclose(probes[:, 0], 0.5)) == 1


def test_expert_review_threshold_respects_floor_and_control_tail() -> None:
    policy = FAUStackConfig(
        expert_review_control_quantile=0.999,
        expert_review_minimum_motif_z=3.5,
    )

    below_floor = _expert_review_motif_threshold(
        np.linspace(-1.0, 3.0, 1000), policy
    )
    tail_limited = _expert_review_motif_threshold(
        np.asarray([0.0] * 998 + [8.0, 10.0]), policy
    )

    assert below_floor == 3.5
    assert tail_limited > 8.0


def test_expert_review_boundary_warning_must_be_positive() -> None:
    with pytest.raises(ValueError, match="boundary_warning_cells"):
        FAUStackConfig(expert_review_boundary_warning_cells=0.0).validate()


def test_expert_review_members_cluster_into_context_regions() -> None:
    members = [
        {"x_pixels": 0.0, "y_pixels": 0.0, "name": "first"},
        {"x_pixels": 3.0, "y_pixels": 4.0, "name": "second"},
        {"x_pixels": 20.0, "y_pixels": 0.0, "name": "third"},
    ]

    groups = _cluster_expert_review_members(members, radius_pixels=6.0)

    assert sorted(len(group) for group in groups) == [1, 2]
    assert {item["name"] for group in groups for item in group} == {
        "first", "second", "third"
    }


def test_aligned_site_attribution_recovers_peak_excluded_from_framework_fit() -> None:
    fit = _template_fit_with_basis(np.eye(2))
    fit.expected_points_xy = np.asarray([[10.0, 10.0], [20.0, 10.0]])
    fit.expected_species = np.asarray(["O", "Si"], dtype=object)
    peaks = SimpleNamespace(
        points_xy=np.asarray([[10.3, 9.8], [20.2, 10.1]]),
        species_brightness_class=np.asarray(["bright", "mid"], dtype=object),
    )

    observed, expected, distances = _associate_all_peaks_with_aligned_sites(
        peaks, fit, 1.0
    )

    assert set(zip(observed.tolist(), expected.tolist())) == {(0, 0), (1, 1)}
    assert np.all(distances < 0.4)


def test_same_motif_contrast_normalization_is_affine_invariant() -> None:
    repeats = 20
    site_ids = np.repeat(np.arange(4), repeats)
    baseline = np.repeat(np.asarray([-4.0, 0.0, 5.0, 11.0]), repeats)
    within_site = np.tile(np.linspace(-0.6, 0.6, repeats), 4)
    values = baseline + within_site
    values[7] += 8.0
    associated = np.ones(len(values), dtype=bool)

    original = _normalize_repeated_site_contrast(
        values, site_ids, associated
    )
    gain_and_offset = _normalize_repeated_site_contrast(
        3.7 * values + 23.0, site_ids, associated
    )

    assert np.allclose(original, gain_and_offset)
    assert original[7] > 5.0
    assert np.max(np.delete(original, 7)) < 5.0


def test_local_same_species_contrast_is_gain_invariant() -> None:
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    points = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    values = 2.0 + 0.1 * np.sin(np.arange(len(points)))
    values[12] += 2.0
    species = np.asarray(["O"] * len(points), dtype=object)
    valid = np.ones(len(points), dtype=bool)

    original, original_count = _normalize_local_same_type_contrast(
        values,
        points,
        species,
        valid,
        radius_pixels=3.0,
        minimum_references=8,
    )
    transformed, transformed_count = _normalize_local_same_type_contrast(
        3.7 * values + 23.0,
        points,
        species,
        valid,
        radius_pixels=3.0,
        minimum_references=8,
    )

    finite = np.isfinite(original)
    assert np.allclose(original[finite], transformed[finite])
    assert np.array_equal(original_count, transformed_count)
    assert original[12] > 3.5


def test_two_O_profile_prefers_center_component_only_when_link_is_added() -> None:
    rows, columns = np.indices((41, 41), dtype=float)
    center = np.asarray([20.0, 20.0])
    endpoints = np.asarray([[20.0, 16.0], [20.0, 24.0]])

    def spot(x: float, y: float, amplitude: float) -> np.ndarray:
        return amplitude * np.exp(
            -0.5 * ((columns - x) ** 2 + (rows - y) ** 2) / 1.3**2
        )

    framework_only = 0.15 + spot(20, 16, 1.0) + spot(20, 24, 1.0)
    with_center = framework_only + spot(20, 20, 1.2)

    null_fit = _fit_template_empty_center_profile(
        framework_only, center, endpoints, sigma_pixels=1.3
    )
    occupied_fit = _fit_template_empty_center_profile(
        with_center, center, endpoints, sigma_pixels=1.3
    )

    assert occupied_fit[2] > 6.0
    assert occupied_fit[2] > null_fit[2]
    assert occupied_fit[3] > 0


def _skeleton_site_fixture(
    delta_bic: np.ndarray,
    local_z: np.ndarray,
) -> SkeletonFirstSiteEvidence:
    count = len(delta_bic)
    return SkeletonFirstSiteEvidence(
        expected_xy=np.column_stack([np.arange(count), np.zeros(count)]),
        canonical_site_id=np.zeros(count, dtype=int),
        oxygen_endpoint_xy=np.zeros((count, 2, 2), dtype=float),
        model_valid=np.ones(count, dtype=bool),
        best_plane=np.zeros(count, dtype=int),
        template_only_bic=np.zeros(count),
        template_plus_site_bic=-np.asarray(delta_bic, dtype=float),
        delta_bic=np.asarray(delta_bic, dtype=float),
        extra_component_amplitude=np.ones(count),
        local_equivalent_site_z=np.asarray(local_z, dtype=float),
        local_reference_count=np.full(count, 12),
        distance_to_image_edge_pixels=np.full(count, 100.0),
        inside_reliable_interior=np.ones(count, dtype=bool),
        raw_structural_screen_positive=np.zeros(count, dtype=bool),
        structural_review=np.zeros(count, dtype=bool),
        local_brightness_anomaly=np.zeros(count, dtype=bool),
        candidate=np.zeros(count, dtype=bool),
        negative_control_tail_probability=np.full(count, np.nan),
        calibrated_delta_bic_threshold=np.nan,
    )


def test_skeleton_first_keeps_every_fixed_threshold_structural_positive() -> None:
    reference = SimpleNamespace(
        label="Y",
        skeleton_first_sites=_skeleton_site_fixture(
            np.linspace(-2.0, 2.0, 100), np.zeros(100)
        ),
    )
    moving = SimpleNamespace(
        label="RA",
        skeleton_first_sites=_skeleton_site_fixture(
            np.asarray([1.0, 10.0, 10.0]),
            np.asarray([0.0, 6.0, 0.8]),
        ),
    )

    audit = apply_skeleton_first_reference_audit(
        [reference, moving], "Y", FAUStackConfig()
    )
    site = moving.skeleton_first_sites

    assert site.structural_review.tolist() == [False, True, True]
    assert site.candidate.tolist() == [False, True, True]
    assert not np.any(site.local_brightness_anomaly)
    assert audit["within_image_structural_quantile_gate"] is False
    assert audit["local_brightness_gate"] is False
    assert audit["images"]["RA"]["structure_supported_candidate_count"] == 2


def test_aligned_site_candidate_gate_rejects_cell_scaled_image_boundary() -> None:
    policy = FAUStackConfig(
        border_pixels=4,
        aligned_site_candidate_border_cells=0.25,
    )
    points = np.asarray([[24.9, 50.0], [25.1, 50.0], [50.0, 74.0]])

    distances, reliable = _aligned_site_interior_mask(
        points,
        (101, 101),
        np.asarray([[100.0, 0.0], [0.0, 100.0]]),
        policy,
    )

    assert np.allclose(distances, [24.9, 25.1, 26.0])
    assert reliable.tolist() == [False, True, True]


def test_aligned_site_candidate_residual_cannot_exceed_match_tolerance() -> None:
    policy = FAUStackConfig(
        framework_match_tolerance_pixels=2.0,
        aligned_site_candidate_max_residual_pixels=3.0,
    )

    with pytest.raises(ValueError, match="cannot exceed match tolerance"):
        policy.validate()


def test_rare_earth_free_template_never_receives_special_site_markers() -> None:
    analysis = SimpleNamespace(
        reference_label="Y",
        reference_is_rare_earth_free=True,
    )

    assert not _aligned_site_markers_allowed(
        analysis, SimpleNamespace(label="Y")
    )
    assert _aligned_site_markers_allowed(
        analysis, SimpleNamespace(label="RA")
    )


def test_rare_earth_guest_is_never_consumed_by_framework_match() -> None:
    expected = np.asarray([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
    expected_species = np.asarray(["O", "Si", "O", "Si"], dtype=object)
    observed = np.asarray(
        [[0.2, 0.1], [5.2, 0.1], [0.1, 5.2], [5.1, 5.1], [4.8, 0.1]]
    )
    observed_species = np.asarray(
        ["O", "Si", "O", "Si", "rare_earth"], dtype=object
    )

    observed_indices, _, distances = match_framework_sites(
        observed, observed_species, expected, expected_species, 1.0
    )

    assert set(observed_indices.tolist()) == {0, 1, 2, 3}
    assert 4 not in observed_indices
    assert np.all(distances < 0.5)
    assert np.linalg.norm(observed[4] - expected[0]) > 4.5


def test_brightness_mixture_does_not_leave_dim_population_empty() -> None:
    rng = np.random.default_rng(12)
    values = np.concatenate(
        [
            rng.normal(34.0, 14.0, 2400),
            rng.normal(70.0, 8.0, 500),
            rng.normal(88.0, 10.0, 700),
        ]
    )

    mixture = fit_constrained_brightness_mixture(values)
    counts = np.bincount(mixture.labels, minlength=3)

    assert np.all(counts >= 0.01 * len(values))
    assert np.min(mixture.separation_sigma) >= 0.8
    assert np.all(np.diff(mixture.means) > 0)


def test_stack_search_keeps_a_slice_local_bright_peak() -> None:
    shape = (112, 112)
    base = np.zeros(shape, dtype=float)
    positions = []
    for row in range(10, 104, 11):
        for column in range(10, 104, 11):
            positions.append((row, column))
            base[row, column] = 0.20 + 0.30 * ((row + column) % 3)
    guest_yx = (59, 59)
    stack = np.stack(
        [
            gaussian_filter(base, sigma)
            for sigma in (1.6, 0.8, 0.25, 0.8, 1.6)
        ]
    ).astype(np.float32)
    stack[0, guest_yx[0], guest_yx[1]] = 1.5
    policy = FAUStackConfig(
        focus_fraction_of_best=0.08,
        peak_detection_z=0.5,
        persistence_z=5.0,
        minimum_persistent_planes=2,
    )

    result = detect_persistent_stack_peaks(stack, policy)

    distance = np.linalg.norm(
        result.points_xy - np.asarray(guest_yx[::-1]), axis=1
    )
    selected = int(np.argmin(distance))
    assert result.best_plane_index == 2
    assert len(result.selected_plane_indices) >= 3
    assert float(np.min(distance)) < 0.6
    assert int(result.persistence_count[selected]) == 1
    assert not bool(result.framework_eligible[selected])
    assert result.species_brightness_class[selected] == "bright"


def test_distortion_and_extreme_brightness_intersection_keeps_only_b() -> None:
    x, y = np.meshgrid(np.arange(0, 50, 5), np.arange(0, 50, 5))
    framework = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    point_a = np.asarray([[80.0, 80.0]])
    point_b = np.asarray([[25.5, 25.5]])
    point_c = np.asarray([[30.5, 25.5]])
    points = np.vstack([framework, point_a, point_b, point_c])
    reference_brightness = 3.0 + 0.15 * np.sin(np.arange(len(framework)))
    center = float(np.median(reference_brightness))
    scale = float(1.4826 * np.median(np.abs(reference_brightness - center)))
    brightness = np.concatenate(
        [reference_brightness, [center + 8 * scale, center + 8 * scale, center + 0.8 * scale]]
    )
    count = len(points)
    peaks = StackPeakResult(
        stack_shape=(4, 96, 96),
        best_plane_index=2,
        selected_plane_indices=np.asarray([1, 2, 3]),
        focus_scores=np.ones(4),
        best_image=np.zeros((96, 96)),
        fused_score=np.zeros((96, 96)),
        points_xy=points,
        best_plane_by_peak=np.full(count, 2),
        persistence_count=np.concatenate([np.full(100, 3), [1, 1, 3]]),
        framework_eligible=np.concatenate([np.ones(100, dtype=bool), np.zeros(3, dtype=bool)]),
        standardized_brightness=brightness,
        local_contrast=brightness,
        species_brightness_class=np.asarray(["mid"] * 100 + ["bright"] * 3, dtype=object),
        assignment_probability=np.ones(count),
        mixture_means=np.asarray([1.0, 3.0, 6.0]),
        mixture_standard_deviations=np.ones(3),
        mixture_separation_sigma=np.ones(2),
    )
    fit = TemplateFit(
        direct_basis_pixels=np.asarray([[50.0, 0.0], [0.0, 50.0]]),
        zone_axis=(1, 0, 1),
        zone_scores={"[101]": 1.0},
        zone_score_margin=1.0,
        unit_cell_transform=np.eye(2),
        phase_shift_uv=np.zeros(2),
        expected_points_xy=framework,
        expected_species=np.asarray(["Si"] * 100, dtype=object),
        matched_observed_indices=np.arange(100),
        matched_expected_indices=np.arange(100),
        match_distances_pixels=np.zeros(100),
        framework_observed_count=100,
        template_ncc=1.0,
        angstrom_per_pixel=0.2,
    )
    distortion = np.full(100, 0.002)
    maximum_shear = np.full(100, 0.001)
    hotspot_index = 55
    distortion[hotspot_index] = 0.08
    maximum_shear[hotspot_index] = 0.04
    deformation = LocalDeformation(
        exx=np.zeros(100),
        eyy=np.zeros(100),
        exy=np.zeros(100),
        rotation=np.zeros(100),
        distortion_magnitude=distortion,
        maximum_shear=maximum_shear,
        fit_rmse_pixels=np.zeros(100),
        neighbor_count=np.full(100, 20),
        reliable=np.ones(100, dtype=bool),
        nearest_guest_distance_pixels=np.full(100, np.inf),
    )

    evidence = identify_distortion_gated_rare_earth(
        peaks, fit, deformation, FAUStackConfig()
    )

    assert evidence.extreme_brightness[-3:].tolist() == [True, True, False]
    assert evidence.inside_hotspot_region[-3:].tolist() == [False, True, True]
    assert evidence.candidate[-3:].tolist() == [False, True, False]
    assert peaks.persistence_count[-2] == 1


def test_per_image_normalization_keeps_moving_candidate_below_control_absolute_tail() -> None:
    reference = SimpleNamespace(
        label="Y",
        rare_earth_evidence=_minimal_evidence(np.asarray([1.0, 2.0, 5.4])),
    )
    moving = SimpleNamespace(
        label="H2",
        rare_earth_evidence=_minimal_evidence(np.asarray([5.2])),
    )

    audit = apply_rare_earth_free_reference_audit([reference, moving], "Y")

    assert not np.any(reference.rare_earth_evidence.candidate)
    assert bool(moving.rare_earth_evidence.candidate[0])
    assert audit["cross_image_absolute_intensity_gate"] is False
    assert moving.rare_earth_evidence.negative_control_tail_probability[0] > 0


def test_global_shape_strain_removes_unknown_zoom_and_rotation() -> None:
    angle = np.deg2rad(7.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    reference_basis = np.asarray([[52.0, 4.0], [-2.0, 49.0]])
    moving_basis = 1.35 * rotation @ reference_basis
    reference = SimpleNamespace(
        template_fit=_template_fit_with_basis(reference_basis)
    )
    moving = SimpleNamespace(template_fit=_template_fit_with_basis(moving_basis))

    comparison = _global_comparison(reference, moving)

    assert np.allclose(
        comparison["scale_free_principal_shape_strain_percent"],
        [0.0, 0.0],
        atol=1e-10,
    )
    assert comparison["absolute_isotropic_lattice_change"] is None
    assert comparison["absolute_projected_area_change"] is None


def _synthetic_structure() -> CIFStructure:
    rng = np.random.default_rng(4)
    coordinates = rng.random((80, 3))
    species = np.asarray(["O"] * 56 + ["Si"] * 24, dtype=object)
    return CIFStructure(
        cell_lengths_angstrom=(20.0, 20.0, 20.0),
        cell_angles_degrees=(90.0, 90.0, 90.0),
        space_group_number=1,
        fractional_coordinates=coordinates,
        species=species,
        symmetry_operation_count=1,
    )


def test_zone_search_recovers_synthetic_110_projection() -> None:
    structure = _synthetic_structure()
    axes = ((0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 1, 2))
    templates = [project_structure(structure, axis) for axis in axes]
    target = templates[1]
    basis = np.asarray([[52.0, 12.0], [0.0, 48.0]])
    expected, species = _generate_expected_points(
        (256, 256), basis, target, np.eye(2, dtype=int), np.asarray([0.21, 0.34])
    )
    image = np.zeros((256, 256), dtype=float)
    rounded = np.rint(expected).astype(int)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < 256)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < 256)
    )
    amplitudes = np.where(species[inside] == "Si", 2.0, 1.0)
    np.add.at(image, (rounded[inside, 1], rounded[inside, 0]), amplitudes)
    image = gaussian_filter(image, 1.1)
    policy = FAUStackConfig(unit_cell_bins=64, template_blur_sigmas=(1.0, 2.0))

    selected, scores, _, _, _ = score_zone_axes(image, basis, templates, policy)

    assert selected.zone_axis == (1, 1, 0)
    assert scores["[110]"] > max(value for key, value in scores.items() if key != "[110]")


def test_local_affine_fit_recovers_framework_strain_without_guest() -> None:
    x, y = np.meshgrid(np.arange(0, 101, 10), np.arange(0, 101, 10))
    expected = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    strain = np.asarray([[0.025, 0.010], [0.010, -0.015]])
    observed = expected + expected @ strain.T + np.asarray([0.3, -0.2])
    species = np.asarray(["O"] * len(expected), dtype=object)
    peaks = StackPeakResult(
        stack_shape=(3, 112, 112),
        best_plane_index=1,
        selected_plane_indices=np.asarray([0, 1, 2]),
        focus_scores=np.ones(3),
        best_image=np.zeros((112, 112)),
        fused_score=np.zeros((112, 112)),
        points_xy=observed,
        best_plane_by_peak=np.ones(len(expected), dtype=int),
        persistence_count=np.full(len(expected), 3),
        framework_eligible=np.ones(len(expected), dtype=bool),
        standardized_brightness=np.ones(len(expected)),
        local_contrast=np.ones(len(expected)),
        species_brightness_class=species,
        assignment_probability=np.ones(len(expected)),
        mixture_means=np.asarray([0.2, 0.5, 0.9]),
        mixture_standard_deviations=np.ones(3),
        mixture_separation_sigma=np.ones(2),
    )
    fit = TemplateFit(
        direct_basis_pixels=np.asarray([[50.0, 0.0], [0.0, 50.0]]),
        zone_axis=(1, 1, 0),
        zone_scores={"[110]": 1.0},
        zone_score_margin=1.0,
        unit_cell_transform=np.eye(2),
        phase_shift_uv=np.zeros(2),
        expected_points_xy=expected,
        expected_species=species,
        matched_observed_indices=np.arange(len(expected)),
        matched_expected_indices=np.arange(len(expected)),
        match_distances_pixels=np.linalg.norm(observed - expected, axis=1),
        framework_observed_count=len(expected),
        template_ncc=1.0,
        angstrom_per_pixel=0.2,
    )
    policy = FAUStackConfig(
        local_fit_radius_cells=1.2,
        local_fit_minimum_neighbors=8,
        local_fit_maximum_rmse_pixels=0.01,
    )

    result = estimate_local_deformation(peaks, fit, policy)

    assert np.mean(result.reliable) > 0.9
    assert np.allclose(np.nanmedian(result.exx), 0.025, atol=1e-5)
    assert np.allclose(np.nanmedian(result.eyy), -0.015, atol=1e-5)
    assert np.allclose(np.nanmedian(result.exy), 0.010, atol=1e-5)


def test_guest_coordinates_remove_image_cell_basis_and_phase_nuisance() -> None:
    canonical = np.asarray([[0.12, 0.34], [0.83, 0.61]])
    basis = np.asarray([[51.0, 8.0], [-4.0, 49.0]])
    transform = np.asarray([[1, -1], [0, 1]])
    shift = np.asarray([0.27, 0.41])
    image_points = (
        basis @ (transform @ canonical.T + shift[:, None])
    ).T
    fit = TemplateFit(
        direct_basis_pixels=basis,
        zone_axis=(1, 1, 0),
        zone_scores={"[110]": 1.0},
        zone_score_margin=1.0,
        unit_cell_transform=transform,
        phase_shift_uv=shift,
        expected_points_xy=np.empty((0, 2)),
        expected_species=np.empty(0, dtype=object),
        matched_observed_indices=np.empty(0, dtype=int),
        matched_expected_indices=np.empty(0, dtype=int),
        match_distances_pixels=np.empty(0),
        framework_observed_count=0,
        template_ncc=1.0,
        angstrom_per_pixel=0.2,
    )

    recovered = guest_coordinates_in_cif_cell(image_points, fit)

    assert np.allclose(recovered, canonical)
