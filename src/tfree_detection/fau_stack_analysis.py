"""Structure-first FAU z-stack analysis against a guest-free CIF framework.

The TIFF first axis is treated as a reconstruction/focus stack because no
physical z calibration is present.  The current site-I candidate rule retains
every valid positive-center two-O fit whose fixed delta-BIC is at least six.
There is no within-image top-quantile or local-brightness gate; brightness and
GPA remain audit-only measurements.  The pure-silica FAU CIF is used only to
model the Si/O framework, and a declared rare-earth-free reference is kept at
zero interpreted guest candidates.

Different fields of view do not support tracking an individual atom between
Y, RA, and H2.  Cross-image migration is therefore reported only as a change
of the ensemble guest-position distribution in the periodic unit cell.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import itertools
import json
import math
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from scipy.linalg import logm, polar
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from skimage.feature import peak_local_max

from .atomic_site_inventory import (
    AtomicSiteInventoryConfig,
    OrderedIntensityMixture,
    fit_ordered_intensity_mixture,
)
from .config import SpectralConfig
from .features.lattice import direct_basis_from_peaks, select_reciprocal_basis
from .features.spectral import estimate_global_reciprocal_peaks
from .reference_deformation import match_lattice_bases


ZONE_AXIS_CANDIDATES = (
    (0, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 0),
    (1, 2, 1),
    (1, 2, 2),
    (1, 2, 3),
)

BRIGHTNESS_CLASS_ORDER = ("dim", "mid", "bright")


@dataclass(frozen=True)
class FAUStackConfig:
    """Numerical and abstention policy for the FAU stack workflow."""

    focus_sigma: float = 0.75
    background_sigma: float = 3.0
    focus_fraction_of_best: float = 0.70
    peak_detection_z: float = 1.50
    persistence_z: float = 1.00
    minimum_persistent_planes: int = 2
    peak_min_distance_pixels: int = 2
    border_pixels: int = 4
    unit_cell_bins: int = 96
    template_blur_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    framework_match_tolerance_pixels: float = 4.0
    local_fit_radius_cells: float = 1.25
    local_fit_minimum_neighbors: int = 12
    local_fit_maximum_rmse_pixels: float = 2.75
    local_fit_maximum_strain: float = 0.15
    hotspot_quantile: float = 0.90
    hotspot_radius_cells: float = 0.20
    local_brightness_neighbors: int = 64
    local_brightness_z_threshold: float = 5.0
    global_brightness_z_threshold: float = 5.0
    framework_brightness_quantile: float = 0.995
    aligned_site_contrast_z_threshold: float = 5.0
    aligned_site_candidate_max_residual_pixels: float = 3.0
    aligned_site_candidate_border_cells: float = 0.25
    local_same_species_radius_cells: float = 1.50
    local_same_species_minimum_references: int = 8
    local_same_species_z_threshold: float = 3.5
    site_i_profile_sigma_angstrom: float = 0.32
    site_i_minimum_framework_separation_angstrom: float = 0.45
    site_i_endpoint_radius_angstrom: float = 1.25
    site_i_minimum_delta_bic: float = 6.0
    site_i_control_quantile: float = 0.999
    site_i_local_radius_cells: float = 2.50
    site_i_local_minimum_references: int = 6
    site_i_candidate_border_cells: float = 0.50
    expert_review_control_quantile: float = 0.999
    expert_review_minimum_motif_z: float = 3.5
    expert_review_cluster_radius_cells: float = 0.25
    expert_review_display_radius_cells: float = 0.18
    expert_review_boundary_warning_cells: float = 0.75
    near_guest_radius_cells: float = 0.18
    far_guest_radius_cells: float = 0.35

    def validate(self) -> None:
        if self.focus_sigma <= 0:
            raise ValueError("focus_sigma must be positive")
        if self.background_sigma <= self.focus_sigma:
            raise ValueError("background_sigma must exceed focus_sigma")
        if not 0 < self.focus_fraction_of_best <= 1:
            raise ValueError("focus_fraction_of_best must be in (0, 1]")
        if self.minimum_persistent_planes < 1:
            raise ValueError("minimum_persistent_planes must be positive")
        if self.peak_min_distance_pixels < 1:
            raise ValueError("peak_min_distance_pixels must be positive")
        if self.unit_cell_bins < 32:
            raise ValueError("unit_cell_bins must be at least 32")
        if self.framework_match_tolerance_pixels <= 0:
            raise ValueError("framework match tolerance must be positive")
        if self.local_fit_minimum_neighbors < 6:
            raise ValueError("local fit requires at least six neighbors")
        if not 0.5 < self.hotspot_quantile < 1:
            raise ValueError("hotspot_quantile must be in (0.5, 1)")
        if self.hotspot_radius_cells <= 0:
            raise ValueError("hotspot_radius_cells must be positive")
        if self.local_brightness_neighbors < 12:
            raise ValueError("local brightness reference requires at least 12 neighbors")
        if self.local_brightness_z_threshold <= 0:
            raise ValueError("local brightness z threshold must be positive")
        if self.global_brightness_z_threshold <= 0:
            raise ValueError("global brightness z threshold must be positive")
        if not 0.9 <= self.framework_brightness_quantile < 1:
            raise ValueError("framework_brightness_quantile must be in [0.9, 1)")
        if self.aligned_site_contrast_z_threshold <= 0:
            raise ValueError("aligned_site_contrast_z_threshold must be positive")
        if self.aligned_site_candidate_max_residual_pixels <= 0:
            raise ValueError(
                "aligned_site_candidate_max_residual_pixels must be positive"
            )
        if (
            self.aligned_site_candidate_max_residual_pixels
            > self.framework_match_tolerance_pixels
        ):
            raise ValueError(
                "Aligned-site candidate residual cannot exceed match tolerance"
            )
        if self.aligned_site_candidate_border_cells <= 0:
            raise ValueError("aligned_site_candidate_border_cells must be positive")
        if self.local_same_species_radius_cells <= 0:
            raise ValueError("local_same_species_radius_cells must be positive")
        if self.local_same_species_minimum_references < 4:
            raise ValueError(
                "local_same_species_minimum_references must be at least four"
            )
        if self.local_same_species_z_threshold <= 0:
            raise ValueError("local_same_species_z_threshold must be positive")
        if self.site_i_profile_sigma_angstrom <= 0:
            raise ValueError("site_i_profile_sigma_angstrom must be positive")
        if self.site_i_minimum_framework_separation_angstrom <= 0:
            raise ValueError(
                "site_i_minimum_framework_separation_angstrom must be positive"
            )
        if (
            self.site_i_endpoint_radius_angstrom
            <= self.site_i_minimum_framework_separation_angstrom
        ):
            raise ValueError(
                "site_i_endpoint_radius_angstrom must exceed the minimum "
                "framework separation"
            )
        if self.site_i_minimum_delta_bic <= 0:
            raise ValueError("site_i_minimum_delta_bic must be positive")
        if not 0.99 <= self.site_i_control_quantile < 1.0:
            raise ValueError("site_i_control_quantile must be in [0.99, 1)")
        if self.site_i_local_radius_cells <= 0:
            raise ValueError("site_i_local_radius_cells must be positive")
        if self.site_i_local_minimum_references < 4:
            raise ValueError(
                "site_i_local_minimum_references must be at least four"
            )
        if self.site_i_candidate_border_cells <= 0:
            raise ValueError("site_i_candidate_border_cells must be positive")
        if not 0.99 <= self.expert_review_control_quantile < 1.0:
            raise ValueError("expert_review_control_quantile must be in [0.99, 1)")
        if self.expert_review_minimum_motif_z <= 0:
            raise ValueError("expert_review_minimum_motif_z must be positive")
        if self.expert_review_cluster_radius_cells <= 0:
            raise ValueError("expert_review_cluster_radius_cells must be positive")
        if self.expert_review_display_radius_cells <= 0:
            raise ValueError("expert_review_display_radius_cells must be positive")
        if self.expert_review_boundary_warning_cells <= 0:
            raise ValueError("expert_review_boundary_warning_cells must be positive")


@dataclass(frozen=True)
class CIFStructure:
    """Expanded crystallographic structure in fractional coordinates."""

    cell_lengths_angstrom: tuple[float, float, float]
    cell_angles_degrees: tuple[float, float, float]
    space_group_number: int
    fractional_coordinates: np.ndarray
    species: np.ndarray
    symmetry_operation_count: int
    symmetry_operations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectedTemplate:
    """One primitive projected cell for a candidate zone axis."""

    zone_axis: tuple[int, int, int]
    phase_uv: np.ndarray
    species: np.ndarray
    projected_basis_cell_fraction: np.ndarray


@dataclass(frozen=True)
class ProjectedFrameworkSkeleton:
    """Periodic Si-O connectivity projected into one canonical CIF cell."""

    zone_axis: tuple[int, int, int]
    atom_uv: np.ndarray
    species: np.ndarray
    bond_segments_uv: np.ndarray
    three_dimensional_bond_count: int
    bond_cutoff_angstrom: float


@dataclass(frozen=True)
class RectangularProjectedFramework:
    """Physical-metric [101] FAU supercell with orthogonal display axes."""

    atom_fractional_xy: np.ndarray
    species: np.ndarray
    bond_segments_fractional_xy: np.ndarray
    width_angstrom: float
    height_angstrom: float
    three_dimensional_bond_count: int
    bond_cutoff_angstrom: float


@dataclass
class StackPeakResult:
    """Plane-wise peaks; persistence is evidence, not a universal gate."""

    stack_shape: tuple[int, int, int]
    best_plane_index: int
    selected_plane_indices: np.ndarray
    focus_scores: np.ndarray
    best_image: np.ndarray
    fused_score: np.ndarray
    points_xy: np.ndarray
    best_plane_by_peak: np.ndarray
    persistence_count: np.ndarray
    framework_eligible: np.ndarray
    standardized_brightness: np.ndarray
    local_contrast: np.ndarray
    species_brightness_class: np.ndarray
    assignment_probability: np.ndarray
    mixture_means: np.ndarray
    mixture_standard_deviations: np.ndarray
    mixture_separation_sigma: np.ndarray


@dataclass
class TemplateFit:
    """Periodic image-to-CIF projection fit for one stack."""

    direct_basis_pixels: np.ndarray
    zone_axis: tuple[int, int, int]
    zone_scores: dict[str, float]
    zone_score_margin: float
    unit_cell_transform: np.ndarray
    phase_shift_uv: np.ndarray
    expected_points_xy: np.ndarray
    expected_species: np.ndarray
    matched_observed_indices: np.ndarray
    matched_expected_indices: np.ndarray
    match_distances_pixels: np.ndarray
    framework_observed_count: int
    template_ncc: float
    angstrom_per_pixel: float


@dataclass
class LocalDeformation:
    """Local affine strain fitted at matched framework sites."""

    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    rotation: np.ndarray
    distortion_magnitude: np.ndarray
    maximum_shear: np.ndarray
    fit_rmse_pixels: np.ndarray
    neighbor_count: np.ndarray
    reliable: np.ndarray
    nearest_guest_distance_pixels: np.ndarray


@dataclass
class RareEarthEvidence:
    """Auditable intersection of distortion and extreme-brightness evidence."""

    screen_positive: np.ndarray
    candidate: np.ndarray
    inside_hotspot_region: np.ndarray
    extreme_brightness: np.ndarray
    local_brightness_z: np.ndarray
    global_brightness_z: np.ndarray
    nearest_hotspot_distance_pixels: np.ndarray
    hotspot_framework_site: np.ndarray
    distortion_threshold: float
    maximum_shear_threshold: float
    framework_brightness_threshold: float
    evidence_score: np.ndarray
    negative_control_tail_probability: np.ndarray


@dataclass
class AlignedSiteContrast:
    """Contrast measured at every CIF-projected O/Si site in one field."""

    expected_xy: np.ndarray
    canonical_uv: np.ndarray
    canonical_site_id: np.ndarray
    expected_species: np.ndarray
    matched: np.ndarray
    observed_peak_index: np.ndarray
    measurement_xy: np.ndarray
    residual_pixels: np.ndarray
    association_reliable: np.ndarray
    best_plane: np.ndarray
    persistence_planes: np.ndarray
    standardized_contrast: np.ndarray
    raw_dog_contrast: np.ndarray
    raw_detector_intensity: np.ndarray
    brightness_class: np.ndarray
    motif_normalized_contrast_z: np.ndarray
    local_same_species_contrast_z: np.ndarray
    local_same_species_reference_count: np.ndarray
    inside_hotspot_region: np.ndarray
    distance_to_image_edge_pixels: np.ndarray
    inside_reliable_interior: np.ndarray
    contrast_anomaly: np.ndarray
    screen_positive: np.ndarray
    candidate: np.ndarray
    overlaps_peak_gate_screen_positive: np.ndarray
    negative_control_tail_probability: np.ndarray


@dataclass
class SkeletonFirstSiteEvidence:
    """Template-empty FAU site-I probes evaluated before intensity support."""

    expected_xy: np.ndarray
    canonical_site_id: np.ndarray
    oxygen_endpoint_xy: np.ndarray
    model_valid: np.ndarray
    best_plane: np.ndarray
    template_only_bic: np.ndarray
    template_plus_site_bic: np.ndarray
    delta_bic: np.ndarray
    extra_component_amplitude: np.ndarray
    local_equivalent_site_z: np.ndarray
    local_reference_count: np.ndarray
    distance_to_image_edge_pixels: np.ndarray
    inside_reliable_interior: np.ndarray
    raw_structural_screen_positive: np.ndarray
    structural_review: np.ndarray
    local_brightness_anomaly: np.ndarray
    candidate: np.ndarray
    negative_control_tail_probability: np.ndarray
    calibrated_delta_bic_threshold: float


@dataclass
class FAUImageAnalysis:
    """All image-only and CIF-conditioned evidence for one sample."""

    label: str
    source_path: Path
    source_sha256: str
    peaks: StackPeakResult
    template_fit: TemplateFit
    deformation: LocalDeformation
    rare_earth_evidence: RareEarthEvidence
    aligned_site_contrast: AlignedSiteContrast | None = None
    skeleton_first_sites: SkeletonFirstSiteEvidence | None = None


@dataclass
class FAUStackAnalysis:
    """Y-referenced comparison of several independently sampled fields."""

    cif_path: Path
    cif_sha256: str
    structure: CIFStructure
    reference_label: str
    reference_is_rare_earth_free: bool
    zone_axis_source: str
    images: list[FAUImageAnalysis]
    global_comparisons: dict[str, dict[str, Any]]
    local_deformation_comparisons: dict[str, dict[str, Any]]
    guest_onset_comparisons: dict[str, dict[str, Any]]
    guest_distribution_comparisons: dict[str, dict[str, Any]]
    negative_control_audit: dict[str, Any] | None
    aligned_site_negative_control_audit: dict[str, Any] | None
    skeleton_first_negative_control_audit: dict[str, Any] | None
    config: FAUStackConfig


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strip_number_uncertainty(value: str) -> float:
    return float(re.sub(r"\([^)]*\)$", "", value))


def _read_cif_tokens(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append(shlex.split(stripped, comments=True, posix=True))
    return rows


def _cif_loops(rows: Sequence[Sequence[str]]) -> list[tuple[list[str], list[list[str]]]]:
    loops: list[tuple[list[str], list[list[str]]]] = []
    index = 0
    while index < len(rows):
        if rows[index] != ["loop_"]:
            index += 1
            continue
        index += 1
        headers: list[str] = []
        while index < len(rows) and rows[index] and rows[index][0].startswith("_"):
            headers.extend(rows[index])
            index += 1
        values: list[str] = []
        while index < len(rows):
            first = rows[index][0]
            if first == "loop_" or first.startswith("_") or first.startswith("data_"):
                break
            values.extend(rows[index])
            index += 1
        if headers:
            if len(values) % len(headers):
                raise ValueError("CIF loop row width is inconsistent")
            data = [
                values[offset : offset + len(headers)]
                for offset in range(0, len(values), len(headers))
            ]
            loops.append((headers, data))
    return loops


def _safe_coordinate_expression(expression: str, xyz: np.ndarray) -> float:
    """Evaluate one CIF symmetry coordinate with a tiny AST whitelist."""

    node = ast.parse(expression, mode="eval").body
    variables = {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}

    def visit(item: ast.AST) -> float:
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return float(item.value)
        if isinstance(item, ast.Name) and item.id in variables:
            return variables[item.id]
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
            value = visit(item.operand)
            return value if isinstance(item.op, ast.UAdd) else -value
        if isinstance(item, ast.BinOp) and isinstance(
            item.op, (ast.Add, ast.Sub, ast.Div)
        ):
            left, right = visit(item.left), visit(item.right)
            if isinstance(item.op, ast.Add):
                return left + right
            if isinstance(item.op, ast.Sub):
                return left - right
            if abs(right) < 1e-15:
                raise ValueError("Division by zero in CIF symmetry operation")
            return left / right
        raise ValueError(f"Unsafe CIF symmetry expression: {expression!r}")

    return visit(node)


def parse_and_expand_cif(path: str | Path) -> CIFStructure:
    """Parse the explicit CIF symmetry loop and expand unique atoms safely."""

    source = Path(path).resolve()
    rows = _read_cif_tokens(source)
    scalar: dict[str, str] = {}
    for row in rows:
        if len(row) >= 2 and row[0].startswith("_"):
            scalar[row[0].lower()] = row[1]
    required = (
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    )
    if any(name not in scalar for name in required):
        raise ValueError("CIF is missing required unit-cell fields")
    lengths = tuple(_strip_number_uncertainty(scalar[name]) for name in required[:3])
    angles = tuple(_strip_number_uncertainty(scalar[name]) for name in required[3:])
    group_text = scalar.get("_symmetry_int_tables_number", "0")
    space_group = int(float(group_text))

    symops: list[str] | None = None
    asymmetric: list[tuple[str, np.ndarray]] | None = None
    for headers, data in _cif_loops(rows):
        lowered = [item.lower() for item in headers]
        if "_symmetry_equiv_pos_as_xyz" in lowered:
            column = lowered.index("_symmetry_equiv_pos_as_xyz")
            symops = [row[column] for row in data]
        if "_atom_site_type_symbol" in lowered:
            needed = (
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
            )
            if any(name not in lowered for name in needed):
                raise ValueError("CIF atom loop lacks fractional coordinates")
            columns = [lowered.index(name) for name in needed]
            asymmetric = [
                (
                    row[columns[0]],
                    np.asarray(
                        [_strip_number_uncertainty(row[column]) for column in columns[1:]],
                        dtype=float,
                    ),
                )
                for row in data
            ]
    if not symops:
        raise ValueError("CIF must contain explicit symmetry operations")
    if not asymmetric:
        raise ValueError("CIF contains no atom sites")

    expanded: dict[tuple[str, int, int, int], tuple[str, np.ndarray]] = {}
    for species, coordinate in asymmetric:
        for operation in symops:
            components = [item.strip() for item in operation.split(",")]
            if len(components) != 3:
                raise ValueError(f"Invalid CIF symmetry operation: {operation!r}")
            transformed = np.mod(
                [_safe_coordinate_expression(item, coordinate) for item in components],
                1.0,
            )
            transformed[np.isclose(transformed, 1.0, atol=1e-8)] = 0.0
            key = (species, *(np.rint(transformed * 10**7).astype(int).tolist()))
            expanded[key] = (species, transformed)
    species = np.asarray([item[0] for item in expanded.values()], dtype=object)
    coordinates = np.asarray([item[1] for item in expanded.values()], dtype=float)
    if any(str(item).lower().startswith(("re", "y", "la", "ce", "nd", "sm")) for item in species):
        raise ValueError("This workflow requires a guest-free Si/O framework CIF")
    return CIFStructure(
        cell_lengths_angstrom=lengths,
        cell_angles_degrees=angles,
        space_group_number=space_group,
        fractional_coordinates=coordinates,
        species=species,
        symmetry_operation_count=len(symops),
        symmetry_operations=tuple(symops),
    )


def _projection_axes(zone_axis: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beam = np.asarray(zone_axis, dtype=float)
    if beam.shape != (3,) or np.linalg.norm(beam) <= 0:
        raise ValueError("Zone axis must be a non-zero three-vector")
    beam /= np.linalg.norm(beam)
    anchor = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(anchor, beam))) > 0.9:
        anchor = np.asarray([0.0, 1.0, 0.0])
    first = np.cross(anchor, beam)
    first /= np.linalg.norm(first)
    second = np.cross(beam, first)
    return first, second, beam


def _projected_bravais_basis(zone_axis: Sequence[int]) -> np.ndarray:
    first, second, _ = _projection_axes(zone_axis)
    centers = np.asarray(
        ([0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]),
        dtype=float,
    )
    translations = []
    for integer in itertools.product(range(-2, 3), repeat=3):
        for center in centers:
            vector = np.asarray(integer, dtype=float) + center
            translations.append([np.dot(vector, first), np.dot(vector, second)])
    projected = np.unique(np.round(np.asarray(translations), 9), axis=0)
    nonzero = projected[np.linalg.norm(projected, axis=1) > 1e-8]
    order = np.argsort(np.linalg.norm(nonzero, axis=1))
    candidates = nonzero[order[: min(80, len(order))]]
    best: tuple[float, float, np.ndarray] | None = None
    for index, first_vector in enumerate(candidates):
        for second_vector in candidates[index + 1 :]:
            basis = np.column_stack([first_vector, second_vector])
            determinant = abs(float(np.linalg.det(basis)))
            if determinant < 1e-8:
                continue
            cosine = abs(
                float(np.dot(first_vector, second_vector))
                / (np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
            )
            ratio = max(np.linalg.norm(first_vector), np.linalg.norm(second_vector)) / min(
                np.linalg.norm(first_vector), np.linalg.norm(second_vector)
            )
            if cosine > 0.87 or ratio > 3.0:
                continue
            coefficients = (np.linalg.inv(basis) @ projected.T).T
            if float(np.max(np.abs(coefficients - np.rint(coefficients)))) > 2e-7:
                continue
            score = (determinant, float(np.sum(np.linalg.norm(basis, axis=0))))
            if best is None or score[:2] < best[:2]:
                best = (score[0], score[1], basis)
    if best is None:
        raise ValueError(f"Could not derive projected Bravais basis for {zone_axis}")
    return best[2]


def project_structure(
    structure: CIFStructure, zone_axis: Sequence[int]
) -> ProjectedTemplate:
    """Collapse an expanded cubic structure into one primitive projected cell."""

    if not np.allclose(structure.cell_lengths_angstrom, structure.cell_lengths_angstrom[0]):
        raise ValueError("Current projection implementation requires a cubic cell")
    if not np.allclose(structure.cell_angles_degrees, 90.0):
        raise ValueError("Current projection implementation requires orthogonal axes")
    first, second, _ = _projection_axes(zone_axis)
    basis = _projected_bravais_basis(zone_axis)
    projected = np.column_stack(
        [structure.fractional_coordinates @ first, structure.fractional_coordinates @ second]
    )
    phase = (np.linalg.inv(basis) @ projected.T).T % 1.0
    unique: dict[tuple[str, int, int], tuple[np.ndarray, str]] = {}
    for point, species in zip(phase, structure.species):
        wrapped = point % 1.0
        wrapped[np.isclose(wrapped, 1.0, atol=1e-7)] = 0.0
        key = (str(species), *(np.rint(wrapped * 10**6).astype(int).tolist()))
        unique[key] = (wrapped, str(species))
    points = np.asarray([item[0] for item in unique.values()], dtype=float)
    species = np.asarray([item[1] for item in unique.values()], dtype=object)
    return ProjectedTemplate(
        zone_axis=tuple(int(item) for item in zone_axis),
        phase_uv=points,
        species=species,
        projected_basis_cell_fraction=basis,
    )


def _expand_cif_special_position(
    structure: CIFStructure,
    fractional_seed: Sequence[float],
) -> np.ndarray:
    """Expand one non-framework crystallographic seed with the CIF operators."""

    seed = np.asarray(fractional_seed, dtype=float)
    if seed.shape != (3,):
        raise ValueError("fractional_seed must contain three coordinates")
    if not structure.symmetry_operations:
        raise ValueError(
            "Special-position projection requires retained CIF symmetry operations"
        )
    expanded: dict[tuple[int, int, int], np.ndarray] = {}
    for operation in structure.symmetry_operations:
        components = [item.strip() for item in operation.split(",")]
        if len(components) != 3:
            raise ValueError(f"Invalid CIF symmetry operation: {operation!r}")
        transformed = np.mod(
            [
                _safe_coordinate_expression(item, seed)
                for item in components
            ],
            1.0,
        )
        transformed[np.isclose(transformed, 1.0, atol=1e-8)] = 0.0
        key = tuple(np.rint(transformed * 10**7).astype(int).tolist())
        expanded[key] = transformed
    return np.asarray(list(expanded.values()), dtype=float)


def project_fau_site_i_template(
    structure: CIFStructure,
    zone_axis: Sequence[int] = (1, 0, 1),
) -> ProjectedTemplate:
    """Project the FAU 16c/site-I D6R centers into one primitive cell.

    The guest-free framework CIF does not list charge-balancing cations.  In
    the supplied FAU setting, site I is the 16c special position with seed
    ``(0, 0, 0)``.  It is used only as a template-empty structural probe; a
    positive image signal at the projection is not, by itself, a chemical ID.
    """

    coordinates = _expand_cif_special_position(structure, (0.0, 0.0, 0.0))
    first, second, _ = _projection_axes(zone_axis)
    basis = _projected_bravais_basis(zone_axis)
    projected = np.column_stack(
        [coordinates @ first, coordinates @ second]
    )
    phase = (np.linalg.inv(basis) @ projected.T).T % 1.0
    unique: dict[tuple[int, int], np.ndarray] = {}
    for point in phase:
        wrapped = point % 1.0
        wrapped[np.isclose(wrapped, 1.0, atol=1e-7)] = 0.0
        key = tuple(np.rint(wrapped * 10**6).astype(int).tolist())
        unique[key] = wrapped
    points = np.asarray(list(unique.values()), dtype=float)
    return ProjectedTemplate(
        zone_axis=tuple(int(item) for item in zone_axis),
        phase_uv=points,
        species=np.asarray(["FAU_site_I"] * len(points), dtype=object),
        projected_basis_cell_fraction=basis,
    )


def _clip_segment_to_unit_square(
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray | None:
    """Clip one line segment to [0, 1] x [0, 1] with Liang-Barsky bounds."""

    first = np.asarray(start, dtype=float)
    last = np.asarray(end, dtype=float)
    delta = last - first
    lower_t, upper_t = 0.0, 1.0
    for coordinate in range(2):
        if abs(float(delta[coordinate])) <= 1e-12:
            if first[coordinate] < 0.0 or first[coordinate] > 1.0:
                return None
            continue
        entry = (0.0 - first[coordinate]) / delta[coordinate]
        leave = (1.0 - first[coordinate]) / delta[coordinate]
        if entry > leave:
            entry, leave = leave, entry
        lower_t = max(lower_t, float(entry))
        upper_t = min(upper_t, float(leave))
        if lower_t > upper_t:
            return None
    clipped = np.stack([first + lower_t * delta, first + upper_t * delta])
    if float(np.linalg.norm(clipped[1] - clipped[0])) <= 1e-10:
        return None
    return np.clip(clipped, 0.0, 1.0)


def _periodic_segment_fragments(
    start_uv: np.ndarray,
    end_uv: np.ndarray,
) -> np.ndarray:
    """Represent a short periodic bond without drawing a false cell-spanning line."""

    start = np.asarray(start_uv, dtype=float) % 1.0
    raw_delta = np.asarray(end_uv, dtype=float) - np.asarray(start_uv, dtype=float)
    delta = (raw_delta + 0.5) % 1.0 - 0.5
    end = start + delta
    fragments: list[np.ndarray] = []
    for shift in itertools.product((-1.0, 0.0, 1.0), repeat=2):
        translated_start = start + np.asarray(shift, dtype=float)
        clipped = _clip_segment_to_unit_square(
            translated_start,
            translated_start + delta,
        )
        if clipped is not None:
            fragments.append(clipped)
    if not fragments:
        return np.empty((0, 2, 2), dtype=float)
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for fragment in fragments:
        key = tuple(np.round(fragment.ravel(), 8).tolist())
        reverse = tuple(np.round(fragment[::-1].ravel(), 8).tolist())
        unique[min(key, reverse)] = fragment
    return np.asarray(list(unique.values()), dtype=float)


def _periodic_si_o_bond_endpoints(
    structure: CIFStructure,
    bond_cutoff_angstrom: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return minimum-image 3D endpoints after validating FAU Si coordination."""

    if bond_cutoff_angstrom <= 0:
        raise ValueError("bond_cutoff_angstrom must be positive")
    silicon_indices = np.flatnonzero(structure.species == "Si")
    oxygen_indices = np.flatnonzero(structure.species == "O")
    if not len(silicon_indices) or not len(oxygen_indices):
        raise ValueError("Projected FAU skeleton requires both Si and O sites")
    silicon = structure.fractional_coordinates[silicon_indices]
    oxygen = structure.fractional_coordinates[oxygen_indices]
    displacement = oxygen[None, :, :] - silicon[:, None, :]
    displacement -= np.rint(displacement)
    lengths = np.asarray(structure.cell_lengths_angstrom, dtype=float)
    distances = np.linalg.norm(displacement * lengths[None, None, :], axis=2)
    bonded = distances <= bond_cutoff_angstrom
    coordination = np.sum(bonded, axis=1)
    if not np.all(coordination == 4):
        raise ValueError(
            "FAU Si-O cutoff must recover four O neighbors for every Si; "
            f"observed coordination values: {np.unique(coordination).tolist()}"
        )
    pairs = np.argwhere(bonded)
    starts = silicon[pairs[:, 0]]
    ends = starts + displacement[pairs[:, 0], pairs[:, 1]]
    return starts, ends, int(len(pairs))


def project_fau_framework_skeleton(
    structure: CIFStructure,
    zone_axis: Sequence[int],
    *,
    bond_cutoff_angstrom: float = 2.1,
) -> ProjectedFrameworkSkeleton:
    """Infer periodic 3D Si-O bonds and project them into one FAU cell."""

    template = project_structure(structure, zone_axis)
    starts, ends, bond_count = _periodic_si_o_bond_endpoints(
        structure, bond_cutoff_angstrom
    )

    first_axis, second_axis, _ = _projection_axes(zone_axis)
    inverse_projected_basis = np.linalg.inv(
        template.projected_basis_cell_fraction
    )
    segments: dict[tuple[float, ...], np.ndarray] = {}
    for start_xyz, end_xyz in zip(starts, ends):
        start_projected = np.asarray(
            [np.dot(start_xyz, first_axis), np.dot(start_xyz, second_axis)]
        )
        end_projected = np.asarray(
            [np.dot(end_xyz, first_axis), np.dot(end_xyz, second_axis)]
        )
        start_uv = inverse_projected_basis @ start_projected
        end_uv = inverse_projected_basis @ end_projected
        for fragment in _periodic_segment_fragments(start_uv, end_uv):
            key = tuple(np.round(fragment.ravel(), 6).tolist())
            reverse = tuple(np.round(fragment[::-1].ravel(), 6).tolist())
            segments[min(key, reverse)] = fragment
    return ProjectedFrameworkSkeleton(
        zone_axis=tuple(int(item) for item in zone_axis),
        atom_uv=template.phase_uv,
        species=template.species,
        bond_segments_uv=np.asarray(list(segments.values()), dtype=float),
        three_dimensional_bond_count=bond_count,
        bond_cutoff_angstrom=float(bond_cutoff_angstrom),
    )


def project_fau_101_rectangular_framework(
    structure: CIFStructure,
    *,
    bond_cutoff_angstrom: float = 2.1,
) -> RectangularProjectedFramework:
    """Project FAU along [101] into an orthogonal [bar101] x [010] supercell."""

    if not np.allclose(structure.cell_lengths_angstrom, structure.cell_lengths_angstrom[0]):
        raise ValueError("Physical rectangular [101] projection requires a cubic cell")
    starts, ends, bond_count = _periodic_si_o_bond_endpoints(
        structure, bond_cutoff_angstrom
    )

    def rectangular_fractional(coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinates, dtype=float)
        return np.column_stack(
            [
                (-values[:, 0] + values[:, 2]) / 2.0,
                values[:, 1],
            ]
        )

    # The [-101] x [010] rectangle has area four times the primitive projected
    # Bravais cell. A single conventional-cell coordinate list does not tile
    # that rectangle after projection: translating the crystal by [001]
    # advances the projected coordinate by half of the rectangular width.
    # Include that neighboring-cell copy before wrapping, otherwise the
    # central vertical O columns (and their bonds) are silently omitted.
    rectangular_tiling_translations = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=float
    )
    tiled_coordinates = np.concatenate(
        [
            structure.fractional_coordinates + translation
            for translation in rectangular_tiling_translations
        ],
        axis=0,
    )
    atom_xy = rectangular_fractional(tiled_coordinates) % 1.0
    tiled_species = np.tile(
        structure.species, len(rectangular_tiling_translations)
    )
    unique_atoms: dict[tuple[str, int, int], tuple[np.ndarray, str]] = {}
    for point, species in zip(atom_xy, tiled_species):
        key = (str(species), *(np.rint(point * 10**6).astype(int).tolist()))
        unique_atoms[key] = (point, str(species))

    tiled_starts = np.concatenate(
        [starts + translation for translation in rectangular_tiling_translations],
        axis=0,
    )
    tiled_ends = np.concatenate(
        [ends + translation for translation in rectangular_tiling_translations],
        axis=0,
    )
    start_xy = rectangular_fractional(tiled_starts)
    end_xy = rectangular_fractional(tiled_ends)
    segments: dict[tuple[float, ...], np.ndarray] = {}
    for start, end in zip(start_xy, end_xy):
        for fragment in _periodic_segment_fragments(start, end):
            key = tuple(np.round(fragment.ravel(), 6).tolist())
            reverse = tuple(np.round(fragment[::-1].ravel(), 6).tolist())
            segments[min(key, reverse)] = fragment
    lattice_constant = float(structure.cell_lengths_angstrom[0])
    return RectangularProjectedFramework(
        atom_fractional_xy=np.asarray(
            [item[0] for item in unique_atoms.values()], dtype=float
        ),
        species=np.asarray(
            [item[1] for item in unique_atoms.values()], dtype=object
        ),
        bond_segments_fractional_xy=np.asarray(
            list(segments.values()), dtype=float
        ),
        width_angstrom=math.sqrt(2.0) * lattice_constant,
        height_angstrom=lattice_constant,
        three_dimensional_bond_count=(
            bond_count * len(rectangular_tiling_translations)
        ),
        bond_cutoff_angstrom=float(bond_cutoff_angstrom),
    )


def project_fau_101_rectangular_site_i(
    structure: CIFStructure,
) -> np.ndarray:
    """Return FAU 16c/site-I probes in the physical [bar101] x [010] cell."""

    coordinates = _expand_cif_special_position(
        structure, (0.0, 0.0, 0.0)
    )
    translations = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=float
    )
    tiled = np.concatenate(
        [coordinates + translation for translation in translations], axis=0
    )
    projected = np.column_stack(
        [(-tiled[:, 0] + tiled[:, 2]) / 2.0, tiled[:, 1]]
    ) % 1.0
    return np.unique(np.round(projected, 7), axis=0)


def _fold_image_to_rectangular_101_cell(
    image: np.ndarray,
    fit: TemplateFit,
    structure: CIFStructure,
    bins_y: int,
) -> np.ndarray:
    """Fold an image into the physical [bar101] x [010] rectangular supercell."""

    if tuple(fit.zone_axis) != (1, 0, 1):
        raise ValueError("Rectangular physical overlay currently requires zone axis [101]")
    template = project_structure(structure, fit.zone_axis)
    height, width = image.shape
    rows, columns = np.indices((height, width), dtype=np.float32)
    xy = np.stack([columns.ravel(), rows.ravel()])
    image_uv = np.linalg.inv(fit.direct_basis_pixels) @ xy
    canonical_uv = np.linalg.inv(fit.unit_cell_transform) @ (
        image_uv - fit.phase_shift_uv[:, None]
    )
    projected_plane = template.projected_basis_cell_fraction @ canonical_uv
    rectangle_x = (projected_plane[1] / math.sqrt(2.0)) % 1.0
    rectangle_y = projected_plane[0] % 1.0
    bins_x = int(round(math.sqrt(2.0) * bins_y))
    indices_x = np.minimum((rectangle_x * bins_x).astype(int), bins_x - 1)
    indices_y = np.minimum((rectangle_y * bins_y).astype(int), bins_y - 1)
    linear = indices_y * bins_x + indices_x
    weighted = np.bincount(
        linear,
        weights=np.asarray(image, dtype=float).ravel(),
        minlength=bins_y * bins_x,
    )
    counts = np.bincount(linear, minlength=bins_y * bins_x)
    return (weighted / np.maximum(counts, 1)).reshape(bins_y, bins_x)


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if scale <= 1e-8:
        scale = max(float(np.std(values)), 1e-8)
    return center, scale


def _em_mixture_from_quantiles(
    samples: np.ndarray,
    quantiles: Sequence[float],
) -> OrderedIntensityMixture:
    """Fit one ordered three-Gaussian candidate from a specified start."""

    values = np.asarray(samples, dtype=float).reshape(-1)
    total_variance = max(float(np.var(values)), 1e-8)
    variance_floor = max(0.002 * total_variance, 1e-8)
    means = np.percentile(values, quantiles)
    variances = np.full(3, 0.15 * total_variance)
    weights = np.full(3, 1.0 / 3.0)
    previous = -np.inf
    for _ in range(400):
        log_density = (
            np.log(weights + 1e-15)[None, :]
            - 0.5 * np.log(2.0 * np.pi * variances)[None, :]
            - 0.5 * (values[:, None] - means[None, :]) ** 2 / variances[None, :]
        )
        maximum = np.max(log_density, axis=1, keepdims=True)
        normalizer = maximum[:, 0] + np.log(
            np.sum(np.exp(log_density - maximum), axis=1)
        )
        responsibilities = np.exp(log_density - normalizer[:, None])
        effective_count = np.sum(responsibilities, axis=0) + 1e-12
        weights = effective_count / len(values)
        means = np.sum(responsibilities * values[:, None], axis=0) / effective_count
        variances = np.maximum(
            np.sum(
                responsibilities * (values[:, None] - means[None, :]) ** 2,
                axis=0,
            )
            / effective_count,
            variance_floor,
        )
        log_likelihood = float(np.sum(normalizer))
        if abs(log_likelihood - previous) <= 1e-7 * max(1.0, abs(log_likelihood)):
            break
        previous = log_likelihood
    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    weights = weights[order]
    responsibilities = responsibilities[:, order]
    labels = np.argmax(responsibilities, axis=1)
    pooled = np.sqrt(0.5 * (variances[:-1] + variances[1:]))
    return OrderedIntensityMixture(
        means=means,
        standard_deviations=np.sqrt(variances),
        weights=weights,
        responsibilities=responsibilities,
        labels=labels,
        separation_sigma=np.diff(means) / np.maximum(pooled, 1e-12),
        log_likelihood=log_likelihood,
    )


def fit_constrained_brightness_mixture(values: np.ndarray) -> OrderedIntensityMixture:
    """Keep three occupied, separated brightness populations when supported.

    Maximum likelihood alone can increase its score by splitting the dominant
    dim population and leaving another component with no hard assignments.  A
    chemical name must never be attached to such an empty population.  We
    therefore select the best-likelihood restart among candidates whose three
    hard populations are occupied and whose adjacent means are at least 0.8
    pooled standard deviations apart.
    """

    samples = np.asarray(values, dtype=float).reshape(-1)
    ordinary = fit_ordered_intensity_mixture(
        samples, AtomicSiteInventoryConfig(minimum_sites_per_image=60)
    )
    candidates = [ordinary]
    for quantiles in (
        (5, 45, 88),
        (10, 55, 92),
        (20, 60, 95),
        (25, 70, 97),
        (10, 75, 95),
        (20, 72, 92),
        (30, 75, 93),
        (50, 75, 90),
    ):
        candidates.append(_em_mixture_from_quantiles(samples, quantiles))
    minimum_count = max(20, int(math.ceil(0.01 * len(samples))))

    def supported(candidate: OrderedIntensityMixture) -> bool:
        counts = np.bincount(candidate.labels, minlength=3)
        return bool(
            np.all(counts >= minimum_count)
            and np.min(candidate.separation_sigma) >= 0.8
        )

    admissible = [candidate for candidate in candidates if supported(candidate)]
    if not admissible:
        raise ValueError(
            "Three occupied, separated brightness populations are not supported"
        )
    return max(admissible, key=lambda candidate: candidate.log_likelihood)


def _subpixel_centroids(response: np.ndarray, coordinates_yx: np.ndarray) -> np.ndarray:
    refined = np.empty((len(coordinates_yx), 2), dtype=float)
    height, width = response.shape
    for index, (row, column) in enumerate(coordinates_yx):
        row_min, row_max = max(0, row - 2), min(height, row + 3)
        column_min, column_max = max(0, column - 2), min(width, column + 3)
        patch = response[row_min:row_max, column_min:column_max]
        floor = float(np.min(patch))
        weights = np.clip(patch - floor, 0.0, None)
        if float(np.sum(weights)) <= 1e-12:
            refined[index] = (float(column), float(row))
            continue
        rows, columns = np.indices(patch.shape, dtype=float)
        total = float(np.sum(weights))
        refined[index] = (
            float(np.sum(weights * columns) / total + column_min),
            float(np.sum(weights * rows) / total + row_min),
        )
    return refined


def load_focus_stack(path: str | Path) -> np.ndarray:
    """Read a 2-D image or QYX TIFF stack without changing its values."""

    try:
        import tifffile
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise ImportError("FAU z-stack analysis requires tifffile") from error
    values = np.asarray(tifffile.imread(Path(path)))
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError("Expected a 2-D image or a QYX TIFF stack")
    if min(values.shape[1:]) < 64:
        raise ValueError("Atomic z-stack frames are unexpectedly small")
    if not np.all(np.isfinite(values)):
        raise ValueError("TIFF stack contains non-finite values")
    return values.astype(np.float32)


def detect_persistent_stack_peaks(
    stack: np.ndarray, config: FAUStackConfig | None = None
) -> StackPeakResult:
    """Detect every plane while reserving persistent focus peaks for the framework.

    The maximum response is taken over all planes so a genuinely slice-local
    bright atom is observable.  Only the framework fit requires repetition in
    the selected focus window; the repetition count remains available as soft
    evidence for every rare-earth candidate.
    """

    policy = config or FAUStackConfig()
    policy.validate()
    values = np.asarray(stack, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Stack peak detection requires a QYX array")
    responses = np.stack(
        [
            gaussian_filter(frame, policy.focus_sigma, mode="reflect")
            - gaussian_filter(frame, policy.background_sigma, mode="reflect")
            for frame in values
        ]
    )
    border = max(policy.border_pixels, 1)
    interior = (slice(None), slice(border, -border), slice(border, -border))
    focus_scores = np.mean(responses[interior] ** 2, axis=(1, 2))
    best = int(np.argmax(focus_scores))
    threshold = policy.focus_fraction_of_best * float(focus_scores[best])
    left = best
    while left > 0 and focus_scores[left - 1] >= threshold:
        left -= 1
    right = best
    while right + 1 < len(focus_scores) and focus_scores[right + 1] >= threshold:
        right += 1
    selected = np.arange(left, right + 1, dtype=int)
    standardized = []
    for index in range(len(responses)):
        center, scale = _robust_center_scale(responses[index, border:-border, border:-border])
        standardized.append((responses[index] - center) / scale)
    standardized_stack = np.asarray(standardized, dtype=np.float32)
    fused = np.max(standardized_stack, axis=0)
    valid = np.zeros(fused.shape, dtype=bool)
    valid[border:-border, border:-border] = True
    coordinates = peak_local_max(
        fused,
        min_distance=policy.peak_min_distance_pixels,
        threshold_abs=policy.peak_detection_z,
        exclude_border=False,
        labels=valid.astype(np.uint8),
    )
    points = _subpixel_centroids(fused, coordinates)
    sampled = np.stack(
        [
            map_coordinates(
                plane,
                [points[:, 1], points[:, 0]],
                order=1,
                mode="nearest",
            )
            for plane in standardized_stack
        ],
        axis=1,
    )
    persistence = np.sum(sampled >= policy.persistence_z, axis=1)
    focus_persistence = np.sum(
        sampled[:, selected] >= policy.persistence_z,
        axis=1,
    )
    keep = persistence >= 1
    points = points[keep]
    sampled = sampled[keep]
    persistence = persistence[keep]
    focus_persistence = focus_persistence[keep]
    if len(points) < 60:
        raise ValueError("Too few atomic peaks were detected across the stack")
    best_by_peak = np.argmax(sampled, axis=1)
    standardized_brightness = np.max(sampled, axis=1)
    local_contrast = map_coordinates(
        responses,
        [best_by_peak.astype(float), points[:, 1], points[:, 0]],
        order=1,
        mode="nearest",
    )
    mixture = fit_constrained_brightness_mixture(standardized_brightness)
    classes = np.asarray(
        [BRIGHTNESS_CLASS_ORDER[int(index)] for index in mixture.labels],
        dtype=object,
    )
    required_focus_planes = min(policy.minimum_persistent_planes, len(selected))
    framework_eligible = focus_persistence >= required_focus_planes
    best_image = values[best].astype(np.float32)
    lower, upper = np.percentile(best_image, (1.0, 99.0))
    best_image = np.clip((best_image - lower) / max(float(upper - lower), 1e-8), 0, 1)
    return StackPeakResult(
        stack_shape=tuple(int(item) for item in values.shape),
        best_plane_index=best,
        selected_plane_indices=selected,
        focus_scores=focus_scores,
        best_image=best_image,
        fused_score=fused,
        points_xy=points,
        best_plane_by_peak=best_by_peak,
        persistence_count=persistence,
        framework_eligible=framework_eligible,
        standardized_brightness=standardized_brightness,
        local_contrast=local_contrast,
        species_brightness_class=classes,
        assignment_probability=np.max(mixture.responsibilities, axis=1),
        mixture_means=mixture.means,
        mixture_standard_deviations=mixture.standard_deviations,
        mixture_separation_sigma=mixture.separation_sigma,
    )


def estimate_macro_direct_basis(image: np.ndarray) -> np.ndarray:
    """Estimate the ~70-pixel projected FAU cell rather than atomic harmonics."""

    spectral = SpectralConfig(
        min_frequency=0.008,
        max_frequency=0.025,
        global_peak_count=16,
        global_peak_min_separation=0.003,
        global_peak_radius=0.003,
    )
    reciprocal = estimate_global_reciprocal_peaks(np.asarray(image, dtype=float), spectral)
    selected = select_reciprocal_basis(reciprocal, minimum_angle_degrees=35.0)
    height, width = image.shape
    standardized = (image - np.mean(image)) / (np.std(image) + 1e-12)
    power = np.abs(
        np.fft.fftshift(
            np.fft.fft2(standardized * np.outer(np.hanning(height), np.hanning(width)))
        )
    ) ** 2
    log_power = np.log(power + np.percentile(power, 10) + 1e-12)
    frequencies_y = np.fft.fftshift(np.fft.fftfreq(height))
    frequencies_x = np.fft.fftshift(np.fft.fftfreq(width))

    def refine(peak: tuple[float, float]) -> tuple[float, float]:
        row = int(np.argmin(np.abs(frequencies_y - peak[0])))
        column = int(np.argmin(np.abs(frequencies_x - peak[1])))
        if row <= 0 or row + 1 >= height or column <= 0 or column + 1 >= width:
            return peak

        def offset(first: float, center: float, last: float) -> float:
            denominator = first - 2.0 * center + last
            if denominator >= -1e-12:
                return 0.0
            return float(np.clip(0.5 * (first - last) / denominator, -0.5, 0.5))

        delta_y = offset(log_power[row - 1, column], log_power[row, column], log_power[row + 1, column])
        delta_x = offset(log_power[row, column - 1], log_power[row, column], log_power[row, column + 1])
        return (
            float(frequencies_y[row] + delta_y / height),
            float(frequencies_x[column] + delta_x / width),
        )

    basis = direct_basis_from_peaks([refine(selected[0]), refine(selected[1])])
    if not 2500 <= abs(float(np.linalg.det(basis))) <= 10000:
        raise ValueError("Estimated FAU projected-cell area is outside the supported range")
    return basis


def _fold_image_to_cell(image: np.ndarray, basis: np.ndarray, bins: int) -> np.ndarray:
    height, width = image.shape
    rows, columns = np.indices((height, width), dtype=np.float32)
    xy = np.stack([columns.ravel(), rows.ravel()])
    uv = (np.linalg.inv(basis) @ xy) % 1.0
    indices_x = np.minimum((uv[0] * bins).astype(int), bins - 1)
    indices_y = np.minimum((uv[1] * bins).astype(int), bins - 1)
    linear = indices_y * bins + indices_x
    weighted = np.bincount(linear, weights=image.ravel(), minlength=bins * bins)
    counts = np.bincount(linear, minlength=bins * bins)
    folded = (weighted / np.maximum(counts, 1)).reshape(bins, bins)
    return folded


def _fold_image_to_canonical_cif_cell(
    image: np.ndarray,
    fit: TemplateFit,
    bins: int,
) -> np.ndarray:
    """Average an arbitrary field of view into the fitted canonical CIF cell."""

    height, width = image.shape
    rows, columns = np.indices((height, width), dtype=np.float32)
    xy = np.stack([columns.ravel(), rows.ravel()])
    image_uv = np.linalg.inv(fit.direct_basis_pixels) @ xy
    canonical_uv = (
        np.linalg.inv(fit.unit_cell_transform)
        @ (image_uv - fit.phase_shift_uv[:, None])
    ) % 1.0
    indices_x = np.minimum((canonical_uv[0] * bins).astype(int), bins - 1)
    indices_y = np.minimum((canonical_uv[1] * bins).astype(int), bins - 1)
    linear = indices_y * bins + indices_x
    weighted = np.bincount(
        linear, weights=np.asarray(image, dtype=float).ravel(), minlength=bins * bins
    )
    counts = np.bincount(linear, minlength=bins * bins)
    return (weighted / np.maximum(counts, 1)).reshape(bins, bins)


def _unimodular_matrices() -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for values in itertools.product(range(-2, 3), repeat=4):
        matrix = np.asarray(values, dtype=int).reshape(2, 2)
        if abs(int(round(np.linalg.det(matrix)))) == 1:
            matrices.append(matrix)
    return matrices


def _render_template(
    template: ProjectedTemplate,
    transform: np.ndarray,
    bins: int,
    sigma: float,
) -> np.ndarray:
    phase = (transform @ template.phase_uv.T).T % 1.0
    x = np.minimum((phase[:, 0] * bins).astype(int), bins - 1)
    y = np.minimum((phase[:, 1] * bins).astype(int), bins - 1)
    weights = np.where(template.species == "Si", 2.0, 1.0)
    rendered = np.zeros((bins, bins), dtype=float)
    np.add.at(rendered, (y, x), weights)
    return gaussian_filter(rendered, sigma=sigma, mode="wrap")


def _circular_ncc(reference: np.ndarray, moving: np.ndarray) -> tuple[float, np.ndarray]:
    first = reference - np.mean(reference)
    second = moving - np.mean(moving)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second)) + 1e-12
    correlation = np.fft.ifft2(np.fft.fft2(first) * np.conj(np.fft.fft2(second))).real
    index = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    return float(correlation[index] / denominator), np.asarray(index, dtype=float)


def score_zone_axes(
    image: np.ndarray,
    basis: np.ndarray,
    templates: Sequence[ProjectedTemplate],
    config: FAUStackConfig,
) -> tuple[ProjectedTemplate, dict[str, float], np.ndarray, np.ndarray, float]:
    """Select a common low-index projection by periodic unit-cell evidence."""

    folded = _fold_image_to_cell(image, basis, config.unit_cell_bins)
    folded = gaussian_filter(folded, 0.7, mode="wrap")
    matrices = _unimodular_matrices()
    best_by_zone: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    template_by_name: dict[str, ProjectedTemplate] = {}
    for template in templates:
        name = "[" + "".join(str(item) for item in template.zone_axis) + "]"
        template_by_name[name] = template
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for matrix in matrices:
            for sigma in config.template_blur_sigmas:
                rendered = _render_template(template, matrix, config.unit_cell_bins, sigma)
                score, shift_yx = _circular_ncc(folded, rendered)
                if best is None or score > best[0]:
                    best = (score, matrix.copy(), shift_yx.copy())
        assert best is not None
        best_by_zone[name] = best
    ordered = sorted(best_by_zone, key=lambda name: best_by_zone[name][0], reverse=True)
    selected_name = ordered[0]
    selected_score, matrix, shift_yx = best_by_zone[selected_name]
    scores = {name: float(best_by_zone[name][0]) for name in ordered}
    margin = (
        float(selected_score - best_by_zone[ordered[1]][0])
        if len(ordered) > 1
        else math.nan
    )
    phase_shift_uv = np.asarray(
        [shift_yx[1] / config.unit_cell_bins, shift_yx[0] / config.unit_cell_bins]
    )
    return template_by_name[selected_name], scores, matrix, phase_shift_uv, margin


def _generate_expected_points(
    shape: tuple[int, int],
    basis: np.ndarray,
    template: ProjectedTemplate,
    unit_cell_transform: np.ndarray,
    phase_shift_uv: np.ndarray,
    margin: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    corners = np.asarray(((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)), dtype=float)
    corner_uv = (np.linalg.inv(basis) @ corners.T).T
    minimum = np.floor(np.min(corner_uv, axis=0)).astype(int) - 2
    maximum = np.ceil(np.max(corner_uv, axis=0)).astype(int) + 2
    phase = (unit_cell_transform @ template.phase_uv.T).T % 1.0
    points: list[np.ndarray] = []
    species: list[str] = []
    for first in range(minimum[0], maximum[0] + 1):
        for second in range(minimum[1], maximum[1] + 1):
            uv = phase + phase_shift_uv + np.asarray([first, second])
            xy = (basis @ uv.T).T
            inside = (
                (xy[:, 0] >= margin)
                & (xy[:, 0] < width - margin)
                & (xy[:, 1] >= margin)
                & (xy[:, 1] < height - margin)
            )
            points.extend(xy[inside])
            species.extend(template.species[inside].tolist())
    return np.asarray(points, dtype=float), np.asarray(species, dtype=object)


def _generate_expected_points_with_site_ids(
    shape: tuple[int, int],
    basis: np.ndarray,
    template: ProjectedTemplate,
    unit_cell_transform: np.ndarray,
    phase_shift_uv: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Tile a projected special-position template while retaining motif IDs."""

    height, width = shape
    corners = np.asarray(
        ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)),
        dtype=float,
    )
    corner_uv = (np.linalg.inv(basis) @ corners.T).T
    minimum = np.floor(np.min(corner_uv, axis=0)).astype(int) - 2
    maximum = np.ceil(np.max(corner_uv, axis=0)).astype(int) + 2
    phase = (unit_cell_transform @ template.phase_uv.T).T % 1.0
    points: list[np.ndarray] = []
    site_ids: list[int] = []
    motif_ids = np.arange(len(phase), dtype=int)
    for first in range(minimum[0], maximum[0] + 1):
        for second in range(minimum[1], maximum[1] + 1):
            uv = phase + phase_shift_uv + np.asarray([first, second])
            xy = (basis @ uv.T).T
            inside = (
                (xy[:, 0] >= margin)
                & (xy[:, 0] < width - margin)
                & (xy[:, 1] >= margin)
                & (xy[:, 1] < height - margin)
            )
            points.extend(xy[inside])
            site_ids.extend(motif_ids[inside].tolist())
    return np.asarray(points, dtype=float), np.asarray(site_ids, dtype=int)


def match_framework_sites(
    observed_xy: np.ndarray,
    observed_species: np.ndarray,
    expected_xy: np.ndarray,
    expected_species: np.ndarray,
    tolerance_pixels: float,
    observed_eligible: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily match geometry without turning brightness into chemistry.

    Legacy callers that provide explicit O/Si/rare-earth labels retain exact
    species matching and exclude rare-earth points.  Chemically neutral
    dim/mid/bright observations can match either framework species, subject to
    an explicit eligibility mask.
    """

    observed = np.asarray(observed_xy, dtype=float)
    observed_names = np.asarray(observed_species, dtype=object)
    expected = np.asarray(expected_xy, dtype=float)
    expected_names = np.asarray(expected_species, dtype=object)
    eligible = (
        np.ones(len(observed), dtype=bool)
        if observed_eligible is None
        else np.asarray(observed_eligible, dtype=bool)
    )
    if eligible.shape != (len(observed),):
        raise ValueError("observed_eligible must align with observed_xy")
    explicit_species = set(str(item) for item in np.unique(observed_names)).issubset(
        {"O", "Si", "rare_earth"}
    )
    candidates: list[tuple[float, int, int]] = []
    for species in ("O", "Si"):
        observed_indices = np.flatnonzero(
            eligible & ((observed_names == species) if explicit_species else True)
        )
        expected_indices = np.flatnonzero(expected_names == species)
        if not len(observed_indices) or not len(expected_indices):
            continue
        tree = cKDTree(expected[expected_indices])
        neighbors = min(4, len(expected_indices))
        distances, local_indices = tree.query(observed[observed_indices], k=neighbors)
        distances = np.atleast_2d(distances) if neighbors > 1 else distances[:, None]
        local_indices = np.atleast_2d(local_indices) if neighbors > 1 else local_indices[:, None]
        for row, observed_index in enumerate(observed_indices):
            for distance, local_index in zip(distances[row], local_indices[row]):
                if float(distance) <= tolerance_pixels:
                    candidates.append(
                        (float(distance), int(observed_index), int(expected_indices[int(local_index)]))
                    )
    used_observed: set[int] = set()
    used_expected: set[int] = set()
    accepted: list[tuple[float, int, int]] = []
    for distance, observed_index, expected_index in sorted(candidates):
        if observed_index in used_observed or expected_index in used_expected:
            continue
        used_observed.add(observed_index)
        used_expected.add(expected_index)
        accepted.append((distance, observed_index, expected_index))
    if not accepted:
        return (
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty(0, dtype=float),
        )
    return (
        np.asarray([item[1] for item in accepted], dtype=int),
        np.asarray([item[2] for item in accepted], dtype=int),
        np.asarray([item[0] for item in accepted], dtype=float),
    )


def _refine_phase_shift(
    peaks: StackPeakResult,
    basis: np.ndarray,
    template: ProjectedTemplate,
    transform: np.ndarray,
    initial_shift: np.ndarray,
    config: FAUStackConfig,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    best: tuple[int, float, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray] | None = None
    for delta_x in np.linspace(-0.06, 0.06, 7):
        for delta_y in np.linspace(-0.06, 0.06, 7):
            shift = (initial_shift + [delta_x, delta_y]) % 1.0
            expected_xy, expected_species = _generate_expected_points(
                peaks.best_image.shape, basis, template, transform, shift
            )
            matches = match_framework_sites(
                peaks.points_xy,
                peaks.species_brightness_class,
                expected_xy,
                expected_species,
                config.framework_match_tolerance_pixels,
                peaks.framework_eligible,
            )
            distances = matches[2]
            count = len(distances)
            median = float(np.median(distances)) if count else math.inf
            candidate = (count, -median, shift, matches, expected_xy, expected_species)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    return best[2], best[3], best[4], best[5]


def fit_template_to_stack(
    peaks: StackPeakResult,
    structure: CIFStructure,
    templates: Sequence[ProjectedTemplate],
    config: FAUStackConfig,
    *,
    fixed_zone_axis: tuple[int, int, int] | None = None,
) -> TemplateFit:
    basis = estimate_macro_direct_basis(peaks.best_image)
    inferred, scores, inferred_transform, inferred_shift, _ = score_zone_axes(
        peaks.best_image, basis, templates, config
    )
    if fixed_zone_axis is None:
        template = inferred
        transform = inferred_transform
        phase_shift = inferred_shift
        ordered_scores = sorted(scores.values(), reverse=True)
        margin = float(ordered_scores[0] - ordered_scores[1])
        template_ncc = float(ordered_scores[0])
    else:
        template = next(
            item for item in templates if item.zone_axis == fixed_zone_axis
        )
        _, _, transform, phase_shift, _ = score_zone_axes(
            peaks.best_image, basis, [template], config
        )
        fixed_name = "[" + "".join(str(item) for item in fixed_zone_axis) + "]"
        template_ncc = float(scores[fixed_name])
        alternatives = [
            score for name, score in scores.items() if name != fixed_name
        ]
        margin = (
            float(template_ncc - max(alternatives))
            if alternatives
            else math.nan
        )
    phase_shift, matches, expected_xy, expected_species = _refine_phase_shift(
        peaks, basis, template, transform, phase_shift, config
    )
    observed_indices, expected_indices, distances = matches
    framework_count = int(np.sum(peaks.framework_eligible))
    physical_basis = template.projected_basis_cell_fraction * structure.cell_lengths_angstrom[0]
    angstrom_per_pixel = math.sqrt(
        abs(float(np.linalg.det(physical_basis))) / abs(float(np.linalg.det(basis)))
    )
    return TemplateFit(
        direct_basis_pixels=basis,
        zone_axis=template.zone_axis,
        zone_scores=scores,
        zone_score_margin=margin,
        unit_cell_transform=transform,
        phase_shift_uv=phase_shift,
        expected_points_xy=expected_xy,
        expected_species=expected_species,
        matched_observed_indices=observed_indices,
        matched_expected_indices=expected_indices,
        match_distances_pixels=distances,
        framework_observed_count=framework_count,
        template_ncc=template_ncc,
        angstrom_per_pixel=angstrom_per_pixel,
    )


def estimate_local_deformation(
    peaks: StackPeakResult,
    fit: TemplateFit,
    config: FAUStackConfig,
) -> LocalDeformation:
    observed = peaks.points_xy[fit.matched_observed_indices]
    expected = fit.expected_points_xy[fit.matched_expected_indices]
    count = len(expected)
    arrays = [np.full(count, np.nan, dtype=float) for _ in range(7)]
    exx, eyy, exy, rotation, distortion, maximum_shear, rmse = arrays
    neighbors_count = np.zeros(count, dtype=int)
    cell_scale = math.sqrt(abs(float(np.linalg.det(fit.direct_basis_pixels))))
    radius = config.local_fit_radius_cells * cell_scale
    tree = cKDTree(expected)
    displacement = observed - expected
    for index, center in enumerate(expected):
        neighbors = np.asarray(tree.query_ball_point(center, radius), dtype=int)
        if len(neighbors) < config.local_fit_minimum_neighbors:
            continue
        relative = expected[neighbors] - center
        design = np.column_stack([relative, np.ones(len(relative))])
        keep = np.ones(len(neighbors), dtype=bool)
        coefficients = None
        residual = None
        for _ in range(3):
            if int(np.sum(keep)) < config.local_fit_minimum_neighbors:
                break
            coefficients = np.linalg.lstsq(
                design[keep], displacement[neighbors][keep], rcond=None
            )[0]
            prediction = design @ coefficients
            residual = np.linalg.norm(displacement[neighbors] - prediction, axis=1)
            center_residual, scale_residual = _robust_center_scale(residual[keep])
            keep = residual <= center_residual + 2.8 * scale_residual
        if coefficients is None or residual is None:
            continue
        gradient = coefficients[:2, :].T
        strain = 0.5 * (gradient + gradient.T)
        exx[index], eyy[index], exy[index] = strain[0, 0], strain[1, 1], strain[0, 1]
        rotation[index] = 0.5 * (gradient[1, 0] - gradient[0, 1])
        distortion[index] = float(np.sqrt(np.sum(strain**2)))
        mean_normal = 0.5 * float(strain[0, 0] + strain[1, 1])
        maximum_shear[index] = float(
            np.sqrt((0.5 * (strain[0, 0] - strain[1, 1])) ** 2 + strain[0, 1] ** 2)
        )
        rmse[index] = float(np.sqrt(np.mean(residual[keep] ** 2)))
        neighbors_count[index] = int(np.sum(keep))
    reliable = (
        np.isfinite(distortion)
        & (neighbors_count >= config.local_fit_minimum_neighbors)
        & (rmse <= config.local_fit_maximum_rmse_pixels)
        & (distortion <= config.local_fit_maximum_strain)
    )
    return LocalDeformation(
        exx=exx,
        eyy=eyy,
        exy=exy,
        rotation=rotation,
        distortion_magnitude=distortion,
        maximum_shear=maximum_shear,
        fit_rmse_pixels=rmse,
        neighbor_count=neighbors_count,
        reliable=reliable,
        nearest_guest_distance_pixels=np.full(count, np.inf),
    )


def _rematch_framework(
    peaks: StackPeakResult,
    fit: TemplateFit,
    eligible: np.ndarray,
    config: FAUStackConfig,
) -> TemplateFit:
    """Rematch a fixed CIF pose after excluding high-confidence guests."""

    matches = match_framework_sites(
        peaks.points_xy,
        peaks.species_brightness_class,
        fit.expected_points_xy,
        fit.expected_species,
        config.framework_match_tolerance_pixels,
        eligible,
    )
    return TemplateFit(
        direct_basis_pixels=fit.direct_basis_pixels,
        zone_axis=fit.zone_axis,
        zone_scores=fit.zone_scores,
        zone_score_margin=fit.zone_score_margin,
        unit_cell_transform=fit.unit_cell_transform,
        phase_shift_uv=fit.phase_shift_uv,
        expected_points_xy=fit.expected_points_xy,
        expected_species=fit.expected_species,
        matched_observed_indices=matches[0],
        matched_expected_indices=matches[1],
        match_distances_pixels=matches[2],
        framework_observed_count=int(np.sum(eligible)),
        template_ncc=fit.template_ncc,
        angstrom_per_pixel=fit.angstrom_per_pixel,
    )


def identify_distortion_gated_rare_earth(
    peaks: StackPeakResult,
    fit: TemplateFit,
    deformation: LocalDeformation,
    config: FAUStackConfig,
) -> RareEarthEvidence:
    """Require both a reliable hotspot and extreme local brightness.

    The brightness reference is formed from geometrically matched Si columns,
    falling back to all matched framework columns only when too few Si anchors
    are available.  Persistence is represented in the evidence score but is
    deliberately absent from the candidate gate.
    """

    count = len(peaks.points_xy)
    reliable = deformation.reliable
    finite_distortion = deformation.distortion_magnitude[reliable]
    finite_shear = deformation.maximum_shear[reliable]
    if not len(finite_distortion):
        return RareEarthEvidence(
            screen_positive=np.zeros(count, dtype=bool),
            candidate=np.zeros(count, dtype=bool),
            inside_hotspot_region=np.zeros(count, dtype=bool),
            extreme_brightness=np.zeros(count, dtype=bool),
            local_brightness_z=np.full(count, np.nan),
            global_brightness_z=np.full(count, np.nan),
            nearest_hotspot_distance_pixels=np.full(count, np.inf),
            hotspot_framework_site=np.zeros(len(reliable), dtype=bool),
            distortion_threshold=math.nan,
            maximum_shear_threshold=math.nan,
            framework_brightness_threshold=math.nan,
            evidence_score=np.zeros(count, dtype=float),
            negative_control_tail_probability=np.full(count, np.nan),
        )

    distortion_threshold = float(
        np.quantile(finite_distortion, config.hotspot_quantile)
    )
    shear_threshold = float(np.quantile(finite_shear, config.hotspot_quantile))
    distortion_center = float(np.median(finite_distortion))
    shear_center = float(np.median(finite_shear))
    hotspot = reliable & (
        (
            (deformation.distortion_magnitude >= distortion_threshold)
            & (deformation.distortion_magnitude > distortion_center)
        )
        | (
            (deformation.maximum_shear >= shear_threshold)
            & (deformation.maximum_shear > shear_center)
        )
    )
    matched_points = peaks.points_xy[fit.matched_observed_indices]
    hotspot_points = matched_points[hotspot]
    nearest_hotspot = (
        cKDTree(hotspot_points).query(peaks.points_xy, k=1)[0]
        if len(hotspot_points)
        else np.full(count, np.inf)
    )
    cell_scale = math.sqrt(abs(float(np.linalg.det(fit.direct_basis_pixels))))
    hotspot_radius = config.hotspot_radius_cells * cell_scale
    inside_hotspot = nearest_hotspot <= hotspot_radius

    matched_expected_species = fit.expected_species[fit.matched_expected_indices]
    silicon = matched_expected_species == "Si"
    reference_indices = fit.matched_observed_indices[silicon]
    if len(reference_indices) < 12:
        reference_indices = fit.matched_observed_indices
    reference_brightness = peaks.standardized_brightness[reference_indices]
    if len(reference_brightness) < 12:
        raise ValueError("Too few matched framework peaks for brightness calibration")
    global_center, global_scale = _robust_center_scale(reference_brightness)
    global_z = (
        peaks.standardized_brightness - global_center
    ) / max(global_scale, 1e-8)
    framework_threshold = max(
        float(np.quantile(reference_brightness, config.framework_brightness_quantile)),
        global_center + config.global_brightness_z_threshold * global_scale,
    )

    reference_points = peaks.points_xy[reference_indices]
    reference_tree = cKDTree(reference_points)
    neighbor_count = min(config.local_brightness_neighbors + 1, len(reference_indices))
    _, local_neighbors = reference_tree.query(peaks.points_xy, k=neighbor_count)
    local_neighbors = np.asarray(local_neighbors)
    if local_neighbors.ndim == 1:
        local_neighbors = local_neighbors[:, None]
    local_z = np.full(count, np.nan, dtype=float)
    for index, neighbors in enumerate(local_neighbors):
        local_reference = reference_brightness[np.asarray(neighbors, dtype=int)]
        if len(local_reference) > 12:
            nearest_coordinates = reference_points[np.asarray(neighbors, dtype=int)]
            self_match = np.linalg.norm(
                nearest_coordinates - peaks.points_xy[index], axis=1
            ) < 1e-6
            if np.any(self_match) and int(np.sum(~self_match)) >= 12:
                local_reference = local_reference[~self_match]
        local_center, local_scale = _robust_center_scale(local_reference)
        conservative_scale = max(local_scale, 0.5 * global_scale, 1e-8)
        local_z[index] = (
            peaks.standardized_brightness[index] - local_center
        ) / conservative_scale

    # Local contrast is the hard brightness evidence because thickness and
    # background vary over the field.  The global z score and 99.5% framework
    # envelope remain audit columns, not gates that can suppress a true atom in
    # a locally thin region.
    extreme = (
        (local_z >= config.local_brightness_z_threshold)
        & (peaks.species_brightness_class == "bright")
    )
    candidate = inside_hotspot & extreme
    brightness_strength = local_z / config.local_brightness_z_threshold
    spatial_strength = np.clip(1.0 - nearest_hotspot / max(hotspot_radius, 1e-8), 0, 1)
    persistence_strength = np.log1p(peaks.persistence_count) / np.log1p(
        max(int(np.max(peaks.persistence_count)), 1)
    )
    evidence_score = np.clip(
        0.60 * np.clip(brightness_strength, 0, 1.5) / 1.5
        + 0.15 * peaks.assignment_probability
        + 0.15 * spatial_strength
        + 0.10 * persistence_strength,
        0,
        1,
    )
    return RareEarthEvidence(
        screen_positive=candidate.copy(),
        candidate=candidate,
        inside_hotspot_region=inside_hotspot,
        extreme_brightness=extreme,
        local_brightness_z=local_z,
        global_brightness_z=global_z,
        nearest_hotspot_distance_pixels=nearest_hotspot,
        hotspot_framework_site=hotspot,
        distortion_threshold=distortion_threshold,
        maximum_shear_threshold=shear_threshold,
        framework_brightness_threshold=framework_threshold,
        evidence_score=evidence_score,
        negative_control_tail_probability=np.full(count, np.nan),
    )


def _attach_candidate_distances(
    peaks: StackPeakResult,
    deformation: LocalDeformation,
    evidence: RareEarthEvidence,
    fit: TemplateFit,
) -> None:
    guest_points = peaks.points_xy[evidence.candidate]
    observed = peaks.points_xy[fit.matched_observed_indices]
    deformation.nearest_guest_distance_pixels = (
        cKDTree(guest_points).query(observed, k=1)[0]
        if len(guest_points)
        else np.full(len(observed), np.inf)
    )


def apply_rare_earth_free_reference_audit(
    images: Sequence[FAUImageAnalysis],
    reference_label: str,
) -> dict[str, Any]:
    """Use a known guest-free reference as a soft, normalized false-positive audit.

    Each image keeps its own local Si-based normalization.  The reference tail
    therefore never becomes a cross-image absolute-intensity veto.  Raw
    intersection positives in the reference are retained as screen positives
    but reinterpreted as false positives rather than rare-earth candidates.
    """

    reference = next(item for item in images if item.label == reference_label)
    reference_evidence = reference.rare_earth_evidence
    control_mask = (
        reference_evidence.inside_hotspot_region
        & np.isfinite(reference_evidence.local_brightness_z)
    )
    control_scores = reference_evidence.local_brightness_z[control_mask]
    if not len(control_scores):
        raise ValueError("Rare-earth-free reference has no auditable hotspot peaks")
    ordered = np.sort(control_scores)
    for image in images:
        scores = image.rare_earth_evidence.local_brightness_z
        finite = np.isfinite(scores)
        probabilities = np.full(len(scores), np.nan, dtype=float)
        if np.any(finite):
            first_not_less = np.searchsorted(ordered, scores[finite], side="left")
            exceedances = len(ordered) - first_not_less
            probabilities[finite] = (1.0 + exceedances) / (1.0 + len(ordered))
        image.rare_earth_evidence.negative_control_tail_probability = probabilities

    screen_positive_count = int(np.sum(reference_evidence.screen_positive))
    reference_evidence.candidate = np.zeros_like(
        reference_evidence.screen_positive, dtype=bool
    )
    return {
        "reference": reference_label,
        "declared_rare_earth_free": True,
        "normalization": (
            "per-plane robust image normalization followed by per-image local "
            "Si-column robust z scores"
        ),
        "cross_image_absolute_intensity_gate": False,
        "reference_hotspot_peak_count": int(len(control_scores)),
        "reference_hotspot_local_z_max": float(np.max(control_scores)),
        "reference_raw_intersection_screen_positive_count": screen_positive_count,
        "reference_interpreted_rare_earth_candidate_count": 0,
        "interpretation": (
            "Reference screen positives are false-positive controls; their empirical "
            "tail probabilities audit moving-image candidates but do not veto them."
        ),
    }


def guest_coordinates_in_cif_cell(
    points_xy: np.ndarray,
    fit: TemplateFit,
) -> np.ndarray:
    """Express image points in the canonical projected-CIF unit cell."""

    image_uv = (
        np.linalg.inv(fit.direct_basis_pixels)
        @ np.asarray(points_xy, dtype=float).T
    ).T
    # Template fitting uses image_uv = M @ cif_uv + phase_shift. Undo both
    # nuisance choices before comparing independently sampled fields.
    return (
        np.linalg.inv(fit.unit_cell_transform)
        @ (image_uv - fit.phase_shift_uv).T
    ).T % 1.0


def _associate_all_peaks_with_aligned_sites(
    peaks: StackPeakResult,
    fit: TemplateFit,
    tolerance_pixels: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associate all detected peaks after fixing the framework registration.

    The deformation fit deliberately excludes provisional guest candidates.
    Contrast attribution must not inherit that exclusion, otherwise an
    exceptionally bright peak replacing or overlapping an O/Si column could
    never be audited at its corresponding CIF site.
    """

    return match_framework_sites(
        peaks.points_xy,
        peaks.species_brightness_class,
        fit.expected_points_xy,
        fit.expected_species,
        tolerance_pixels,
        np.ones(len(peaks.points_xy), dtype=bool),
    )


def _normalize_repeated_site_contrast(
    values: np.ndarray,
    canonical_site_id: np.ndarray,
    associated: np.ndarray,
    *,
    minimum_repeats: int = 12,
) -> np.ndarray:
    """Robustly compare each column with repeats of the same CIF motif."""

    contrast = np.asarray(values, dtype=float)
    site_ids = np.asarray(canonical_site_id, dtype=int)
    valid = np.asarray(associated, dtype=bool) & np.isfinite(contrast)
    if contrast.shape != site_ids.shape or contrast.shape != valid.shape:
        raise ValueError("Contrast, site IDs, and association mask must align")
    if minimum_repeats < 3:
        raise ValueError("At least three repeats are required")

    residual = np.full(len(contrast), np.nan, dtype=float)
    local_scale = np.full(len(contrast), np.nan, dtype=float)
    for site_id in np.unique(site_ids[site_ids >= 0]):
        calibration = valid & (site_ids == site_id)
        if int(np.sum(calibration)) < minimum_repeats:
            continue
        center, scale = _robust_center_scale(contrast[calibration])
        rows = site_ids == site_id
        residual[rows] = contrast[rows] - center
        local_scale[rows] = scale

    calibration = valid & np.isfinite(residual)
    if int(np.sum(calibration)) < 60:
        raise ValueError("Too few repeated CIF sites for contrast calibration")
    _, global_residual_scale = _robust_center_scale(residual[calibration])
    conservative_scale = np.maximum(local_scale, 0.5 * global_residual_scale)
    return residual / np.maximum(conservative_scale, 1e-8)


def _normalize_local_same_type_contrast(
    values: np.ndarray,
    points_xy: np.ndarray,
    type_ids: np.ndarray,
    valid: np.ndarray,
    *,
    radius_pixels: float,
    minimum_references: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare each value only with nearby valid sites of the same type."""

    contrast = np.asarray(values, dtype=float)
    points = np.asarray(points_xy, dtype=float)
    kinds = np.asarray(type_ids)
    usable = np.asarray(valid, dtype=bool) & np.isfinite(contrast)
    if points.shape != (len(contrast), 2):
        raise ValueError("points_xy must have shape (N, 2)")
    if kinds.shape != contrast.shape or usable.shape != contrast.shape:
        raise ValueError("values, type_ids, and valid must align")
    if radius_pixels <= 0:
        raise ValueError("radius_pixels must be positive")
    if minimum_references < 3:
        raise ValueError("minimum_references must be at least three")

    z_scores = np.full(len(contrast), np.nan, dtype=float)
    reference_count = np.zeros(len(contrast), dtype=int)
    for kind in np.unique(kinds):
        type_rows = np.flatnonzero(usable & (kinds == kind))
        if len(type_rows) < minimum_references + 1:
            continue
        _, global_scale = _robust_center_scale(contrast[type_rows])
        tree = cKDTree(points[type_rows])
        for index in np.flatnonzero(kinds == kind):
            neighbors_local = tree.query_ball_point(
                points[index], radius_pixels
            )
            neighbors = type_rows[np.asarray(neighbors_local, dtype=int)]
            neighbors = neighbors[neighbors != index]
            if len(neighbors) < minimum_references:
                continue
            local_center, local_scale = _robust_center_scale(
                contrast[neighbors]
            )
            scale = max(local_scale, 0.5 * global_scale, 1e-8)
            z_scores[index] = (contrast[index] - local_center) / scale
            reference_count[index] = int(len(neighbors))
    return z_scores, reference_count


def _aligned_site_interior_mask(
    points_xy: np.ndarray,
    shape: tuple[int, int],
    direct_basis_pixels: np.ndarray,
    config: FAUStackConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return edge distances and a projected-cell-scaled reliability mask."""

    points = np.asarray(points_xy, dtype=float)
    height, width = shape
    edge_distance = np.minimum.reduce(
        [
            points[:, 0],
            points[:, 1],
            width - 1.0 - points[:, 0],
            height - 1.0 - points[:, 1],
        ]
    )
    cell_scale = math.sqrt(abs(float(np.linalg.det(direct_basis_pixels))))
    minimum_edge_distance = max(
        float(config.border_pixels),
        config.aligned_site_candidate_border_cells * cell_scale,
    )
    return edge_distance, edge_distance >= minimum_edge_distance


def _opposed_site_i_oxygen_endpoints(
    site_points_xy: np.ndarray,
    fit: TemplateFit,
    config: FAUStackConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the opposed projected O pair flanking each template-empty site I."""

    sites = np.asarray(site_points_xy, dtype=float)
    oxygen = fit.expected_points_xy[fit.expected_species == "O"]
    endpoints = np.full((len(sites), 2, 2), np.nan, dtype=float)
    valid = np.zeros(len(sites), dtype=bool)
    if len(oxygen) < 2:
        return endpoints, valid
    minimum_distance = (
        config.site_i_minimum_framework_separation_angstrom
        / fit.angstrom_per_pixel
    )
    maximum_distance = (
        config.site_i_endpoint_radius_angstrom / fit.angstrom_per_pixel
    )
    tree = cKDTree(oxygen)
    neighbor_count = min(12, len(oxygen))
    distances, indices = tree.query(sites, k=neighbor_count)
    distances = (
        distances[:, None] if neighbor_count == 1 else np.asarray(distances)
    )
    indices = indices[:, None] if neighbor_count == 1 else np.asarray(indices)
    for site_index, center in enumerate(sites):
        if float(distances[site_index, 0]) < minimum_distance:
            continue
        nearby = np.flatnonzero(distances[site_index] <= maximum_distance)
        best: tuple[float, int, int] | None = None
        for first_local, second_local in itertools.combinations(nearby, 2):
            first_index = int(indices[site_index, first_local])
            second_index = int(indices[site_index, second_local])
            first_vector = oxygen[first_index] - center
            second_vector = oxygen[second_index] - center
            first_length = float(np.linalg.norm(first_vector))
            second_length = float(np.linalg.norm(second_vector))
            if min(first_length, second_length) <= 1e-8:
                continue
            cosine = float(
                np.dot(first_vector, second_vector)
                / (first_length * second_length)
            )
            if cosine > -0.72:
                continue
            midpoint_offset = float(
                np.linalg.norm(
                    0.5 * (oxygen[first_index] + oxygen[second_index])
                    - center
                )
            )
            score = (
                first_length
                + second_length
                + 1.5 * abs(first_length - second_length)
                + 4.0 * midpoint_offset
                + maximum_distance * (cosine + 1.0)
            )
            candidate = (score, first_index, second_index)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            continue
        endpoints[site_index] = oxygen[[best[1], best[2]]]
        valid[site_index] = True
    return endpoints, valid


def _fit_template_empty_center_profile(
    frame: np.ndarray,
    center_xy: np.ndarray,
    endpoint_xy: np.ndarray,
    sigma_pixels: float,
) -> tuple[float, float, float, float]:
    """Compare two-O and two-O-plus-center models along one local bridge."""

    center = np.asarray(center_xy, dtype=float)
    endpoints = np.asarray(endpoint_xy, dtype=float)
    if endpoints.shape != (2, 2) or not np.all(np.isfinite(endpoints)):
        return math.nan, math.nan, math.nan, math.nan
    direction = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(direction))
    if length <= 2.0 or sigma_pixels <= 0:
        return math.nan, math.nan, math.nan, math.nan
    direction /= length
    endpoint_positions = (endpoints - center) @ direction
    lower = float(np.min(endpoint_positions) - 2.5 * sigma_pixels)
    upper = float(np.max(endpoint_positions) + 2.5 * sigma_pixels)
    sample_count = int(np.clip(math.ceil((upper - lower) * 4.0), 31, 81))
    coordinate = np.linspace(lower, upper, sample_count)
    xy = center[None, :] + coordinate[:, None] * direction[None, :]
    profile = map_coordinates(
        np.asarray(frame, dtype=float),
        [xy[:, 1], xy[:, 0]],
        order=1,
        mode="nearest",
    )
    profile_center, profile_scale = _robust_center_scale(profile)
    normalized = (profile - profile_center) / max(profile_scale, 1e-8)
    normalized_coordinate = coordinate / max(0.5 * length, 1e-8)
    gaussian_sigma = sigma_pixels / max(0.5 * length, 1e-8)

    def gaussian(position: float) -> np.ndarray:
        return np.exp(
            -0.5
            * ((normalized_coordinate - position) / gaussian_sigma) ** 2
        )

    endpoint_normalized = endpoint_positions / max(0.5 * length, 1e-8)
    base = np.column_stack(
        [
            np.ones(sample_count),
            normalized_coordinate,
            gaussian(float(endpoint_normalized[0])),
            gaussian(float(endpoint_normalized[1])),
        ]
    )
    expanded = np.column_stack([base, gaussian(0.0)])
    base_coefficients = np.linalg.lstsq(base, normalized, rcond=None)[0]
    expanded_coefficients = np.linalg.lstsq(
        expanded, normalized, rcond=None
    )[0]
    base_residual = normalized - base @ base_coefficients
    expanded_residual = normalized - expanded @ expanded_coefficients
    base_rss = max(float(np.sum(base_residual**2)), 1e-12)
    expanded_rss = max(float(np.sum(expanded_residual**2)), 1e-12)
    base_bic = sample_count * math.log(base_rss / sample_count) + 4 * math.log(
        sample_count
    )
    expanded_bic = sample_count * math.log(
        expanded_rss / sample_count
    ) + 5 * math.log(sample_count)
    center_amplitude = float(expanded_coefficients[-1])
    if center_amplitude <= 0 or np.any(expanded_coefficients[2:4] <= 0):
        return base_bic, expanded_bic, -math.inf, center_amplitude
    return (
        float(base_bic),
        float(expanded_bic),
        float(base_bic - expanded_bic),
        center_amplitude,
    )


def measure_skeleton_first_site_i_evidence(
    stack: np.ndarray,
    peaks: StackPeakResult,
    fit: TemplateFit,
    structure: CIFStructure,
    config: FAUStackConfig,
) -> SkeletonFirstSiteEvidence:
    """Measure template-empty D6R site-I bridges before any brightness gate."""

    template = project_fau_site_i_template(structure, fit.zone_axis)
    cell_scale = math.sqrt(abs(float(np.linalg.det(fit.direct_basis_pixels))))
    sigma_pixels = config.site_i_profile_sigma_angstrom / fit.angstrom_per_pixel
    margin = max(
        float(config.border_pixels),
        config.site_i_candidate_border_cells * cell_scale,
        config.site_i_endpoint_radius_angstrom / fit.angstrom_per_pixel
        + 3.0 * sigma_pixels,
    )
    expected, site_ids = _generate_expected_points_with_site_ids(
        peaks.best_image.shape,
        fit.direct_basis_pixels,
        template,
        fit.unit_cell_transform,
        fit.phase_shift_uv,
        margin,
    )
    endpoints, endpoint_valid = _opposed_site_i_oxygen_endpoints(
        expected, fit, config
    )
    count = len(expected)
    best_plane = np.full(count, -1, dtype=int)
    base_bic = np.full(count, np.nan, dtype=float)
    expanded_bic = np.full(count, np.nan, dtype=float)
    delta_bic = np.full(count, -np.inf, dtype=float)
    amplitude = np.full(count, np.nan, dtype=float)
    values = np.asarray(stack, dtype=np.float32)
    for index in np.flatnonzero(endpoint_valid):
        for plane_index, frame in enumerate(values):
            fit_values = _fit_template_empty_center_profile(
                frame,
                expected[index],
                endpoints[index],
                sigma_pixels,
            )
            if not np.isfinite(fit_values[2]):
                continue
            if fit_values[2] > delta_bic[index]:
                base_bic[index] = fit_values[0]
                expanded_bic[index] = fit_values[1]
                delta_bic[index] = fit_values[2]
                amplitude[index] = fit_values[3]
                best_plane[index] = plane_index
    model_valid = endpoint_valid & (best_plane >= 0) & np.isfinite(delta_bic)
    height, width = peaks.best_image.shape
    edge_distance = np.minimum.reduce(
        [
            expected[:, 0],
            expected[:, 1],
            width - 1.0 - expected[:, 0],
            height - 1.0 - expected[:, 1],
        ]
    )
    inside = edge_distance >= max(
        float(config.border_pixels),
        config.site_i_candidate_border_cells * cell_scale,
    )
    local_z, local_count = _normalize_local_same_type_contrast(
        amplitude,
        expected,
        np.zeros(len(site_ids), dtype=int),
        model_valid & inside,
        radius_pixels=config.site_i_local_radius_cells * cell_scale,
        minimum_references=config.site_i_local_minimum_references,
    )
    false_mask = np.zeros(count, dtype=bool)
    return SkeletonFirstSiteEvidence(
        expected_xy=expected,
        canonical_site_id=site_ids,
        oxygen_endpoint_xy=endpoints,
        model_valid=model_valid,
        best_plane=best_plane,
        template_only_bic=base_bic,
        template_plus_site_bic=expanded_bic,
        delta_bic=delta_bic,
        extra_component_amplitude=amplitude,
        local_equivalent_site_z=local_z,
        local_reference_count=local_count,
        distance_to_image_edge_pixels=edge_distance,
        inside_reliable_interior=inside,
        raw_structural_screen_positive=false_mask.copy(),
        structural_review=false_mask.copy(),
        local_brightness_anomaly=false_mask.copy(),
        candidate=false_mask.copy(),
        negative_control_tail_probability=np.full(count, np.nan),
        calibrated_delta_bic_threshold=math.nan,
    )


def apply_skeleton_first_reference_audit(
    images: Sequence[FAUImageAnalysis],
    reference_label: str,
    config: FAUStackConfig,
) -> dict[str, Any]:
    """Calibrate structure-first site-I evidence with guest-free Y."""

    reference = next(item for item in images if item.label == reference_label)
    if reference.skeleton_first_sites is None:
        raise ValueError("Reference skeleton-first site-I evidence is missing")
    control = reference.skeleton_first_sites
    control_eligible = (
        control.model_valid
        & control.inside_reliable_interior
        & np.isfinite(control.delta_bic)
        & (control.extra_component_amplitude > 0)
    )
    control_scores = control.delta_bic[control_eligible]
    if len(control_scores) < 20:
        raise ValueError("Too few Y site-I probes for structural calibration")
    threshold = float(config.site_i_minimum_delta_bic)
    control_audit_threshold = float(
        np.quantile(control_scores, config.site_i_control_quantile)
    )
    ordered = np.sort(control_scores)
    image_summary: dict[str, Any] = {}
    for image in images:
        site = image.skeleton_first_sites
        if site is None:
            raise ValueError(
                f"Skeleton-first site-I evidence is missing for {image.label}"
            )
        eligible = (
            site.model_valid
            & site.inside_reliable_interior
            & np.isfinite(site.delta_bic)
            & (site.extra_component_amplitude > 0)
        )
        raw_structural = eligible & (site.delta_bic >= threshold)
        probabilities = np.full(len(site.expected_xy), np.nan, dtype=float)
        finite = np.isfinite(site.delta_bic)
        first_not_less = np.searchsorted(
            ordered, site.delta_bic[finite], side="left"
        )
        probabilities[finite] = (
            1.0 + len(ordered) - first_not_less
        ) / (1.0 + len(ordered))
        site.raw_structural_screen_positive = raw_structural
        # Local equivalent-site intensity remains in the audit table as a
        # continuous measurement, but no brightness threshold may select,
        # reject, promote, or rescue a site-I candidate.
        site.local_brightness_anomaly = np.zeros_like(raw_structural)
        site.calibrated_delta_bic_threshold = float(threshold)
        site.negative_control_tail_probability = probabilities
        if image.label == reference_label:
            site.structural_review = np.zeros_like(raw_structural)
            site.candidate = np.zeros_like(raw_structural)
        else:
            # Every valid positive structural comparison is retained.  There
            # is no within-image top-quantile truncation and no B1/B2 split.
            site.structural_review = raw_structural.copy()
            site.candidate = raw_structural.copy()
        image_summary[image.label] = {
            "modeled_template_empty_site_I_count": int(np.sum(site.model_valid)),
            "raw_structural_tail_count": int(np.sum(raw_structural)),
            "structure_supported_candidate_count": int(np.sum(site.candidate)),
        }
    return {
        "schema_version": "fau_skeleton_first_site_i_v3",
        "reference": reference_label,
        "declared_rare_earth_free": True,
        "site_definition": "FAU 16c/site-I D6R center projected along [101]",
        "candidate_order": (
            "all valid template-plus-site-I fits with positive center amplitude "
            "and delta-BIC at or above the fixed minimum are retained"
        ),
        "global_absolute_brightness_gate": False,
        "local_brightness_gate": False,
        "within_image_structural_quantile_gate": False,
        "Y_control_quantile": float(config.site_i_control_quantile),
        "minimum_structural_delta_bic_threshold": float(threshold),
        "Y_control_delta_bic_tail_audit_threshold": float(
            control_audit_threshold
        ),
        "images": image_summary,
        "interpretation": (
            "All fixed-threshold structural positives are site-I occupancy or "
            "projection-overlap hypotheses, not elemental identifications. Local "
            "intensity z is reported only as an audit measurement. Y raw tail "
            "points remain false-positive audit evidence and are never interpreted "
            "as candidates."
        ),
    }


def measure_aligned_framework_site_contrast(
    stack: np.ndarray,
    peaks: StackPeakResult,
    fit: TemplateFit,
    evidence: RareEarthEvidence,
    structure: CIFStructure,
    config: FAUStackConfig,
) -> AlignedSiteContrast:
    """Measure every theoretical O/Si site after CIF registration."""

    expected = np.asarray(fit.expected_points_xy, dtype=float)
    count = len(expected)
    associated = np.zeros(count, dtype=bool)
    observed_index = np.full(count, -1, dtype=int)
    residual_pixels = np.full(count, np.nan, dtype=float)
    observed_rows, expected_rows, distances = _associate_all_peaks_with_aligned_sites(
        peaks, fit, config.framework_match_tolerance_pixels
    )
    associated[expected_rows] = True
    observed_index[expected_rows] = observed_rows
    residual_pixels[expected_rows] = distances
    measurement_xy = expected.copy()
    measurement_xy[associated] = peaks.points_xy[observed_index[associated]]

    values = np.asarray(stack, dtype=np.float32)
    border = max(config.border_pixels, 1)
    standardized_contrast = np.full(count, -np.inf, dtype=float)
    raw_dog_contrast = np.full(count, np.nan, dtype=float)
    raw_detector_intensity = np.full(count, np.nan, dtype=float)
    best_plane = np.zeros(count, dtype=int)
    persistence = np.zeros(count, dtype=int)
    coordinates = [measurement_xy[:, 1], measurement_xy[:, 0]]
    for plane_index, frame in enumerate(values):
        response = (
            gaussian_filter(frame, config.focus_sigma, mode="reflect")
            - gaussian_filter(frame, config.background_sigma, mode="reflect")
        )
        center, scale = _robust_center_scale(
            response[border:-border, border:-border]
        )
        standardized = (response - center) / scale
        sampled = map_coordinates(
            standardized, coordinates, order=1, mode="nearest"
        )
        persistence += sampled >= config.persistence_z
        update = sampled > standardized_contrast
        if np.any(update):
            sampled_raw_dog = map_coordinates(
                response, coordinates, order=1, mode="nearest"
            )
            sampled_raw_intensity = map_coordinates(
                frame, coordinates, order=1, mode="nearest"
            )
            standardized_contrast[update] = sampled[update]
            raw_dog_contrast[update] = sampled_raw_dog[update]
            raw_detector_intensity[update] = sampled_raw_intensity[update]
            best_plane[update] = plane_index

    canonical_uv = guest_coordinates_in_cif_cell(expected, fit)
    template = project_structure(structure, fit.zone_axis)
    canonical_site_id = np.full(count, -1, dtype=int)
    for species in ("O", "Si"):
        rows = np.flatnonzero(fit.expected_species == species)
        template_rows = np.flatnonzero(template.species == species)
        if not len(rows) or not len(template_rows):
            continue
        delta = (
            canonical_uv[rows, None, :]
            - template.phase_uv[template_rows][None, :, :]
            + 0.5
        ) % 1.0 - 0.5
        physical_delta = np.einsum(
            "ij,nkj->nki", template.projected_basis_cell_fraction, delta
        )
        nearest = np.argmin(np.linalg.norm(physical_delta, axis=2), axis=1)
        canonical_site_id[rows] = template_rows[nearest]
    if np.any(canonical_site_id < 0):
        raise ValueError("Some aligned framework sites could not be assigned to CIF motifs")

    motif_z = _normalize_repeated_site_contrast(
        standardized_contrast,
        canonical_site_id,
        associated,
    )

    brightness_class = np.full(count, "unmatched", dtype=object)
    brightness_class[associated] = peaks.species_brightness_class[
        observed_index[associated]
    ]
    inside_hotspot = np.zeros(count, dtype=bool)
    inside_hotspot[associated] = evidence.inside_hotspot_region[
        observed_index[associated]
    ]
    association_reliable = (
        associated
        & np.isfinite(residual_pixels)
        & (
            residual_pixels
            <= config.aligned_site_candidate_max_residual_pixels
        )
    )
    edge_distance, inside_reliable_interior = _aligned_site_interior_mask(
        measurement_xy,
        peaks.best_image.shape,
        fit.direct_basis_pixels,
        config,
    )
    cell_scale = math.sqrt(
        abs(float(np.linalg.det(fit.direct_basis_pixels)))
    )
    local_same_species_z, local_reference_count = (
        _normalize_local_same_type_contrast(
            standardized_contrast,
            measurement_xy,
            np.asarray(fit.expected_species, dtype=object),
            association_reliable & inside_reliable_interior,
            radius_pixels=(
                config.local_same_species_radius_cells * cell_scale
            ),
            minimum_references=(
                config.local_same_species_minimum_references
            ),
        )
    )
    contrast_anomaly = (
        association_reliable
        & np.isfinite(local_same_species_z)
        & (
            local_same_species_z
            >= config.local_same_species_z_threshold
        )
    )
    screen_positive = (
        contrast_anomaly
        & inside_hotspot
        & association_reliable
        & inside_reliable_interior
    )
    overlaps_peak_gate = np.zeros(count, dtype=bool)
    overlaps_peak_gate[associated] = evidence.screen_positive[
        observed_index[associated]
    ]
    return AlignedSiteContrast(
        expected_xy=expected,
        canonical_uv=canonical_uv,
        canonical_site_id=canonical_site_id,
        expected_species=np.asarray(fit.expected_species, dtype=object),
        matched=associated,
        observed_peak_index=observed_index,
        measurement_xy=measurement_xy,
        residual_pixels=residual_pixels,
        association_reliable=association_reliable,
        best_plane=best_plane,
        persistence_planes=persistence,
        standardized_contrast=standardized_contrast,
        raw_dog_contrast=raw_dog_contrast,
        raw_detector_intensity=raw_detector_intensity,
        brightness_class=brightness_class,
        motif_normalized_contrast_z=motif_z,
        local_same_species_contrast_z=local_same_species_z,
        local_same_species_reference_count=local_reference_count,
        inside_hotspot_region=inside_hotspot,
        distance_to_image_edge_pixels=edge_distance,
        inside_reliable_interior=inside_reliable_interior,
        contrast_anomaly=contrast_anomaly,
        screen_positive=screen_positive,
        candidate=screen_positive.copy(),
        overlaps_peak_gate_screen_positive=overlaps_peak_gate,
        negative_control_tail_probability=np.full(count, np.nan),
    )


def apply_aligned_site_reference_audit(
    images: Sequence[FAUImageAnalysis],
    reference_label: str,
) -> dict[str, Any]:
    """Use guest-free Y to audit, but not hard-veto, aligned-site anomalies."""

    reference = next(item for item in images if item.label == reference_label)
    if reference.aligned_site_contrast is None:
        raise ValueError("Reference aligned-site contrast has not been measured")
    reference_sites = reference.aligned_site_contrast
    control = (
        reference_sites.matched
        & reference_sites.association_reliable
        & reference_sites.inside_hotspot_region
        & reference_sites.inside_reliable_interior
        & np.isfinite(reference_sites.local_same_species_contrast_z)
    )
    scores = np.sort(reference_sites.local_same_species_contrast_z[control])
    if not len(scores):
        raise ValueError("No aligned Y sites are available for negative-control audit")
    for image in images:
        if image.aligned_site_contrast is None:
            raise ValueError(f"Aligned-site contrast is missing for {image.label}")
        site = image.aligned_site_contrast
        finite = np.isfinite(site.local_same_species_contrast_z)
        probabilities = np.full(len(site.expected_xy), np.nan, dtype=float)
        first_not_less = np.searchsorted(
            scores, site.local_same_species_contrast_z[finite], side="left"
        )
        probabilities[finite] = (
            1.0 + len(scores) - first_not_less
        ) / (1.0 + len(scores))
        site.negative_control_tail_probability = probabilities
    raw_positive = int(np.sum(reference_sites.screen_positive))
    reference_sites.candidate = np.zeros_like(
        reference_sites.screen_positive, dtype=bool
    )
    return {
        "reference": reference_label,
        "declared_rare_earth_free": True,
        "control_site_count": int(len(scores)),
        "control_max_local_same_species_contrast_z": float(np.max(scores)),
        "control_max_motif_normalized_contrast_z": float(np.max(scores)),
        "reference_raw_screen_positive_count": raw_positive,
        "reference_interpreted_candidate_count": 0,
        "candidate_logic": (
            "matched CIF O/Si site AND nearby same-element local contrast z >= "
            "threshold AND distortion-hotspot support AND strict association "
            "residual AND reliable image interior; global bright-class membership "
            "is audit-only"
        ),
        "interpretation": (
            "Aligned-site anomalies are rare-earth occupancy/overlap candidates, "
            "not chemical identifications; Y tail probabilities are audit-only."
        ),
    }


def _expert_review_motif_threshold(
    control_scores: np.ndarray,
    config: FAUStackConfig,
) -> float:
    """Calibrate a sensitive review tier from the rare-earth-free Y tail."""

    values = np.asarray(control_scores, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("Expert review threshold requires finite Y control scores")
    return max(
        float(config.expert_review_minimum_motif_z),
        float(np.quantile(values, config.expert_review_control_quantile)),
    )


def _cluster_expert_review_members(
    members: Sequence[Mapping[str, Any]],
    radius_pixels: float,
) -> list[list[dict[str, Any]]]:
    """Merge nearby point evidence into the larger regions an expert reviews."""

    if radius_pixels <= 0:
        raise ValueError("Expert review clustering radius must be positive")
    records = [dict(item) for item in members]
    if not records:
        return []
    points = np.asarray(
        [[item["x_pixels"], item["y_pixels"]] for item in records],
        dtype=float,
    )
    parents = list(range(len(records)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merge(first: int, second: int) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first, second in cKDTree(points).query_pairs(radius_pixels):
        merge(int(first), int(second))
    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(root(index), []).append(record)
    return list(grouped.values())


def build_expert_guided_candidate_regions(
    analysis: FAUStackAnalysis,
) -> dict[str, Any]:
    """Build a sensitive, Y-calibrated review pool around distorted skeletons.

    This tier does not replace the strict rare-earth assignments.  It combines
    two complementary point sources inside reliable distortion regions:
    template-site contrast above the empirical Y 99.9th-percentile tail and
    the existing non-template extreme-bright intersection.  Nearby points are
    merged into cell-scaled regions so an expert sees context rather than an
    artificially precise atom-sized circle.
    """

    if not analysis.reference_is_rare_earth_free:
        raise ValueError("Expert-guided review requires a declared guest-free Y reference")
    reference = next(
        item for item in analysis.images if item.label == analysis.reference_label
    )
    if reference.aligned_site_contrast is None:
        raise ValueError("Reference aligned-site contrast is missing")
    reference_site = reference.aligned_site_contrast
    control_eligible = (
        reference_site.matched
        & reference_site.association_reliable
        & reference_site.inside_hotspot_region
        & reference_site.inside_reliable_interior
        & np.isfinite(reference_site.local_same_species_contrast_z)
    )
    control_scores = reference_site.local_same_species_contrast_z[
        control_eligible
    ]
    threshold = _expert_review_motif_threshold(control_scores, analysis.config)
    image_records: dict[str, Any] = {}
    for image in analysis.images:
        if image.aligned_site_contrast is None:
            raise ValueError(f"Aligned-site contrast is missing for {image.label}")
        site = image.aligned_site_contrast
        cell_scale = math.sqrt(
            abs(float(np.linalg.det(image.template_fit.direct_basis_pixels)))
        )
        eligible = (
            site.matched
            & site.association_reliable
            & site.inside_hotspot_region
            & site.inside_reliable_interior
            & np.isfinite(site.local_same_species_contrast_z)
        )
        review_site_indices = np.flatnonzero(
            eligible & (site.local_same_species_contrast_z >= threshold)
        )
        control_tail_count = int(len(review_site_indices))
        members: list[dict[str, Any]] = []
        if image.label != analysis.reference_label:
            for index in review_site_indices:
                members.append(
                    {
                        "source": "template_site_contrast",
                        "source_index": int(index),
                        "x_pixels": float(site.measurement_xy[index, 0]),
                        "y_pixels": float(site.measurement_xy[index, 1]),
                        "best_plane_zero_based": int(site.best_plane[index]),
                        "nominal_template_species": str(
                            site.expected_species[index]
                        ),
                        "motif_contrast_z": float(
                            site.motif_normalized_contrast_z[index]
                        ),
                        "local_same_species_contrast_z": float(
                            site.local_same_species_contrast_z[index]
                        ),
                        "local_brightness_z": None,
                        "association_residual_pixels": float(
                            site.residual_pixels[index]
                        ),
                        "persistence_planes": int(
                            site.persistence_planes[index]
                        ),
                        "distance_to_image_edge_pixels": float(
                            site.distance_to_image_edge_pixels[index]
                        ),
                        "strict_rule_support": bool(site.candidate[index]),
                        "review_score": float(
                            site.local_same_species_contrast_z[index]
                        ),
                    }
                )
            evidence = image.rare_earth_evidence
            image_height, image_width = image.peaks.best_image.shape
            for index in np.flatnonzero(evidence.candidate):
                peak_x = float(image.peaks.points_xy[index, 0])
                peak_y = float(image.peaks.points_xy[index, 1])
                members.append(
                    {
                        "source": "non_template_extreme_bright_peak",
                        "source_index": int(index),
                        "x_pixels": peak_x,
                        "y_pixels": peak_y,
                        "best_plane_zero_based": int(
                            image.peaks.best_plane_by_peak[index]
                        ),
                        "nominal_template_species": None,
                        "motif_contrast_z": None,
                        "local_brightness_z": float(
                            evidence.local_brightness_z[index]
                        ),
                        "association_residual_pixels": None,
                        "persistence_planes": int(
                            image.peaks.persistence_count[index]
                        ),
                        "distance_to_image_edge_pixels": float(
                            min(
                                peak_x,
                                peak_y,
                                image_width - 1 - peak_x,
                                image_height - 1 - peak_y,
                            )
                        ),
                        "strict_rule_support": True,
                        "review_score": float(
                            evidence.local_brightness_z[index]
                        ),
                    }
                )
        cluster_radius = (
            analysis.config.expert_review_cluster_radius_cells * cell_scale
        )
        groups = _cluster_expert_review_members(members, cluster_radius)
        regions: list[dict[str, Any]] = []
        base_display_radius = (
            analysis.config.expert_review_display_radius_cells * cell_scale
        )
        for group in groups:
            points = np.asarray(
                [[item["x_pixels"], item["y_pixels"]] for item in group],
                dtype=float,
            )
            center = np.mean(points, axis=0)
            spread = float(
                np.max(np.linalg.norm(points - center[None, :], axis=1))
            )
            strict = any(bool(item["strict_rule_support"]) for item in group)
            representative = max(
                group, key=lambda item: float(item["review_score"])
            )
            minimum_edge_distance = float(
                min(item["distance_to_image_edge_pixels"] for item in group)
            )
            boundary_warning = minimum_edge_distance < (
                analysis.config.expert_review_boundary_warning_cells * cell_scale
            )
            sources = sorted({str(item["source"]) for item in group})
            if len(sources) == 2:
                spatial_interpretation = "mixed_template_and_non_template_support"
            elif sources == ["template_site_contrast"]:
                spatial_interpretation = "template_site_occupancy_or_beam_overlap"
            else:
                spatial_interpretation = "template_empty_or_unassociated_bright_peak"
            regions.append(
                {
                    "center_x_pixels": float(center[0]),
                    "center_y_pixels": float(center[1]),
                    "radius_pixels": float(
                        max(base_display_radius, spread + 0.07 * cell_scale)
                    ),
                    "radius_projected_cells": float(
                        max(base_display_radius, spread + 0.07 * cell_scale)
                        / cell_scale
                    ),
                    "member_count": int(len(group)),
                    "member_sources": sources,
                    "strict_rule_support": strict,
                    "review_tier": (
                        "primary_existing_rule_supported"
                        if strict
                        else "secondary_expert_review_only"
                    ),
                    "spatial_interpretation": spatial_interpretation,
                    "best_plane_zero_based": int(
                        representative["best_plane_zero_based"]
                    ),
                    "maximum_review_score": float(
                        max(float(item["review_score"]) for item in group)
                    ),
                    "minimum_distance_to_image_edge_pixels": minimum_edge_distance,
                    "boundary_adjacent_for_review": bool(boundary_warning),
                    "members": group,
                }
            )
        regions.sort(
            key=lambda item: (
                not bool(item["strict_rule_support"]),
                -float(item["maximum_review_score"]),
                float(item["center_y_pixels"]),
                float(item["center_x_pixels"]),
            )
        )
        for ordinal, region in enumerate(regions, 1):
            region["region_id"] = ordinal
        image_records[image.label] = {
            "role": (
                "rare_earth_free_control"
                if image.label == analysis.reference_label
                else "moving_sample"
            ),
            "eligible_template_site_count": int(np.sum(eligible)),
            "template_site_tail_count_at_review_threshold": control_tail_count,
            "point_evidence_count": int(len(members)),
            "review_region_count": int(len(regions)),
            "primary_region_count": int(
                sum(bool(item["strict_rule_support"]) for item in regions)
            ),
            "secondary_region_count": int(
                sum(not bool(item["strict_rule_support"]) for item in regions)
            ),
            "boundary_adjacent_region_count": int(
                sum(bool(item["boundary_adjacent_for_review"]) for item in regions)
            ),
            "regions": regions,
        }
    return {
        "schema_version": "fau_expert_guided_candidate_regions_v1",
        "reference": analysis.reference_label,
        "reference_interpreted_region_count": 0,
        "Y_control_quantile": float(
            analysis.config.expert_review_control_quantile
        ),
        "minimum_motif_z": float(
            analysis.config.expert_review_minimum_motif_z
        ),
        "calibrated_motif_z_threshold": float(threshold),
        "Y_control_score_count": int(len(control_scores)),
        "Y_control_tail_count_at_threshold": int(np.sum(control_scores >= threshold)),
        "cluster_radius_projected_cells": float(
            analysis.config.expert_review_cluster_radius_cells
        ),
        "minimum_display_radius_projected_cells": float(
            analysis.config.expert_review_display_radius_cells
        ),
        "boundary_warning_distance_projected_cells": float(
            analysis.config.expert_review_boundary_warning_cells
        ),
        "logic": (
            "reliable distortion region AND bright class AND either Y-tail-calibrated "
            "same-CIF-motif contrast or strict non-template extreme-bright evidence; "
            "nearby point evidence is merged into review regions"
        ),
        "interpretation": (
            "Expanded regions are expert-review proposals, not additional final "
            "rare-earth assignments or independent chemical identification."
        ),
        "images": image_records,
    }


def _final_rare_earth_points(image: FAUImageAnalysis) -> np.ndarray:
    """Return structure-first candidates, falling back only for legacy callers."""

    if image.skeleton_first_sites is not None:
        return image.skeleton_first_sites.expected_xy[
            image.skeleton_first_sites.candidate
        ]
    return image.peaks.points_xy[image.rare_earth_evidence.candidate]


def _guest_cell_histogram(image: FAUImageAnalysis, bins: int = 32) -> np.ndarray:
    points = _final_rare_earth_points(image)
    uv = guest_coordinates_in_cif_cell(points, image.template_fit)
    histogram, _, _ = np.histogram2d(
        uv[:, 1], uv[:, 0], bins=bins, range=((0, 1), (0, 1))
    )
    histogram = gaussian_filter(histogram, 1.0, mode="wrap")
    histogram /= float(np.sum(histogram)) + 1e-12
    return histogram


def _global_comparison(reference: FAUImageAnalysis, moving: FAUImageAnalysis) -> dict[str, Any]:
    """Compare cell shape while separating unknown image magnification."""

    reference_basis = reference.template_fit.direct_basis_pixels
    moving_basis = moving.template_fit.direct_basis_pixels
    correspondence = match_lattice_bases(reference_basis, moving_basis)
    matched_moving = moving_basis @ correspondence["basis_transform"]
    deformation_gradient = matched_moving @ np.linalg.inv(reference_basis)
    rotation, stretch = polar(deformation_gradient)
    log_strain = np.asarray(logm(stretch).real, dtype=float)
    isotropic_log_scale = 0.5 * float(np.trace(log_strain))
    deviatoric_log_strain = log_strain - isotropic_log_scale * np.eye(2)
    principal_deviatoric = np.linalg.eigvalsh(deviatoric_log_strain)
    return {
        "apparent_pixel_basis_deformation_gradient": deformation_gradient.tolist(),
        "rigid_rotation_degrees_removed": float(
            np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))
        ),
        "apparent_isotropic_log_scale_percent": float(
            100.0 * isotropic_log_scale
        ),
        "scale_free_shape_log_strain_tensor_percent": (
            100.0 * deviatoric_log_strain
        ).tolist(),
        "scale_free_principal_shape_strain_percent": (
            100.0 * principal_deviatoric
        ).tolist(),
        "scale_free_principal_strain_difference_percent": float(
            100.0 * np.ptp(principal_deviatoric)
        ),
        "absolute_isotropic_lattice_change": None,
        "absolute_projected_area_change": None,
        "identifiability": (
            "Different image sizes/fields of view do not establish a common physical "
            "pixel size. Isotropic scale and area change are confounded with magnification; "
            "only shape anisotropy is reported, assuming no differential scan anisotropy."
        ),
    }


def _local_deformation_comparison(
    reference: FAUImageAnalysis,
    moving: FAUImageAnalysis,
) -> dict[str, Any]:
    """Compare Y-referenced local distributions without spatial point pairing."""

    reference_valid = reference.deformation.reliable
    moving_valid = moving.deformation.reliable
    reference_distortion = (
        100.0 * reference.deformation.distortion_magnitude[reference_valid]
    )
    moving_distortion = (
        100.0 * moving.deformation.distortion_magnitude[moving_valid]
    )
    reference_shear = 100.0 * reference.deformation.maximum_shear[reference_valid]
    moving_shear = 100.0 * moving.deformation.maximum_shear[moving_valid]
    reference_p95 = float(np.percentile(reference_distortion, 95))
    return {
        "status": "ensemble_distribution_comparison",
        "reference": reference.label,
        "moving": moving.label,
        "reference_reliable_site_count": int(len(reference_distortion)),
        "moving_reliable_site_count": int(len(moving_distortion)),
        "distortion_percent": {
            "reference_median": float(np.median(reference_distortion)),
            "moving_median": float(np.median(moving_distortion)),
            "median_difference_vs_reference": float(
                np.median(moving_distortion) - np.median(reference_distortion)
            ),
            "wasserstein_distance": float(
                wasserstein_distance(reference_distortion, moving_distortion)
            ),
            "reference_p95": reference_p95,
            "moving_fraction_above_reference_p95": float(
                np.mean(moving_distortion > reference_p95)
            ),
        },
        "maximum_shear_strain_proxy_percent": {
            "reference_median": float(np.median(reference_shear)),
            "moving_median": float(np.median(moving_shear)),
            "median_difference_vs_reference": float(
                np.median(moving_shear) - np.median(reference_shear)
            ),
            "wasserstein_distance": float(
                wasserstein_distance(reference_shear, moving_shear)
            ),
        },
        "spatial_interpretation": (
            "CIF-aligned ensemble distributions only; different fields of view "
            "prevent same-atom or same-pixel displacement tracking."
        ),
    }


def _guest_onset_comparison(
    reference: FAUImageAnalysis,
    moving: FAUImageAnalysis,
) -> dict[str, Any]:
    reference_count = int(len(_final_rare_earth_points(reference)))
    moving_count = int(len(_final_rare_earth_points(moving)))
    reference_cells = np.prod(reference.peaks.best_image.shape) / abs(
        float(np.linalg.det(reference.template_fit.direct_basis_pixels))
    )
    moving_cells = np.prod(moving.peaks.best_image.shape) / abs(
        float(np.linalg.det(moving.template_fit.direct_basis_pixels))
    )
    return {
        "status": "appearance_vs_rare_earth_free_reference",
        "reference": reference.label,
        "moving": moving.label,
        "reference_candidate_count": reference_count,
        "moving_candidate_count": moving_count,
        "reference_candidates_per_projected_cell": float(
            reference_count / reference_cells
        ),
        "moving_candidates_per_projected_cell": float(moving_count / moving_cells),
        "migration_from_reference_identifiable": False,
        "interpretation": (
            "The reference contains no rare earth, so this is candidate appearance/"
            "occupancy relative to a negative control, not migration of the same atom."
        ),
    }


def _distribution_comparison(reference: FAUImageAnalysis, moving: FAUImageAnalysis) -> dict[str, Any]:
    reference_count = int(len(_final_rare_earth_points(reference)))
    moving_count = int(len(_final_rare_earth_points(moving)))
    if min(reference_count, moving_count) < 2:
        return {
            "status": "insufficient_candidates",
            "reference_candidate_count": reference_count,
            "moving_candidate_count": moving_count,
            "periodic_distribution_shift_uv": None,
            "periodic_distribution_shift_pixels_in_moving_image": None,
            "periodic_distribution_ncc": None,
            "jensen_shannon_divergence_bits": None,
            "interpretation": (
                "At least two intersection-qualified candidates are required in "
                "both fields; no distribution change was estimated."
            ),
        }
    reference_histogram = _guest_cell_histogram(reference)
    moving_histogram = _guest_cell_histogram(moving)
    score, shift_yx = _circular_ncc(reference_histogram, moving_histogram)
    bins = reference_histogram.shape[0]
    signed_shift = np.asarray([shift_yx[1], shift_yx[0]], dtype=float)
    signed_shift[signed_shift > bins / 2] -= bins
    shift_uv = signed_shift / bins
    divergence = float(
        jensenshannon(reference_histogram.ravel(), moving_histogram.ravel(), base=2.0) ** 2
    )
    return {
        "status": "estimated",
        "reference_candidate_count": reference_count,
        "moving_candidate_count": moving_count,
        "periodic_distribution_shift_uv": shift_uv.tolist(),
        "periodic_distribution_shift_pixels_in_moving_image": (
            moving.template_fit.direct_basis_pixels
            @ moving.template_fit.unit_cell_transform
            @ shift_uv
        ).tolist(),
        "periodic_distribution_ncc": score,
        "jensen_shannon_divergence_bits": divergence,
        "interpretation": (
            "ensemble unit-cell distribution only; different fields of view do not "
            "support individual-atom trajectories"
        ),
    }


def analyze_fau_stacks(
    cif_path: str | Path,
    stacks: Mapping[str, str | Path],
    *,
    reference_label: str = "Y",
    reference_is_rare_earth_free: bool = False,
    zone_axis: Sequence[int] | None = None,
    config: FAUStackConfig | None = None,
) -> FAUStackAnalysis:
    """Analyze independently sampled FAU stacks against a pure Si/O CIF."""

    policy = config or FAUStackConfig()
    policy.validate()
    if reference_label not in stacks:
        raise ValueError(f"Reference label {reference_label!r} is not present")
    if len(stacks) < 2:
        raise ValueError("At least one reference and one moving stack are required")
    structure = parse_and_expand_cif(cif_path)
    fixed_zone_axis: tuple[int, int, int] | None = None
    candidate_axes = list(ZONE_AXIS_CANDIDATES)
    if zone_axis is not None:
        raw_axis = np.asarray(zone_axis, dtype=int)
        if raw_axis.shape != (3,) or not np.any(raw_axis):
            raise ValueError("zone_axis must be a non-zero integer three-vector")
        divisor = math.gcd(*(abs(int(item)) for item in raw_axis))
        fixed_zone_axis = tuple(int(item // max(divisor, 1)) for item in raw_axis)

        def equivalent(first: Sequence[int], second: Sequence[int]) -> bool:
            return sorted(abs(int(item)) for item in first) == sorted(
                abs(int(item)) for item in second
            )

        candidate_axes = [fixed_zone_axis] + [
            axis for axis in candidate_axes
            if not equivalent(axis, fixed_zone_axis)
        ]
    templates = [project_structure(structure, axis) for axis in candidate_axes]
    intermediate: list[tuple[str, Path, StackPeakResult, TemplateFit]] = []
    for label, raw_path in stacks.items():
        path = Path(raw_path).resolve()
        peaks = detect_persistent_stack_peaks(load_focus_stack(path), policy)
        fit = fit_template_to_stack(
            peaks,
            structure,
            templates,
            policy,
            fixed_zone_axis=fixed_zone_axis,
        )
        intermediate.append((label, path, peaks, fit))

    # A zone axis is a property of the common specimen orientation.  Require a
    # common vote and use the aggregate evidence if a weak single field differs.
    aggregate: dict[str, float] = {}
    for _, _, _, fit in intermediate:
        for name, score in fit.zone_scores.items():
            aggregate[name] = aggregate.get(name, 0.0) + score
    if fixed_zone_axis is not None:
        common_axis = fixed_zone_axis
    else:
        common_name = max(aggregate, key=aggregate.get)
        common_axis = tuple(int(item) for item in common_name.strip("[]"))
    if any(fit.zone_axis != common_axis for _, _, _, fit in intermediate):
        common_template = next(item for item in templates if item.zone_axis == common_axis)
        rebuilt = []
        for label, path, peaks, fit in intermediate:
            scores = fit.zone_scores
            # Re-run the selected zone fit while keeping the already estimated
            # image basis; score_zone_axes is not needed a second time.
            folded = _fold_image_to_cell(peaks.best_image, fit.direct_basis_pixels, policy.unit_cell_bins)
            best: tuple[float, np.ndarray, np.ndarray] | None = None
            for matrix in _unimodular_matrices():
                for sigma in policy.template_blur_sigmas:
                    rendered = _render_template(common_template, matrix, policy.unit_cell_bins, sigma)
                    score, shift = _circular_ncc(folded, rendered)
                    if best is None or score > best[0]:
                        best = (score, matrix, shift)
            assert best is not None
            initial = np.asarray([best[2][1], best[2][0]]) / policy.unit_cell_bins
            shift, matches, expected_xy, expected_species = _refine_phase_shift(
                peaks, fit.direct_basis_pixels, common_template, best[1], initial, policy
            )
            physical_basis = (
                common_template.projected_basis_cell_fraction
                * structure.cell_lengths_angstrom[0]
            )
            new_fit = TemplateFit(
                direct_basis_pixels=fit.direct_basis_pixels,
                zone_axis=common_axis,
                zone_scores=scores,
                zone_score_margin=fit.zone_score_margin,
                unit_cell_transform=best[1],
                phase_shift_uv=shift,
                expected_points_xy=expected_xy,
                expected_species=expected_species,
                matched_observed_indices=matches[0],
                matched_expected_indices=matches[1],
                match_distances_pixels=matches[2],
                framework_observed_count=fit.framework_observed_count,
                template_ncc=float(best[0]),
                angstrom_per_pixel=math.sqrt(
                    abs(float(np.linalg.det(physical_basis)))
                    / abs(float(np.linalg.det(fit.direct_basis_pixels)))
                ),
            )
            rebuilt.append((label, path, peaks, new_fit))
        intermediate = rebuilt

    images: list[FAUImageAnalysis] = []
    for label, path, peaks, initial_fit in intermediate:
        fit = initial_fit
        deformation = estimate_local_deformation(peaks, fit, policy)
        evidence = identify_distortion_gated_rare_earth(
            peaks, fit, deformation, policy
        )
        # Remove only the intersection-qualified candidates, then recompute the
        # framework deformation.  Two short fixed-point passes are enough for
        # the sparse exclusion set while keeping the procedure deterministic.
        for _ in range(2):
            eligible = peaks.framework_eligible & ~evidence.candidate
            refined_fit = _rematch_framework(peaks, fit, eligible, policy)
            refined_deformation = estimate_local_deformation(
                peaks, refined_fit, policy
            )
            refined_evidence = identify_distortion_gated_rare_earth(
                peaks, refined_fit, refined_deformation, policy
            )
            unchanged = np.array_equal(
                evidence.candidate, refined_evidence.candidate
            )
            fit, deformation, evidence = (
                refined_fit,
                refined_deformation,
                refined_evidence,
            )
            if unchanged:
                break
        _attach_candidate_distances(peaks, deformation, evidence, fit)
        images.append(
            FAUImageAnalysis(
                label=label,
                source_path=path,
                source_sha256=_sha256_file(path),
                peaks=peaks,
                template_fit=fit,
                deformation=deformation,
                rare_earth_evidence=evidence,
            )
        )
    reference = next(item for item in images if item.label == reference_label)
    negative_control_audit = (
        apply_rare_earth_free_reference_audit(images, reference_label)
        if reference_is_rare_earth_free
        else None
    )
    if reference_is_rare_earth_free:
        for image in images:
            _attach_candidate_distances(
                image.peaks,
                image.deformation,
                image.rare_earth_evidence,
                image.template_fit,
            )
    for image in images:
        stack = load_focus_stack(image.source_path)
        image.aligned_site_contrast = measure_aligned_framework_site_contrast(
            stack,
            image.peaks,
            image.template_fit,
            image.rare_earth_evidence,
            structure,
            policy,
        )
        image.skeleton_first_sites = measure_skeleton_first_site_i_evidence(
            stack,
            image.peaks,
            image.template_fit,
            structure,
            policy,
        )
    aligned_site_negative_control_audit = (
        apply_aligned_site_reference_audit(images, reference_label)
        if reference_is_rare_earth_free
        else None
    )
    skeleton_first_negative_control_audit = (
        apply_skeleton_first_reference_audit(
            images, reference_label, policy
        )
        if reference_is_rare_earth_free
        else None
    )
    if reference_is_rare_earth_free:
        for image in images:
            final_points = _final_rare_earth_points(image)
            observed_framework = image.peaks.points_xy[
                image.template_fit.matched_observed_indices
            ]
            image.deformation.nearest_guest_distance_pixels = (
                cKDTree(final_points).query(observed_framework, k=1)[0]
                if len(final_points)
                else np.full(len(observed_framework), np.inf)
            )
    global_comparisons = {
        item.label: _global_comparison(reference, item)
        for item in images
        if item.label != reference_label
    }
    local_comparisons = {
        item.label: _local_deformation_comparison(reference, item)
        for item in images
        if item.label != reference_label
    }
    guest_onset = {
        item.label: _guest_onset_comparison(reference, item)
        for item in images
        if item.label != reference_label
    }
    if reference_is_rare_earth_free:
        guest_images = [item for item in images if item.label != reference_label]
        guest_comparisons = {
            f"{moving.label}_vs_{guest_reference.label}": _distribution_comparison(
                guest_reference, moving
            )
            for guest_reference, moving in itertools.combinations(guest_images, 2)
        }
    else:
        guest_comparisons = {
            item.label: _distribution_comparison(reference, item)
            for item in images
            if item.label != reference_label
        }
    return FAUStackAnalysis(
        cif_path=Path(cif_path).resolve(),
        cif_sha256=_sha256_file(Path(cif_path).resolve()),
        structure=structure,
        reference_label=reference_label,
        reference_is_rare_earth_free=reference_is_rare_earth_free,
        zone_axis_source=(
            "user_supplied" if fixed_zone_axis is not None else "image_search"
        ),
        images=images,
        global_comparisons=global_comparisons,
        local_deformation_comparisons=local_comparisons,
        guest_onset_comparisons=guest_onset,
        guest_distribution_comparisons=guest_comparisons,
        negative_control_audit=negative_control_audit,
        aligned_site_negative_control_audit=(
            aligned_site_negative_control_audit
        ),
        skeleton_first_negative_control_audit=(
            skeleton_first_negative_control_audit
        ),
        config=policy,
    )


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return {
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
    }


def _image_summary(image: FAUImageAnalysis, config: FAUStackConfig) -> dict[str, Any]:
    peaks, fit, deformation = image.peaks, image.template_fit, image.deformation
    evidence = image.rare_earth_evidence
    guest = evidence.candidate
    candidate_control_p = evidence.negative_control_tail_probability[guest]
    match_fraction = len(fit.matched_observed_indices) / max(fit.framework_observed_count, 1)
    cell_scale = math.sqrt(abs(float(np.linalg.det(fit.direct_basis_pixels))))
    reliable = deformation.reliable
    near = reliable & (
        deformation.nearest_guest_distance_pixels <= config.near_guest_radius_cells * cell_scale
    )
    far = reliable & (
        deformation.nearest_guest_distance_pixels >= config.far_guest_radius_cells * cell_scale
    )
    quality = "periodicity_and_local_fit_supported__chemical_identity_unvalidated" if (
        fit.template_ncc >= 0.45
        and fit.zone_score_margin >= 0.05
        and match_fraction >= 0.50
        and len(fit.match_distances_pixels) > 0
        and float(np.median(fit.match_distances_pixels)) <= 2.0
        and float(np.mean(reliable)) >= 0.70
    ) else "exploratory_only"
    aligned = image.aligned_site_contrast
    aligned_summary: dict[str, Any] | None = None
    if aligned is not None:
        associated = aligned.matched
        aligned_candidates = aligned.candidate
        aligned_candidate_control_p = aligned.negative_control_tail_probability[
            aligned_candidates
        ]
        aligned_summary = {
            "theoretical_site_count": int(len(aligned.expected_xy)),
            "theoretical_site_counts_by_species": {
                species: int(np.sum(aligned.expected_species == species))
                for species in ("O", "Si")
            },
            "peak_associated_site_count": int(np.sum(associated)),
            "peak_association_fraction": float(np.mean(associated)),
            "peak_associated_counts_by_species": {
                species: int(
                    np.sum(associated & (aligned.expected_species == species))
                )
                for species in ("O", "Si")
            },
            "motif_normalized_contrast_z": _percentiles(
                aligned.motif_normalized_contrast_z[associated]
            ),
            "local_same_species_contrast_z": _percentiles(
                aligned.local_same_species_contrast_z[associated]
            ),
            "contrast_anomaly_count_before_hotspot_gate": int(
                np.sum(aligned.contrast_anomaly)
            ),
            "hotspot_supported_anomaly_count_before_border_gate": int(
                np.sum(
                    aligned.contrast_anomaly
                    & aligned.inside_hotspot_region
                )
            ),
            "association_rejected_hotspot_anomaly_count": int(
                np.sum(
                    aligned.contrast_anomaly
                    & aligned.inside_hotspot_region
                    & ~aligned.association_reliable
                )
            ),
            "boundary_rejected_hotspot_anomaly_count": int(
                np.sum(
                    aligned.contrast_anomaly
                    & aligned.inside_hotspot_region
                    & aligned.association_reliable
                    & ~aligned.inside_reliable_interior
                )
            ),
            "raw_hotspot_supported_screen_positive_count": int(
                np.sum(aligned.screen_positive)
            ),
            "interpreted_rare_earth_candidate_count": int(
                np.sum(aligned_candidates)
            ),
            "candidate_counts_by_nominal_template_species": {
                species: int(
                    np.sum(
                        aligned_candidates
                        & (aligned.expected_species == species)
                    )
                )
                for species in ("O", "Si")
            },
            "candidate_overlap_with_peak_gate_count": int(
                np.sum(
                    aligned_candidates
                    & aligned.overlaps_peak_gate_screen_positive
                )
            ),
            "candidate_negative_control_tail_probability": (
                _percentiles(aligned_candidate_control_p)
                if len(aligned_candidate_control_p)
                else None
            ),
            "gate": {
                "logic": (
                    "detected peak associated with a fixed CIF O/Si site AND "
                    "nearby same-element local contrast z >= threshold AND "
                    "distortion-hotspot support AND strict association residual "
                    "AND reliable image interior; global brightness class is "
                    "audit-only"
                ),
                "local_same_species_contrast_z_threshold": (
                    config.local_same_species_z_threshold
                ),
                "local_radius_projected_cells": (
                    config.local_same_species_radius_cells
                ),
                "minimum_local_same_species_references": (
                    config.local_same_species_minimum_references
                ),
                "maximum_candidate_association_residual_pixels": (
                    config.aligned_site_candidate_max_residual_pixels
                ),
                "minimum_edge_distance_projected_cells": (
                    config.aligned_site_candidate_border_cells
                ),
                "cross_plane_persistence_role": "reported_not_required",
            },
            "interpretation": (
                "A candidate marks anomalous intensity at the nominal O/Si "
                "template position. It can reflect rare-earth occupancy or "
                "column overlap, but is not chemical confirmation."
            ),
        }
    skeleton = image.skeleton_first_sites
    skeleton_summary: dict[str, Any] | None = None
    final_candidate_count = int(np.sum(guest))
    priority_review_count = final_candidate_count
    final_single_plane_count: int | None = int(
        np.sum(guest & (peaks.persistence_count == 1))
    )
    if skeleton is not None:
        final_candidate_count = int(np.sum(skeleton.candidate))
        priority_review_count = int(np.sum(skeleton.structural_review))
        # Each probe is searched over every reconstructed plane and keeps an
        # argmax plane.  That is not evidence that the site is "single-plane".
        final_single_plane_count = None
        skeleton_summary = {
            "site_definition": "FAU 16c/site-I D6R center projected along [101]",
            "template_empty_probe_count": int(len(skeleton.expected_xy)),
            "valid_two_O_bridge_model_count": int(
                np.sum(skeleton.model_valid)
            ),
            "raw_structural_screen_positive_count": int(
                np.sum(skeleton.raw_structural_screen_positive)
            ),
            "structure_supported_candidate_count": final_candidate_count,
            "delta_bic": _percentiles(
                skeleton.delta_bic[skeleton.model_valid]
            ),
            "local_equivalent_site_center_component_z": _percentiles(
                skeleton.local_equivalent_site_z[
                    skeleton.model_valid
                ]
            ),
            "calibrated_delta_bic_threshold": float(
                skeleton.calibrated_delta_bic_threshold
            ),
            "within_image_structural_quantile_gate": False,
            "local_brightness_gate": False,
            "interpretation": (
                "Every fixed-threshold structural positive is retained as a "
                "site-I occupancy/projection-overlap hypothesis, not elemental "
                "confirmation."
            ),
        }
    return {
        "source": str(image.source_path),
        "sha256": image.source_sha256,
        "stack_shape_qyx": list(peaks.stack_shape),
        "z_axis_interpretation": "reconstruction/focus planes; physical z spacing absent",
        "best_focus_plane_zero_based": peaks.best_plane_index,
        "selected_focus_planes_zero_based": peaks.selected_plane_indices.tolist(),
        "detected_peak_count_all_planes": int(len(peaks.points_xy)),
        "framework_eligible_persistent_peak_count": int(
            np.sum(peaks.framework_eligible)
        ),
        "single_plane_detected_peak_count": int(np.sum(peaks.persistence_count == 1)),
        "brightness_class_counts": {
            name: int(np.sum(peaks.species_brightness_class == name))
            for name in BRIGHTNESS_CLASS_ORDER
        },
        "rare_earth_candidate_count": final_candidate_count,
        "possible_rare_earth_priority_review_count": priority_review_count,
        "legacy_peak_gate_candidate_count": int(np.sum(guest)),
        "raw_intersection_screen_positive_count": int(
            np.sum(evidence.screen_positive)
        ),
        "single_plane_rare_earth_candidate_count": final_single_plane_count,
        "hotspot_region_peak_count": int(np.sum(evidence.inside_hotspot_region)),
        "extreme_brightness_peak_count": int(np.sum(evidence.extreme_brightness)),
        "outside_hotspot_extreme_brightness_count": int(
            np.sum(evidence.extreme_brightness & ~evidence.inside_hotspot_region)
        ),
        "rare_earth_candidates_per_projected_cell": float(
            final_candidate_count
            / (
                np.prod(peaks.best_image.shape)
                / abs(float(np.linalg.det(fit.direct_basis_pixels)))
            )
        ),
        "rare_earth_identity": (
            "Every structure-supported FAU site-I occupancy/projection-overlap "
            "location is retained; local intensity is audit-only and the result "
            "is not chemical proof"
        ),
        "candidate_negative_control_tail_probability": (
            _percentiles(candidate_control_p) if len(candidate_control_p) else None
        ),
        "candidate_gate": {
            "owner": "skeleton_first_site_I",
            "logic": (
                "retain every valid positive-center template-plus-site-I fit with "
                "delta-BIC at or above the fixed minimum; Y is forced to zero as "
                "a declared negative control; brightness and GPA are not gates"
            ),
            "minimum_structural_delta_bic": (
                config.site_i_minimum_delta_bic
            ),
            "local_equivalent_site_audit_radius_projected_cells": (
                config.site_i_local_radius_cells
            ),
            "minimum_local_equivalent_site_audit_references": (
                config.site_i_local_minimum_references
            ),
            "within_image_structural_quantile_gate": "absent",
            "local_equivalent_site_intensity_gate": "absent_audit_only",
            "global_absolute_brightness_role": "absent",
            "gpa_hotspot_role": "posthoc_support_only_not_a_gate",
        },
        "legacy_peak_gate": {
            "logic": (
                "reliable hotspot region AND local brightness z >= threshold "
                "AND neutral brightness class == bright"
            ),
            "hotspot_distortion_threshold_percent": float(
                100.0 * evidence.distortion_threshold
            ),
            "hotspot_maximum_shear_threshold_percent": float(
                100.0 * evidence.maximum_shear_threshold
            ),
            "hotspot_radius_projected_cells": config.hotspot_radius_cells,
            "local_brightness_z_threshold": config.local_brightness_z_threshold,
            "global_brightness_z_audit_reference_not_gate": (
                config.global_brightness_z_threshold
            ),
            "framework_brightness_quantile": config.framework_brightness_quantile,
            "framework_standardized_brightness_audit_threshold_not_gate": float(
                evidence.framework_brightness_threshold
            ),
            "persistence_role": "confidence_only_not_required",
            "negative_control_tail_probability_role": (
                "audit_only_not_cross_image_intensity_gate"
            ),
        },
        "mixture_means_dim_to_bright": peaks.mixture_means.tolist(),
        "mixture_separation_sigma": peaks.mixture_separation_sigma.tolist(),
        "zone_axis": list(fit.zone_axis),
        "zone_scores": fit.zone_scores,
        "zone_score_margin": fit.zone_score_margin,
        "template_ncc": fit.template_ncc,
        "unit_cell_transform_cif_to_image_uv": fit.unit_cell_transform.tolist(),
        "phase_shift_image_uv": fit.phase_shift_uv.tolist(),
        "projected_cell_basis_xy_pixels": fit.direct_basis_pixels.T.tolist(),
        "projected_cell_area_pixels2": abs(float(np.linalg.det(fit.direct_basis_pixels))),
        "cif_calibrated_angstrom_per_pixel": fit.angstrom_per_pixel,
        "matched_framework_count": int(len(fit.matched_observed_indices)),
        "matched_fraction_of_framework_eligible_peaks": float(match_fraction),
        "framework_match_distance_pixels": _percentiles(fit.match_distances_pixels),
        "local_fit_reliable_fraction": float(np.mean(reliable)) if len(reliable) else 0.0,
        "local_distortion_percent": (
            _percentiles(100.0 * deformation.distortion_magnitude[reliable])
        ),
        "near_guest_distortion_percent": (
            _percentiles(100.0 * deformation.distortion_magnitude[near])
        ),
        "far_from_guest_distortion_percent": (
            _percentiles(100.0 * deformation.distortion_magnitude[far])
        ),
        "scientific_interpretation_grade": quality,
        "stress_status": (
            "physical stress not computed; maximum shear strain is a hotspot proxy only"
        ),
        "aligned_O_Si_site_contrast": aligned_summary,
        "skeleton_first_site_I_evidence": skeleton_summary,
    }


def _serializable_summary(analysis: FAUStackAnalysis) -> dict[str, Any]:
    species, counts = np.unique(analysis.structure.species, return_counts=True)
    return {
        "schema_version": "fau_stack_analysis_v6",
        "reference": analysis.reference_label,
        "reference_is_rare_earth_free": analysis.reference_is_rare_earth_free,
        "negative_control_audit": analysis.negative_control_audit,
        "aligned_site_negative_control_audit": (
            analysis.aligned_site_negative_control_audit
        ),
        "skeleton_first_negative_control_audit": (
            analysis.skeleton_first_negative_control_audit
        ),
        "zone_axis_source": analysis.zone_axis_source,
        "cif": {
            "path": str(analysis.cif_path),
            "sha256": analysis.cif_sha256,
            "space_group_number": analysis.structure.space_group_number,
            "cell_lengths_angstrom": list(analysis.structure.cell_lengths_angstrom),
            "cell_angles_degrees": list(analysis.structure.cell_angles_degrees),
            "expanded_atom_counts": {
                str(name): int(count) for name, count in zip(species, counts)
            },
            "symmetry_operation_count": analysis.structure.symmetry_operation_count,
            "rare_earth_in_cif": False,
        },
        "hard_invariant": (
            "The candidate owner is structure-first at the template-empty #4/#10-"
            "type projected FAU site-I bridge. Every valid positive-center fit at "
            "or above the fixed delta-BIC minimum is retained. No within-image "
            "quantile or local-brightness gate is applied. Y remains a declared "
            "zero-candidate control, and GPA is absent from candidate generation."
        ),
        "images": {
            image.label: _image_summary(image, analysis.config)
            for image in analysis.images
        },
        "moving_vs_Y_global_lattice": analysis.global_comparisons,
        "moving_vs_Y_local_deformation_distributions": (
            analysis.local_deformation_comparisons
        ),
        "moving_vs_Y_guest_onset": analysis.guest_onset_comparisons,
        "guest_distribution_between_nonreference_fields": (
            analysis.guest_distribution_comparisons
        ),
        "limits": [
            "Different fields of view prevent individual-atom trajectory matching.",
            "Different field sizes do not establish a common physical pixel size; absolute isotropic lattice and area changes are not identifiable.",
            "No TIFF physical z step is available, so this is not a 3-D coordinate reconstruction.",
            "No elastic tensor is supplied, so stress in Pa/GPa is not identifiable.",
            "CIF-conditioned per-site strain remains exploratory when match coverage/ residuals fail the declared gate.",
            "Candidate identity remains image-based and requires EDS/EELS or calibrated multislice validation.",
            "An anomalous nominal O/Si column can represent substitution, projected column overlap, thickness, channeling, or reconstruction artifacts; it is not elemental proof.",
        ],
        "config": asdict(analysis.config),
    }


def _write_rare_earth_csv(analysis: FAUStackAnalysis, path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        fields = (
            "image", "candidate_id", "x_pixels", "y_pixels",
            "canonical_CIF_cell_u", "canonical_CIF_cell_v",
            "best_focus_plane_zero_based", "persistence_planes",
            "brightness_class", "brightness_assignment_probability",
            "standardized_brightness", "local_contrast",
            "local_brightness_z", "global_brightness_z",
            "negative_control_tail_probability_audit",
            "nearest_hotspot_pixels", "evidence_score_not_probability",
            "nearest_CIF_framework_site_pixels", "status",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image in analysis.images:
            evidence = image.rare_earth_evidence
            selected = np.flatnonzero(evidence.candidate)
            nearest = cKDTree(image.template_fit.expected_points_xy).query(
                image.peaks.points_xy[selected], k=1
            )[0]
            canonical_uv = guest_coordinates_in_cif_cell(
                image.peaks.points_xy[selected], image.template_fit
            )
            for ordinal, (index, distance, uv) in enumerate(
                zip(selected, nearest, canonical_uv), 1
            ):
                writer.writerow({
                    "image": image.label,
                    "candidate_id": ordinal,
                    "x_pixels": float(image.peaks.points_xy[index, 0]),
                    "y_pixels": float(image.peaks.points_xy[index, 1]),
                    "canonical_CIF_cell_u": float(uv[0]),
                    "canonical_CIF_cell_v": float(uv[1]),
                    "best_focus_plane_zero_based": int(image.peaks.best_plane_by_peak[index]),
                    "persistence_planes": int(image.peaks.persistence_count[index]),
                    "brightness_class": str(image.peaks.species_brightness_class[index]),
                    "brightness_assignment_probability": float(image.peaks.assignment_probability[index]),
                    "standardized_brightness": float(image.peaks.standardized_brightness[index]),
                    "local_contrast": float(image.peaks.local_contrast[index]),
                    "local_brightness_z": float(evidence.local_brightness_z[index]),
                    "global_brightness_z": float(evidence.global_brightness_z[index]),
                    "negative_control_tail_probability_audit": float(
                        evidence.negative_control_tail_probability[index]
                    ),
                    "nearest_hotspot_pixels": float(
                        evidence.nearest_hotspot_distance_pixels[index]
                    ),
                    "evidence_score_not_probability": float(
                        evidence.evidence_score[index]
                    ),
                    "nearest_CIF_framework_site_pixels": float(distance),
                    "status": "distortion_gated_extreme_brightness_candidate",
                })


def _write_rare_earth_screening_csv(
    analysis: FAUStackAnalysis, path: Path
) -> None:
    """Write every peak that passed either half of the intersection gate."""

    with path.open("x", encoding="utf-8", newline="") as stream:
        fields = (
            "image", "peak_id", "x_pixels", "y_pixels", "best_plane_zero_based",
            "persistence_planes", "brightness_class", "standardized_brightness",
            "local_brightness_z", "global_brightness_z", "nearest_hotspot_pixels",
            "negative_control_tail_probability_audit", "inside_hotspot_region",
            "extreme_brightness", "screen_positive", "candidate",
            "screening_status",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image in analysis.images:
            peaks = image.peaks
            evidence = image.rare_earth_evidence
            selected = np.flatnonzero(
                evidence.inside_hotspot_region | evidence.extreme_brightness
            )
            for index in selected:
                if evidence.screen_positive[index] and not evidence.candidate[index]:
                    status = "negative_control_screen_positive_not_guest"
                elif evidence.candidate[index]:
                    status = "candidate_intersection"
                elif evidence.extreme_brightness[index]:
                    status = "rejected_outside_hotspot"
                else:
                    status = "rejected_not_extreme_brightness"
                writer.writerow({
                    "image": image.label,
                    "peak_id": int(index + 1),
                    "x_pixels": float(peaks.points_xy[index, 0]),
                    "y_pixels": float(peaks.points_xy[index, 1]),
                    "best_plane_zero_based": int(peaks.best_plane_by_peak[index]),
                    "persistence_planes": int(peaks.persistence_count[index]),
                    "brightness_class": str(peaks.species_brightness_class[index]),
                    "standardized_brightness": float(peaks.standardized_brightness[index]),
                    "local_brightness_z": float(evidence.local_brightness_z[index]),
                    "global_brightness_z": float(evidence.global_brightness_z[index]),
                    "nearest_hotspot_pixels": float(
                        evidence.nearest_hotspot_distance_pixels[index]
                    ),
                    "negative_control_tail_probability_audit": float(
                        evidence.negative_control_tail_probability[index]
                    ),
                    "inside_hotspot_region": bool(evidence.inside_hotspot_region[index]),
                    "extreme_brightness": bool(evidence.extreme_brightness[index]),
                    "screen_positive": bool(evidence.screen_positive[index]),
                    "candidate": bool(evidence.candidate[index]),
                    "screening_status": status,
                })


def _aligned_site_screening_status(
    site: AlignedSiteContrast,
    index: int,
) -> str:
    if site.candidate[index]:
        return "rare_earth_occupancy_or_overlap_candidate"
    if site.screen_positive[index]:
        return "negative_control_screen_positive_not_guest"
    if (
        site.contrast_anomaly[index]
        and site.inside_hotspot_region[index]
        and not site.association_reliable[index]
    ):
        return "rejected_large_template_association_residual"
    if (
        site.contrast_anomaly[index]
        and site.inside_hotspot_region[index]
        and not site.inside_reliable_interior[index]
    ):
        return "rejected_unreliable_image_boundary"
    if site.contrast_anomaly[index]:
        return "contrast_anomaly_outside_distortion_hotspot"
    if site.matched[index]:
        return "peak_associated_not_contrast_anomalous"
    return "no_detected_peak_within_template_tolerance"


def _write_aligned_site_contrast_csv(
    analysis: FAUStackAnalysis,
    path: Path,
    *,
    candidates_only: bool = False,
) -> None:
    """Write one auditable row per projected CIF site, or candidates only."""

    fields = (
        "image", "expected_site_id", "canonical_CIF_motif_id",
        "nominal_template_species", "expected_x_pixels", "expected_y_pixels",
        "measurement_x_pixels", "measurement_y_pixels",
        "canonical_CIF_cell_u", "canonical_CIF_cell_v",
        "peak_associated", "observed_peak_id", "association_residual_pixels",
        "candidate_association_reliable",
        "best_plane_zero_based", "persistence_planes",
        "per_plane_standardized_DoG_contrast",
        "raw_DoG_contrast", "raw_detector_intensity",
        "neutral_brightness_class", "same_motif_contrast_z",
        "local_same_species_contrast_z", "local_same_species_reference_count",
        "inside_distortion_hotspot", "distance_to_image_edge_pixels",
        "inside_reliable_image_interior", "contrast_anomaly",
        "raw_screen_positive", "interpreted_candidate",
        "overlaps_original_peak_gate_screen_positive",
        "Y_negative_control_tail_probability_audit", "screening_status",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image in analysis.images:
            site = image.aligned_site_contrast
            if site is None:
                raise ValueError(
                    f"Aligned-site contrast is missing for {image.label}"
                )
            indices = (
                np.flatnonzero(site.candidate)
                if candidates_only
                else np.arange(len(site.expected_xy), dtype=int)
            )
            for index in indices:
                observed_id: int | str = ""
                if site.observed_peak_index[index] >= 0:
                    observed_id = int(site.observed_peak_index[index] + 1)
                writer.writerow({
                    "image": image.label,
                    "expected_site_id": int(index + 1),
                    "canonical_CIF_motif_id": int(
                        site.canonical_site_id[index] + 1
                    ),
                    "nominal_template_species": str(
                        site.expected_species[index]
                    ),
                    "expected_x_pixels": float(site.expected_xy[index, 0]),
                    "expected_y_pixels": float(site.expected_xy[index, 1]),
                    "measurement_x_pixels": float(
                        site.measurement_xy[index, 0]
                    ),
                    "measurement_y_pixels": float(
                        site.measurement_xy[index, 1]
                    ),
                    "canonical_CIF_cell_u": float(site.canonical_uv[index, 0]),
                    "canonical_CIF_cell_v": float(site.canonical_uv[index, 1]),
                    "peak_associated": bool(site.matched[index]),
                    "observed_peak_id": observed_id,
                    "association_residual_pixels": float(
                        site.residual_pixels[index]
                    ),
                    "candidate_association_reliable": bool(
                        site.association_reliable[index]
                    ),
                    "best_plane_zero_based": int(site.best_plane[index]),
                    "persistence_planes": int(site.persistence_planes[index]),
                    "per_plane_standardized_DoG_contrast": float(
                        site.standardized_contrast[index]
                    ),
                    "raw_DoG_contrast": float(site.raw_dog_contrast[index]),
                    "raw_detector_intensity": float(
                        site.raw_detector_intensity[index]
                    ),
                    "neutral_brightness_class": str(
                        site.brightness_class[index]
                    ),
                    "same_motif_contrast_z": float(
                        site.motif_normalized_contrast_z[index]
                    ),
                    "local_same_species_contrast_z": float(
                        site.local_same_species_contrast_z[index]
                    ),
                    "local_same_species_reference_count": int(
                        site.local_same_species_reference_count[index]
                    ),
                    "inside_distortion_hotspot": bool(
                        site.inside_hotspot_region[index]
                    ),
                    "distance_to_image_edge_pixels": float(
                        site.distance_to_image_edge_pixels[index]
                    ),
                    "inside_reliable_image_interior": bool(
                        site.inside_reliable_interior[index]
                    ),
                    "contrast_anomaly": bool(site.contrast_anomaly[index]),
                    "raw_screen_positive": bool(site.screen_positive[index]),
                    "interpreted_candidate": bool(site.candidate[index]),
                    "overlaps_original_peak_gate_screen_positive": bool(
                        site.overlaps_peak_gate_screen_positive[index]
                    ),
                    "Y_negative_control_tail_probability_audit": float(
                        site.negative_control_tail_probability[index]
                    ),
                    "screening_status": _aligned_site_screening_status(
                        site, int(index)
                    ),
                })


def _skeleton_first_site_status(
    image_label: str,
    reference_label: str,
    site: SkeletonFirstSiteEvidence,
    index: int,
) -> str:
    if not site.model_valid[index]:
        return "model_unavailable"
    if not site.inside_reliable_interior[index]:
        return "rejected_image_boundary"
    if image_label == reference_label and site.raw_structural_screen_positive[index]:
        return "Y_control_structural_tail_not_guest"
    if site.candidate[index]:
        return "structure_supported_site_I_occupancy_or_overlap_review"
    return "excluded_below_fixed_delta_BIC_threshold"


def _write_skeleton_first_site_i_csv(
    analysis: FAUStackAnalysis,
    path: Path,
    *,
    review_only: bool = False,
    candidates_only: bool = False,
) -> None:
    """Write every site-I structural probe or only review/candidate rows."""

    fields = (
        "image", "site_I_probe_id", "canonical_site_I_motif_id",
        "x_pixels", "y_pixels", "oxygen_endpoint_1_x_pixels",
        "oxygen_endpoint_1_y_pixels", "oxygen_endpoint_2_x_pixels",
        "oxygen_endpoint_2_y_pixels", "model_valid",
        "best_plane_zero_based", "template_only_bic",
        "template_plus_site_I_bic", "delta_bic_template_plus_site_I",
        "extra_site_I_component_amplitude_local_units",
        "local_equivalent_site_I_center_component_z",
        "local_equivalent_site_I_brightness_z_legacy_name",
        "local_reference_count",
        "distance_to_image_edge_pixels", "inside_reliable_image_interior",
        "calibrated_Y_delta_bic_threshold", "raw_structural_screen_positive",
        "structure_supported_candidate", "selection_rule",
        "Y_negative_control_tail_probability_audit", "status",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image in analysis.images:
            site = image.skeleton_first_sites
            if site is None:
                raise ValueError(
                    f"Skeleton-first site-I evidence is missing for {image.label}"
                )
            if candidates_only:
                selected = site.candidate
            elif review_only:
                selected = site.structural_review | site.candidate
            else:
                selected = np.ones(len(site.expected_xy), dtype=bool)
            for index in np.flatnonzero(selected):
                endpoints = site.oxygen_endpoint_xy[index]
                writer.writerow({
                    "image": image.label,
                    "site_I_probe_id": int(index + 1),
                    "canonical_site_I_motif_id": int(
                        site.canonical_site_id[index] + 1
                    ),
                    "x_pixels": float(site.expected_xy[index, 0]),
                    "y_pixels": float(site.expected_xy[index, 1]),
                    "oxygen_endpoint_1_x_pixels": float(endpoints[0, 0]),
                    "oxygen_endpoint_1_y_pixels": float(endpoints[0, 1]),
                    "oxygen_endpoint_2_x_pixels": float(endpoints[1, 0]),
                    "oxygen_endpoint_2_y_pixels": float(endpoints[1, 1]),
                    "model_valid": bool(site.model_valid[index]),
                    "best_plane_zero_based": int(site.best_plane[index]),
                    "template_only_bic": float(site.template_only_bic[index]),
                    "template_plus_site_I_bic": float(
                        site.template_plus_site_bic[index]
                    ),
                    "delta_bic_template_plus_site_I": float(
                        site.delta_bic[index]
                    ),
                    "extra_site_I_component_amplitude_local_units": float(
                        site.extra_component_amplitude[index]
                    ),
                    "local_equivalent_site_I_center_component_z": float(
                        site.local_equivalent_site_z[index]
                    ),
                    "local_equivalent_site_I_brightness_z_legacy_name": float(
                        site.local_equivalent_site_z[index]
                    ),
                    "local_reference_count": int(
                        site.local_reference_count[index]
                    ),
                    "distance_to_image_edge_pixels": float(
                        site.distance_to_image_edge_pixels[index]
                    ),
                    "inside_reliable_image_interior": bool(
                        site.inside_reliable_interior[index]
                    ),
                    "calibrated_Y_delta_bic_threshold": float(
                        site.calibrated_delta_bic_threshold
                    ),
                    "raw_structural_screen_positive": bool(
                        site.raw_structural_screen_positive[index]
                    ),
                    "structure_supported_candidate": bool(
                        site.candidate[index]
                    ),
                    "selection_rule": (
                        "valid_positive_center_and_delta_BIC_at_least_fixed_minimum"
                        if site.candidate[index]
                        else "not_selected"
                    ),
                    "Y_negative_control_tail_probability_audit": float(
                        site.negative_control_tail_probability[index]
                    ),
                    "status": _skeleton_first_site_status(
                        image.label,
                        analysis.reference_label,
                        site,
                        int(index),
                    ),
                })


def _write_expert_guided_candidate_regions_csv(
    audit: Mapping[str, Any],
    path: Path,
) -> None:
    """Write one row per expanded expert-review region."""

    fields = (
        "image", "region_id", "center_x_pixels", "center_y_pixels",
        "radius_pixels", "radius_projected_cells", "review_tier",
        "strict_rule_support", "spatial_interpretation",
        "best_plane_zero_based", "maximum_review_score", "member_count",
        "member_sources", "member_source_indices", "member_coordinates_pixels",
        "minimum_distance_to_image_edge_pixels", "boundary_adjacent_for_review",
        "calibrated_Y_control_motif_z_threshold", "status",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, image in audit["images"].items():
            for region in image["regions"]:
                members = region["members"]
                writer.writerow(
                    {
                        "image": label,
                        "region_id": int(region["region_id"]),
                        "center_x_pixels": float(region["center_x_pixels"]),
                        "center_y_pixels": float(region["center_y_pixels"]),
                        "radius_pixels": float(region["radius_pixels"]),
                        "radius_projected_cells": float(
                            region["radius_projected_cells"]
                        ),
                        "review_tier": region["review_tier"],
                        "strict_rule_support": bool(
                            region["strict_rule_support"]
                        ),
                        "spatial_interpretation": region[
                            "spatial_interpretation"
                        ],
                        "best_plane_zero_based": int(
                            region["best_plane_zero_based"]
                        ),
                        "maximum_review_score": float(
                            region["maximum_review_score"]
                        ),
                        "member_count": int(region["member_count"]),
                        "member_sources": ";".join(
                            str(item) for item in region["member_sources"]
                        ),
                        "member_source_indices": ";".join(
                            f"{item['source']}:{int(item['source_index']) + 1}"
                            for item in members
                        ),
                        "member_coordinates_pixels": ";".join(
                            f"({float(item['x_pixels']):.3f},"
                            f"{float(item['y_pixels']):.3f})"
                            for item in members
                        ),
                        "minimum_distance_to_image_edge_pixels": float(
                            region["minimum_distance_to_image_edge_pixels"]
                        ),
                        "boundary_adjacent_for_review": bool(
                            region["boundary_adjacent_for_review"]
                        ),
                        "calibrated_Y_control_motif_z_threshold": float(
                            audit["calibrated_motif_z_threshold"]
                        ),
                        "status": "expert_review_region_not_chemical_assignment",
                    }
                )


def _write_framework_csv(analysis: FAUStackAnalysis, path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        fields = (
            "image", "match_id", "species", "observed_x_pixels", "observed_y_pixels",
            "expected_x_pixels", "expected_y_pixels", "residual_pixels", "residual_angstrom",
            "local_exx", "local_eyy", "local_exy", "local_rotation_radians",
            "local_distortion_magnitude", "maximum_shear_stress_proxy",
            "local_fit_rmse_pixels", "local_fit_neighbor_count", "local_fit_reliable",
            "nearest_rare_earth_candidate_pixels",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image in analysis.images:
            fit, deformation = image.template_fit, image.deformation
            observed = image.peaks.points_xy[fit.matched_observed_indices]
            expected = fit.expected_points_xy[fit.matched_expected_indices]
            species = fit.expected_species[fit.matched_expected_indices]
            for index in range(len(observed)):
                writer.writerow({
                    "image": image.label,
                    "match_id": index + 1,
                    "species": species[index],
                    "observed_x_pixels": float(observed[index, 0]),
                    "observed_y_pixels": float(observed[index, 1]),
                    "expected_x_pixels": float(expected[index, 0]),
                    "expected_y_pixels": float(expected[index, 1]),
                    "residual_pixels": float(fit.match_distances_pixels[index]),
                    "residual_angstrom": float(fit.match_distances_pixels[index] * fit.angstrom_per_pixel),
                    "local_exx": float(deformation.exx[index]),
                    "local_eyy": float(deformation.eyy[index]),
                    "local_exy": float(deformation.exy[index]),
                    "local_rotation_radians": float(deformation.rotation[index]),
                    "local_distortion_magnitude": float(deformation.distortion_magnitude[index]),
                    "maximum_shear_stress_proxy": float(deformation.maximum_shear[index]),
                    "local_fit_rmse_pixels": float(deformation.fit_rmse_pixels[index]),
                    "local_fit_neighbor_count": int(deformation.neighbor_count[index]),
                    "local_fit_reliable": bool(deformation.reliable[index]),
                    "nearest_rare_earth_candidate_pixels": float(deformation.nearest_guest_distance_pixels[index]),
                })


def _hotspot_region_mask(
    image: FAUImageAnalysis, config: FAUStackConfig
) -> np.ndarray:
    """Rasterize hotspot seeds using the declared projected-cell radius."""

    shape = image.peaks.best_image.shape
    seeds = np.zeros(shape, dtype=bool)
    observed = image.peaks.points_xy[
        image.template_fit.matched_observed_indices
    ]
    points = observed[image.rare_earth_evidence.hotspot_framework_site]
    rounded = np.rint(points).astype(int)
    if len(rounded):
        inside = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < shape[1])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < shape[0])
        )
        seeds[rounded[inside, 1], rounded[inside, 0]] = True
    if not np.any(seeds):
        return seeds
    cell_scale = math.sqrt(
        abs(float(np.linalg.det(image.template_fit.direct_basis_pixels)))
    )
    return distance_transform_edt(~seeds) <= config.hotspot_radius_cells * cell_scale


def _save_final_rare_earth_assignment_figure(
    analysis: FAUStackAnalysis,
    path: Path,
    dpi: int,
) -> None:
    """Plot only final rare-earth assignments, without hotspot audit marks."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite rare-earth assignment figure: {path}")
    figure, axes = plt.subplots(
        1,
        len(analysis.images),
        figsize=(5.4 * len(analysis.images), 5.8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, image in zip(axes, analysis.images):
        axis.imshow(image.peaks.best_image, cmap="gray", vmin=0, vmax=1)
        selected_points = _final_rare_earth_points(image)
        if len(selected_points):
            axis.scatter(
                selected_points[:, 0],
                selected_points[:, 1],
                s=260,
                facecolors="none",
                edgecolors="#FF2FA3",
                linewidths=1.8,
            )
            axis.scatter(
                selected_points[:, 0],
                selected_points[:, 1],
                s=95,
                marker="*",
                c="#FF2FA3",
                edgecolors="white",
                linewidths=0.55,
            )
            for ordinal, (x, y) in enumerate(selected_points, 1):
                axis.annotate(
                    str(ordinal),
                    (x, y),
                    xytext=(9, -9),
                    textcoords="offset points",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "circle,pad=0.18",
                        "facecolor": "#FF2FA3",
                        "edgecolor": "white",
                        "linewidth": 0.6,
                    },
                )
        else:
            message = "NO RARE-EARTH SITES\nfinal assignment = 0"
            if (
                image.label == analysis.reference_label
                and analysis.reference_is_rare_earth_free
            ):
                message = "RARE-EARTH-FREE REFERENCE\nfinal assignment = 0"
            axis.text(
                0.5,
                0.08,
                message,
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                color="white",
                fontsize=10,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": "#173B2E",
                    "edgecolor": "white",
                    "alpha": 0.90,
                },
            )
        role = (
            "rare-earth-free Y reference"
            if image.label == analysis.reference_label
            and analysis.reference_is_rare_earth_free
            else "sample"
        )
        axis.set_title(
            f"{image.label} ({role})\n"
            f"final rare-earth assignments: {len(selected_points)}"
        )
        axis.set_axis_off()
    figure.suptitle(
        "Final rare-earth site assignments only\n"
        "magenta circle/star = assigned candidate; no hotspot or rejected-point marks",
        fontsize=13,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_cif_projection_agreement_figure(
    analysis: FAUStackAnalysis,
    path: Path,
    dpi: int,
) -> None:
    """Show the canonical experimental cell and quantitative CIF agreement."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite CIF agreement figure: {path}")
    bins = analysis.config.unit_cell_bins
    colors = {"O": "#00E5FF", "Si": "#FFC400"}
    figure, axes = plt.subplots(
        2,
        len(analysis.images),
        figsize=(5.2 * len(analysis.images), 9.0),
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"height_ratios": (1.45, 1.0)},
    )
    for column, image in enumerate(analysis.images):
        fit = image.template_fit
        template = project_structure(analysis.structure, fit.zone_axis)
        cell = _fold_image_to_canonical_cif_cell(
            image.peaks.best_image, fit, bins
        )
        lower, upper = np.percentile(cell, (2.0, 98.0))
        cell = np.clip((cell - lower) / max(float(upper - lower), 1e-8), 0, 1)

        image_axis = axes[0, column]
        image_axis.imshow(
            cell,
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        for species, marker in (("O", "o"), ("Si", "s")):
            selected = template.species == species
            points = template.phase_uv[selected] * bins
            image_axis.scatter(
                points[:, 0],
                points[:, 1],
                s=22 if species == "O" else 28,
                marker=marker,
                facecolors="none",
                edgecolors=colors[species],
                linewidths=0.8,
                label=f"FAU.cif projected {species}",
            )
        matched_fraction = len(fit.matched_observed_indices) / max(
            fit.framework_observed_count, 1
        )
        image_axis.set_title(
            f"{image.label}: experimental canonical cell + FAU.cif [101]\n"
            f"periodic NCC={fit.template_ncc:.3f}; matched={matched_fraction:.1%}"
        )
        image_axis.set(
            xlabel="canonical CIF u bin",
            ylabel="canonical CIF v bin",
            xlim=(0, bins - 1),
            ylim=(0, bins - 1),
        )
        image_axis.legend(loc="upper right", fontsize=7)

        residual_axis = axes[1, column]
        residuals = np.asarray(fit.match_distances_pixels, dtype=float)
        residual_axis.hist(
            residuals,
            bins=np.linspace(0, analysis.config.framework_match_tolerance_pixels, 25),
            color="#5271A3",
            alpha=0.85,
        )
        median = float(np.median(residuals))
        p95 = float(np.percentile(residuals, 95))
        residual_axis.axvline(
            median,
            color="#FFB000",
            linewidth=2,
            label=f"median={median:.2f}px",
        )
        residual_axis.axvline(
            p95,
            color="#D1495B",
            linewidth=2,
            linestyle="--",
            label=f"P95={p95:.2f}px",
        )
        residual_axis.axvline(
            analysis.config.framework_match_tolerance_pixels,
            color="#222222",
            linewidth=1.2,
            linestyle=":",
            label=(
                "matching limit="
                f"{analysis.config.framework_match_tolerance_pixels:.1f}px"
            ),
        )
        residual_axis.set(
            title=(
                f"matched framework columns: {len(residuals):,} / "
                f"{fit.framework_observed_count:,}"
            ),
            xlabel="observed-to-CIF projected-site residual (pixels)",
            ylabel="matched columns",
            xlim=(0, analysis.config.framework_match_tolerance_pixels * 1.04),
        )
        residual_axis.legend(fontsize=8)
    figure.suptitle(
        "FAU.cif projection agreement after canonical-cell registration\n"
        "markers are theoretical O/Si projected columns; agreement is structural, "
        "not chemical validation",
        fontsize=13,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _draw_projected_framework_skeleton(
    axis: plt.Axes,
    skeleton: ProjectedFrameworkSkeleton,
    *,
    overlay: bool,
) -> None:
    """Draw one periodic projected framework with consistent atom ownership."""

    colors = {"O": "#00DFFF", "Si": "#FFB000"}
    bonds = LineCollection(
        skeleton.bond_segments_uv,
        colors="#35F28B" if overlay else "#D9E2EA",
        linewidths=0.9 if overlay else 1.45,
        alpha=0.78 if overlay else 0.88,
        zorder=3,
    )
    axis.add_collection(bonds)
    for species, marker in (("O", "o"), ("Si", "s")):
        selected = skeleton.species == species
        points = skeleton.atom_uv[selected]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=(14 if species == "O" else 22) if overlay else (
                32 if species == "O" else 48
            ),
            marker=marker,
            facecolors=colors[species],
            edgecolors="#102027" if overlay else "white",
            linewidths=0.45 if overlay else 0.65,
            alpha=0.92,
            zorder=4,
            label=f"{species} projected column",
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")


def _save_projected_framework_overlay_figure(
    analysis: FAUStackAnalysis,
    path: Path,
    dpi: int,
) -> None:
    """Overlay a bond-connected FAU projection on all registered cell images."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite framework overlay figure: {path}")
    zone_axis = next(
        image.template_fit.zone_axis
        for image in analysis.images
        if image.label == analysis.reference_label
    )
    skeleton = project_fau_framework_skeleton(analysis.structure, zone_axis)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 11.5),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    theory_axis = flat_axes[0]
    theory_axis.set_facecolor("#101820")
    _draw_projected_framework_skeleton(theory_axis, skeleton, overlay=False)
    theory_axis.set_title(
        "FAU.cif [101] projected Si-O framework\n"
        f"78 unique projected columns; {len(skeleton.bond_segments_uv)} visible bond segments"
    )
    theory_axis.set(xlabel="canonical CIF u", ylabel="canonical CIF v")
    theory_axis.tick_params(colors="#27313A")
    theory_axis.legend(loc="upper right", fontsize=8)
    theory_axis.text(
        0.02,
        0.02,
        f"3D Si-O bonds: {skeleton.three_dimensional_bond_count}\n"
        f"cutoff: {skeleton.bond_cutoff_angstrom:.1f} Å; every Si is four-coordinate",
        transform=theory_axis.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#263640",
            "edgecolor": "white",
            "alpha": 0.9,
        },
    )

    bins = analysis.config.unit_cell_bins
    for axis, image in zip(flat_axes[1:], analysis.images):
        fit = image.template_fit
        cell = _fold_image_to_canonical_cif_cell(
            image.peaks.best_image, fit, bins
        )
        lower, upper = np.percentile(cell, (2.0, 98.0))
        cell = np.clip((cell - lower) / max(float(upper - lower), 1e-8), 0, 1)
        axis.imshow(
            cell,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        _draw_projected_framework_skeleton(axis, skeleton, overlay=True)
        matched_fraction = len(fit.matched_observed_indices) / max(
            fit.framework_observed_count, 1
        )
        median = float(np.median(fit.match_distances_pixels))
        axis.set_title(
            f"{image.label}: experiment + identical FAU.cif skeleton\n"
            f"NCC={fit.template_ncc:.3f}; matched={matched_fraction:.1%}; "
            f"median residual={median:.2f}px"
        )
        axis.set(xlabel="canonical CIF u", ylabel="canonical CIF v")
    figure.suptitle(
        "Bond-connected FAU.cif [101] framework registered to experimental cells\n"
        "green lines = projected Si-O connectivity; cyan circles = O; amber squares = Si",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _draw_rectangular_projected_framework(
    axis: plt.Axes,
    framework: RectangularProjectedFramework,
    *,
    overlay: bool,
) -> None:
    """Draw the [101] framework in physical projected coordinates (angstrom)."""

    scale = np.asarray(
        [framework.width_angstrom, framework.height_angstrom], dtype=float
    )
    segments = framework.bond_segments_fractional_xy * scale[None, None, :]
    bonds = LineCollection(
        segments,
        colors="#35F28B" if overlay else "#D9E2EA",
        linewidths=0.55 if overlay else 0.85,
        alpha=0.68 if overlay else 0.82,
        zorder=3,
    )
    axis.add_collection(bonds)
    colors = {"O": "#00DFFF", "Si": "#FFB000"}
    for species, marker in (("O", "o"), ("Si", "s")):
        selected = framework.species == species
        points = framework.atom_fractional_xy[selected] * scale
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=(6 if species == "O" else 11) if overlay else (
                12 if species == "O" else 20
            ),
            marker=marker,
            facecolors=colors[species],
            edgecolors="#102027" if overlay else "white",
            linewidths=0.25 if overlay else 0.4,
            alpha=0.88,
            zorder=4,
            label=f"{species} projected column",
        )
    axis.set(
        xlim=(0, framework.width_angstrom),
        ylim=(0, framework.height_angstrom),
        aspect="equal",
        xlabel=r"[$\bar{1}$01] projected distance ($\AA$)",
        ylabel=r"[010] projected distance ($\AA$)",
    )


def _save_fau_site_i_search_template_figure(
    structure: CIFStructure,
    path: Path,
    dpi: int,
) -> None:
    """Show where the structure-first search probes sit in the FAU skeleton."""

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite site-I search template figure: {path}"
        )
    framework = project_fau_101_rectangular_framework(structure)
    probes = project_fau_101_rectangular_site_i(structure)
    scale = np.asarray(
        [framework.width_angstrom, framework.height_angstrom], dtype=float
    )
    probe_xy = probes * scale
    figure, axes = plt.subplots(
        1, 2, figsize=(14.5, 5.5), constrained_layout=True
    )
    for axis in axes:
        axis.set_facecolor("#101820")
        _draw_rectangular_projected_framework(
            axis, framework, overlay=False
        )
    axes[0].set_title(
        "Pure FAU.cif [101] projected framework\n"
        "cyan = O; amber = Si; lines = projected Si-O bonds"
    )
    axes[1].scatter(
        probe_xy[:, 0],
        probe_xy[:, 1],
        s=145,
        marker="o",
        facecolors="none",
        edgecolors="#FF2FA3",
        linewidths=1.7,
        zorder=6,
        label="template-empty FAU 16c/site-I probe",
    )
    axes[1].scatter(
        probe_xy[:, 0],
        probe_xy[:, 1],
        s=45,
        marker="+",
        c="#FF2FA3",
        linewidths=1.2,
        zorder=7,
    )
    for ordinal, (x, y) in enumerate(probe_xy, 1):
        axes[1].annotate(
            str(ordinal),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            color="#FF80BF",
            fontsize=6.5,
            zorder=8,
        )
    axes[1].set_title(
        "Structure-first search positions derived from FAU 16c/site-I\n"
        "magenta marks are empty-site probes, not detected atoms"
    )
    axes[1].legend(loc="upper right", fontsize=8)
    for axis in axes:
        axis.tick_params(colors="#27313A")
    figure.suptitle(
        "Physical-metric FAU [101] skeleton and the repeated red-box-type "
        "site-I search template\n"
        "the image algorithm tests these positions first, then compares local "
        "equivalent-site brightness",
        fontsize=13,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_physical_rectangular_framework_overlay_figure(
    analysis: FAUStackAnalysis,
    path: Path,
    dpi: int,
) -> None:
    """Save a metric-preserving rectangular [101] structure overlay page."""

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite physical rectangular overlay figure: {path}"
        )
    framework = project_fau_101_rectangular_framework(analysis.structure)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 10.5),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    theory_axis = flat_axes[0]
    theory_axis.set_facecolor("#101820")
    _draw_rectangular_projected_framework(
        theory_axis, framework, overlay=False
    )
    theory_axis.set_title(
        "FAU.cif [101] rectangular projected framework\n"
        f"physical cell: {framework.width_angstrom:.2f} Å × "
        f"{framework.height_angstrom:.2f} Å"
    )
    theory_axis.legend(loc="upper right", fontsize=7)
    theory_axis.text(
        0.02,
        0.02,
        "orthogonal axes; equal Å scale\n"
        "four primitive projected cells\n"
        "neighbor-cell copies retained at half-period",
        transform=theory_axis.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#263640",
            "edgecolor": "white",
            "alpha": 0.9,
        },
    )

    bins_y = analysis.config.unit_cell_bins
    for axis, image in zip(flat_axes[1:], analysis.images):
        cell = _fold_image_to_rectangular_101_cell(
            image.peaks.best_image,
            image.template_fit,
            analysis.structure,
            bins_y,
        )
        lower, upper = np.percentile(cell, (2.0, 98.0))
        cell = np.clip((cell - lower) / max(float(upper - lower), 1e-8), 0, 1)
        axis.imshow(
            cell,
            origin="lower",
            extent=(
                0,
                framework.width_angstrom,
                0,
                framework.height_angstrom,
            ),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="equal",
        )
        _draw_rectangular_projected_framework(axis, framework, overlay=True)
        fit = image.template_fit
        matched_fraction = len(fit.matched_observed_indices) / max(
            fit.framework_observed_count, 1
        )
        axis.set_title(
            f"{image.label}: physical-metric experiment + FAU.cif\n"
            f"NCC={fit.template_ncc:.3f}; stable-peak coverage={matched_fraction:.1%}; "
            f"median residual={np.median(fit.match_distances_pixels):.2f}px"
        )
    figure.suptitle(
        "Metric-preserving rectangular FAU [101] projection overlay\n"
        r"horizontal [$\bar{1}$01] = $\sqrt{2}a$; vertical [010] = $a$; "
        "unmarked pore intensity is not automatically an atom",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _aligned_site_markers_allowed(
    analysis: FAUStackAnalysis,
    image: FAUImageAnalysis,
) -> bool:
    """Never mark special sites on a declared rare-earth-free template."""

    return not (
        analysis.reference_is_rare_earth_free
        and image.label == analysis.reference_label
    )


def _save_aligned_site_contrast_figure(
    analysis: FAUStackAnalysis,
    path: Path,
    dpi: int,
) -> None:
    """Show associated CIF sites using nearby same-element contrast."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite aligned-site figure: {path}")
    columns = len(analysis.images)
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(6.0 * columns, 10.0),
        constrained_layout=True,
        squeeze=False,
    )
    display_limit = analysis.config.local_same_species_z_threshold + 2.0
    for column, image in enumerate(analysis.images):
        site = image.aligned_site_contrast
        if site is None:
            raise ValueError(f"Aligned-site contrast is missing for {image.label}")
        map_axis = axes[0, column]
        map_axis.imshow(image.peaks.best_image, cmap="gray", vmin=0, vmax=1)
        associated = site.matched & np.isfinite(
            site.local_same_species_contrast_z
        )
        markers_allowed = _aligned_site_markers_allowed(analysis, image)
        scatter = map_axis.scatter(
            site.measurement_xy[associated, 0],
            site.measurement_xy[associated, 1],
            c=np.clip(
                site.local_same_species_contrast_z[associated],
                -3.0,
                display_limit,
            ),
            s=2.0,
            cmap="turbo",
            vmin=-3.0,
            vmax=display_limit,
            linewidths=0,
            alpha=0.75,
        )
        outside = (
            site.contrast_anomaly & ~site.inside_hotspot_region
            if markers_allowed
            else np.zeros(len(site.expected_xy), dtype=bool)
        )
        if np.any(outside):
            map_axis.scatter(
                site.measurement_xy[outside, 0],
                site.measurement_xy[outside, 1],
                s=30,
                marker="x",
                c="#FF9F1C",
                linewidths=0.75,
                label="contrast anomaly outside hotspot",
            )
        boundary_rejected = (
            site.contrast_anomaly
            & site.inside_hotspot_region
            & site.association_reliable
            & ~site.inside_reliable_interior
            if markers_allowed
            else np.zeros(len(site.expected_xy), dtype=bool)
        )
        if np.any(boundary_rejected):
            map_axis.scatter(
                site.measurement_xy[boundary_rejected, 0],
                site.measurement_xy[boundary_rejected, 1],
                s=48,
                marker="^",
                facecolors="none",
                edgecolors="#B56CFF",
                linewidths=1.0,
                label="rejected image-boundary anomaly",
            )
        association_rejected = (
            site.contrast_anomaly
            & site.inside_hotspot_region
            & ~site.association_reliable
            if markers_allowed
            else np.zeros(len(site.expected_xy), dtype=bool)
        )
        if np.any(association_rejected):
            map_axis.scatter(
                site.measurement_xy[association_rejected, 0],
                site.measurement_xy[association_rejected, 1],
                s=52,
                marker="D",
                facecolors="none",
                edgecolors="#FFFFFF",
                linewidths=1.0,
                label="rejected large association residual",
            )
        raw_control = np.zeros(len(site.expected_xy), dtype=bool)
        if np.any(raw_control):
            map_axis.scatter(
                site.measurement_xy[raw_control, 0],
                site.measurement_xy[raw_control, 1],
                s=70,
                marker="s",
                facecolors="none",
                edgecolors="#00E5FF",
                linewidths=1.2,
                label="Y-control raw positive",
            )
        candidates = (
            site.candidate
            if markers_allowed
            else np.zeros(len(site.expected_xy), dtype=bool)
        )
        if np.any(candidates):
            map_axis.scatter(
                site.measurement_xy[candidates, 0],
                site.measurement_xy[candidates, 1],
                s=115,
                facecolors="none",
                edgecolors="#FF2FA3",
                linewidths=1.5,
                label="interpreted RE-site candidate",
            )
        figure.colorbar(
            scatter,
            ax=map_axis,
            fraction=0.046,
            label="nearby same-element contrast z (display clipped)",
        )
        title = (
            f"{image.label}: template-aligned O/Si contrast\n"
            f"associated {int(np.sum(associated))}/{len(site.expected_xy)}; "
            f"anomaly {int(np.sum(site.contrast_anomaly))}; "
            f"candidate {int(np.sum(candidates))}"
        )
        if not markers_allowed:
            title = (
                f"{image.label}: rare-earth-free template / negative control\n"
                f"associated {int(np.sum(associated))}/{len(site.expected_xy)}; "
                "no special sites marked; final RE sites = 0"
            )
        map_axis.set_title(title)
        if (
            np.any(outside)
            or np.any(boundary_rejected)
            or np.any(association_rejected)
            or np.any(raw_control)
            or np.any(candidates)
        ):
            map_axis.legend(loc="upper right", fontsize=6)
        map_axis.set_axis_off()

        histogram_axis = axes[1, column]
        for species, color in (("O", "#00BFD8"), ("Si", "#E99B00")):
            selected = associated & (site.expected_species == species)
            values = site.local_same_species_contrast_z[selected]
            values = values[np.isfinite(values)]
            if len(values):
                histogram_axis.hist(
                    np.clip(values, -6.0, 10.0),
                    bins=np.linspace(-6.0, 10.0, 65),
                    histtype="step",
                    linewidth=1.4,
                    color=color,
                    label=f"nominal {species} (n={len(values)})",
                )
        histogram_axis.axvline(
            analysis.config.local_same_species_z_threshold,
            color="#FF2FA3",
            linestyle="--",
            linewidth=1.4,
            label="anomaly threshold",
        )
        histogram_axis.set(
            xlabel="nearby same-element robust contrast z",
            ylabel="associated site count",
            title=(
                "Per-image, local same-element normalization\n"
                "right-edge values at z>=10 are clipped for display"
            ),
        )
        histogram_axis.set_yscale("symlog", linthresh=1.0)
        histogram_axis.legend(fontsize=7)
    figure.suptitle(
        "All template-aligned O/Si-site local contrasts and legacy anomaly audit\n"
        "Y is an unmarked rare-earth-free template; magenta circles on moving samples "
        "are candidates, not chemical proof",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_aligned_site_candidate_zoom_figure(
    analysis: FAUStackAnalysis,
    path: Path,
    dpi: int,
) -> None:
    """Audit the strongest aligned-site candidates on their own best plane."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite site-candidate zoom: {path}")
    records: list[tuple[float, FAUImageAnalysis, int]] = []
    for image in analysis.images:
        site = image.aligned_site_contrast
        if site is None:
            raise ValueError(f"Aligned-site contrast is missing for {image.label}")
        records.extend(
            (
                float(site.local_same_species_contrast_z[index]),
                image,
                int(index),
            )
            for index in np.flatnonzero(site.candidate)
        )
    shown = sorted(records, key=lambda item: item[0], reverse=True)[:12]
    columns = min(3, max(len(shown), 1))
    rows = max(1, int(math.ceil(len(shown) / columns)))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.4 * columns, 4.3 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    if not shown:
        flat_axes[0].text(
            0.5,
            0.5,
            "No aligned-site candidate passed\ncontrast + hotspot gate",
            ha="center",
            va="center",
            transform=flat_axes[0].transAxes,
        )
        flat_axes[0].set_axis_off()
    stack_cache: dict[Path, np.ndarray] = {}
    for ordinal, (axis, (_, image, index)) in enumerate(
        zip(flat_axes, shown), 1
    ):
        site = image.aligned_site_contrast
        assert site is not None
        if image.source_path not in stack_cache:
            stack_cache[image.source_path] = load_focus_stack(image.source_path)
        plane_index = int(site.best_plane[index])
        plane = np.asarray(stack_cache[image.source_path][plane_index], dtype=float)
        lower, upper = np.percentile(plane, (1.0, 99.0))
        plane = np.clip(
            (plane - lower) / max(float(upper - lower), 1e-8), 0, 1
        )
        x, y = site.measurement_xy[index]
        expected_x, expected_y = site.expected_xy[index]
        radius = 34
        x0, x1 = max(0, int(x) - radius), min(
            plane.shape[1], int(x) + radius + 1
        )
        y0, y1 = max(0, int(y) - radius), min(
            plane.shape[0], int(y) + radius + 1
        )
        axis.imshow(
            plane[y0:y1, x0:x1],
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=(x0, x1, y1, y0),
        )
        axis.scatter(
            [expected_x],
            [expected_y],
            s=95,
            marker="+",
            c="#00E5FF",
            linewidths=1.4,
            label="CIF site",
        )
        axis.scatter(
            [x],
            [y],
            s=155,
            facecolors="none",
            edgecolors="#FF2FA3",
            linewidths=1.7,
            label="associated peak",
        )
        axis.set_title(
            f"{ordinal}. {image.label} nominal {site.expected_species[index]} | "
            f"plane {plane_index}\n"
            f"local same-element z={site.local_same_species_contrast_z[index]:.2f}; "
            f"residual={site.residual_pixels[index]:.2f}px; "
            f"persistence={site.persistence_planes[index]}"
        )
        axis.set(xlabel="x (px)", ylabel="y (px)")
        axis.legend(fontsize=7)
    for axis in flat_axes[len(shown):]:
        axis.set_axis_off()
    figure.suptitle(
        "Template-site rare-earth occupancy/overlap candidates on their brightest plane\n"
        "cyan + = nominal O/Si site; magenta circle = associated peak; chemical validation required",
        fontsize=12,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_expert_guided_candidate_region_figure(
    analysis: FAUStackAnalysis,
    audit: Mapping[str, Any],
    path: Path,
    dpi: int,
) -> None:
    """Show cell-scaled review regions without promoting them to final labels."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite expert-review figure: {path}")
    figure, axes = plt.subplots(
        1,
        len(analysis.images),
        figsize=(6.2 * len(analysis.images), 6.2),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, image in zip(axes, analysis.images):
        axis.imshow(image.peaks.best_image, cmap="gray", vmin=0, vmax=1)
        image_audit = audit["images"][image.label]
        if image.label == analysis.reference_label:
            axis.text(
                0.5,
                0.06,
                "RARE-EARTH-FREE Y CONTROL\n"
                f"{audit['Y_control_tail_count_at_threshold']} statistical-tail "
                "points used only for calibration\ninterpreted review regions = 0",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": "#173B2E",
                    "edgecolor": "white",
                    "alpha": 0.92,
                },
            )
        for region in image_audit["regions"]:
            primary = bool(region["strict_rule_support"])
            color = "#FF2FA3" if primary else "#FFD23F"
            circle = Circle(
                (
                    region["center_x_pixels"],
                    region["center_y_pixels"],
                ),
                radius=region["radius_pixels"],
                fill=False,
                edgecolor=color,
                linewidth=2.0 if primary else 1.7,
                linestyle="-" if primary else "--",
            )
            axis.add_patch(circle)
            axis.annotate(
                f"R{region['region_id']}",
                (
                    region["center_x_pixels"],
                    region["center_y_pixels"],
                ),
                xytext=(7, -7),
                textcoords="offset points",
                color="white",
                fontsize=8,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": color,
                    "edgecolor": "white",
                    "linewidth": 0.5,
                    "alpha": 0.92,
                },
            )
            if region["boundary_adjacent_for_review"]:
                axis.annotate(
                    "EDGE",
                    (
                        region["center_x_pixels"],
                        region["center_y_pixels"],
                    ),
                    xytext=(7, 8),
                    textcoords="offset points",
                    color="#FF7043",
                    fontsize=7,
                    fontweight="bold",
                )
        axis.set_title(
            f"{image.label}: expert-review skeleton regions\n"
            f"primary {image_audit['primary_region_count']}; "
            f"secondary {image_audit['secondary_region_count']}; "
            f"total {image_audit['review_region_count']}; "
            f"edge-warning {image_audit['boundary_adjacent_region_count']}"
        )
        axis.set_axis_off()
    axes[-1].plot(
        [], [], color="#FF2FA3", linewidth=2.0,
        label="primary: existing strict rule support",
    )
    axes[-1].plot(
        [], [], color="#FFD23F", linewidth=1.7, linestyle="--",
        label="secondary: Y-tail-calibrated expert review only",
    )
    axes[-1].legend(loc="lower right", fontsize=7)
    figure.suptitle(
        "Expanded expert-review regions around distorted FAU skeletons\n"
        f"Y-control calibrated same-motif threshold z>="
        f"{audit['calibrated_motif_z_threshold']:.2f}; "
        "large circles show review context, not atomic radius or chemical proof",
        fontsize=13,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_expert_guided_candidate_region_zoom_figure(
    analysis: FAUStackAnalysis,
    audit: Mapping[str, Any],
    path: Path,
    dpi: int,
) -> None:
    """Show every expanded review region on its representative raw plane."""

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite expert-review zoom figure: {path}"
        )
    by_label = {image.label: image for image in analysis.images}
    records = [
        (label, region)
        for label, image_audit in audit["images"].items()
        for region in image_audit["regions"]
    ]
    columns = min(3, max(len(records), 1))
    rows = max(1, int(math.ceil(len(records) / columns)))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 4.4 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    if not records:
        flat_axes[0].text(
            0.5,
            0.5,
            "No expanded expert-review region",
            ha="center",
            va="center",
            transform=flat_axes[0].transAxes,
        )
        flat_axes[0].set_axis_off()
    stack_cache: dict[Path, np.ndarray] = {}
    for axis, (label, region) in zip(flat_axes, records):
        image = by_label[label]
        if image.source_path not in stack_cache:
            stack_cache[image.source_path] = load_focus_stack(image.source_path)
        plane_index = int(region["best_plane_zero_based"])
        plane = np.asarray(
            stack_cache[image.source_path][plane_index], dtype=float
        )
        lower, upper = np.percentile(plane, (1.0, 99.0))
        plane = np.clip(
            (plane - lower) / max(float(upper - lower), 1e-8), 0, 1
        )
        x = float(region["center_x_pixels"])
        y = float(region["center_y_pixels"])
        review_radius = float(region["radius_pixels"])
        crop_radius = max(34, int(math.ceil(2.4 * review_radius)))
        x0, x1 = max(0, int(x) - crop_radius), min(
            plane.shape[1], int(x) + crop_radius + 1
        )
        y0, y1 = max(0, int(y) - crop_radius), min(
            plane.shape[0], int(y) + crop_radius + 1
        )
        axis.imshow(
            plane[y0:y1, x0:x1],
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=(x0, x1, y1, y0),
        )
        fit = image.template_fit
        expected = fit.expected_points_xy[fit.matched_expected_indices]
        observed = image.peaks.points_xy[fit.matched_observed_indices]
        species = fit.expected_species[fit.matched_expected_indices]
        nearby_framework = (
            (expected[:, 0] >= x0)
            & (expected[:, 0] <= x1)
            & (expected[:, 1] >= y0)
            & (expected[:, 1] <= y1)
        )
        for name, marker, framework_color in (
            ("O", ".", "#00DFFF"),
            ("Si", "s", "#FFB000"),
        ):
            chosen = nearby_framework & (species == name)
            axis.scatter(
                expected[chosen, 0],
                expected[chosen, 1],
                s=8 if name == "O" else 11,
                marker=marker,
                facecolors=framework_color,
                edgecolors="black" if name == "Si" else "none",
                linewidths=0.2,
                alpha=0.72,
            )
        arrow_indices = np.flatnonzero(nearby_framework)[::2]
        if len(arrow_indices):
            axis.quiver(
                expected[arrow_indices, 0],
                expected[arrow_indices, 1],
                observed[arrow_indices, 0] - expected[arrow_indices, 0],
                observed[arrow_indices, 1] - expected[arrow_indices, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="#FF4F4F",
                width=0.006,
                alpha=0.75,
            )
        primary = bool(region["strict_rule_support"])
        color = "#FF2FA3" if primary else "#FFD23F"
        axis.add_patch(
            Circle(
                (x, y),
                review_radius,
                fill=False,
                edgecolor=color,
                linewidth=2.1,
                linestyle="-" if primary else "--",
            )
        )
        for member in region["members"]:
            if member["source"] == "template_site_contrast":
                axis.scatter(
                    [member["x_pixels"]],
                    [member["y_pixels"]],
                    marker="+",
                    s=95,
                    c="#00E5FF",
                    linewidths=1.5,
                )
            else:
                axis.scatter(
                    [member["x_pixels"]],
                    [member["y_pixels"]],
                    marker="*",
                    s=105,
                    c="#FF2FA3",
                    edgecolors="white",
                    linewidths=0.6,
                )
        tier = "PRIMARY" if primary else "SECONDARY REVIEW"
        edge_note = " | EDGE-WARNING" if region["boundary_adjacent_for_review"] else ""
        axis.set_title(
            f"{label}-R{region['region_id']} | {tier}{edge_note} | plane {plane_index}\n"
            f"score={region['maximum_review_score']:.2f}; "
            f"radius={region['radius_projected_cells']:.2f} cell; "
            f"members={region['member_count']}",
            fontsize=9,
        )
        axis.set(xlabel="x (px)", ylabel="y (px)")
    for axis in flat_axes[len(records):]:
        axis.set_axis_off()
    figure.suptitle(
        "Expert-review region audit on representative raw stack planes\n"
        "cyan + = anomalous nominal O/Si site; magenta star = strict non-template "
        "bright peak; small cyan/gold = CIF O/Si; red arrows = observed displacement; "
        "outer circle = review context",
        fontsize=12,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_skeleton_first_site_i_figure(
    analysis: FAUStackAnalysis,
    path: Path,
    dpi: int,
) -> None:
    """Show every #4/#10-type site-I bridge and every structural positive."""

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite skeleton-first site-I figure: {path}"
        )
    figure, axes = plt.subplots(
        1,
        len(analysis.images),
        figsize=(6.2 * len(analysis.images), 6.2),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, image in zip(axes, analysis.images):
        site = image.skeleton_first_sites
        if site is None:
            raise ValueError(
                f"Skeleton-first site-I evidence is missing for {image.label}"
            )
        axis.imshow(image.peaks.best_image, cmap="gray", vmin=0, vmax=1)
        if image.label == analysis.reference_label:
            axis.text(
                0.5,
                0.05,
                "RARE-EARTH-FREE Y CONTROL\n"
                f"raw structural tail={int(np.sum(site.raw_structural_screen_positive))}; "
                "interpreted sites=0",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "#173B2E",
                    "edgecolor": "white",
                    "alpha": 0.92,
                },
            )
        else:
            modeled_motif_ids = np.unique(
                site.canonical_site_id[site.model_valid]
            )
            target_motif = np.isin(
                site.canonical_site_id, modeled_motif_ids
            )
            tested = site.model_valid & site.inside_reliable_interior
            unavailable = target_motif & ~tested
            if np.any(unavailable):
                axis.scatter(
                    site.expected_xy[unavailable, 0],
                    site.expected_xy[unavailable, 1],
                    s=18,
                    marker="+",
                    c="#56D9E9",
                    linewidths=0.65,
                    alpha=0.55,
                    label="theoretical #4/#10; model unavailable",
                )
            tested_negative = tested & ~site.candidate
            if np.any(tested_negative):
                axis.scatter(
                    site.expected_xy[tested_negative, 0],
                    site.expected_xy[tested_negative, 1],
                    s=24,
                    marker="+",
                    c="#FFFFFF",
                    linewidths=0.8,
                    alpha=0.85,
                    label="modeled; delta-BIC below 6",
                )
            if np.any(site.candidate):
                axis.scatter(
                    site.expected_xy[site.candidate, 0],
                    site.expected_xy[site.candidate, 1],
                    s=84,
                    facecolors="none",
                    edgecolors="#39FF14",
                    linewidths=1.35,
                    label="structure-supported candidate: delta-BIC >= 6",
                )
        axis.set_title(
            f"{image.label}: skeleton-first FAU site-I screen\n"
            f"modeled #4/#10 repeats={int(np.sum(site.model_valid & site.inside_reliable_interior))}; "
            f"all delta-BIC>=6 retained={int(np.sum(site.candidate))}"
        )
        if image.label != analysis.reference_label:
            axis.legend(loc="upper right", fontsize=7)
        axis.set_axis_off()
    figure.suptitle(
        "FAU #4/#10-type site-I occupancy/projection-overlap audit\n"
        "all valid positive-center fits with delta-BIC >= 6 retained; "
        "no top-20% or local-brightness gate",
        fontsize=13,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_skeleton_first_site_i_zoom_figure(
    analysis: FAUStackAnalysis,
    path: Path,
    dpi: int,
) -> None:
    """Audit every structure-first review point on its selected raw plane."""

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite skeleton-first zoom figure: {path}"
        )
    records: list[tuple[FAUImageAnalysis, int]] = []
    for image in analysis.images:
        if image.label == analysis.reference_label:
            continue
        site = image.skeleton_first_sites
        if site is None:
            continue
        records.extend(
            (image, int(index))
            for index in np.flatnonzero(site.structural_review)
        )
    records.sort(
        key=lambda item: -float(
            item[0].skeleton_first_sites.delta_bic[item[1]]
        )
    )
    # Keep the audit sheet readable. This display-only limit never changes
    # the candidate CSV or the complete overview figure.
    shown = records[:24]
    columns = min(3, max(len(shown), 1))
    rows = max(1, int(math.ceil(len(shown) / columns)))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 4.3 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    if not shown:
        flat_axes[0].text(
            0.5,
            0.5,
            "No moving-sample site-I structural review point passed the Y control",
            ha="center",
            va="center",
            transform=flat_axes[0].transAxes,
        )
        flat_axes[0].set_axis_off()
    stack_cache: dict[Path, np.ndarray] = {}
    for ordinal, (axis, (image, index)) in enumerate(
        zip(flat_axes, shown), 1
    ):
        site = image.skeleton_first_sites
        assert site is not None
        if image.source_path not in stack_cache:
            stack_cache[image.source_path] = load_focus_stack(
                image.source_path
            )
        plane_index = int(site.best_plane[index])
        plane = np.asarray(
            stack_cache[image.source_path][plane_index], dtype=float
        )
        lower, upper = np.percentile(plane, (1.0, 99.0))
        plane = np.clip(
            (plane - lower) / max(float(upper - lower), 1e-8), 0, 1
        )
        center = site.expected_xy[index]
        radius = 28
        x0 = max(0, int(center[0]) - radius)
        x1 = min(plane.shape[1], int(center[0]) + radius + 1)
        y0 = max(0, int(center[1]) - radius)
        y1 = min(plane.shape[0], int(center[1]) + radius + 1)
        axis.imshow(
            plane[y0:y1, x0:x1],
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=(x0, x1, y1, y0),
        )
        endpoints = site.oxygen_endpoint_xy[index]
        axis.plot(
            endpoints[:, 0],
            endpoints[:, 1],
            color="#00E5FF",
            linewidth=1.0,
            alpha=0.85,
        )
        axis.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            s=58,
            facecolors="none",
            edgecolors="#00E5FF",
            linewidths=1.2,
            label="template O endpoints",
        )
        axis.scatter(
            [center[0]],
            [center[1]],
            s=150,
            marker="o",
            facecolors="none",
            edgecolors="#39FF14",
            linewidths=1.6,
            label="site-I probe",
        )
        axis.set_title(
            f"{ordinal}. {image.label} | STRUCTURE-SUPPORTED | plane {plane_index}\n"
            f"delta-BIC={site.delta_bic[index]:.2f}; "
            f"local z={site.local_equivalent_site_z[index]:.2f} (audit only); "
            f"local n={site.local_reference_count[index]}",
            fontsize=8.5,
        )
        axis.set(xlabel="x (px)", ylabel="y (px)")
    for axis in flat_axes[len(shown):]:
        axis.set_axis_off()
    figure.suptitle(
        "Strongest 24 structure-supported FAU #4/#10-type site-I locations\n"
        "display subset only; all candidates remain in figure 18 and CSV; "
        "local brightness is not a gate",
        fontsize=12,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_figures(analysis: FAUStackAnalysis, output: Path, dpi: int) -> dict[str, str]:
    colors = {"O": "#00D7FF", "Si": "#FFB000", "rare_earth": "#FF3CAC"}
    rows = len(analysis.images)
    figure, axes = plt.subplots(rows, 2, figsize=(13, 4.3 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, image in enumerate(analysis.images):
        peak_axis, image_axis = axes[row]
        scores = image.peaks.focus_scores / np.max(image.peaks.focus_scores)
        peak_axis.plot(np.arange(len(scores)), scores, "o-", color="#3957A5")
        peak_axis.axvspan(
            image.peaks.selected_plane_indices[0] - 0.4,
            image.peaks.selected_plane_indices[-1] + 0.4,
            color="#67C587", alpha=0.25, label="selected focus window",
        )
        peak_axis.axvline(image.peaks.best_plane_index, color="#D1495B", label="best")
        peak_axis.set(title=f"{image.label}: raw detail by stack plane", xlabel="zero-based plane", ylabel="relative detail")
        peak_axis.legend(fontsize=8)
        image_axis.imshow(image.peaks.best_image, cmap="gray", vmin=0, vmax=1)
        evidence = image.rare_earth_evidence
        hotspot_mask = _hotspot_region_mask(image, analysis.config)
        if np.any(hotspot_mask):
            image_axis.contour(
                hotspot_mask.astype(float), levels=[0.5], colors=["#00FF8C"],
                linewidths=0.55, alpha=0.75,
            )
        rejected_bright = evidence.extreme_brightness & ~evidence.inside_hotspot_region
        image_axis.scatter(
            image.peaks.points_xy[rejected_bright, 0],
            image.peaks.points_xy[rejected_bright, 1],
            s=4,
            c="#FF9F1C",
            marker="x",
            linewidths=0.35,
            label="extreme but outside hotspot",
        )
        selected = evidence.candidate
        image_axis.scatter(
            image.peaks.points_xy[selected, 0], image.peaks.points_xy[selected, 1],
            s=28, facecolors="none", edgecolors=colors["rare_earth"], linewidths=0.8,
            label="intersection candidate",
        )
        image_axis.set_title(
            f"{image.label}: {int(np.sum(selected))} distortion-gated candidates\n"
            f"single-plane allowed: {int(np.sum(selected & (image.peaks.persistence_count == 1)))}"
        )
        if np.any(selected) or np.any(rejected_bright):
            image_axis.legend(loc="upper right", fontsize=6)
        image_axis.set_axis_off()
    focus_path = output / "01_stack_focus_and_rare_earth_candidates.png"
    figure.savefig(focus_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, rows, figsize=(6 * rows, 6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, image in zip(axes, analysis.images):
        fit = image.template_fit
        observed = image.peaks.points_xy[fit.matched_observed_indices]
        expected = fit.expected_points_xy[fit.matched_expected_indices]
        center = np.asarray([image.peaks.best_image.shape[1] / 2, image.peaks.best_image.shape[0] / 2])
        radius = min(190, min(image.peaks.best_image.shape) / 2 - 5)
        crop = (slice(int(center[1] - radius), int(center[1] + radius)), slice(int(center[0] - radius), int(center[0] + radius)))
        axis.imshow(image.peaks.best_image[crop], cmap="gray", extent=(center[0]-radius, center[0]+radius, center[1]+radius, center[1]-radius), vmin=0, vmax=1)
        inside = (np.abs(expected[:, 0] - center[0]) < radius) & (np.abs(expected[:, 1] - center[1]) < radius)
        names = fit.expected_species[fit.matched_expected_indices]
        for species in ("O", "Si"):
            chosen = inside & (names == species)
            axis.scatter(expected[chosen, 0], expected[chosen, 1], s=10, c=colors[species], marker="+", linewidths=0.6, label=f"CIF {species}")
        chosen_indices = np.flatnonzero(inside)[::3]
        axis.quiver(
            expected[chosen_indices, 0], expected[chosen_indices, 1],
            observed[chosen_indices, 0] - expected[chosen_indices, 0],
            observed[chosen_indices, 1] - expected[chosen_indices, 1],
            angles="xy", scale_units="xy", scale=1, color="#FF4F4F", width=0.004,
        )
        fraction = len(observed) / max(fit.framework_observed_count, 1)
        axis.set_title(f"{image.label}: FAU {fit.zone_axis} framework\nmatched {fraction:.1%}; residual median {np.median(fit.match_distances_pixels):.2f}px")
        axis.set_xlim(center[0]-radius, center[0]+radius)
        axis.set_ylim(center[1]+radius, center[1]-radius)
        axis.legend(fontsize=8)
        axis.set_axis_off()
    template_path = output / "02_cif_framework_matches_and_residuals.png"
    figure.savefig(template_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, rows, figsize=(6 * rows, 5.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, image in zip(axes, analysis.images):
        fit, deformation = image.template_fit, image.deformation
        evidence = image.rare_earth_evidence
        observed = image.peaks.points_xy[fit.matched_observed_indices]
        reliable = deformation.reliable
        axis.imshow(image.peaks.best_image, cmap="gray", vmin=0, vmax=1)
        scatter = axis.scatter(
            observed[reliable, 0], observed[reliable, 1],
            c=100 * deformation.distortion_magnitude[reliable], s=5,
            cmap="magma", vmin=0, vmax=min(12, float(np.nanpercentile(100 * deformation.distortion_magnitude[reliable], 98))) if np.any(reliable) else 1,
        )
        figure.colorbar(scatter, ax=axis, fraction=0.04, label="local strain-tensor norm (%)")
        hotspot_mask = _hotspot_region_mask(image, analysis.config)
        if np.any(hotspot_mask):
            axis.contourf(
                hotspot_mask.astype(float), levels=[0.5, 1.5],
                colors=["#00FF8C"], alpha=0.08,
            )
            axis.contour(
                hotspot_mask.astype(float), levels=[0.5], colors=["#00FF8C"],
                linewidths=0.75, alpha=0.9,
            )
        candidates = image.peaks.points_xy[evidence.candidate]
        axis.scatter(
            candidates[:, 0], candidates[:, 1],
            s=62, marker="*", c="#FF3CAC", edgecolors="white", linewidths=0.45,
            label="extreme-bright intersection",
        )
        for ordinal, point in enumerate(candidates, 1):
            axis.annotate(
                str(ordinal), point, xytext=(5, 5), textcoords="offset points",
                color="white", fontsize=7, weight="bold",
                bbox={"boxstyle": "circle,pad=0.15", "fc": "#B0006D", "ec": "white", "lw": 0.4},
            )
        axis.set_title(f"{image.label}: framework distortion hotspots\nreliable local fits {np.mean(reliable):.1%}")
        if len(candidates):
            axis.legend(loc="upper right", fontsize=6)
        axis.set_axis_off()
    strain_path = output / "03_local_framework_distortion_hotspots.png"
    figure.savefig(strain_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    zone_names = list(analysis.images[0].template_fit.zone_scores)
    width = 0.8 / rows
    positions = np.arange(len(zone_names))
    for index, image in enumerate(analysis.images):
        axes[0].bar(
            positions + (index - (rows - 1) / 2) * width,
            [image.template_fit.zone_scores[name] for name in zone_names],
            width=width, label=image.label,
        )
    axes[0].set_xticks(positions, zone_names, rotation=35)
    axes[0].set(ylabel="periodic unit-cell NCC", title="CIF low-index zone-axis evidence")
    axes[0].legend()
    labels = list(analysis.global_comparisons)
    principal = np.asarray([
        analysis.global_comparisons[label][
            "scale_free_principal_shape_strain_percent"
        ]
        for label in labels
    ])
    x = np.arange(len(labels))
    axes[1].bar(x - 0.18, principal[:, 0], width=0.36, label="principal min")
    axes[1].bar(x + 0.18, principal[:, 1], width=0.36, label="principal max")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xticks(x, [f"{label} vs {analysis.reference_label}" for label in labels])
    axes[1].set(
        ylabel="scale-free principal shape strain (%)",
        title="Y-referenced cell-shape anisotropy\n(isotropic scale not identifiable)",
    )
    axes[1].legend()
    comparison_path = output / "04_zone_axis_and_Y_comparison.png"
    figure.savefig(comparison_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    histograms = [_guest_cell_histogram(image) for image in analysis.images]
    reference_index = next(
        index for index, image in enumerate(analysis.images)
        if image.label == analysis.reference_label
    )
    reference_histogram = histograms[reference_index]
    difference_limit = max(1e-12, max(
        float(np.max(np.abs(histogram - reference_histogram)))
        for histogram in histograms
    ))
    figure, axes = plt.subplots(
        2, rows, figsize=(5.2 * rows, 8.7), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    density_limit = max(
        1e-12, max(float(np.max(histogram)) for histogram in histograms)
    )
    for column, (image, histogram) in enumerate(zip(analysis.images, histograms)):
        density = axes[0, column].imshow(
            histogram,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="viridis",
            vmin=0,
            vmax=density_limit,
        )
        figure.colorbar(density, ax=axes[0, column], fraction=0.046)
        axes[0, column].set_title(
            f"{image.label}: gated-candidate density\nin canonical CIF cell"
        )
        difference = histogram - reference_histogram
        change = axes[1, column].imshow(
            difference,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="coolwarm",
            vmin=-difference_limit,
            vmax=difference_limit,
        )
        figure.colorbar(change, ax=axes[1, column], fraction=0.046)
        axes[1, column].set_title(f"{image.label} minus {analysis.reference_label}")
        for axis in axes[:, column]:
            axis.set(xlabel="canonical u", ylabel="canonical v")
    guest_path = output / "05_rare_earth_distribution_in_cif_cell.png"
    figure.savefig(guest_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(
        1, rows, figsize=(5.6 * rows, 4.8), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, image in zip(axes, analysis.images):
        deformation = image.deformation
        reliable = deformation.reliable
        cell_scale = math.sqrt(
            abs(float(np.linalg.det(image.template_fit.direct_basis_pixels)))
        )
        distance = deformation.nearest_guest_distance_pixels[reliable] / cell_scale
        distortion = 100.0 * deformation.distortion_magnitude[reliable]
        finite = np.isfinite(distance) & np.isfinite(distortion)
        distance = distance[finite]
        distortion = distortion[finite]
        if not len(distance):
            axis.text(
                0.5, 0.5, "No intersection-qualified candidate",
                ha="center", va="center", transform=axis.transAxes,
            )
            axis.set(
                title=f"{image.label}: no candidate-distance estimate",
                xlabel="nearest candidate distance / projected-cell scale",
                ylabel="local strain-tensor norm (%)",
            )
            continue
        axis.hexbin(
            distance,
            distortion,
            gridsize=38,
            mincnt=1,
            cmap="Blues",
            bins="log",
        )
        edges = np.linspace(0, max(0.45, float(np.percentile(distance, 99))), 13)
        centers = 0.5 * (edges[:-1] + edges[1:])
        medians = np.full(len(centers), np.nan)
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            selected = (distance >= lower) & (distance < upper)
            if int(np.sum(selected)) >= 12:
                medians[index] = float(np.median(distortion[selected]))
        axis.plot(centers, medians, "o-", color="#D1495B", linewidth=2, label="bin median")
        axis.axvline(
            analysis.config.near_guest_radius_cells,
            color="#A020F0",
            linestyle="--",
            linewidth=1,
            label="near threshold",
        )
        axis.axvline(
            analysis.config.far_guest_radius_cells,
            color="#333333",
            linestyle=":",
            linewidth=1,
            label="far threshold",
        )
        axis.set(
            title=f"{image.label}: distortion vs gated RE candidate",
            xlabel="nearest candidate distance / projected-cell scale",
            ylabel="local strain-tensor norm (%)",
        )
        axis.legend(fontsize=8)
    relation_path = output / "06_distortion_vs_rare_earth_distance.png"
    figure.savefig(relation_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    candidate_records: list[tuple[FAUImageAnalysis, int, int]] = []
    for image in analysis.images:
        for ordinal, index in enumerate(
            np.flatnonzero(image.rare_earth_evidence.candidate), 1
        ):
            candidate_records.append((image, ordinal, int(index)))
    shown = sorted(
        candidate_records,
        key=lambda item: item[0].rare_earth_evidence.evidence_score[item[2]],
        reverse=True,
    )[:12]
    columns = min(3, max(len(shown), 1))
    rows_zoom = max(1, int(math.ceil(len(shown) / columns)))
    figure, axes = plt.subplots(
        rows_zoom, columns,
        figsize=(4.4 * columns, 4.4 * rows_zoom),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    if not shown:
        flat_axes[0].text(
            0.5, 0.5, "No intersection-qualified candidate",
            ha="center", va="center", transform=flat_axes[0].transAxes,
        )
        flat_axes[0].set_axis_off()
    stack_cache: dict[Path, np.ndarray] = {}
    for axis, (image, ordinal, index) in zip(flat_axes, shown):
        if image.source_path not in stack_cache:
            stack_cache[image.source_path] = load_focus_stack(image.source_path)
        stack = stack_cache[image.source_path]
        plane_index = int(image.peaks.best_plane_by_peak[index])
        plane = np.asarray(stack[plane_index], dtype=float)
        lower, upper = np.percentile(plane, (1.0, 99.0))
        plane = np.clip(
            (plane - lower) / max(float(upper - lower), 1e-8), 0, 1
        )
        x, y = image.peaks.points_xy[index]
        radius = 34
        x0, x1 = max(0, int(x) - radius), min(plane.shape[1], int(x) + radius + 1)
        y0, y1 = max(0, int(y) - radius), min(plane.shape[0], int(y) + radius + 1)
        axis.imshow(
            plane[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1,
            extent=(x0, x1, y1, y0),
        )
        hotspot_mask = _hotspot_region_mask(image, analysis.config)
        local_mask = hotspot_mask[y0:y1, x0:x1]
        if np.any(local_mask) and np.any(~local_mask):
            axis.contour(
                np.arange(x0, x1), np.arange(y0, y1),
                local_mask.astype(float), levels=[0.5], colors=["#00FF8C"],
                linewidths=1.1,
            )
        axis.scatter(
            [x], [y], s=150, marker="*", c="#FF3CAC",
            edgecolors="white", linewidths=0.8,
        )
        evidence = image.rare_earth_evidence
        axis.set_title(
            f"{image.label}-{ordinal} | plane {plane_index} | "
            f"local z={evidence.local_brightness_z[index]:.2f}\n"
            f"persistence={int(image.peaks.persistence_count[index])} | "
            f"hotspot distance={evidence.nearest_hotspot_distance_pixels[index]:.1f}px",
            fontsize=9,
        )
        axis.set(xlabel="x (px)", ylabel="y (px)")
    for axis in flat_axes[len(shown):]:
        axis.set_axis_off()
    figure.suptitle(
        "Distortion-gated extreme-bright candidates on their own brightest planes\n"
        "green = hotspot boundary; magenta star = candidate",
        fontsize=12,
    )
    zoom_path = output / "07_candidate_zoom_audit.png"
    figure.savefig(zoom_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    canonical_cells = [
        _fold_image_to_canonical_cif_cell(
            image.peaks.best_image, image.template_fit, analysis.config.unit_cell_bins
        )
        for image in analysis.images
    ]
    normalized_cells = []
    for cell in canonical_cells:
        lower, upper = np.percentile(cell, (2.0, 98.0))
        normalized_cells.append(
            np.clip((cell - lower) / max(float(upper - lower), 1e-8), 0, 1)
        )
    reference_index = next(
        index for index, image in enumerate(analysis.images)
        if image.label == analysis.reference_label
    )
    reference_cell = normalized_cells[reference_index]
    difference_limit = max(
        1e-6,
        max(float(np.max(np.abs(cell - reference_cell))) for cell in normalized_cells),
    )
    figure, axes = plt.subplots(
        2, len(analysis.images), figsize=(4.5 * len(analysis.images), 8.0),
        constrained_layout=True, squeeze=False,
    )
    for column, (image, cell) in enumerate(zip(analysis.images, normalized_cells)):
        axes[0, column].imshow(cell, origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(
            f"{image.label}: canonical [101] CIF cell\n"
            f"source FOV {image.peaks.best_image.shape[1]}x{image.peaks.best_image.shape[0]} px"
        )
        difference = cell - reference_cell
        axes[1, column].imshow(
            difference, origin="lower", cmap="coolwarm",
            vmin=-difference_limit, vmax=difference_limit,
        )
        axes[1, column].set_title(
            f"{image.label} minus {analysis.reference_label} (normalized)"
        )
        for axis in axes[:, column]:
            axis.set(xlabel="canonical u bin", ylabel="canonical v bin")
    figure.suptitle(
        "Different fields of view aligned by periodic CIF coordinates\n"
        "per-image contrast normalized; not same-atom/same-pixel registration",
        fontsize=13,
    )
    canonical_path = output / "08_Y_reference_canonical_cell_alignment.png"
    figure.savefig(canonical_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    assignment_path = output / "09_final_rare_earth_site_assignments.png"
    _save_final_rare_earth_assignment_figure(analysis, assignment_path, dpi)
    agreement_path = output / "10_FAU_CIF_projection_registration_agreement.png"
    _save_cif_projection_agreement_figure(analysis, agreement_path, dpi)
    framework_overlay_path = output / "11_FAU_CIF_projected_framework_overlay.png"
    _save_projected_framework_overlay_figure(analysis, framework_overlay_path, dpi)
    physical_overlay_path = (
        output / "12_FAU_CIF_101_physical_rectangular_overlay.png"
    )
    _save_physical_rectangular_framework_overlay_figure(
        analysis, physical_overlay_path, dpi
    )
    aligned_site_path = output / "13_aligned_O_Si_site_contrast_anomalies.png"
    _save_aligned_site_contrast_figure(analysis, aligned_site_path, dpi)
    aligned_site_zoom_path = output / "14_aligned_site_candidate_zoom_audit.png"
    _save_aligned_site_candidate_zoom_figure(
        analysis, aligned_site_zoom_path, dpi
    )
    return {
        "focus_and_rare_earth_figure": str(focus_path.resolve()),
        "cif_framework_figure": str(template_path.resolve()),
        "local_distortion_figure": str(strain_path.resolve()),
        "Y_comparison_figure": str(comparison_path.resolve()),
        "rare_earth_distribution_figure": str(guest_path.resolve()),
        "distortion_vs_rare_earth_distance_figure": str(
            relation_path.resolve()
        ),
        "candidate_zoom_audit_figure": str(zoom_path.resolve()),
        "Y_reference_canonical_alignment_figure": str(canonical_path.resolve()),
        "final_rare_earth_site_assignment_figure": str(assignment_path.resolve()),
        "FAU_CIF_projection_registration_agreement_figure": str(
            agreement_path.resolve()
        ),
        "FAU_CIF_projected_framework_overlay_figure": str(
            framework_overlay_path.resolve()
        ),
        "FAU_CIF_101_physical_rectangular_overlay_figure": str(
            physical_overlay_path.resolve()
        ),
        "aligned_O_Si_site_contrast_anomaly_figure": str(
            aligned_site_path.resolve()
        ),
        "aligned_site_candidate_zoom_audit_figure": str(
            aligned_site_zoom_path.resolve()
        ),
    }


def _write_report_legacy_v4(summary: dict[str, Any], path: Path) -> None:
    images = summary["images"]
    projection_statement = (
        "三组图像均按实验者给定的投影方向拟合，且该方向相对非等价低指数候选获得最高周期单胞相关；"
        if summary["zone_axis_source"] == "user_supplied"
        else "三组图像的周期单胞均由低指数投影搜索自动匹配；"
    )
    lines = [
        "# FAU 4D-STEM z-stack：畸变区约束的极亮稀土候选分析",
        "",
        "## 一句话结论",
        "",
        projection_statement + "流程同时保留非模板局部极亮峰与模板 O/Si 位点异常两类候选；二者均要求可靠畸变热点支持，必须分别报告，一致时才形成更强的图像证据，且仍不是化学确证。",
        "",
        "## 输入与物理边界",
        "",
        f"- 参考：`{summary['reference']}`。",
        f"- Y 无稀土负对照：`{summary['reference_is_rare_earth_free']}`；各图先独立归一化，Y 的绝对亮度尾部只作假阳性审计，不跨图硬否决。",
        f"- 投影方向来源：`{summary['zone_axis_source']}`。",
        f"- CIF：空间群 {summary['cif']['space_group_number']}，展开后 {summary['cif']['expanded_atom_counts']}；CIF 中没有稀土。",
        "- 所有 TIFF 层都参与极亮峰搜索；选焦窗口只负责提供稳定的 Si/O 骨架。单层峰允许成为候选。",
        "- TIFF 第一轴没有物理 z 步长，因此仍不能把层编号换算为三维原子坐标。",
        "- 三个尺寸和视野不同，只按周期 CIF 坐标对齐；不能跟踪同一个原子，也不能逐像素相减。",
        "- 未提供各图独立 Å/px 标定，因此全局等向缩放和面积变化与拍摄倍率混淆，只报告去除等向尺度后的晶格形状畸变。",
        "- 没有弹性张量，因此不能输出 Pa/GPa；图中的 maximum shear 只是应变热点代理。",
        "",
        "## 分图结果",
        "",
    ]
    for label, item in images.items():
        aligned = item["aligned_O_Si_site_contrast"]
        if aligned is None:
            aligned_note = "- 尚未生成逐 CIF 位点衬度审计。"
        elif label == summary["reference"] and summary[
            "reference_is_rare_earth_free"
        ]:
            aligned_note = (
                f"- Y 是纯 FAU 模板和无稀土负对照：覆盖 "
                f"{aligned['theoretical_site_count']} 个理论 O/Si 位点，"
                f"其中 {aligned['peak_associated_site_count']} 个关联到检测峰；"
                "衬度尾部只用于校准，不在图上标记任何特殊点，最终稀土点位固定为 0。"
            )
        else:
            species_counts = aligned[
                "candidate_counts_by_nominal_template_species"
            ]
            aligned_note = (
                f"- 模板覆盖 {aligned['theoretical_site_count']} 个理论 O/Si 位点，"
                f"其中 {aligned['peak_associated_site_count']} 个关联到检测峰；"
                f"同位点类型衬度异常 {aligned['contrast_anomaly_count_before_hotspot_gate']} 个，"
                f"其中热点支持但配准残差过大而拒绝 "
                f"{aligned['association_rejected_hotspot_anomaly_count']} 个，"
                f"其中热点支持但因图像边界不可靠而拒绝 "
                f"{aligned['boundary_rejected_hotspot_anomaly_count']} 个；"
                f"再经畸变热点与内区约束后解释为 {aligned['interpreted_rare_earth_candidate_count']} 个"
                f"稀土占位/投影重叠候选（名义 O 位 {species_counts['O']}，"
                f"名义 Si 位 {species_counts['Si']}）。"
            )
        screen_note = (
            f"- 原始交集筛选阳性 {item['raw_intersection_screen_positive_count']} 个；Y 是无稀土负对照，因此这些只能登记为假阳性。"
            if label == summary["reference"] and summary["reference_is_rare_earth_free"]
            else f"- 原始交集筛选阳性 {item['raw_intersection_screen_positive_count']} 个；逐图归一化后按本图局部阈值解释。"
        )
        lines.extend([
            f"### {label}",
            "",
            f"- 最佳焦面（从 0 开始）：{item['best_focus_plane_zero_based']}；选中窗口 {item['selected_focus_planes_zero_based']}。",
            f"- 全层共检出 {item['detected_peak_count_all_planes']} 个峰，其中 {item['framework_eligible_persistent_peak_count']} 个可用于稳定骨架拟合。",
            f"- 热点区内峰 {item['hotspot_region_peak_count']} 个，极端亮峰 {item['extreme_brightness_peak_count']} 个；两者交集得到稀土候选 {item['rare_earth_candidate_count']} 个，其中单层候选 {item['single_plane_rare_earth_candidate_count']} 个。",
            screen_note,
            f"- 热点外另有 {item['outside_hotspot_extreme_brightness_count']} 个极亮峰，按规则拒绝，不计为稀土。",
            f"- CIF 投影 `{item['zone_axis']}`，NCC={item['template_ncc']:.3f}，相对下一候选优势={item['zone_score_margin']:.3f}。",
            f"- Si/O 骨架匹配 {item['matched_framework_count']} 个，占稳定骨架候选的 {item['matched_fraction_of_framework_eligible_peaks']:.1%}。",
            aligned_note,
            f"- CIF 反推采样尺度：{item['cif_calibrated_angstrom_per_pixel']:.4f} Å/px，仅用于配准诊断，不是独立仪器像素标定。",
            f"- 科学解释等级：`{item['scientific_interpretation_grade']}`。",
            "",
        ])
    if summary["negative_control_audit"] is not None:
        audit = summary["negative_control_audit"]
        lines.extend([
            "## Y 无稀土负对照审计",
            "",
            f"- Y 热点内用于审计的峰 {audit['reference_hotspot_peak_count']} 个，局部 z 最大值 {audit['reference_hotspot_local_z_max']:.3f}。",
            f"- Y 原始交集筛选阳性 {audit['reference_raw_intersection_screen_positive_count']} 个；解释后的稀土候选固定为 0。",
            "- Y 的经验尾概率写入 RA/H2 候选表，但遵照逐图归一化原则，只作旁证而不是跨图绝对阈值。",
            "",
        ])
    if summary["aligned_site_negative_control_audit"] is not None:
        audit = summary["aligned_site_negative_control_audit"]
        lines.extend([
            "## 全 O/Si 模板位点的 Y 负对照审计",
            "",
            f"- Y 中有 {audit['control_site_count']} 个峰关联且落在热点内的模板位点可形成经验亮度尾部；同位点归一化 z 最大值为 {audit['control_max_motif_normalized_contrast_z']:.3f}。",
            f"- Y 原始模板位点筛选阳性 {audit['reference_raw_screen_positive_count']} 个，但因已知无稀土，解释后的候选固定为 0。",
            "- 每个理论位点先与本图同一 CIF motif 的重复单胞比较；Y 尾概率只用于假阳性审计，不把不同视野的绝对灰度强行相减。",
            "",
        ])
    lines.extend(["## 两类稀土候选证据的一致性", ""])
    total_overlap = 0
    for label, item in images.items():
        aligned = item["aligned_O_Si_site_contrast"]
        if aligned is None:
            continue
        overlap = aligned["candidate_overlap_with_peak_gate_count"]
        total_overlap += overlap
        lines.append(
            f"- {label}：局部 Si 参照的非模板峰候选 {item['rare_earth_candidate_count']} 个；"
            f"同 CIF motif 参照的模板位点候选 "
            f"{aligned['interpreted_rare_earth_candidate_count']} 个；严格重合 {overlap} 个。"
        )
    lines.extend([
        f"- 三图两种检测规则的严格重合总数为 {total_overlap}。两类不一致点都保留给专家/EDS/EELS 复核，不将任一规则单独当作元素身份。",
        "",
    ])
    expert_review = summary.get("expert_guided_skeleton_region_review")
    if expert_review is not None:
        lines.extend([
            "## 专家式骨架差异区域复核层",
            "",
            f"- 使用无稀土 Y 的 {expert_review['Y_control_quantile']:.3%} 尾部标定，"
            f"同 motif 复核门槛为 z≥{expert_review['calibrated_motif_z_threshold']:.2f}；"
            f"Y 有 {expert_review['Y_control_tail_count_at_threshold']} 个尾部点只计作假阳性校准，不在图上圈出。",
        ])
        for label, item in expert_review["images"].items():
            if label == expert_review["reference"]:
                continue
            lines.append(
                f"- {label}：合并后 {item['review_region_count']} 个大范围复核区，"
                f"其中原严格规则已支持 {item['primary_region_count']} 个，"
                f"仅供专家二次审核 {item['secondary_region_count']} 个；"
                f"其中 {item['boundary_adjacent_region_count']} 个带视野边缘风险警告。"
            )
        lines.extend([
            "- 这一层把异常点按投影单胞尺度合并成区域，用于对照周围骨架。大圈是审核上下文，不是稀土原子的物理半径。",
            f"- 距图像边缘少于 {expert_review['boundary_warning_distance_projected_cells']:.2f} 个投影单胞的区域保留供查看，但标记 EDGE-WARNING；边缘裁切、DoG 边界与不完整骨架都可能制造假异常。",
            "- 次级区域不加入最终稀土数量或迁移统计，只有专家点选或 EDS/EELS/模拟确认后才能升级。",
            "",
        ])
    lines.extend(["## RA/H2 相对 Y 的全局投影晶格形状", ""])
    for label, item in summary["moving_vs_Y_global_lattice"].items():
        principal = item["scale_free_principal_shape_strain_percent"]
        lines.append(
            f"- {label}：去除刚体旋转和未知等向倍率后，主形状应变 {principal[0]:.3f}% / {principal[1]:.3f}%，主方向差 {item['scale_free_principal_strain_difference_percent']:.3f}%；绝对等向膨胀和面积变化不可识别。"
        )
    lines.extend(["", "## RA/H2 相对 Y 的局部畸变分布", ""])
    for label, item in summary["moving_vs_Y_local_deformation_distributions"].items():
        distortion = item["distortion_percent"]
        shear = item["maximum_shear_strain_proxy_percent"]
        lines.append(
            f"- {label}：局部畸变中位数相对 Y 变化 {distortion['median_difference_vs_reference']:+.3f} 个百分点，Wasserstein 距离 {distortion['wasserstein_distance']:.3f}；超过 Y 的 P95 的比例 {distortion['moving_fraction_above_reference_p95']:.1%}。最大剪应变代理中位数变化 {shear['median_difference_vs_reference']:+.3f} 个百分点。"
        )
    lines.extend(["", "## 稀土出现与可识别迁移", ""])
    for label, item in summary["moving_vs_Y_guest_onset"].items():
        lines.append(
            f"- {label}：相对无稀土 Y 检出 {item['moving_candidate_count']} 个候选（{item['moving_candidates_per_projected_cell']:.5f}/投影单胞）；这表示候选出现/占位，不是同一原子从 Y 迁移。"
        )
    comparisons = summary["guest_distribution_between_nonreference_fields"]
    if not comparisons:
        lines.append("- 非参考样品不足两个，无法比较稀土候选总体位置迁移。")
    for label, item in comparisons.items():
        if item["status"] == "estimated":
            lines.append(
                f"- {label}：周期单胞分布平移 uv={item['periodic_distribution_shift_uv']}，Jensen–Shannon 散度={item['jensen_shannon_divergence_bits']:.4f} bit；这是总体分布，不是单原子轨迹。"
            )
        else:
            lines.append(
                f"- {label}：两侧候选不足（{item['reference_candidate_count']} / {item['moving_candidate_count']}），主动放弃 RA–H2 总体迁移估计。"
            )
    lines.extend(["", "## 稀土附近的骨架畸变", ""])
    for label, item in images.items():
        near = item["near_guest_distortion_percent"]
        far = item["far_from_guest_distortion_percent"]
        if near is not None and far is not None:
            lines.append(
                f"- {label}：近稀土候选中位畸变 {near['p50']:.3f}%，远处 {far['p50']:.3f}%。"
            )
    lines.append(
        "- 热点是候选筛选的先验条件，因此这里的近邻关系不能再用于声称‘稀土导致畸变’的独立因果验证。"
    )
    lines.extend([
        "",
        "## 如何读图",
        "",
        "1. `01`：所有层都搜索亮点；洋红圆圈是热点与极亮度交集，橙色叉是因位于热点外而拒绝的极亮峰。",
        "2. `02`：青/黄十字是 CIF 的 O/Si 期望位点，红箭头是已匹配骨架的残差。",
        "3. `03`：底色越亮表示局部对称应变张量范数越大；绿圈是热点种子，洋红星是最终交集候选。",
        "4. `04`：左图审计投影选择，右图只给出去除未知等向倍率后的 Y 参考晶格形状畸变。",
        "5. `05`：仅在候选数足够时，统一到 CIF 单胞坐标比较候选总体位置分布。",
        "6. `06`：直接检查骨架畸变随最近稀土候选距离如何变化；红线是分箱中位数。",
        "7. `07`：逐个放大候选在其自身最亮切片中的原图；绿线是热点边界，洋红星是候选中心。",
        "8. `08`：将不同大小和视野折叠到统一的周期 CIF 单胞；这是晶格坐标对齐，不是同一位置配准。",
        "9. `09`：只显示最终稀土点位指认；Y 是无稀土参考，严格为 0，不混入热点轮廓或被拒绝亮点。",
        "10. `10`：在与 `08` 相同的规范单胞坐标上叠加 FAU.cif `[101]` 理论 O/Si 投影，并同步给出 NCC、骨架匹配率和残差分布。",
        "11. `11`：先单独显示带 Si-O 键的 FAU.cif `[101]` 投影结构，再将完全相同的周期骨架叠加到 Y、RA、H2 实验规范单胞上。",
        "12. `12`：用正交的 `[\\bar101] × [010]` 矩形投影超胞和真实 Å 比例显示骨架及实验叠加；图中的长度和角度不再被分数坐标正方形拉伸，并保留相邻晶胞的半周期原子副本。",
        "13. `13`：在三幅原图上显示所有已关联的理论 O/Si 位点衬度；颜色是同一 CIF motif 在本图重复单胞间的稳健 z 分数，橙叉是热点外异常、白色空菱形是配准残差过大、紫色空三角是图像边界不可靠，洋红圈是全部门槛共同支持的最终候选。",
        "14. `14`：逐个在自身最亮层放大模板位点候选；青色加号是理论 O/Si 位置，洋红圈是关联峰，标题记录名义元素、层号、z 分数和配准残差。",
        "15. `15`：以大圈显示骨架畸变区内的扩展专家复核区；实线洋红是原严格规则已支持，虚线黄色只是 Y 尾部标定的次级复核提案。",
        "16. `16`：在各自代表层放大每个专家复核区；青色加号是异常名义 O/Si 位，洋红星是非模板极亮峰，外圈是需要连同骨架一起判读的范围。",
        "",
        "## 仍需补充",
        "",
        "当前候选仍不是化学确证。模板位点异常可能来自稀土替位、沿束流方向与 O/Si 柱重叠、厚度/通道效应或重建伪影。若要升级为可发表的原子级定量，需要：真实扫描像素标定、少量专家/EDS/EELS 点标、厚度或多重散射模拟、物理 z 步长；若要计算真实应力，还需要对应材料的取向相关弹性常数。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(summary: dict[str, Any], path: Path) -> None:
    """Write the fixed-threshold structure-first site-I report."""

    images = summary["images"]
    audit = summary["skeleton_first_negative_control_audit"]
    lines = [
        "# FAU 4D-STEM：骨架优先的 site-I 稀土候选分析",
        "",
        "## 一句话结论",
        "",
        "最终候选只使用 FAU.cif `[101]` 骨架规定的 #4/#10 类空 site-I/D6R 中心"
        "结构比较：凡有效两 O 模型中加入正中心成分后达到固定 ΔBIC 门槛的位置全部保留。"
        "不再使用图内最高 20% 截断，也不使用局部亮度 z 门槛；Y 作为已知无稀土负对照"
        "固定为 0。",
        "",
        "## 当前单一候选定义",
        "",
        "- `structure_supported_candidate`：有效、位于可靠内部、中心成分为正且 ΔBIC 达到固定门槛。",
        "- 不再划分 B1/B2，也没有按图内分位数限制候选数量。",
        "- 局部亮度 z 和 GPA 只作后验审计，不参与候选生成或否决。",
        "",
        "## 候选规则",
        "",
        f"1. CIF 骨架先确定理论空 site-I（{audit['site_definition']}）。",
        "2. 沿两侧理论 O 投影柱的连接方向，比较“两 O 骨架模型”和“两 O + 中心成分模型”。",
        f"3. 额外中心模型至少改善 ΔBIC={audit['minimum_structural_delta_bic_threshold']:.1f} 才进入结构阳性池。",
        "4. RA/H2 的全部结构阳性均保留，不再执行最高 20% 截断。",
        "5. 局部等价 site-I 中心强度 z 只记录为连续审计量，不设入选阈值。",
        "6. Y 的 ΔBIC 尾部只记录成像假阳性背景；因 Y 已知无稀土，解释候选固定为 0。",
        "",
        "## 分图结果",
        "",
    ]
    for label, item in images.items():
        evidence = item["skeleton_first_site_I_evidence"]
        lines.extend([
            f"### {label}",
            "",
            f"- 原始堆栈：`{item['stack_shape_qyx']}`；骨架代表焦面（从 0 开始）为 {item['best_focus_plane_zero_based']}。",
            f"- 理论 site-I 探针 {evidence['template_empty_probe_count']} 个，其中 {evidence['valid_two_O_bridge_model_count']} 个具有有效的两 O 对向骨架模型。",
            f"- 固定 ΔBIC 结构阳性 {evidence['raw_structural_screen_positive_count']} 个；最终保留的结构支持候选 {evidence['structure_supported_candidate_count']} 个。",
            f"- 模板 NCC={item['template_ncc']:.3f}；稳定骨架匹配率 {item['matched_fraction_of_framework_eligible_peaks']:.1%}；匹配距离 {item['framework_match_distance_pixels']}。",
        ])
        structural_fraction = (
            evidence["raw_structural_screen_positive_count"]
            / max(evidence["valid_two_O_bridge_model_count"], 1)
        )
        if structural_fraction >= 0.80:
            lines.append(
                f"- 警告：结构存在门通过率为 {structural_fraction:.1%}，"
                "在本图接近饱和；它约束搜索位置，但不能单独提供高特异性。"
            )
        if label == summary["reference"] and summary["reference_is_rare_earth_free"]:
            lines.append(
                "- Y 已知不含稀土：结构模型的尾部仅作负对照审计，图中不圈特殊点，最终候选强制为 0。"
            )
        else:
            lines.append(
                "- 这些位置都是 site-I 占位或沿束流投影重叠假设，不等同于稀土元素化学确证。"
            )
        lines.append("")

    lines.extend([
        "## Y 负对照审计",
        "",
        f"- Y 有效 #4/#10 类 site-I 模型 {audit['images']['Y']['modeled_template_empty_site_I_count']} 个；固定 ΔBIC 下原始结构阳性 {audit['images']['Y']['raw_structural_tail_count']} 个。",
        f"- Y 的 {audit['Y_control_quantile']:.3%} ΔBIC 尾部阈值为 {audit['Y_control_delta_bic_tail_audit_threshold']:.3f}，只作审计，因为三幅图的焦点、厚度和重建质量不同。",
        "- Y 解释后的结构复核点和稀土候选均为 0；这一约束阻止模板自身被错误标成稀土。",
        "",
        "## 相对 Y 的晶格形状与局部畸变",
        "",
    ])
    for label, item in summary["moving_vs_Y_global_lattice"].items():
        principal = item["scale_free_principal_shape_strain_percent"]
        lines.append(
            f"- {label}：移除刚体旋转和未知等向倍率后，主形状应变为 "
            f"{principal[0]:+.3f}% / {principal[1]:+.3f}%，主方向差 "
            f"{item['scale_free_principal_strain_difference_percent']:.3f}%。"
        )
    for label, item in summary["moving_vs_Y_local_deformation_distributions"].items():
        distortion = item["distortion_percent"]
        shear = item["maximum_shear_strain_proxy_percent"]
        lines.append(
            f"- {label}：局部畸变中位数相对 Y 变化 "
            f"{distortion['median_difference_vs_reference']:+.3f} 个百分点；"
            f"超过 Y P95 的比例 {distortion['moving_fraction_above_reference_p95']:.1%}；"
            f"最大剪应变代理中位数变化 {shear['median_difference_vs_reference']:+.3f} 个百分点。"
        )
    lines.extend([
        "- 上述为本流程的晶格统计；候选与独立 GPA 像素热点的同坐标交叉验证由后续 GPA bundle 单独保存，避免循环论证。",
        "- 最大剪应变仅是无量纲应力热点代理；没有弹性常数、厚度模型和独立像素标定时不能换算为 Pa/GPa。",
        "",
        "## 主要输出怎么读",
        "",
        "- `12_FAU_CIF_101_physical_rectangular_overlay.png`：保持 Å 比例的 `[101]` 骨架配准审计。",
        "- `17_FAU_CIF_101_site_I_search_template.png`：纯 CIF 骨架与 16c/site-I 重复搜索位置；标记是模板空位探针，不是检测原子。",
        "- `18_skeleton_first_site_I_candidates.png`：显示全部目标理论位置、模型不可用位置和所有 ΔBIC≥6 的结构支持候选；Y 不解释为稀土。",
        "- `19_skeleton_first_site_I_zoom_audit.png`：按 ΔBIC 排序展示最强 24 个位置；它只是可视化子集，不改变完整候选表。",
        "- `rare_earth_candidates.csv`、`rare_earth_priority_review.csv` 与 `skeleton_first_site_I_review.csv`：均保存全部结构支持候选。",
        "- `legacy_peak_gate_candidates.csv` 和 `13/14` 图只保留为旧方法审计，不拥有最终候选解释权。",
        "",
        "## 物理边界",
        "",
        "- RA、H2、Y 不是同一视野，不能声称同一颗原子的逐点迁移，也不能直接逐像素相减。",
        "- TIFF 第一轴没有物理 z 步长，层号只是重建/焦面顺序，不是三维原子坐标。",
        "- 候选仍可能来自客体占位、沿束流柱重叠、厚度/通道效应或重建伪影；最终元素身份需要 EDS/EELS、厚度标定或多重散射模拟。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_fau_stack_analysis(
    analysis: FAUStackAnalysis,
    output_directory: str | Path,
    *,
    dpi: int = 170,
) -> dict[str, str]:
    """Write a new auditable result bundle and refuse any overwrite."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite FAU stack analysis: {output}")
    output.mkdir(parents=True, exist_ok=True)
    expert_region_audit = build_expert_guided_candidate_regions(analysis)
    summary = _serializable_summary(analysis)
    summary["expert_guided_skeleton_region_review"] = expert_region_audit
    summary_path = output / "summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    rare_earth_path = output / "rare_earth_candidates.csv"
    priority_review_path = output / "rare_earth_priority_review.csv"
    legacy_peak_candidate_path = output / "legacy_peak_gate_candidates.csv"
    screening_path = output / "rare_earth_screening.csv"
    framework_path = output / "matched_framework_sites.csv"
    aligned_site_path = output / "aligned_O_Si_site_contrast.csv"
    aligned_candidate_path = output / "aligned_site_rare_earth_candidates.csv"
    expert_region_path = output / "expert_guided_candidate_regions.csv"
    site_i_all_path = output / "skeleton_first_site_I_all_probes.csv"
    site_i_review_path = output / "skeleton_first_site_I_review.csv"
    _write_skeleton_first_site_i_csv(
        analysis, rare_earth_path, candidates_only=True
    )
    _write_skeleton_first_site_i_csv(
        analysis, priority_review_path, review_only=True
    )
    _write_rare_earth_csv(analysis, legacy_peak_candidate_path)
    _write_rare_earth_screening_csv(analysis, screening_path)
    _write_framework_csv(analysis, framework_path)
    _write_aligned_site_contrast_csv(analysis, aligned_site_path)
    _write_aligned_site_contrast_csv(
        analysis, aligned_candidate_path, candidates_only=True
    )
    _write_expert_guided_candidate_regions_csv(
        expert_region_audit, expert_region_path
    )
    _write_skeleton_first_site_i_csv(analysis, site_i_all_path)
    _write_skeleton_first_site_i_csv(
        analysis, site_i_review_path, review_only=True
    )
    arrays_path = output / "analysis_arrays.npz"
    arrays: dict[str, np.ndarray] = {}
    for image in analysis.images:
        prefix = image.label
        arrays[f"{prefix}_points_xy"] = image.peaks.points_xy
        arrays[f"{prefix}_species_brightness_class"] = image.peaks.species_brightness_class.astype(str)
        arrays[f"{prefix}_persistence_count"] = image.peaks.persistence_count
        arrays[f"{prefix}_framework_eligible"] = image.peaks.framework_eligible
        arrays[f"{prefix}_standardized_brightness"] = image.peaks.standardized_brightness
        arrays[f"{prefix}_rare_earth_screen_positive"] = image.rare_earth_evidence.screen_positive
        arrays[f"{prefix}_rare_earth_candidate"] = image.rare_earth_evidence.candidate
        arrays[f"{prefix}_inside_hotspot_region"] = image.rare_earth_evidence.inside_hotspot_region
        arrays[f"{prefix}_extreme_brightness"] = image.rare_earth_evidence.extreme_brightness
        arrays[f"{prefix}_local_brightness_z"] = image.rare_earth_evidence.local_brightness_z
        arrays[f"{prefix}_global_brightness_z"] = image.rare_earth_evidence.global_brightness_z
        arrays[f"{prefix}_negative_control_tail_probability"] = image.rare_earth_evidence.negative_control_tail_probability
        arrays[f"{prefix}_nearest_hotspot_distance_pixels"] = image.rare_earth_evidence.nearest_hotspot_distance_pixels
        arrays[f"{prefix}_matched_observed_indices"] = image.template_fit.matched_observed_indices
        arrays[f"{prefix}_matched_expected_indices"] = image.template_fit.matched_expected_indices
        arrays[f"{prefix}_expected_points_xy"] = image.template_fit.expected_points_xy
        arrays[f"{prefix}_local_distortion"] = image.deformation.distortion_magnitude
        arrays[f"{prefix}_local_fit_reliable"] = image.deformation.reliable
        site = image.aligned_site_contrast
        if site is None:
            raise ValueError(f"Aligned-site contrast is missing for {image.label}")
        arrays[f"{prefix}_aligned_expected_xy"] = site.expected_xy
        arrays[f"{prefix}_aligned_measurement_xy"] = site.measurement_xy
        arrays[f"{prefix}_aligned_canonical_uv"] = site.canonical_uv
        arrays[f"{prefix}_aligned_canonical_site_id"] = site.canonical_site_id
        arrays[f"{prefix}_aligned_expected_species"] = (
            site.expected_species.astype(str)
        )
        arrays[f"{prefix}_aligned_peak_associated"] = site.matched
        arrays[f"{prefix}_aligned_observed_peak_index"] = (
            site.observed_peak_index
        )
        arrays[f"{prefix}_aligned_residual_pixels"] = site.residual_pixels
        arrays[f"{prefix}_aligned_association_reliable"] = (
            site.association_reliable
        )
        arrays[f"{prefix}_aligned_best_plane"] = site.best_plane
        arrays[f"{prefix}_aligned_persistence_planes"] = site.persistence_planes
        arrays[f"{prefix}_aligned_standardized_contrast"] = (
            site.standardized_contrast
        )
        arrays[f"{prefix}_aligned_raw_dog_contrast"] = site.raw_dog_contrast
        arrays[f"{prefix}_aligned_raw_detector_intensity"] = (
            site.raw_detector_intensity
        )
        arrays[f"{prefix}_aligned_brightness_class"] = (
            site.brightness_class.astype(str)
        )
        arrays[f"{prefix}_aligned_motif_contrast_z"] = (
            site.motif_normalized_contrast_z
        )
        arrays[f"{prefix}_aligned_local_same_species_contrast_z"] = (
            site.local_same_species_contrast_z
        )
        arrays[f"{prefix}_aligned_local_same_species_reference_count"] = (
            site.local_same_species_reference_count
        )
        arrays[f"{prefix}_aligned_inside_hotspot"] = (
            site.inside_hotspot_region
        )
        arrays[f"{prefix}_aligned_distance_to_image_edge_pixels"] = (
            site.distance_to_image_edge_pixels
        )
        arrays[f"{prefix}_aligned_inside_reliable_interior"] = (
            site.inside_reliable_interior
        )
        arrays[f"{prefix}_aligned_contrast_anomaly"] = site.contrast_anomaly
        arrays[f"{prefix}_aligned_screen_positive"] = site.screen_positive
        arrays[f"{prefix}_aligned_candidate"] = site.candidate
        arrays[f"{prefix}_aligned_peak_gate_overlap"] = (
            site.overlaps_peak_gate_screen_positive
        )
        arrays[f"{prefix}_aligned_negative_control_tail_probability"] = (
            site.negative_control_tail_probability
        )
        skeleton_site = image.skeleton_first_sites
        if skeleton_site is None:
            raise ValueError(
                f"Skeleton-first site-I evidence is missing for {image.label}"
            )
        arrays[f"{prefix}_site_I_expected_xy"] = skeleton_site.expected_xy
        arrays[f"{prefix}_site_I_canonical_site_id"] = (
            skeleton_site.canonical_site_id
        )
        arrays[f"{prefix}_site_I_oxygen_endpoint_xy"] = (
            skeleton_site.oxygen_endpoint_xy
        )
        arrays[f"{prefix}_site_I_model_valid"] = skeleton_site.model_valid
        arrays[f"{prefix}_site_I_best_plane"] = skeleton_site.best_plane
        arrays[f"{prefix}_site_I_delta_bic"] = skeleton_site.delta_bic
        arrays[f"{prefix}_site_I_extra_component_amplitude"] = (
            skeleton_site.extra_component_amplitude
        )
        arrays[f"{prefix}_site_I_local_equivalent_z"] = (
            skeleton_site.local_equivalent_site_z
        )
        arrays[f"{prefix}_site_I_local_reference_count"] = (
            skeleton_site.local_reference_count
        )
        arrays[f"{prefix}_site_I_structural_review"] = (
            skeleton_site.structural_review
        )
        arrays[f"{prefix}_site_I_local_brightness_anomaly"] = (
            skeleton_site.local_brightness_anomaly
        )
        arrays[f"{prefix}_site_I_candidate"] = skeleton_site.candidate
    np.savez_compressed(arrays_path, **arrays)
    figures = _save_figures(analysis, output, dpi)
    expert_region_figure_path = (
        output / "15_expert_guided_skeleton_candidate_regions.png"
    )
    _save_expert_guided_candidate_region_figure(
        analysis, expert_region_audit, expert_region_figure_path, dpi
    )
    expert_region_zoom_path = (
        output / "16_expert_guided_candidate_region_zoom_audit.png"
    )
    _save_expert_guided_candidate_region_zoom_figure(
        analysis, expert_region_audit, expert_region_zoom_path, dpi
    )
    site_i_template_path = (
        output / "17_FAU_CIF_101_site_I_search_template.png"
    )
    _save_fau_site_i_search_template_figure(
        analysis.structure, site_i_template_path, dpi
    )
    skeleton_first_figure_path = (
        output / "18_skeleton_first_site_I_candidates.png"
    )
    _save_skeleton_first_site_i_figure(
        analysis, skeleton_first_figure_path, dpi
    )
    skeleton_first_zoom_path = (
        output / "19_skeleton_first_site_I_zoom_audit.png"
    )
    _save_skeleton_first_site_i_zoom_figure(
        analysis, skeleton_first_zoom_path, dpi
    )
    report_path = output / "RESULTS_zh.md"
    _write_report(summary, report_path)
    return {
        "summary": str(summary_path.resolve()),
        "rare_earth_candidates": str(rare_earth_path.resolve()),
        "rare_earth_priority_review": str(priority_review_path.resolve()),
        "legacy_peak_gate_candidates": str(
            legacy_peak_candidate_path.resolve()
        ),
        "rare_earth_screening": str(screening_path.resolve()),
        "matched_framework_sites": str(framework_path.resolve()),
        "aligned_O_Si_site_contrast": str(aligned_site_path.resolve()),
        "aligned_site_rare_earth_candidates": str(
            aligned_candidate_path.resolve()
        ),
        "expert_guided_candidate_regions": str(expert_region_path.resolve()),
        "skeleton_first_site_I_all_probes": str(site_i_all_path.resolve()),
        "skeleton_first_site_I_review": str(site_i_review_path.resolve()),
        "arrays": str(arrays_path.resolve()),
        "report": str(report_path.resolve()),
        "expert_guided_skeleton_candidate_region_figure": str(
            expert_region_figure_path.resolve()
        ),
        "expert_guided_candidate_region_zoom_audit_figure": str(
            expert_region_zoom_path.resolve()
        ),
        "FAU_CIF_101_site_I_search_template_figure": str(
            site_i_template_path.resolve()
        ),
        "skeleton_first_site_I_candidate_figure": str(
            skeleton_first_figure_path.resolve()
        ),
        "skeleton_first_site_I_zoom_audit_figure": str(
            skeleton_first_zoom_path.resolve()
        ),
        **figures,
    }
