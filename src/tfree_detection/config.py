"""Configuration objects and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml


@dataclass
class PreprocessConfig:
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    background_sigma: float = 0.0


@dataclass
class SpectralConfig:
    patch_size: int = 128
    stride: int = 24
    min_frequency: float = 0.035
    max_frequency: float = 0.32
    radial_bin_width: float = 0.008
    peak_fraction: float = 0.004
    autocorrelation_min_lag: int = 8
    autocorrelation_max_lag: int = 24
    global_peak_count: int = 12
    global_peak_min_separation: float = 0.012
    global_peak_radius: float = 0.012


@dataclass
class FusionConfig:
    top_k: int = 3
    smooth_sigma_grid: float = 1.0
    threshold_quantile: float = 0.94
    threshold_mad: float = 2.5
    support_z: float = 1.25
    minimum_evidence_groups: int = 2


@dataclass
class PostprocessConfig:
    border_margin: Optional[int] = None
    exclude_border_touching: bool = True
    morphology_radius: int = 6
    minimum_region_area: int = 2500
    minimum_hole_area: int = 1500


@dataclass
class OutputConfig:
    save_feature_maps: bool = True
    save_numpy_arrays: bool = True
    figure_dpi: int = 160


@dataclass
class EvidenceProtocolConfig:
    """Shared reliability and abstention policy for all physical tasks."""

    enabled: bool = True
    reliable_threshold: float = 0.60
    invalid_threshold: float = 0.20
    minimum_confirmed_evidence_groups: int = 1


@dataclass
class ResolverConfig:
    """Transparent open-world hypothesis resolution policy."""

    enabled: bool = True
    minimum_hypothesis_score: float = 0.15
    minimum_score_margin: float = 0.05
    minimum_region_support_fraction: float = 0.20
    maximum_ranked_hypotheses: int = 5


@dataclass
class PhaseAnalysisConfig:
    enabled: bool = True
    frequency_filter_sigma: float = 0.009
    derivative_smoothing_sigma: float = 3.0
    amplitude_background_sigma: float = 120.0
    fft_padding: int = 128
    minimum_basis_angle_degrees: float = 15.0
    minimum_coherence: float = 0.65
    transition_inner_radius: int = 12
    transition_outer_radius: int = 90
    reference_inner_radius: int = 110
    reference_outer_radius: int = 190


@dataclass
class DINOConfig:
    enabled: bool = False
    model_id: str = "facebook/dino-vits8"
    local_model_directory: Optional[str] = None
    input_size: int = 224
    microscopy_patch_size: int = 128
    batch_size: int = 8
    samples_per_zone: int = 16
    device: str = "auto"
    random_seed: int = 13


@dataclass
class SAM3RefinementConfig:
    """Physics-prompted, proposal-only SAM 2/3 boundary refinement."""

    enabled: bool = False
    backend: str = "sam3"
    supported_tasks: list[str] = field(
        default_factory=lambda: ["amorphous_region", "phase_boundary"]
    )
    inference_mode: str = "isolated_process"
    exchange_directory: Optional[str] = None
    worker_python: Optional[str] = None
    worker_script: str = "scripts/run_sam3_boundary_worker.py"
    checkpoint_path: Optional[str] = None
    model_variant: str = "sam3"
    device: str = "auto"
    timeout_seconds: int = 600
    ensemble_size: int = 8
    positive_points_per_candidate: int = 3
    negative_points_per_candidate: int = 8
    box_padding_pixels: int = 24
    box_padding_lattice_cells: float = 2.0
    use_coarse_mask_prompt: bool = True
    core_score_quantile: float = 0.75
    stable_core_threshold: float = 0.80
    selection_threshold: float = 0.50
    possible_extent_threshold: float = 0.20
    minimum_seed_core_recall: float = 0.95
    maximum_negative_leakage: float = 0.05
    minimum_area_ratio: float = 0.50
    maximum_area_ratio: float = 2.00
    minimum_physics_contrast: float = 0.0
    minimum_prompt_stability: float = 0.50
    minimum_boundary_evidence_alignment: float = 0.05
    allow_component_merging: bool = False
    preserve_candidate_topology: bool = True
    interface_geometry_mode: str = "paired_domains"
    interface_max_gap_pixels: int = 8
    interface_half_width_pixels: int = 5
    boundary_tolerance_pixels: float = 2.0
    fallback_to_coarse: bool = True
    random_seed: int = 31


@dataclass
class Stage2Config:
    enabled: bool = False
    phase: PhaseAnalysisConfig = field(default_factory=PhaseAnalysisConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    sam3: SAM3RefinementConfig = field(default_factory=SAM3RefinementConfig)


@dataclass
class AtomicPointDefectConfig:
    """Parameters for lattice-site residual analysis."""

    polarity: str = "auto"
    peak_sigma: float = 1.2
    peak_min_distance: int = 4
    peak_threshold_quantile: float = 0.72
    matching_tolerance_fraction: float = 0.34
    site_anomaly_z: float = 3.5
    displacement_z: float = 3.0
    minimum_reference_sites: int = 16
    render_radius_fraction: float = 0.28
    offset_search_steps: int = 24
    max_auto_sublattices: int = 4
    fractional_histogram_bins: int = 32
    nonrigid_smoothing_cells: float = 4.0
    nonrigid_iterations: int = 2
    minimum_displacement_fraction: float = 0.12


@dataclass
class LineDefectConfig:
    """Parameters for phase-slip, stacking-fault, and dislocation lines."""

    frequency_filter_sigma: float = 0.012
    derivative_smoothing_sigma: float = 2.0
    score_smoothing_sigma: float = 1.5
    threshold_quantile: float = 0.965
    threshold_mad: float = 3.5
    minimum_coherence: float = 0.30
    morphology_radius: int = 2
    minimum_region_area: int = 40
    valid_border_margin: int = 16


@dataclass
class PhaseBoundaryConfig:
    """Parameters for local reciprocal-signature domain separation."""

    patch_size: int = 64
    stride: int = 12
    radial_bins: int = 5
    orientation_bins: int = 12
    minimum_phase_fraction: float = 0.08
    minimum_signature_separation: float = 0.8
    boundary_width: int = 5
    minimum_region_area: int = 80
    minimum_boundary_length: float = 64.0
    clustering_iterations: int = 40


@dataclass
class SymmetryRelationConfig:
    """Reference-anchored, group-aware Class1 analysis parameters."""

    motif_bins: int = 8
    reference_patch_cells: float = 4.5
    analysis_stride_cells: float = 1.15
    maximum_references: int = 5
    minimum_references: int = 3
    reference_quality_quantile: float = 0.25
    identity_shift_tolerance_bins: int = 1
    minimum_motif_effect: float = 0.08
    maximum_aligned_residual: float = 0.42
    domain_score_threshold: float = 3.5
    orientation_score_threshold: float = 3.5
    phase_slip_score_threshold: float = 4.0
    burgers_score_threshold: float = 5.0
    minimum_coherence: float = 0.25
    minimum_domain_cells: int = 3
    minimum_boundary_phase_support: float = 0.25
    boundary_support_radius_cells: float = 2.0
    geometry_width_cells: float = 0.30
    artifact_competition_weight: float = 0.75
    edge_reliability_floor: float = 0.25


@dataclass
class StructureGuidedConfig:
    """Optional projected-site model-selection parameters."""

    sample_role: str = "target_possible"
    minimum_delta_bic: float = 6.0
    profile_sigma_pixels: float = 1.25
    minimum_endpoint_separation_pixels: float = 4.0
    border_pixels: float = 4.0
    candidate_radius_pixels: float = 2.5


@dataclass
class DetectorConfig:
    task: str = "amorphous_region"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evidence_protocol: EvidenceProtocolConfig = field(
        default_factory=EvidenceProtocolConfig
    )
    resolver: ResolverConfig = field(default_factory=ResolverConfig)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    atomic: AtomicPointDefectConfig = field(
        default_factory=AtomicPointDefectConfig
    )
    line_defect: LineDefectConfig = field(default_factory=LineDefectConfig)
    phase_boundary: PhaseBoundaryConfig = field(
        default_factory=PhaseBoundaryConfig
    )
    symmetry_relation: SymmetryRelationConfig = field(
        default_factory=SymmetryRelationConfig
    )
    structure_guided: StructureGuidedConfig = field(
        default_factory=StructureGuidedConfig
    )

    def validate(self) -> None:
        supported_tasks = {
            "amorphous_region",
            "atomic_point_defect",
            "stacking_fault",
            "phase_boundary",
            "symmetry_relation",
            "structure_guided_site",
        }
        if self.task not in supported_tasks:
            raise ValueError(
                f"Unsupported task {self.task!r}; choose one of "
                + ", ".join(sorted(supported_tasks))
            )
        if self.spectral.patch_size < 16 or self.spectral.patch_size % 2:
            raise ValueError("spectral.patch_size must be an even integer >= 16")
        if self.spectral.stride < 1:
            raise ValueError("spectral.stride must be positive")
        if not 0 < self.spectral.min_frequency < self.spectral.max_frequency < 0.5:
            raise ValueError("frequency bounds must satisfy 0 < min < max < 0.5")
        if not 0.5 < self.fusion.threshold_quantile < 1:
            raise ValueError("fusion.threshold_quantile must be between 0.5 and 1")
        if self.fusion.minimum_evidence_groups < 1:
            raise ValueError("minimum_evidence_groups must be positive")
        if self.preprocess.lower_percentile >= self.preprocess.upper_percentile:
            raise ValueError("lower_percentile must be below upper_percentile")
        evidence = self.evidence_protocol
        if not (
            0
            <= evidence.invalid_threshold
            < evidence.reliable_threshold
            <= 1
        ):
            raise ValueError(
                "evidence_protocol thresholds must satisfy "
                "0 <= invalid < reliable <= 1"
            )
        if evidence.minimum_confirmed_evidence_groups < 1:
            raise ValueError(
                "minimum_confirmed_evidence_groups must be positive"
            )
        resolver = self.resolver
        if resolver.minimum_hypothesis_score < 0:
            raise ValueError(
                "resolver.minimum_hypothesis_score cannot be negative"
            )
        if resolver.minimum_score_margin < 0:
            raise ValueError(
                "resolver.minimum_score_margin cannot be negative"
            )
        if not 0 <= resolver.minimum_region_support_fraction <= 1:
            raise ValueError(
                "resolver.minimum_region_support_fraction must be in [0, 1]"
            )
        if resolver.maximum_ranked_hypotheses < 1:
            raise ValueError(
                "resolver.maximum_ranked_hypotheses must be positive"
            )
        if self.stage2.enabled:
            phase = self.stage2.phase
            if not 0 < phase.frequency_filter_sigma < 0.1:
                raise ValueError(
                    "stage2.phase.frequency_filter_sigma must be between 0 and 0.1"
                )
            if not 0 < phase.minimum_basis_angle_degrees < 90:
                raise ValueError(
                    "stage2.phase.minimum_basis_angle_degrees must be between 0 and 90"
                )
            if not (
                0
                <= phase.transition_inner_radius
                < phase.transition_outer_radius
                < phase.reference_inner_radius
                < phase.reference_outer_radius
            ):
                raise ValueError("stage2 phase zone radii must increase strictly")
            if self.stage2.dino.enabled:
                dino = self.stage2.dino
                if dino.input_size < 32 or dino.microscopy_patch_size < 16:
                    raise ValueError("DINO input and microscopy patches are too small")
                if dino.batch_size < 1 or dino.samples_per_zone < 1:
                    raise ValueError("DINO batch size and samples must be positive")
        sam3 = self.stage2.sam3
        if sam3.enabled:
            if sam3.backend not in {"sam2", "sam3"}:
                raise ValueError(
                    "stage2.sam3.backend must be 'sam2' or 'sam3'"
                )
            if sam3.inference_mode not in {
                "isolated_process",
                "prepare_only",
                "response_only",
            }:
                raise ValueError(
                    "stage2.sam3.inference_mode must be isolated_process, "
                    "prepare_only, or response_only"
                )
            if not sam3.supported_tasks:
                raise ValueError(
                    "stage2.sam3.supported_tasks cannot be empty"
                )
            unknown_sam3_tasks = set(sam3.supported_tasks) - supported_tasks
            if unknown_sam3_tasks:
                raise ValueError(
                    "stage2.sam3.supported_tasks contains unknown task(s): "
                    + ", ".join(sorted(unknown_sam3_tasks))
                )
            if sam3.timeout_seconds < 1:
                raise ValueError(
                    "stage2.sam3.timeout_seconds must be positive"
                )
            if sam3.ensemble_size < 2:
                raise ValueError(
                    "stage2.sam3.ensemble_size must be at least 2"
                )
            if (
                sam3.positive_points_per_candidate < 1
                or sam3.negative_points_per_candidate < 1
            ):
                raise ValueError(
                    "stage2.sam3 positive and negative prompt counts must "
                    "be positive"
                )
            if sam3.box_padding_pixels < 0:
                raise ValueError(
                    "stage2.sam3.box_padding_pixels cannot be negative"
                )
            if sam3.box_padding_lattice_cells < 0:
                raise ValueError(
                    "stage2.sam3.box_padding_lattice_cells cannot be negative"
                )
            thresholds = (
                sam3.possible_extent_threshold,
                sam3.selection_threshold,
                sam3.stable_core_threshold,
            )
            if not 0 <= thresholds[0] < thresholds[1] < thresholds[2] <= 1:
                raise ValueError(
                    "stage2.sam3 consensus thresholds must satisfy "
                    "0 <= possible < selection < stable <= 1"
                )
            if not 0.5 <= sam3.core_score_quantile < 1:
                raise ValueError(
                    "stage2.sam3.core_score_quantile must be in [0.5, 1)"
                )
            if not 0.95 <= sam3.minimum_seed_core_recall <= 1:
                raise ValueError(
                    "Policy A requires stage2.sam3.minimum_seed_core_recall "
                    "to remain in [0.95, 1]"
                )
            if not 0 <= sam3.maximum_negative_leakage < 1:
                raise ValueError(
                    "stage2.sam3.maximum_negative_leakage must be in [0, 1)"
                )
            if not (
                0 < sam3.minimum_area_ratio
                <= sam3.maximum_area_ratio
            ):
                raise ValueError(
                    "stage2.sam3 area ratios must satisfy 0 < minimum <= maximum"
                )
            if not 0 <= sam3.minimum_prompt_stability <= 1:
                raise ValueError(
                    "stage2.sam3.minimum_prompt_stability must be in [0, 1]"
                )
            if not 0 <= sam3.minimum_boundary_evidence_alignment <= 1:
                raise ValueError(
                    "stage2.sam3.minimum_boundary_evidence_alignment must "
                    "be in [0, 1]"
                )
            if sam3.interface_geometry_mode != "paired_domains":
                raise ValueError(
                    "stage2.sam3.interface_geometry_mode must be "
                    "'paired_domains'; SAM must not segment an interface "
                    "as one filled region"
                )
            if sam3.interface_max_gap_pixels < 1:
                raise ValueError(
                    "stage2.sam3.interface_max_gap_pixels must be positive"
                )
            if sam3.interface_half_width_pixels < 0:
                raise ValueError(
                    "stage2.sam3.interface_half_width_pixels cannot be negative"
                )
            if sam3.boundary_tolerance_pixels < 0:
                raise ValueError(
                    "stage2.sam3.boundary_tolerance_pixels cannot be negative"
                )
            if not sam3.fallback_to_coarse:
                raise ValueError(
                    "Policy A requires stage2.sam3.fallback_to_coarse=true"
                )
        if self.atomic.polarity not in {"auto", "bright", "dark"}:
            raise ValueError("atomic.polarity must be auto, bright, or dark")
        if self.atomic.peak_sigma <= 0 or self.atomic.peak_min_distance < 1:
            raise ValueError("atomic peak scale and distance must be positive")
        if not 0.5 < self.atomic.peak_threshold_quantile < 1:
            raise ValueError(
                "atomic.peak_threshold_quantile must be between 0.5 and 1"
            )
        if not 0 < self.atomic.matching_tolerance_fraction < 0.75:
            raise ValueError(
                "atomic.matching_tolerance_fraction must be between 0 and 0.75"
            )
        if self.atomic.minimum_reference_sites < 6:
            raise ValueError("atomic.minimum_reference_sites must be at least 6")
        if not 1 <= self.atomic.max_auto_sublattices <= 12:
            raise ValueError(
                "atomic.max_auto_sublattices must be between 1 and 12"
            )
        if self.atomic.fractional_histogram_bins < 12:
            raise ValueError(
                "atomic.fractional_histogram_bins must be at least 12"
            )
        if self.atomic.nonrigid_smoothing_cells <= 0:
            raise ValueError(
                "atomic.nonrigid_smoothing_cells must be positive"
            )
        if self.atomic.nonrigid_iterations < 0:
            raise ValueError("atomic.nonrigid_iterations cannot be negative")
        if not 0 <= self.atomic.minimum_displacement_fraction < 0.75:
            raise ValueError(
                "atomic.minimum_displacement_fraction must be in [0, 0.75)"
            )
        if not 0 < self.line_defect.frequency_filter_sigma < 0.1:
            raise ValueError(
                "line_defect.frequency_filter_sigma must be between 0 and 0.1"
            )
        if not 0.5 < self.line_defect.threshold_quantile < 1:
            raise ValueError(
                "line_defect.threshold_quantile must be between 0.5 and 1"
            )
        if self.phase_boundary.patch_size < 16:
            raise ValueError("phase_boundary.patch_size must be at least 16")
        if self.phase_boundary.stride < 1:
            raise ValueError("phase_boundary.stride must be positive")
        if self.phase_boundary.radial_bins < 2:
            raise ValueError("phase_boundary.radial_bins must be at least 2")
        if self.phase_boundary.orientation_bins < 4:
            raise ValueError(
                "phase_boundary.orientation_bins must be at least 4"
            )
        if self.phase_boundary.minimum_boundary_length < 0:
            raise ValueError(
                "phase_boundary.minimum_boundary_length cannot be negative"
            )
        if not 0 < self.phase_boundary.minimum_phase_fraction < 0.5:
            raise ValueError(
                "phase_boundary.minimum_phase_fraction must be between 0 and 0.5"
            )
        symmetry = self.symmetry_relation
        if symmetry.motif_bins < 4 or symmetry.motif_bins % 2:
            raise ValueError(
                "symmetry_relation.motif_bins must be an even integer >= 4"
            )
        if symmetry.reference_patch_cells < 2:
            raise ValueError(
                "symmetry_relation.reference_patch_cells must be at least 2"
            )
        if symmetry.analysis_stride_cells <= 0:
            raise ValueError(
                "symmetry_relation.analysis_stride_cells must be positive"
            )
        if not (
            3
            <= symmetry.minimum_references
            <= symmetry.maximum_references
        ):
            raise ValueError(
                "symmetry_relation reference counts must satisfy "
                "3 <= minimum <= maximum"
            )
        if not 0 <= symmetry.reference_quality_quantile < 1:
            raise ValueError(
                "symmetry_relation.reference_quality_quantile must be in [0, 1)"
            )
        if not 0 < symmetry.maximum_aligned_residual <= 2:
            raise ValueError(
                "symmetry_relation.maximum_aligned_residual must be in (0, 2]"
            )
        if symmetry.minimum_domain_cells < 1:
            raise ValueError(
                "symmetry_relation.minimum_domain_cells must be positive"
            )
        if not 0 <= symmetry.minimum_boundary_phase_support <= 1:
            raise ValueError(
                "symmetry_relation.minimum_boundary_phase_support must be in [0, 1]"
            )
        if symmetry.boundary_support_radius_cells <= 0:
            raise ValueError(
                "symmetry_relation.boundary_support_radius_cells must be positive"
            )
        if not 0 <= symmetry.artifact_competition_weight <= 1:
            raise ValueError(
                "symmetry_relation.artifact_competition_weight must be in [0, 1]"
            )
        structure = self.structure_guided
        if structure.sample_role not in {
            "target_possible",
            "target_free_reference",
        }:
            raise ValueError(
                "structure_guided.sample_role must be target_possible or "
                "target_free_reference"
            )
        if structure.minimum_delta_bic < 0:
            raise ValueError(
                "structure_guided.minimum_delta_bic cannot be negative"
            )
        if structure.profile_sigma_pixels <= 0:
            raise ValueError(
                "structure_guided.profile_sigma_pixels must be positive"
            )
        if structure.minimum_endpoint_separation_pixels <= 0:
            raise ValueError(
                "structure_guided.minimum_endpoint_separation_pixels must "
                "be positive"
            )
        if structure.border_pixels < 0:
            raise ValueError(
                "structure_guided.border_pixels cannot be negative"
            )
        if structure.candidate_radius_pixels <= 0:
            raise ValueError(
                "structure_guided.candidate_radius_pixels must be positive"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _update_dataclass(instance: T, values: Dict[str, Any]) -> T:
    known = {item.name: item for item in fields(instance)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(
            f"Unknown configuration key(s) for {type(instance).__name__}: "
            + ", ".join(sorted(unknown))
        )
    for key, value in values.items():
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{key} must be a mapping")
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: Optional[str | Path] = None) -> DetectorConfig:
    """Load a detector configuration, merging it over typed defaults."""

    config = DetectorConfig()
    if path is not None:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        if not isinstance(values, dict):
            raise TypeError("Configuration root must be a mapping")
        _update_dataclass(config, values)
    config.validate()
    return config
