"""Counterfactual same-nuisance synthetic twins for structural defects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
from scipy.ndimage import gaussian_filter


SUPPORTED_TWIN_TASKS = (
    "amorphous_region",
    "atomic_point_defect",
    "stacking_fault",
    "phase_boundary",
)


@dataclass(frozen=True)
class NuisanceParameters:
    angle_degrees: float = 11.0
    gaussian_noise_standard_deviation: float = 0.06
    blur_sigma: float = 0.7
    illumination_x: float = 0.10
    illumination_y: float = -0.07
    dose_electrons_per_normalized_unit: float = 0.0

    def validate(self) -> None:
        if self.gaussian_noise_standard_deviation < 0:
            raise ValueError("Gaussian noise cannot be negative")
        if self.blur_sigma < 0:
            raise ValueError("Blur sigma cannot be negative")
        if self.dose_electrons_per_normalized_unit < 0:
            raise ValueError("Dose cannot be negative")


@dataclass(frozen=True)
class CounterfactualTwinPair:
    pair_id: str
    task: str
    normal_image: np.ndarray
    defect_image: np.ndarray
    defect_mask: np.ndarray
    structural_delta: np.ndarray
    nuisance: NuisanceParameters
    seed: int
    metadata: Dict[str, Any]

    def validate(self) -> None:
        shape = self.normal_image.shape
        if self.task not in SUPPORTED_TWIN_TASKS:
            raise ValueError(f"Unsupported twin task {self.task!r}")
        if self.normal_image.ndim != 2:
            raise ValueError("Twin images must be 2-D")
        for name, values in (
            ("defect_image", self.defect_image),
            ("defect_mask", self.defect_mask),
            ("structural_delta", self.structural_delta),
        ):
            if values.shape != shape:
                raise ValueError(f"Twin {name} shape differs from normal image")
        if not np.any(self.defect_mask):
            raise ValueError("Counterfactual twin must contain changed structure")
        if not np.all(np.isfinite(self.normal_image)) or not np.all(
            np.isfinite(self.defect_image)
        ):
            raise ValueError("Counterfactual images contain non-finite values")
        if self.normal_image.dtype != np.uint16 or self.defect_image.dtype != np.uint16:
            raise ValueError("Counterfactual images must preserve uint16 range")
        self.nuisance.validate()

    def manifest(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "task": self.task,
            "shape": list(self.normal_image.shape),
            "seed": self.seed,
            "nuisance": asdict(self.nuisance),
            "changed_pixel_fraction": float(np.mean(self.defect_mask)),
            "source_kind": "digital_twin",
            "counterfactual_rule": (
                "normal and defect share nuisance parameters and random "
                "realization; only the declared structural intervention changes"
            ),
            "metadata": self.metadata,
        }


def _coordinates(
    size: int, angle_degrees: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:size, :size]
    angle = np.deg2rad(angle_degrees)
    xr = np.cos(angle) * xx - np.sin(angle) * yy
    yr = np.sin(angle) * xx + np.cos(angle) * yy
    return yy, xx, xr, yr


def _base_crystal(xr: np.ndarray, yr: np.ndarray) -> np.ndarray:
    return (
        np.cos(2 * np.pi * xr / 15.0)
        + 0.88 * np.cos(2 * np.pi * yr / 13.5 + 0.4)
        + 0.28 * np.cos(2 * np.pi * (xr / 15.0 + yr / 13.5))
    )


def _structures(
    task: str,
    size: int,
    nuisance: NuisanceParameters,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    yy, xx, xr, yr = _coordinates(size, nuisance.angle_degrees)
    normal = _base_crystal(xr, yr)
    metadata: Dict[str, Any]
    if task == "amorphous_region":
        hard = (
            ((xx - 0.48 * size) / (0.16 * size)) ** 2
            + ((yy - 0.55 * size) / (0.12 * size)) ** 2
            <= 1
        )
        blend = gaussian_filter(hard.astype(float), max(size / 100, 1.2))
        disorder = gaussian_filter(rng.normal(size=(size, size)), 1.6)
        disorder = (disorder - np.mean(disorder)) / (
            np.std(disorder) + 1e-12
        )
        defect = (1 - blend) * normal + blend * disorder
        metadata = {"structural_intervention": "regional_periodicity_loss"}
    elif task == "atomic_point_defect":
        spacing = 16
        impulse_normal = np.zeros((size, size), dtype=float)
        for y in range(spacing, size - spacing + 1, spacing):
            for x in range(spacing, size - spacing + 1, spacing):
                impulse_normal[y, x] = 1.0
        impulse_defect = impulse_normal.copy()
        center = int(round(size / (2 * spacing))) * spacing
        impulse_defect[center, center] = 0.0
        impulse_defect[
            min(center + spacing, size - 1),
            min(center + spacing, size - 1),
        ] = 0.35
        normal = gaussian_filter(impulse_normal, 1.4)
        defect = gaussian_filter(impulse_defect, 1.4)
        normal /= max(float(np.max(normal)), 1e-8)
        defect /= max(float(np.max(normal)), 1e-8)
        metadata = {
            "structural_intervention": "vacancy_and_intensity_residual"
        }
    elif task == "stacking_fault":
        fault_y = 0.53 * size + 0.08 * (xx - size / 2)
        shift = 0.34 * 15.0 * (
            0.5 + 0.5 * np.tanh((yy - fault_y) / 1.6)
        )
        defect = (
            np.cos(2 * np.pi * (xr - shift) / 15.0)
            + 0.88 * np.cos(2 * np.pi * yr / 13.5 + 0.4)
            + 0.28
            * np.cos(
                2 * np.pi * ((xr - shift) / 15.0 + yr / 13.5)
            )
        )
        metadata = {"structural_intervention": "translational_phase_slip"}
    elif task == "phase_boundary":
        second_angle = np.deg2rad(nuisance.angle_degrees + 19.0)
        x2 = np.cos(second_angle) * xx - np.sin(second_angle) * yy
        y2 = np.sin(second_angle) * xx + np.cos(second_angle) * yy
        second = (
            np.cos(2 * np.pi * x2 / 12.0 + 0.5)
            + 0.82 * np.cos(2 * np.pi * y2 / 16.5)
            + 0.25 * np.cos(2 * np.pi * (x2 / 12.0 - y2 / 16.5))
        )
        interface_x = 0.51 * size + 0.08 * (yy - size / 2)
        blend = 0.5 + 0.5 * np.tanh((xx - interface_x) / 2.0)
        defect = (1 - blend) * normal + blend * second
        metadata = {"structural_intervention": "second_crystal_domain"}
    else:
        raise ValueError(
            f"Unsupported twin task {task!r}; choose one of "
            + ", ".join(SUPPORTED_TWIN_TASKS)
        )
    return normal, defect, metadata


def _shared_imaging(
    normal: np.ndarray,
    defect: np.ndarray,
    nuisance: NuisanceParameters,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    size = normal.shape[0]
    yy, xx = np.mgrid[:size, :size]
    illumination = (
        nuisance.illumination_x * xx / size
        + nuisance.illumination_y * yy / size
    )
    shared_noise = rng.normal(
        scale=nuisance.gaussian_noise_standard_deviation,
        size=normal.shape,
    )
    normal_observed = normal + illumination + shared_noise
    defect_observed = defect + illumination + shared_noise
    if nuisance.dose_electrons_per_normalized_unit > 0:
        # Common-random-number Gaussian approximation: same realization and
        # dose parameter, signal-dependent shot-noise amplitude.
        shot_standard = rng.normal(size=normal.shape)
        dose = nuisance.dose_electrons_per_normalized_unit
        normal_observed += (
            np.sqrt(np.clip(normal_observed - np.min(normal_observed), 0, None) / dose)
            * shot_standard
        )
        defect_observed += (
            np.sqrt(np.clip(defect_observed - np.min(defect_observed), 0, None) / dose)
            * shot_standard
        )
    if nuisance.blur_sigma > 0:
        normal_observed = gaussian_filter(
            normal_observed, nuisance.blur_sigma, mode="reflect"
        )
        defect_observed = gaussian_filter(
            defect_observed, nuisance.blur_sigma, mode="reflect"
        )
    return normal_observed, defect_observed


def _joint_uint16(
    normal: np.ndarray, defect: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([normal.ravel(), defect.ravel()])
    lower, upper = np.percentile(combined, [0.5, 99.5])

    def convert(values: np.ndarray) -> np.ndarray:
        normalized = np.clip(
            (values - lower) / max(float(upper - lower), 1e-12), 0, 1
        )
        return np.round(65535 * normalized).astype(np.uint16)

    return convert(normal), convert(defect)


def make_counterfactual_twin(
    task: str,
    *,
    size: int = 128,
    seed: int = 0,
    nuisance: NuisanceParameters = NuisanceParameters(),
) -> CounterfactualTwinPair:
    """Generate a normal/defect pair with a coupled imaging realization."""

    if size < 128:
        raise ValueError("Counterfactual twin size must be at least 128")
    nuisance.validate()
    structure_rng = np.random.default_rng(seed)
    normal_structure, defect_structure, metadata = _structures(
        task, size, nuisance, structure_rng
    )
    structural_delta = defect_structure - normal_structure
    imaging_rng = np.random.default_rng(seed + 1_000_003)
    normal_observed, defect_observed = _shared_imaging(
        normal_structure, defect_structure, nuisance, imaging_rng
    )
    normal_uint16, defect_uint16 = _joint_uint16(
        normal_observed, defect_observed
    )
    delta_threshold = max(
        0.02 * float(np.max(np.abs(structural_delta))), 1e-8
    )
    defect_mask = np.abs(structural_delta) >= delta_threshold
    payload = {
        "task": task,
        "size": size,
        "seed": seed,
        "nuisance": asdict(nuisance),
        "intervention": metadata["structural_intervention"],
    }
    pair_id = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:20]
    pair = CounterfactualTwinPair(
        pair_id=f"twin-{pair_id}",
        task=task,
        normal_image=normal_uint16,
        defect_image=defect_uint16,
        defect_mask=defect_mask,
        structural_delta=structural_delta.astype(np.float32),
        nuisance=nuisance,
        seed=seed,
        metadata={
            **metadata,
            "truth_scope": "synthetic_structural_intervention",
            "experimental_accuracy_claim_allowed": False,
            "normal_and_defect_jointly_normalized": True,
            "shared_random_imaging_realization": True,
        },
    )
    pair.validate()
    return pair
