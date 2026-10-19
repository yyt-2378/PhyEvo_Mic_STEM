"""Opt-in expert-student shadow inference alongside unchanged physics results."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from skimage.draw import disk

from .expert_supervision import GEOMETRY_INDEX, contained_path, sha256_file, write_new_json
from .expert_student import point_candidates, predict_scores, restore_checkpoint
from .models import DetectionResult
from .preprocessing import load_grayscale


TASK_GEOMETRY = {"atomic_point_defect": "point", "phase_boundary": "interface",
                 "amorphous_region": "region", "stacking_fault": "line"}


@dataclass
class ExpertShadowPrediction:
    score: np.ndarray
    candidate_mask: np.ndarray
    points_xy: np.ndarray
    metadata: dict[str, Any]


@dataclass
class ExpertAssistedDetection:
    teacher: DetectionResult
    student: ExpertShadowPrediction | None
    student_status: str


class ExpertStudentRuntime:
    def __init__(self, checkpoint: str | Path, *, project_root: str | Path,
                 allow_experimental: bool = False, device: str = "cpu"):
        if not allow_experimental:
            raise ValueError("Expert student is uncalibrated; explicit allow_experimental=True is required")
        self.root = Path(project_root).resolve()
        self.checkpoint = contained_path(self.root, checkpoint)
        self.device = device
        self.model, _, _, self.metadata = restore_checkpoint(self.checkpoint, device=device)
        self.checkpoint_sha256 = sha256_file(self.checkpoint / "checkpoint.json")

    def predict(self, image: np.ndarray, geometry: str, *, point_budget_per_512: int = 50,
                area_budget_fraction: float = 0.10) -> ExpertShadowPrediction:
        if geometry not in self.metadata["geometry_support"]:
            raise ValueError(f"No expert training support for geometry {geometry}")
        if point_budget_per_512 < 1:
            raise ValueError("Point budget must be positive")
        if not math.isfinite(area_budget_fraction) or not 0 < area_budget_fraction <= 1:
            raise ValueError("Area budget must be finite and within (0, 1]")
        score = predict_scores(self.model, image, GEOMETRY_INDEX[geometry], device=self.device)
        points = np.empty((0, 2), dtype=float)
        if geometry == "point":
            budget = max(1, round(point_budget_per_512 * score.size / (512 * 512)))
            points = point_candidates(score, budget)
            mask = np.zeros(score.shape, bool)
            for x, y in points:
                rr, cc = disk((y, x), 3, shape=score.shape)
                mask[rr, cc] = True
        else:
            # Positive/unknown training does not calibrate an absolute 0.5
            # threshold. Use the same fixed area budget as held-out evaluation,
            # not a nearly full-image mask misread as a confirmed defect.
            mask = np.zeros(score.shape, bool)
            count = max(1, math.ceil(area_budget_fraction * score.size))
            mask.ravel()[np.argsort(-score.ravel(), kind="stable")[:count]] = True
        meta = {
            "mode": "experimental_shadow_not_confirmed_defect", "geometry": geometry,
            "checkpoint": str(self.checkpoint.relative_to(self.root)), "checkpoint_sha256": self.checkpoint_sha256,
            "learning_manifest_sha256": self.metadata["learning_manifest_sha256"],
            "coordinate_space": "source_pixels_xy", "image_shape": list(score.shape),
            "score_kind": "uncalibrated_geometry_score_not_physical_probability",
            "candidate_selection": "fixed_budget_ranking" if geometry == "point" else "fixed_area_budget_ranking",
            "point_budget_per_512_square": point_budget_per_512 if geometry == "point" else None,
            "area_budget_fraction": area_budget_fraction if geometry != "point" else None,
            "candidate_area_fraction": float(mask.mean()),
            "candidate_count": len(points) if geometry == "point" else None,
            "teacher_mask_overwritten": False, "teacher_validity_gates_student": False,
            "physical_evidence": "use_original_teacher_evidence_not_unsupervised_student_heads",
            "interface_semantics": "affected_band_not_exact_centerline" if geometry == "interface" else None,
            "unvalidated_geometry": geometry in self.metadata.get("unvalidated_geometries", ["line", "region"]),
            "warnings": ["Not Gold calibrated", "No OOD safety guarantee", "Candidates are not chemical identities",
                         "Top-budget candidates may exist even on a normal image; not a presence/absence decision"],
        }
        return ExpertShadowPrediction(score, mask, points, meta)


def detect_with_expert_student(framework, source, *, checkpoint, project_root,
                              allow_experimental=False, student_device="cpu", **detect_options):
    """Run original detector and a separate learned branch, never mutate its mask."""
    runtime = ExpertStudentRuntime(checkpoint, project_root=project_root,
                                   allow_experimental=allow_experimental, device=student_device)
    if isinstance(source, (str, Path)):
        source = contained_path(Path(project_root), source)
    teacher = framework.detect(source, **detect_options)
    geometry = TASK_GEOMETRY.get(teacher.task)
    if geometry not in runtime.metadata["geometry_support"]:
        return ExpertAssistedDetection(teacher, None, "unsupported_task_physics_retained")
    if isinstance(source, (str, Path)):
        image = load_grayscale(source, frame_index=detect_options.get("frame_index"),
                               projection=detect_options.get("projection"), dataset_path=detect_options.get("dataset_path"))
    else:
        image = np.asarray(source)
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8).tobytes())
    if digest.hexdigest() != teacher.metadata["input_provenance"]["pixel_sha256"]:
        raise ValueError("Input changed between physics and student inference")
    student = runtime.predict(image, geometry)
    return ExpertAssistedDetection(teacher, student, "experimental_shadow")


def save_shadow_prediction(prediction: ExpertShadowPrediction, output: Path, *, project_root: Path):
    output = contained_path(project_root, output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an expert-student prediction")
    output.mkdir(parents=True)
    path = output / "student_shadow.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, score=prediction.score, candidate_mask=prediction.candidate_mask,
                            points_xy=prediction.points_xy)
    write_new_json(output / "student_shadow.json", {**prediction.metadata, "arrays_sha256": sha256_file(path)})
    return {"student_shadow_arrays": str(path), "student_shadow_manifest": str(output / "student_shadow.json")}
