"""Opt-in inference for a sealed synthesized-program lineage.

No source-name dispatch, expert labels, training, or production-default changes.
The caller explicitly selects one research CV/deployment model, just as for the
frozen interface runtime. An outer-audit failure is never silently hidden.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .autonomous_program import top_mask, validate
from .autonomous_rsi import prepare_case, scores
from .expert_student import predict_scores
from .expert_supervision import contained_path, sha256_file
from .interface_release import InterfaceResearchRuntime
from .interface_reliability import qualify_boundary_or_band
from .interface_runtime import choose_interface, physical_consistency


class AutonomousResearchRuntime:
    def __init__(self, project_root, binding, *, device="cpu", allow_experimental=False):
        if not allow_experimental:
            raise ValueError("Explicit experimental opt-in required")
        self.root = Path(project_root).resolve()
        self.binding = json.loads(contained_path(self.root, binding).read_text())
        b = self.binding
        if b.get("format") != "phyevo-autonomous-program-binding-v1" or b.get("production_approved") is not False:
            raise ValueError("Invalid research binding")
        for name, digest in b["runtime_source_hashes"].items():
            if sha256_file(contained_path(self.root, name)) != digest:
                raise ValueError("Bound runtime implementation changed")
        audit = contained_path(self.root, b["audit"])
        if sha256_file(audit) != b["audit_sha256"]:
            raise ValueError("Audit record changed")
        record = json.loads(audit.read_text())
        if record["outer_results_returned_to_proposer"] is not False:
            raise ValueError("Outer-feedback contamination")
        self.stages = record["states"][str(b["job_index"])]
        if len(self.stages) > 3:
            raise ValueError("Lineage exceeds frozen round budget")
        for stage in self.stages:
            for program in stage["programs"]:
                validate(program)
            weights = np.asarray(stage["weights"])
            if weights.shape != (len(stage["programs"]), 3) or not np.isfinite(weights).all() or np.abs(weights).max() > 2.000001:
                raise ValueError("Invalid learned coefficients")
        self.base = InterfaceResearchRuntime(self.root, b["base_binding"], device=device, allow_experimental=True)
        self.device = device

    def predict(self, normalized_image):
        image = np.asarray(normalized_image, np.float32)
        if image.ndim != 2 or min(image.shape) < 4 or not np.isfinite(image).all():
            raise ValueError("Expected finite normalized 2-D image")
        physical, strict = physical_consistency(image, self.base.config)
        fields = {"labels": physical.evidence_maps["domain_label"],
                  "valid": physical.evidence_maps["boundary_validity"] > .5,
                  "score": physical.score_map,
                  "variation": physical.evidence_maps["reciprocal_signature_variation"]}
        guard = qualify_boundary_or_band(fields, strict, patch_size=self.base.config.phase_boundary.patch_size)
        student = predict_scores(self.base.model, image, 4, device=self.device, normalized=True)
        budget = int(np.ceil(.1 * image.size))
        baseline_mask, decision = choose_interface(fields["score"], fields["valid"], guard, student,
            student_admitted=self.base.binding["student_training_admitted"], budget=budget)
        is_physical = decision["branch"] != "student"
        raw = fields["score"] if is_physical else student
        allowed = (raw > 0) & (fields["valid"] if is_physical else np.ones(raw.shape, bool))
        # Placeholder only satisfies the training Case structure. It never enters
        # scores(), the program inputs or the inference decision as a label.
        case = prepare_case(-1, "", image, fields, student, raw, allowed, np.zeros(image.shape, bool),
                            physical_route=is_physical, band=guard["hypothesis"] == "enclosed_texture_band",
                            device=self.device, resolution=self.binding["program_resolution"])
        with torch.no_grad():
            score = scores(case, self.stages, full=True)
            if self.stages and (not torch.isfinite(score).all() or float(score.std(correction=0)) < 1e-5):
                mask = baseline_mask
                status = "invalid_synthesized_score_returned_baseline"
            else:
                mask = top_mask(score, case.allowed).cpu().numpy()
                status = "experimental_program_lineage" if self.stages else "unchanged_baseline"
        return {"mask": mask, "baseline_mask": baseline_mask, "score": score.cpu().numpy(),
                "student_score": student, "physical_score": fields["score"], "physical_valid": fields["valid"],
                "budget": budget, "area_fraction": float(mask.mean()), "baseline_decision": decision,
                "status": status, "stages": len(self.stages), "audit_nonregressing": self.binding["audit_nonregressing"],
                "claims": {"production_approved": False, "candidate_not_confirmed_defect": True,
                           "calibrated": False, "uses_expert_labels_at_inference": False,
                           "new_blind_external_test": False}}
