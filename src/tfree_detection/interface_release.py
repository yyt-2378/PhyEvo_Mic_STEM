"""Source-excluded checkpoint selection and bound interface research runtime."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import load_config
from .expert_student import predict_scores,restore_checkpoint
from .expert_supervision import contained_path,sha256_file
from .interface_reliability import qualify_boundary_or_band
from .interface_runtime import choose_interface,physical_consistency


def training_healthy(run,step):
    rows=[r for r in run["records"] if r["step"]==step and r["is_training_diagnostic"]]
    return bool(rows) and all(r["health"]["eligible"] for r in rows)


def select_retention_step(protocol,runs,outer_group):
    candidates=[]
    for step in protocol["checkpoints"]:
        values=[];receipts=[];eligible=True
        for group in protocol["groups"]:
            if group==outer_group:continue
            scores=[]
            for seed in [23,41,67]:
                matching=[r for r in runs if r["job"]["stage"]=="inner" and
                    r["job"]["outer_group"]==outer_group and r["job"]["validation_group"]==group and r["job"]["seed"]==seed]
                if len(matching)!=1:raise ValueError("Missing or duplicate source-excluded inner run")
                run=matching[0];job=run["job"]
                if outer_group in job["train_groups"] or group in job["train_groups"]:
                    raise ValueError("Inner selection leakage")
                rows=[r for r in run["records"] if r["step"]==step and not r["is_training_diagnostic"]]
                if not rows or any(r["group"]!=group for r in rows):raise ValueError("Wrong validation source")
                # A training-saturated checkpoint is inadmissible. An unhealthy
                # unseen-image output is vetoed by the router for that input;
                # it must not force the student to solve domains handled by
                # qualified physics. Score the actual guarded system instead.
                healthy=training_healthy(run,step)
                eligible &= healthy
                if any("routed_coverage" not in r for r in rows):
                    raise ValueError("Inner selection requires guarded-system evaluation")
                scores.append(float(np.mean([r["routed_coverage"] for r in rows])))
                receipts.append({"job":job["index"],"validation_group":group,
                                 "train_groups":job["train_groups"],"training_healthy":healthy,
                                 "student_output_vetoes":[r["case_number"] for r in rows if not r["health"]["eligible"]]})
            values.append(float(np.mean(scores)))
        candidates.append({"step":step,"inner_macro_coverage":float(np.mean(values)),
                           "eligible":bool(eligible),"receipts":receipts})
    available=[c for c in candidates if c["eligible"]]
    selected=max(available,key=lambda c:(c["inner_macro_coverage"],-c["step"])) if available else None
    return {"outer_group":outer_group,"selected":selected,"candidates":candidates,
            "selection_uses_outer_labels":False,"historical_base_was_test_informed":True,
            "selection_target":"guarded physics/student system, not universal student",
            "tie_break":"earliest training checkpoint",
            "inner_objective_distinguishes_steps":len({round(c["inner_macro_coverage"],12) for c in available})>1}


class InterfaceResearchRuntime:
    """One configured CV/deployment model, no filename-based checkpoint routing."""

    def __init__(self,project_root,binding,*,device="cpu",allow_experimental=False):
        if not allow_experimental:raise ValueError("Explicit experimental runtime opt-in required")
        self.root=Path(project_root).resolve()
        path=contained_path(self.root,binding)
        self.binding=json.loads(path.read_text());b=self.binding
        if b.get("format")!="phyevo-interface-research-binding-v1" or b.get("production_approved") is not False:
            raise ValueError("Invalid interface research binding")
        checkpoint=contained_path(self.root,b["checkpoint"])
        if sha256_file(checkpoint/"checkpoint.json")!=b["checkpoint_sha256"]:raise ValueError("Bound checkpoint changed")
        selector=contained_path(self.root,b["selector"])
        if sha256_file(selector)!=b["selector_sha256"]:raise ValueError("Bound validation selector changed")
        decision=json.loads(selector.read_text())
        if decision["selection_uses_outer_labels"] is not False or decision["selected"] is None:
            raise ValueError("No valid inner selection")
        self.model,_,_,self.meta=restore_checkpoint(checkpoint,device=device)
        if self.meta["step"]!=decision["selected"]["step"] or self.meta["stage"]!="outer":
            raise ValueError("Checkpoint/selection identity mismatch")
        if self.meta["outer_group"]!=decision["outer_group"] or self.meta["outer_group"] in self.meta["train_groups"]:
            raise ValueError("Source isolation mismatch")
        for name,digest in b["runtime_source_hashes"].items():
            if sha256_file(contained_path(self.root,name))!=digest:raise ValueError("Bound runtime source changed")
        self.device=device
        self.config=load_config(contained_path(self.root,"configs/phase_boundary_geometry_aware.yaml"))

    def predict(self,normalized_image):
        image=np.asarray(normalized_image,np.float32)
        if image.ndim!=2 or not np.isfinite(image).all():raise ValueError("Invalid normalized image")
        physical,strict=physical_consistency(image,self.config)
        fields={"labels":physical.evidence_maps["domain_label"],"valid":physical.evidence_maps["boundary_validity"]>.5}
        guard=qualify_boundary_or_band(fields,strict,patch_size=self.config.phase_boundary.patch_size)
        score=predict_scores(self.model,image,4,device=self.device,normalized=True)
        budget=int(np.ceil(.1*image.size))
        mask,decision=choose_interface(physical.score_map,fields["valid"],guard,score,
            student_admitted=bool(self.binding["student_training_admitted"]),budget=budget)
        return {"mask":mask,"student_score":score,"physical_score":physical.score_map,
                "physical_valid":fields["valid"],"guard":guard,"decision":decision,
                "budget":budget,"area_fraction":float(mask.mean()),
                "claims":{"candidate_not_confirmed_defect":True,"calibrated":False,"production_approved":False}}
