"""Conservative adaptation of an existing specialist, not blind retraining.

Frozen feature extractor + trainable geometry readout. Centered-logit retention
protects already useful spatial ordering while the expert/physical objective
can repair the score offset and readout. It is not probability calibration.
"""
from __future__ import annotations

import numpy as np

from .expert_student import geometry_logits, torch
from .interface_student import InterfaceSampler,physical_ranking_loss


class RetentionSampler(InterfaceSampler):
    def __init__(self,entries,arrays,policy,physical,*,excluded_groups=()):
        super().__init__(entries,arrays,policy,physical,excluded_groups=excluded_groups)
        for entry in entries:
            n=entry["case_number"]
            # Physical evidence may rank the well-supported part of a weak ROI;
            # lack of physical validity must never erase an expert positive.
            outside=np.asarray(arrays[n]["human"],bool)&~np.asarray(physical[n]["valid"],bool)
            self.targets[n]["physical_positive"][outside]=1.


def freeze_feature_extractor(model):
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("geometry_head."))
    return model


def retention_loss(output,teacher_output,batch,scaled_output=None):
    loss,parts=physical_ranking_loss(output,batch,scaled_output)
    z=geometry_logits(output,batch["geometry_index"])
    old=geometry_logits(teacher_output,batch["geometry_index"]).detach()
    valid=batch["valid_pixels"]
    count=valid.sum((1,2),keepdim=True).clamp(min=1)
    centered=z-(z*valid).sum((1,2),keepdim=True)/count
    old_centered=old-(old*valid).sum((1,2),keepdim=True)/count
    retain=(((centered-old_centered).square()*valid).sum((1,2))/count[:,0,0]).mean()
    total=loss+10*retain
    return total,{**parts,"centered_logit_retention":retain,"total":total}
