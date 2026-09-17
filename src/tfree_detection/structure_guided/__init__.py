"""Material-agnostic projected-structure anomaly analysis."""

from .analyzer import analyze_projected_site_probes
from .contracts import (
    ProjectedSiteProbe,
    SampleRole,
    StructureGuidedAnalysis,
    StructureGuidedPolicy,
)
from .template import projected_probes_from_skeleton

__all__ = [
    "ProjectedSiteProbe",
    "SampleRole",
    "StructureGuidedAnalysis",
    "StructureGuidedPolicy",
    "analyze_projected_site_probes",
    "projected_probes_from_skeleton",
]
