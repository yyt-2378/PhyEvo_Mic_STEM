"""Training-free microscopy anomaly detection."""

from .config import DetectorConfig, load_config
from .context import DetectionContext, IdealSkeleton
from .dataset_label_adapter import (
    AdaptedLabel,
    DatasetLabelAdapterRegistry,
    EpistemicState,
    ExternalLabel,
    SemanticRole,
    default_label_adapter_registry,
)
from .evidence import (
    DecisionState,
    DefectHypothesis,
    EvidenceBundle,
    EvidenceField,
    EvidenceScale,
    LabelTier,
    ReliabilityBundle,
    SpatialSupport,
    ViolationType,
)
from .framework import TrainingFreeDetectionFramework
from .governance import (
    AnnotationLedger,
    AnnotationRecord,
    CandidateRecord,
    DataAsset,
    DataPartition,
    DataRegistry,
    candidate_records_from_result,
)
from .models import DetectionRegion, DetectionResult
from .preprocessing import list_emd_datasets, load_grayscale
from .resolver import (
    HypothesisResolution,
    ResolverPolicy,
    attach_hypothesis_resolutions,
)

__all__ = [
    "DetectorConfig",
    "DataAsset",
    "DataPartition",
    "DataRegistry",
    "DecisionState",
    "AdaptedLabel",
    "DatasetLabelAdapterRegistry",
    "DefectHypothesis",
    "DetectionContext",
    "DetectionRegion",
    "DetectionResult",
    "EvidenceBundle",
    "EvidenceField",
    "EvidenceScale",
    "EpistemicState",
    "ExternalLabel",
    "IdealSkeleton",
    "LabelTier",
    "AnnotationLedger",
    "AnnotationRecord",
    "CandidateRecord",
    "ReliabilityBundle",
    "ResolverPolicy",
    "HypothesisResolution",
    "SemanticRole",
    "SpatialSupport",
    "TrainingFreeDetectionFramework",
    "ViolationType",
    "candidate_records_from_result",
    "attach_hypothesis_resolutions",
    "default_label_adapter_registry",
    "load_config",
    "load_grayscale",
    "list_emd_datasets",
]

__version__ = "0.4.0"
