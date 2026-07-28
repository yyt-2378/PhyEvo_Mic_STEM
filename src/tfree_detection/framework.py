"""Task registry and end-to-end framework entry point."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Type

import numpy as np

from .config import DetectorConfig
from .context import DetectionContext, IdealSkeleton
from .evidence import attach_evidence_protocol
from .models import DetectionResult
from .line_topology import attach_line_topology
from .preprocessing import load_grayscale, robust_normalize
from .resolver import ResolverPolicy, attach_hypothesis_resolutions
from .tasks import (
    AmorphousRegionTask,
    AtomicPointDefectTask,
    DetectionTask,
    PhaseBoundaryTask,
    StackingFaultTask,
    SymmetryRelationTask,
)


class TrainingFreeDetectionFramework:
    """Run registered physical anomaly tasks behind one stable interface."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._tasks: Dict[str, Type[DetectionTask]] = {
            AmorphousRegionTask.name: AmorphousRegionTask,
            AtomicPointDefectTask.name: AtomicPointDefectTask,
            StackingFaultTask.name: StackingFaultTask,
            PhaseBoundaryTask.name: PhaseBoundaryTask,
            SymmetryRelationTask.name: SymmetryRelationTask,
        }

    def register_task(
        self, name: str, task_class: Type[DetectionTask]
    ) -> None:
        if not name:
            raise ValueError("Task name cannot be empty")
        self._tasks[name] = task_class

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(sorted(self._tasks))

    def detect(
        self,
        source: str | Path | np.ndarray,
        skeleton: str | Path | IdealSkeleton | None = None,
        exemplar: str | Path | np.ndarray | None = None,
        *,
        frame_index: int | None = None,
        projection: str | None = None,
        dataset_path: str | None = None,
        line_topology: bool = False,
    ) -> DetectionResult:
        if isinstance(source, (str, Path)):
            raw_image = load_grayscale(
                source,
                frame_index=frame_index,
                projection=projection,
                dataset_path=dataset_path,
            )
            source_path = Path(source).resolve()
            source_name = str(source_path)
            file_hasher = hashlib.sha256()
            with source_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    file_hasher.update(block)
            source_file_sha256 = file_hasher.hexdigest()
            source_file_bytes = source_path.stat().st_size
        else:
            raw_image = np.asarray(source)
            if raw_image.ndim != 2:
                raise ValueError("Array input must be a 2-D grayscale image")
            source_name = "<array>"
            source_file_sha256 = None
            source_file_bytes = None
        contiguous = np.ascontiguousarray(raw_image)
        pixel_hasher = hashlib.sha256()
        pixel_hasher.update(str(contiguous.dtype).encode("utf-8"))
        pixel_hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        pixel_hasher.update(contiguous.view(np.uint8).tobytes())
        config_sha256 = hashlib.sha256(
            json.dumps(
                self.config.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        normalized, metadata = robust_normalize(raw_image, self.config.preprocess)
        metadata["source"] = source_name
        metadata["input_provenance"] = {
            "source_image": source_name,
            "source_file_sha256": source_file_sha256,
            "source_file_bytes": source_file_bytes,
            "pixel_sha256": pixel_hasher.hexdigest(),
            "source_id": pixel_hasher.hexdigest()[:16],
            "raw_dtype": str(contiguous.dtype),
            "raw_shape": list(contiguous.shape),
            "config_sha256": config_sha256,
            "stack_selection": {
                "frame_index": frame_index,
                "projection": projection,
                "dataset_path": dataset_path,
            },
        }
        if isinstance(skeleton, (str, Path)):
            skeleton_object = IdealSkeleton.from_json(skeleton)
        else:
            skeleton_object = skeleton
        if isinstance(exemplar, (str, Path)):
            exemplar_raw = load_grayscale(exemplar)
            exemplar_image, _ = robust_normalize(
                exemplar_raw, self.config.preprocess
            )
        elif exemplar is None:
            exemplar_image = None
        else:
            exemplar_image, _ = robust_normalize(
                np.asarray(exemplar), self.config.preprocess
            )
        context = DetectionContext(
            skeleton=skeleton_object, exemplar_image=exemplar_image
        )

        try:
            task_class = self._tasks[self.config.task]
        except KeyError as error:
            raise ValueError(
                f"Unknown task {self.config.task!r}; "
                f"available tasks: {', '.join(self.available_tasks)}"
            ) from error
        task = task_class()
        result = task.run(normalized, self.config, metadata, context)
        if line_topology:
            result = attach_line_topology(result)
        protocol = self.config.evidence_protocol
        if protocol.enabled:
            result = attach_evidence_protocol(
                result,
                reliable_threshold=protocol.reliable_threshold,
                invalid_threshold=protocol.invalid_threshold,
                minimum_confirmed_evidence_groups=(
                    protocol.minimum_confirmed_evidence_groups
                ),
            )
            if self.config.resolver.enabled:
                resolver = self.config.resolver
                result = attach_hypothesis_resolutions(
                    result,
                    ResolverPolicy(
                        minimum_hypothesis_score=(
                            resolver.minimum_hypothesis_score
                        ),
                        minimum_score_margin=resolver.minimum_score_margin,
                        minimum_region_support_fraction=(
                            resolver.minimum_region_support_fraction
                        ),
                        maximum_ranked_hypotheses=(
                            resolver.maximum_ranked_hypotheses
                        ),
                    ),
                )
        return result
