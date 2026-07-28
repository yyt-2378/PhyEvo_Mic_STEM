"""Task interface for adding new physical anomaly definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from ..config import DetectorConfig
from ..context import DetectionContext
from ..models import DetectionResult


class DetectionTask(ABC):
    name: str

    @abstractmethod
    def run(
        self,
        normalized_image: np.ndarray,
        config: DetectorConfig,
        input_metadata: Dict[str, Any],
        context: DetectionContext,
    ) -> DetectionResult:
        raise NotImplementedError
