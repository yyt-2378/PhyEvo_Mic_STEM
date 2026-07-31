#!/usr/bin/env python3
"""Run SAM 2 image segmentation from a boundary-protocol request.

The worker intentionally reuses ``tfree-sam3-boundary-v2``.  That protocol
describes candidate-local prompts and proposal masks rather than a particular
SAM generation.  The main process remains responsible for consensus,
physics-based acceptance gates, and per-candidate fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


PROTOCOL_VERSION = "tfree-sam3-boundary-v2"

MODEL_CONFIGS = {
    "sam2_hiera_tiny": "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_small": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_base_plus": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_large": "configs/sam2/sam2_hiera_l.yaml",
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.view(np.uint8).tobytes())
    return hasher.hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _protocol_path(directory: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("Protocol file paths must be relative")
    resolved = (directory / path).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise ValueError("Protocol path traversal rejected")
    return resolved


def _load_array(
    directory: Path, record: Dict[str, Any]
) -> np.ndarray:
    path = _protocol_path(directory, str(record["file"]))
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != record["shape"]:
        raise ValueError(f"Array shape mismatch for {path.name}")
    if _array_sha256(values) != record["sha256"]:
        raise ValueError(f"Array checksum mismatch for {path.name}")
    return values


def _write_array_exclusive(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def _write_json_exclusive(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def _rgb_image(values: np.ndarray) -> np.ndarray:
    image = np.asarray(values, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("SAM2 boundary worker expects a 2-D grayscale image")
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError("Input image contains no finite pixels")
    lower, upper = np.percentile(finite, [0.5, 99.5])
    normalized = np.clip(
        (image - lower) / max(float(upper - lower), 1e-8),
        0,
        1,
    )
    grayscale = np.round(normalized * 255).astype(np.uint8)
    return np.repeat(grayscale[:, :, None], 3, axis=2)


def _mask_logits(component: np.ndarray) -> np.ndarray:
    resized = Image.fromarray(
        component.astype(np.uint8) * 255, mode="L"
    ).resize((256, 256), resample=Image.Resampling.BILINEAR)
    probability = np.asarray(resized, dtype=np.float32) / 255.0
    return ((probability - 0.5) * 12.0)[None, :, :]


def _overlap_fraction(mask: np.ndarray, reference: np.ndarray) -> float:
    denominator = int(reference.sum())
    if not denominator:
        return 0.0
    return float(np.logical_and(mask, reference).sum() / denominator)


def _model_config(model_variant: str) -> str:
    try:
        return MODEL_CONFIGS[model_variant]
    except KeyError as error:
        supported = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(
            f"Unsupported SAM2 model_variant {model_variant!r}; "
            f"choose one of: {supported}"
        ) from error


def run(request_path: Path) -> Dict[str, Any]:
    started_at = time.perf_counter()
    request_path = request_path.resolve()
    directory = request_path.parent
    with request_path.open("r", encoding="utf-8") as stream:
        request = json.load(stream)
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported SAM boundary request protocol")

    worker_config = request["worker"]
    if worker_config.get("backend") != "sam2":
        raise ValueError("SAM2 worker requires request worker.backend='sam2'")

    arrays = request["arrays"]
    image_values = _load_array(directory, arrays["image"])
    coarse = _load_array(directory, arrays["coarse_mask"]).astype(bool)
    physics_core = _load_array(
        directory, arrays["physics_core"]
    ).astype(bool)
    component_labels = _load_array(
        directory, arrays["component_labels"]
    ).astype(np.int32)
    domain_labels = (
        _load_array(directory, arrays["domain_labels"]).astype(np.int8)
        if "domain_labels" in arrays
        else None
    )
    valid_mask = (
        _load_array(directory, arrays["valid_mask"]).astype(bool)
        if "valid_mask" in arrays
        else np.ones(coarse.shape, dtype=bool)
    )
    for name, values in (
        ("image", image_values),
        ("physics_core", physics_core),
        ("component_labels", component_labels),
        ("valid_mask", valid_mask),
    ):
        if values.shape != coarse.shape:
            raise ValueError(
                f"{name} shape {values.shape} does not match "
                f"coarse_mask shape {coarse.shape}"
            )
    if domain_labels is not None and domain_labels.shape != coarse.shape:
        raise ValueError("domain_labels shape does not match coarse_mask")

    import sam2
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    requested_device = str(worker_config.get("device", "auto"))
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for SAM2 but unavailable")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    checkpoint_value = worker_config.get("checkpoint_path")
    if not checkpoint_value:
        raise ValueError("SAM2 requires an explicit checkpoint_path")
    checkpoint = Path(str(checkpoint_value)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    model_variant = str(
        worker_config.get("model_variant", "sam2.1_hiera_large")
    )
    model_config = _model_config(model_variant)
    model = build_sam2(
        model_config,
        str(checkpoint),
        device=device,
        mode="eval",
        apply_postprocessing=True,
    )
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(_rgb_image(image_values))

    prompt_records = request.get("prompts", [])
    proposals = []
    seen_prompt_keys: set[tuple[str, int, str]] = set()
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )
    inference_mode = (
        torch.inference_mode()
        if hasattr(torch, "inference_mode")
        else nullcontext()
    )
    with inference_mode, autocast:
        for prompt_index, prompt in enumerate(prompt_records):
            candidate_id = str(prompt["candidate_id"])
            ensemble_index = int(prompt["ensemble_index"])
            prompt_role = str(prompt.get("prompt_role", "candidate"))
            prompt_key = (candidate_id, ensemble_index, prompt_role)
            if prompt_key in seen_prompt_keys:
                raise ValueError(
                    "Duplicate SAM2 candidate/ensemble prompt in request"
                )
            seen_prompt_keys.add(prompt_key)
            component_label = int(prompt["component_label"])
            box = np.asarray(prompt["box_xyxy"], dtype=np.float32)
            x1, y1, x2, y2 = (
                int(value) for value in prompt["box_xyxy"]
            )
            local_roi = np.zeros(coarse.shape, dtype=bool)
            local_roi[y1:y2, x1:x2] = True
            if prompt_role in {"domain_a", "domain_b"}:
                if domain_labels is None:
                    raise ValueError(
                        "Paired-domain prompt requires domain_labels array"
                    )
                domain_value = 0 if prompt_role == "domain_a" else 1
                component = (
                    (domain_labels == domain_value)
                    & valid_mask
                    & local_roi
                )
                component_core = component
            else:
                component = component_labels == component_label
                component_core = physics_core & component

            positive = prompt.get("positive_points_xy", [])
            negative = prompt.get("negative_points_xy", [])
            points = np.asarray(positive + negative, dtype=np.float32)
            labels = np.asarray(
                [1] * len(positive) + [0] * len(negative),
                dtype=np.int32,
            )
            if not len(points):
                points = None
                labels = None
            mask_input = (
                _mask_logits(component)
                if prompt.get("use_coarse_mask_prompt", False)
                else None
            )
            masks, quality_scores, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                mask_input=mask_input,
                multimask_output=True,
            )
            candidate_masks = np.asarray(masks) > 0
            candidate_scores = np.asarray(
                quality_scores, dtype=float
            ).reshape(-1)
            if candidate_masks.ndim == 2:
                candidate_masks = candidate_masks[None, :, :]
            if not len(candidate_masks):
                raise RuntimeError("SAM2 returned no interactive masks")

            ranking = []
            for index, candidate_mask in enumerate(candidate_masks):
                core_recall = _overlap_fraction(
                    candidate_mask, component_core
                )
                model_score = (
                    float(np.clip(candidate_scores[index], 0.0, 1.0))
                    if index < len(candidate_scores)
                    and math.isfinite(float(candidate_scores[index]))
                    else 0.0
                )
                ranking.append(
                    (0.65 * core_recall + 0.35 * model_score, index)
                )
            _, selected_index = max(ranking)
            selected = candidate_masks[selected_index]
            selected_score = (
                float(np.clip(candidate_scores[selected_index], 0.0, 1.0))
                if selected_index < len(candidate_scores)
                and math.isfinite(float(candidate_scores[selected_index]))
                else 0.0
            )
            filename = (
                f"proposal_candidate_{prompt_index:04d}_"
                f"ensemble_{ensemble_index:03d}.npy"
            )
            path = directory / filename
            _write_array_exclusive(path, selected.astype(bool))
            proposals.append(
                {
                    "proposal_id": (
                        f"sam2-candidate-{prompt_index:04d}-"
                        f"ensemble-{ensemble_index:03d}"
                    ),
                    "candidate_id": candidate_id,
                    "component_label": component_label,
                    "ensemble_index": ensemble_index,
                    "prompt_role": prompt_role,
                    "mask_file": filename,
                    "sha256": _file_sha256(path),
                    "score": selected_score,
                }
            )

    response = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request["request_id"],
        "proposals": proposals,
        "worker": {
            "backend": "sam2",
            "model_variant": model_variant,
            "model_config": model_config,
            "device": device,
            "checkpoint_source": str(checkpoint),
            "sam2_version": getattr(sam2, "__version__", "unknown"),
            "torch_version": torch.__version__,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "interactive_api": "SAM2ImagePredictor.predict",
            "training_or_fitting_used": False,
            "runtime_seconds": float(time.perf_counter() - started_at),
            "peak_process_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "peak_device_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if device.startswith("cuda")
                else 0
            ),
        },
    }
    response_path = _protocol_path(
        directory, str(request["response_file"])
    )
    _write_json_exclusive(response_path, response)
    return {
        "request_id": request["request_id"],
        "proposal_count": len(proposals),
        "response": str(response_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated SAM2 boundary proposal inference"
    )
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    print(json.dumps(run(Path(args.request)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
