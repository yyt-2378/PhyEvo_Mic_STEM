#!/usr/bin/env python3
"""Run SAM 3 interactive image segmentation from a file-protocol request.

This script is intentionally standalone: execute it with the Python 3.12 SAM3
environment, not the main tfree-detection environment.
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


def _pil_image(values: np.ndarray) -> Image.Image:
    image = np.asarray(values, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("SAM3 boundary worker expects a 2-D grayscale image")
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError("Input image contains no finite pixels")
    lower, upper = np.percentile(finite, [0.5, 99.5])
    normalized = np.clip((image - lower) / max(upper - lower, 1e-8), 0, 1)
    grayscale = Image.fromarray(
        np.round(normalized * 255).astype(np.uint8), mode="L"
    )
    return grayscale.convert("RGB")


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


def run(request_path: Path) -> Dict[str, Any]:
    started_at = time.perf_counter()
    request_path = request_path.resolve()
    directory = request_path.parent
    with request_path.open("r", encoding="utf-8") as stream:
        request = json.load(stream)
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported SAM3 request protocol")

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

    import sam3
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    worker_config = request["worker"]
    requested_device = str(worker_config.get("device", "auto"))
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for SAM3 but unavailable")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    checkpoint = worker_config.get("checkpoint_path")
    model_variant = str(worker_config.get("model_variant", "sam3"))
    if model_variant != "sam3" and checkpoint is None:
        raise ValueError(
            "Non-default SAM3 variants require an explicit compatible "
            "checkpoint_path"
        )
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint,
        load_from_HF=checkpoint is None,
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model)
    inference_state = processor.set_image(_pil_image(image_values))

    prompt_records = request.get("prompts", [])
    proposals = []
    seen_prompt_keys: set[tuple[str, int, str]] = set()
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )
    with autocast:
        for prompt_index, prompt in enumerate(prompt_records):
            candidate_id = str(prompt["candidate_id"])
            ensemble_index = int(prompt["ensemble_index"])
            prompt_role = str(prompt.get("prompt_role", "candidate"))
            prompt_key = (candidate_id, ensemble_index, prompt_role)
            if prompt_key in seen_prompt_keys:
                raise ValueError(
                    "Duplicate SAM3 candidate/ensemble prompt in request"
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
                [1] * len(positive) + [0] * len(negative), dtype=np.int32
            )
            if not len(points):
                points = None
                labels = None
            mask_input = (
                _mask_logits(component)
                if prompt.get("use_coarse_mask_prompt", False)
                else None
            )
            masks, quality_scores, _ = model.predict_inst(
                inference_state,
                point_coords=points,
                point_labels=labels,
                box=box,
                mask_input=mask_input,
                multimask_output=True,
            )
            candidate_masks = np.asarray(masks, dtype=bool)
            candidate_scores = np.asarray(
                quality_scores, dtype=float
            ).reshape(-1)
            if candidate_masks.ndim == 2:
                candidate_masks = candidate_masks[None, :, :]
            if not len(candidate_masks):
                raise RuntimeError("SAM3 returned no interactive masks")
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
                ranking.append((0.65 * core_recall + 0.35 * model_score, index))
            _, selected_index = max(ranking)
            selected = candidate_masks[selected_index]
            filename = (
                f"proposal_candidate_{prompt_index:04d}_"
                f"ensemble_{ensemble_index:03d}.npy"
            )
            path = directory / filename
            _write_array_exclusive(path, selected.astype(bool))
            proposals.append(
                {
                    "proposal_id": (
                        f"sam3-candidate-{prompt_index:04d}-"
                        f"ensemble-{ensemble_index:03d}"
                    ),
                    "candidate_id": candidate_id,
                    "component_label": component_label,
                    "ensemble_index": ensemble_index,
                    "prompt_role": prompt_role,
                    "mask_file": filename,
                    "sha256": _file_sha256(path),
                    "score": float(candidate_scores[selected_index]),
                }
            )

    response = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request["request_id"],
        "proposals": proposals,
        "worker": {
            "backend": "sam3",
            "model_variant": model_variant,
            "device": device,
            "checkpoint_source": (
                str(checkpoint) if checkpoint else "huggingface_gated_default"
            ),
            "sam3_version": getattr(sam3, "__version__", "unknown"),
            "torch_version": torch.__version__,
            "interactive_api": "model.predict_inst",
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
        description="Run isolated SAM3 boundary proposal inference"
    )
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    print(json.dumps(run(Path(args.request)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
