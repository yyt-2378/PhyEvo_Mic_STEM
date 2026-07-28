"""Frozen DINO feature verification for candidate cores and transition rings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label

from ..config import DINOConfig, PhaseAnalysisConfig


@dataclass
class DINOAnalysisResult:
    metadata: Dict[str, Any]
    region_metrics: List[Dict[str, Any]]


def _load_original_dino_vits8(checkpoint: Path, device: str):
    import torch
    from transformers import ViTConfig, ViTModel

    config = ViTConfig(
        image_size=224,
        patch_size=8,
        num_channels=3,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=1536,
        qkv_bias=True,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-6,
    )
    model = ViTModel(config, add_pooling_layer=False)
    source = torch.load(str(checkpoint), map_location="cpu")
    if "state_dict" in source:
        source = source["state_dict"]
    source = {
        key.removeprefix("module.").removeprefix("backbone."): value
        for key, value in source.items()
    }
    converted = {
        "embeddings.cls_token": source["cls_token"],
        "embeddings.position_embeddings": source["pos_embed"],
        "embeddings.patch_embeddings.projection.weight": source[
            "patch_embed.proj.weight"
        ],
        "embeddings.patch_embeddings.projection.bias": source[
            "patch_embed.proj.bias"
        ],
        "layernorm.weight": source["norm.weight"],
        "layernorm.bias": source["norm.bias"],
    }
    hidden_size = config.hidden_size
    for index in range(config.num_hidden_layers):
        source_prefix = f"blocks.{index}"
        target_prefix = f"encoder.layer.{index}"
        converted[f"{target_prefix}.layernorm_before.weight"] = source[
            f"{source_prefix}.norm1.weight"
        ]
        converted[f"{target_prefix}.layernorm_before.bias"] = source[
            f"{source_prefix}.norm1.bias"
        ]
        converted[f"{target_prefix}.layernorm_after.weight"] = source[
            f"{source_prefix}.norm2.weight"
        ]
        converted[f"{target_prefix}.layernorm_after.bias"] = source[
            f"{source_prefix}.norm2.bias"
        ]
        qkv_weight = source[f"{source_prefix}.attn.qkv.weight"]
        qkv_bias = source[f"{source_prefix}.attn.qkv.bias"]
        for part, name in enumerate(("query", "key", "value")):
            start = part * hidden_size
            end = (part + 1) * hidden_size
            converted[
                f"{target_prefix}.attention.attention.{name}.weight"
            ] = qkv_weight[start:end]
            converted[
                f"{target_prefix}.attention.attention.{name}.bias"
            ] = qkv_bias[start:end]
        converted[f"{target_prefix}.attention.output.dense.weight"] = source[
            f"{source_prefix}.attn.proj.weight"
        ]
        converted[f"{target_prefix}.attention.output.dense.bias"] = source[
            f"{source_prefix}.attn.proj.bias"
        ]
        converted[f"{target_prefix}.intermediate.dense.weight"] = source[
            f"{source_prefix}.mlp.fc1.weight"
        ]
        converted[f"{target_prefix}.intermediate.dense.bias"] = source[
            f"{source_prefix}.mlp.fc1.bias"
        ]
        converted[f"{target_prefix}.output.dense.weight"] = source[
            f"{source_prefix}.mlp.fc2.weight"
        ]
        converted[f"{target_prefix}.output.dense.bias"] = source[
            f"{source_prefix}.mlp.fc2.bias"
        ]
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"DINO checkpoint conversion mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for DINO but is unavailable")
    return requested


def _resolve_model(config: DINOConfig, device: str):
    if config.local_model_directory:
        directory = Path(config.local_model_directory)
        if not directory.exists() and not directory.is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            directory = project_root / directory
        checkpoint = directory / "dino_deitsmall8_pretrain.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"DINO checkpoint not found: {checkpoint}. Run "
                "`python scripts/download_dino_weights.py` first."
            )
        return _load_original_dino_vits8(checkpoint, device), str(
            checkpoint.resolve()
        )
    from transformers import ViTModel

    model = ViTModel.from_pretrained(config.model_id)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config.model_id


def _sample_coordinates(
    zone: np.ndarray, count: int, rng: np.random.Generator
) -> List[Tuple[int, int]]:
    coordinates = np.argwhere(zone)
    if not len(coordinates):
        return []
    chosen = rng.choice(
        len(coordinates), size=min(count, len(coordinates)), replace=False
    )
    return [
        (int(coordinates[index, 0]), int(coordinates[index, 1]))
        for index in chosen
    ]


def _extract_patches(
    image: np.ndarray,
    coordinates: Sequence[Tuple[int, int]],
    patch_size: int,
) -> np.ndarray:
    half = patch_size // 2
    padded = np.pad(image, ((half, half), (half, half)), mode="reflect")
    patches = []
    for row, column in coordinates:
        patch = padded[row : row + patch_size, column : column + patch_size]
        lower, upper = np.percentile(patch, [1, 99])
        patch = np.clip((patch - lower) / max(upper - lower, 1e-12), 0, 1)
        patches.append(patch.astype(np.float32))
    return (
        np.stack(patches)
        if patches
        else np.empty((0, patch_size, patch_size), dtype=np.float32)
    )


def _embed(
    model,
    patches: np.ndarray,
    config: DINOConfig,
    device: str,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    if not len(patches):
        return np.empty((0, 384), dtype=np.float32)
    means = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    standard_deviations = torch.tensor([0.229, 0.224, 0.225])[
        None, :, None, None
    ]
    embeddings = []
    for start in range(0, len(patches), config.batch_size):
        batch = torch.from_numpy(
            patches[start : start + config.batch_size]
        )[:, None]
        batch = batch.repeat(1, 3, 1, 1)
        batch = functional.interpolate(
            batch,
            size=(config.input_size, config.input_size),
            mode="bilinear",
            align_corners=False,
        )
        batch = ((batch - means) / standard_deviations).to(device)
        with torch.inference_mode():
            features = model(pixel_values=batch).last_hidden_state[:, 0]
            features = functional.normalize(features, dim=1)
        embeddings.append(features.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _distance_summary(values: np.ndarray) -> Dict[str, float | int]:
    if not len(values):
        return {"count": 0, "median": float("nan"), "p90": float("nan")}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _distance_to_prototype(
    embeddings: np.ndarray, prototype: np.ndarray
) -> np.ndarray:
    return 1.0 - embeddings @ prototype


def verify_regions_with_dino(
    image: np.ndarray,
    candidate_mask: np.ndarray,
    phase_config: PhaseAnalysisConfig,
    config: DINOConfig,
    exemplar_image: Optional[np.ndarray] = None,
) -> DINOAnalysisResult:
    """Compare core/ring patches with local normal reference embeddings."""

    device = _resolve_device(config.device)
    model, model_source = _resolve_model(config, device)
    rng = np.random.default_rng(config.random_seed)
    candidate_labels = label(candidate_mask, connectivity=2)
    region_metrics: List[Dict[str, Any]] = []

    exemplar_embedding = None
    if exemplar_image is not None:
        center = (
            exemplar_image.shape[0] // 2,
            exemplar_image.shape[1] // 2,
        )
        exemplar_patches = _extract_patches(
            exemplar_image,
            [center],
            min(config.microscopy_patch_size, min(exemplar_image.shape)),
        )
        exemplar_embedding = _embed(
            model, exemplar_patches, config, device
        )[0]

    for region_id in range(1, int(np.max(candidate_labels)) + 1):
        core = candidate_labels == region_id
        distance = distance_transform_edt(~core)
        transition = (
            (distance > phase_config.transition_inner_radius)
            & (distance <= phase_config.transition_outer_radius)
            & ~candidate_mask
        )
        reference = (
            (distance > phase_config.reference_inner_radius)
            & (distance <= phase_config.reference_outer_radius)
            & ~candidate_mask
        )
        zones = {
            "core": core,
            "transition": transition,
            "reference": reference,
        }
        coordinates = {
            name: _sample_coordinates(
                zone, config.samples_per_zone, rng
            )
            for name, zone in zones.items()
        }
        embeddings = {
            name: _embed(
                model,
                _extract_patches(
                    image, points, config.microscopy_patch_size
                ),
                config,
                device,
            )
            for name, points in coordinates.items()
        }
        reference_embedding = embeddings["reference"]
        if not len(reference_embedding):
            continue
        prototype = np.median(reference_embedding, axis=0)
        prototype /= np.linalg.norm(prototype) + 1e-12
        distances = {
            name: _distance_to_prototype(values, prototype)
            for name, values in embeddings.items()
        }
        reference_center = float(np.median(distances["reference"]))
        reference_scale = float(
            1.4826
            * np.median(
                np.abs(distances["reference"] - reference_center)
            )
        )
        # Very small local reference sets can have nearly identical cosine
        # distances. A physical floor prevents meaningless enormous z-scores.
        reference_scale = max(reference_scale, 0.01)
        core_z = (
            float(np.median(distances["core"])) - reference_center
        ) / reference_scale
        transition_z = (
            float(np.median(distances["transition"])) - reference_center
        ) / reference_scale
        if core_z >= 3 or transition_z >= 3:
            dino_support = "strong"
        elif core_z >= 1.5 or transition_z >= 1.5:
            dino_support = "moderate"
        else:
            dino_support = "weak"
        exemplar_similarity = None
        if exemplar_embedding is not None and len(embeddings["core"]):
            exemplar_similarity = float(
                np.max(embeddings["core"] @ exemplar_embedding)
            )
        region_metrics.append(
            {
                "region_id": region_id,
                "dino_support": dino_support,
                "cosine_distance": {
                    name: _distance_summary(values)
                    for name, values in distances.items()
                },
                "core_robust_z_vs_reference": core_z,
                "transition_robust_z_vs_reference": transition_z,
                "sample_coordinates_yx": {
                    name: [[row, column] for row, column in points]
                    for name, points in coordinates.items()
                },
                "maximum_similarity_to_exemplar": exemplar_similarity,
            }
        )

    return DINOAnalysisResult(
        metadata={
            "model": "DINO ViT-S/8",
            "model_source": model_source,
            "weights_frozen": True,
            "training_or_fitting_used": False,
            "device": device,
            "embedding_dimension": 384,
            "microscopy_patch_size": config.microscopy_patch_size,
            "input_size": config.input_size,
            "cosine_distance_scale_floor": 0.01,
        },
        region_metrics=region_metrics,
    )
