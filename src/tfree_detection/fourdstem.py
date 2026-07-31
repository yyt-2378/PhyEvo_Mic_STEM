"""Strict, provenance-preserving adapters for published 4D-STEM cubes.

The adapter intentionally supports the explicit HDF5 schema used by the
Werbrouck et al. fast phase-mapping archive.  It does not infer axis meanings
from shape alone: the required coordinate datasets and the author-documented
``(real-x, real-y, reciprocal-x, reciprocal-y)`` ordering must be present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


FAST_MAPPING_REQUIRED_PATHS = (
    "X4D",
    "coordinates/x_real",
    "coordinates/y_real",
    "coordinates/x_reciprocal",
    "coordinates/y_reciprocal",
    "coordinates/beamstop",
    "metadata",
)


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "4D-STEM HDF5 support requires the optional 'fourdstem' "
            "dependency (h5py)"
        ) from error
    return h5py


def _decode_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {"value": decoded}
    return {"value_type": type(value).__name__}


def inspect_fast_mapping_h5(path: str | Path) -> Dict[str, Any]:
    """Inspect the published schema without loading the 4D intensity cube."""

    h5py = _h5py()
    source = Path(path)
    with h5py.File(source, "r") as archive:
        missing = [
            dataset_path
            for dataset_path in FAST_MAPPING_REQUIRED_PATHS
            if dataset_path not in archive
        ]
        if missing:
            raise ValueError(
                "HDF5 file does not match the published fast-mapping schema; "
                f"missing: {', '.join(missing)}"
            )
        cube = archive["X4D"]
        if cube.ndim != 4 or not np.issubdtype(cube.dtype, np.number):
            raise ValueError(
                "X4D must be a numeric four-dimensional dataset in "
                "(real-x, real-y, reciprocal-x, reciprocal-y) order"
            )
        if min(cube.shape) < 2:
            raise ValueError(f"X4D dimensions are too small: {cube.shape}")
        coordinate_shapes = {
            name.rsplit("/", 1)[-1]: list(archive[name].shape)
            for name in FAST_MAPPING_REQUIRED_PATHS
            if name.startswith("coordinates/")
        }
        beamstop_shape = tuple(archive["coordinates/beamstop"].shape)
        reciprocal_shape = tuple(int(value) for value in cube.shape[2:])
        if beamstop_shape not in {reciprocal_shape, ()}:
            raise ValueError(
                "Beamstop metadata must be scalar or match the reciprocal "
                f"plane {reciprocal_shape}; observed {beamstop_shape}"
            )
        metadata = _decode_metadata(archive["metadata"][()])
        return {
            "inspection_status": "ok",
            "format": "hdf5_4dstem",
            "schema": "werbrouck_fast_mapping_v1",
            "cube_dataset": "X4D",
            "shape": [int(value) for value in cube.shape],
            "dtype": str(cube.dtype),
            "bytes_uncompressed": int(cube.size * cube.dtype.itemsize),
            "axis_semantics": [
                "real_x",
                "real_y",
                "reciprocal_x",
                "reciprocal_y",
            ],
            "scan_shape": [int(value) for value in cube.shape[:2]],
            "diffraction_shape": [
                int(value) for value in cube.shape[2:]
            ],
            "coordinate_shapes": coordinate_shapes,
            "metadata_keys": sorted(str(key) for key in metadata),
        }


def _reciprocal_signature_masks(
    shape: tuple[int, int],
    *,
    radial_bins: int,
    angular_bins: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    yy, xx = np.indices(shape, dtype=np.float64)
    yy -= (shape[0] - 1) / 2.0
    xx -= (shape[1] - 1) / 2.0
    radius = np.hypot(yy, xx)
    radius /= max(float(np.max(radius)), 1.0)
    angle = np.mod(np.arctan2(yy, xx), 2.0 * np.pi)
    radial_edges = np.linspace(0.0, 1.0 + 1e-9, radial_bins + 1)
    angular_edges = np.linspace(
        0.0, 2.0 * np.pi + 1e-9, angular_bins + 1
    )
    masks = []
    for radial_index in range(radial_bins):
        radial = (
            (radius >= radial_edges[radial_index])
            & (radius < radial_edges[radial_index + 1])
        )
        for angular_index in range(angular_bins):
            masks.append(
                radial
                & (angle >= angular_edges[angular_index])
                & (angle < angular_edges[angular_index + 1])
            )
    return masks, radius


def derive_fast_mapping_maps(
    path: str | Path,
    *,
    chunk_rows: int = 2,
    radial_bins: int = 4,
    angular_bins: int = 12,
) -> Dict[str, np.ndarray]:
    """Derive deterministic virtual-detector and reciprocal-order maps.

    The returned maps preserve the real-space scan grid.  No external labels,
    learned weights, or per-dataset threshold fitting are used.  The
    ``signature_pc1`` map is the first SVD coordinate of fixed polar-bin
    diffraction signatures; its sign is fixed from its largest loading.
    """

    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if radial_bins < 2 or angular_bins < 4:
        raise ValueError(
            "At least two radial and four angular bins are required"
        )
    inspection = inspect_fast_mapping_h5(path)
    scan_shape = tuple(int(value) for value in inspection["scan_shape"])
    diffraction_shape = tuple(
        int(value) for value in inspection["diffraction_shape"]
    )
    masks, radius = _reciprocal_signature_masks(
        diffraction_shape,
        radial_bins=radial_bins,
        angular_bins=angular_bins,
    )
    total = np.zeros(scan_shape, dtype=np.float64)
    entropy = np.zeros(scan_shape, dtype=np.float64)
    radial_moment = np.zeros(scan_shape, dtype=np.float64)
    peak_fraction = np.zeros(scan_shape, dtype=np.float64)
    signatures = np.zeros(
        (scan_shape[0], scan_shape[1], len(masks)),
        dtype=np.float64,
    )
    h5py = _h5py()
    with h5py.File(path, "r") as archive:
        cube = archive["X4D"]
        for row_start in range(0, scan_shape[0], chunk_rows):
            row_stop = min(row_start + chunk_rows, scan_shape[0])
            block = np.asarray(
                cube[row_start:row_stop, :, :, :], dtype=np.float64
            )
            block = np.nan_to_num(
                block, nan=0.0, posinf=0.0, neginf=0.0
            )
            np.maximum(block, 0.0, out=block)
            block_total = np.sum(block, axis=(-2, -1))
            safe_total = np.maximum(block_total, 1e-12)
            total[row_start:row_stop] = block_total
            xlogx = np.zeros_like(block)
            positive = block > 0
            xlogx[positive] = (
                block[positive] * np.log(block[positive])
            )
            entropy_block = (
                np.log(safe_total)
                - np.sum(xlogx, axis=(-2, -1)) / safe_total
            )
            entropy[row_start:row_stop] = entropy_block / np.log(
                float(np.prod(diffraction_shape))
            )
            radial_moment[row_start:row_stop] = (
                np.sum(block * radius, axis=(-2, -1)) / safe_total
            )
            peak_fraction[row_start:row_stop] = (
                np.max(block, axis=(-2, -1)) / safe_total
            )
            for index, mask in enumerate(masks):
                signatures[row_start:row_stop, :, index] = np.sum(
                    block[..., mask], axis=-1
                )

    signature_total = np.sum(signatures, axis=-1, keepdims=True)
    signatures /= np.maximum(signature_total, 1e-12)
    signatures = np.sqrt(signatures)
    flat = signatures.reshape(-1, signatures.shape[-1])
    median = np.median(flat, axis=0)
    scale = 1.4826 * np.median(np.abs(flat - median), axis=0)
    active = scale > 1e-8
    if int(np.sum(active)) < 2:
        pc1 = np.zeros(flat.shape[0], dtype=np.float64)
        anomaly = np.zeros(flat.shape[0], dtype=np.float64)
    else:
        standardized = np.clip(
            (flat[:, active] - median[active]) / scale[active],
            -8.0,
            8.0,
        )
        centered = standardized - np.median(standardized, axis=0)
        _, _, components = np.linalg.svd(centered, full_matrices=False)
        loading = components[0]
        anchor = int(np.argmax(np.abs(loading)))
        if loading[anchor] < 0:
            loading = -loading
        pc1 = centered @ loading
        anomaly = np.sqrt(np.mean(centered**2, axis=1))
    return {
        "integrated_intensity": total.astype(np.float32),
        "diffraction_entropy": entropy.astype(np.float32),
        "radial_first_moment": radial_moment.astype(np.float32),
        "diffraction_peak_fraction": peak_fraction.astype(np.float32),
        "signature_pc1": pc1.reshape(scan_shape).astype(np.float32),
        "signature_anomaly": anomaly.reshape(scan_shape).astype(np.float32),
    }
