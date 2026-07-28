"""Quantitative-safe image loading and preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple
from zipfile import BadZipFile, ZipFile

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread

from .config import PreprocessConfig


def _nested_value(values: Dict[str, Any], *path: str) -> Any:
    current: Any = values
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decode_emd_metadata_frame(dataset: Any, frame_index: int) -> Dict[str, Any]:
    if dataset.ndim == 2:
        column = np.asarray(dataset[:, frame_index], dtype=np.uint8)
    else:
        column = np.asarray(dataset[...], dtype=np.uint8)
    raw = column.tobytes().rstrip(b"\x00")
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("EMD metadata frame must decode to a JSON object")
    return decoded


def _summarize_emd_metadata(
    archive: Any,
    data_path: str,
    frame_count: int,
) -> Dict[str, Any]:
    metadata_path = f"{data_path.rsplit('/', 1)[0]}/Metadata"
    if metadata_path not in archive:
        return {"metadata_status": "absent"}
    metadata = archive[metadata_path]
    try:
        first = _decode_emd_metadata_frame(metadata, 0)
        last_index = (
            min(max(frame_count - 1, 0), metadata.shape[1] - 1)
            if metadata.ndim == 2
            else 0
        )
        last = _decode_emd_metadata_frame(metadata, last_index)
    except (IndexError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {"metadata_status": "present_unparsed"}

    binary = first.get("BinaryResult", {})
    pixel_size = (
        binary.get("PixelSize", {}) if isinstance(binary, dict) else {}
    )
    summary: Dict[str, Any] = {
        "metadata_status": "parsed",
        "detector_name": _nested_value(first, "BinaryResult", "Detector"),
        "detector_index": _nested_value(
            first, "BinaryResult", "DetectorIndex"
        ),
        "instrument_model": _nested_value(
            first, "Instrument", "InstrumentModel"
        ),
        "acceleration_voltage_v": _as_float(
            _nested_value(first, "Optics", "AccelerationVoltage")
        ),
        "illumination_mode": _nested_value(
            first, "Optics", "IlluminationMode"
        ),
        "camera_length_m": _as_float(
            _nested_value(first, "Optics", "CameraLength")
        ),
        "pixel_size_x": _as_float(pixel_size.get("width")),
        "pixel_size_y": _as_float(pixel_size.get("height")),
        "pixel_unit_x": binary.get("PixelUnitX"),
        "pixel_unit_y": binary.get("PixelUnitY"),
        "metadata_frame_count": (
            int(metadata.shape[1]) if metadata.ndim == 2 else 1
        ),
    }
    last_detector = _nested_value(last, "BinaryResult", "Detector")
    summary["detector_stable_across_stack"] = (
        summary["detector_name"] == last_detector
    )
    return {
        key: value for key, value in summary.items() if value is not None
    }


def list_emd_datasets(path: str | Path) -> list[Dict[str, Any]]:
    """List numeric Velox/EMD image datasets without loading full arrays."""

    image_path = Path(path)
    if image_path.suffix.lower() != ".emd":
        raise ValueError(f"Expected an .emd file: {image_path}")
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Reading EMD files requires h5py") from error
    candidates: list[Dict[str, Any]] = []
    with h5py.File(image_path, "r") as archive:
        def visit(name: str, value: Any) -> None:
            if not isinstance(value, h5py.Dataset):
                return
            if (
                name.startswith("Data/Image/")
                and name.endswith("/Data")
                and value.ndim in {2, 3}
                and min(value.shape[:2]) >= 16
                and np.issubdtype(value.dtype, np.number)
            ):
                frame_count = int(value.shape[2]) if value.ndim == 3 else 1
                candidates.append(
                    {
                        "dataset_path": name,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "bytes": int(value.size * value.dtype.itemsize),
                        "frame_axis": 2 if value.ndim == 3 else None,
                        **_summarize_emd_metadata(
                            archive, name, frame_count
                        ),
                    }
                )

        archive.visititems(visit)
    candidates.sort(key=lambda item: item["dataset_path"])
    return candidates


def _load_emd(
    path: Path, dataset_path: str | None
) -> tuple[np.ndarray, int | None]:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Reading EMD files requires h5py") from error
    candidates = list_emd_datasets(path)
    if not candidates:
        raise ValueError(f"No numeric Data/Image/*/Data dataset found: {path}")
    if dataset_path is None:
        if len(candidates) != 1:
            names = ", ".join(item["dataset_path"] for item in candidates)
            raise ValueError(
                "EMD contains multiple image datasets; choose dataset_path "
                f"explicitly. Available: {names}"
            )
        selected = str(candidates[0]["dataset_path"])
    else:
        selected = dataset_path.lstrip("/")
        known = {str(item["dataset_path"]) for item in candidates}
        if selected not in known:
            raise ValueError(
                f"Unknown EMD dataset_path {dataset_path!r}; available: "
                + ", ".join(sorted(known))
            )
    with h5py.File(path, "r") as archive:
        image = np.asarray(archive[selected])
    frame_axis = 2 if image.ndim == 3 else None
    return image, frame_axis


def load_microscopy_stack(
    path: str | Path,
    *,
    dataset_path: str | None = None,
) -> np.ndarray:
    """Load a declared microscopy stack as ``(frame, y, x)``.

    This API never chooses an EMD detector channel implicitly.
    """

    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix == ".ndata1":
        try:
            with ZipFile(image_path) as archive:
                candidates = [
                    name
                    for name in archive.namelist()
                    if name == "data.npy" or name.endswith("/data.npy")
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "NData 1 archive must contain exactly one data.npy"
                    )
                with archive.open(candidates[0]) as stream:
                    image = np.asarray(np.load(stream, allow_pickle=False))
        except BadZipFile as error:
            raise ValueError(
                f"Invalid NData 1 archive: {image_path}"
            ) from error
        frame_axis = 0
    elif suffix == ".emd":
        image, frame_axis = _load_emd(image_path, dataset_path)
        if frame_axis is None:
            raise ValueError(f"EMD dataset is not a stack: {image_path}")
    else:
        raise ValueError(
            "Temporal stack analysis currently supports .ndata1 and .emd"
        )
    if image.ndim != 3:
        raise ValueError(
            f"Expected a 3-D microscopy stack; got shape {image.shape}"
        )
    stack = np.moveaxis(image, frame_axis, 0)
    if stack.shape[0] < 3 or min(stack.shape[1:]) < 16:
        raise ValueError(
            f"Microscopy stack is too small for temporal analysis: "
            f"{stack.shape}"
        )
    if not np.all(np.isfinite(stack)):
        raise ValueError("Microscopy stack contains NaN or infinite values")
    return stack


def _select_microscopy_plane(
    image: np.ndarray,
    *,
    frame_axis: int,
    frame_index: int | None,
    projection: str | None,
    source: Path,
) -> np.ndarray:
    if frame_index is not None and projection is not None:
        raise ValueError("Choose either frame_index or projection, not both")
    if frame_index is None and projection is None:
        raise ValueError(
            f"Microscopy stack {source} has shape {image.shape}; choose "
            "frame_index or projection explicitly"
        )
    if frame_index is not None:
        count = image.shape[frame_axis]
        if not -count <= frame_index < count:
            raise IndexError(
                f"frame_index {frame_index} is outside stack length {count}"
            )
        return np.take(image, frame_index, axis=frame_axis)
    if projection not in {"mean", "median", "max"}:
        raise ValueError("projection must be one of: mean, median, max")
    if projection == "mean":
        return np.mean(image, axis=frame_axis)
    if projection == "median":
        return np.median(image, axis=frame_axis)
    return np.max(image, axis=frame_axis)


def load_grayscale(
    path: str | Path,
    *,
    frame_index: int | None = None,
    projection: str | None = None,
    dataset_path: str | None = None,
) -> np.ndarray:
    """Load one explicit 2-D microscopy plane without silent stack guessing."""

    image_path = Path(path)
    frame_axis: int | None = None
    if image_path.suffix.lower() == ".ndata1":
        try:
            with ZipFile(image_path) as archive:
                candidates = [
                    name
                    for name in archive.namelist()
                    if name == "data.npy" or name.endswith("/data.npy")
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "NData 1 archive must contain exactly one data.npy"
                    )
                with archive.open(candidates[0]) as stream:
                    image = np.asarray(np.load(stream, allow_pickle=False))
        except BadZipFile as error:
            raise ValueError(f"Invalid NData 1 archive: {image_path}") from error
        if image.ndim == 3:
            frame_axis = 0
    elif image_path.suffix.lower() == ".emd":
        image, frame_axis = _load_emd(image_path, dataset_path)
    else:
        if dataset_path is not None:
            raise ValueError("dataset_path is only valid for EMD inputs")
        image = np.asarray(imread(str(image_path)))
    if frame_axis is not None:
        image = _select_microscopy_plane(
            image,
            frame_axis=frame_axis,
            frame_index=frame_index,
            projection=projection,
            source=image_path,
        )
    elif frame_index is not None or projection is not None:
        raise ValueError(
            "frame_index/projection were provided for a non-stack input"
        )
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB/RGBA image or explicit microscopy stack "
                f"selection; got shape {image.shape}"
            )
        image = np.tensordot(
            image.astype(np.float64), np.asarray([0.2126, 0.7152, 0.0722]), axes=([-1], [0])
        )
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image; got shape {image.shape}")
    if min(image.shape) < 16:
        raise ValueError(f"Image is too small for structural analysis: {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("Input image contains NaN or infinite values")
    return image


def robust_normalize(
    image: np.ndarray, config: PreprocessConfig
) -> Tuple[np.ndarray, dict]:
    """Return a robust [0, 1] view while recording the original scale."""

    values = image.astype(np.float64, copy=False)
    low, high = np.percentile(
        values, [config.lower_percentile, config.upper_percentile]
    )
    if high <= low:
        raise ValueError("Input image has no usable intensity range")
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    if config.background_sigma > 0:
        background = gaussian_filter(normalized, config.background_sigma)
        residual = normalized - background
        r_low, r_high = np.percentile(residual, [1.0, 99.0])
        normalized = np.clip((residual - r_low) / max(r_high - r_low, 1e-12), 0, 1)
    metadata = {
        "input_dtype": str(image.dtype),
        "input_min": float(np.min(values)),
        "input_max": float(np.max(values)),
        "normalization_low": float(low),
        "normalization_high": float(high),
    }
    return normalized.astype(np.float32), metadata
