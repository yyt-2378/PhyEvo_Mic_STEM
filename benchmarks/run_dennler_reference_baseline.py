#!/usr/bin/env python
"""Reproduce the released Dennler Fourier baseline on the public archive.

This runner implements the training-free Bragg-filtering branch released with
Dennler et al. (Micron 146, 103069, 2021) and evaluates its masks with the same
pixel-level definitions used by this repository.  The published VGG16-guided
eigenfilter method is deliberately not represented here: the official source
repository does not contain trained checkpoints, and reproducing it would
therefore require a separate retraining study.

The original implementation is MIT licensed and available at
https://github.com/nkdnnlr/TEMDefectClassification (commit pinned below).
Compatibility changes here are limited to current SciPy/scikit-image APIs,
numerically safe normalization, in-memory TIFF decoding, and silent execution.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any, Iterable
from zipfile import ZipFile

import numpy as np
from scipy import fftpack, signal
from scipy.signal import medfilt2d
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    PROJECT_ROOT
    / "datasets/public/dennler_2021_hrstem_defects/"
    "hrstem_defects_dataset.zip"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "benchmarks/paper_baselines/dennler_fourier_reference_v1.json"
)
EXPECTED_ARCHIVE_SHA256 = (
    "abef46a0183ba08b925c64807d5e914cd97967444d9fd3e0799924896f0614ec"
)
UPSTREAM_REPOSITORY = "https://github.com/nkdnnlr/TEMDefectClassification"
UPSTREAM_COMMIT = "72d37c3ca5a84cfa39337c7243a80e3ffcc0f648"


@dataclass(frozen=True)
class Sample:
    """One image and its pixel labels from the public Dennler archive."""

    filename: str
    category: str
    image: np.ndarray
    truth: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("Output must stay inside the project") from error
    return resolved


def _decode_grayscale_tiff(payload: bytes) -> np.ndarray:
    """Decode a TIFF using the grayscale convention of OpenCV IMREAD_GRAYSCALE."""

    try:
        import tifffile
    except ImportError as error:  # pragma: no cover - environment guidance
        raise RuntimeError(
            "This benchmark requires the optional 'tifffile' dependency"
        ) from error
    image = np.asarray(tifffile.imread(BytesIO(payload)))
    if image.ndim == 3 and image.shape[-1] in {3, 4}:
        rgb = image[..., :3].astype(np.float64)
        image = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if image.ndim != 2:
        raise ValueError(f"Expected a two-dimensional image, got {image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("Image contains non-finite values")
    return np.rint(image).astype(np.uint8)


def _decode_label(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
    label_image = _decode_grayscale_tiff(payload)
    if label_image.shape != shape:
        raise ValueError(f"Label shape {label_image.shape} != image shape {shape}")
    raw_values = np.unique(label_image)
    if not set(raw_values.tolist()).issubset({0, 127, 255}):
        raise ValueError(f"Unexpected Dennler label values: {raw_values.tolist()}")
    output = np.zeros(shape, dtype=np.uint8)
    output[label_image == 127] = 1
    output[label_image == 255] = 2
    return output


def _iter_samples(archive_path: Path) -> Iterable[Sample]:
    image_prefixes = (
        "hrstem_defects_dataset/defective/images/",
        "hrstem_defects_dataset/non_defective/images/",
    )
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        image_names = sorted(
            name
            for name in names
            if name.lower().endswith((".tif", ".tiff"))
            and name.startswith(image_prefixes)
        )
        if len(image_names) != 61:
            raise ValueError(f"Expected 61 STEM images, found {len(image_names)}")
        for image_name in image_names:
            image = _decode_grayscale_tiff(archive.read(image_name))
            defective = "/defective/images/" in image_name
            if defective:
                label_name = image_name.replace("/images/", "/labels/")
                if label_name not in names:
                    raise FileNotFoundError(f"Missing label for {image_name}")
                truth = _decode_label(archive.read(label_name), image.shape)
                category = "defective"
            else:
                truth = np.zeros(image.shape, dtype=np.uint8)
                category = "normal"
            yield Sample(
                filename=Path(image_name).name,
                category=category,
                image=image,
                truth=truth,
            )


def _gaussian_kernel(length: int = 9, sigma: float = 1.0) -> np.ndarray:
    gaussian = signal.windows.gaussian(length, std=sigma).reshape(length, 1)
    return np.outer(gaussian, gaussian)


def _centered_crop(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if values.shape == shape:
        return values
    extra_y = values.shape[0] - shape[0]
    extra_x = values.shape[1] - shape[1]
    if extra_y < 0 or extra_x < 0:
        raise ValueError(f"Cannot crop {values.shape} to {shape}")
    start_y = extra_y // 2
    start_x = extra_x // 2
    return values[start_y : start_y + shape[0], start_x : start_x + shape[1]]


def _peak_map(
    image: np.ndarray,
    *,
    kernel_size: int = 9,
    sigma: float = 1.0,
    cut_edge: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fourier = fftpack.fftshift(fftpack.fft2(image))
    power = np.maximum(np.abs(fourier) ** 2, np.finfo(np.float64).tiny)
    log_power = np.log(power)
    background = medfilt2d(log_power, kernel_size=25)
    difference = log_power - background
    difference[difference < 2 * background.std()] = 0
    kernel = _gaussian_kernel(kernel_size, sigma)
    peaks = signal.convolve2d(difference, kernel, mode="full")
    peaks = _centered_crop(peaks, image.shape)
    edge_y = min(cut_edge, max(image.shape[0] // 4, 1))
    edge_x = min(cut_edge, max(image.shape[1] // 4, 1))
    peaks[:edge_y, :] = 0
    peaks[-edge_y:, :] = 0
    peaks[:, :edge_x] = 0
    peaks[:, -edge_x:] = 0
    return fourier, peaks, kernel


def _circular_mask(shape: tuple[int, int], radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy = shape[0] / 2
    cx = shape[1] / 2
    return (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2


def bragg_filter_amorphous(image: np.ndarray, radius: int = 50) -> np.ndarray:
    """Published low-frequency Bragg-filter branch for class-2 regions."""

    fourier, peaks, _ = _peak_map(image)
    return _amorphous_from_peak_map(fourier, peaks, radius=radius)


def _amorphous_from_peak_map(
    fourier: np.ndarray,
    peaks: np.ndarray,
    *,
    radius: int = 50,
) -> np.ndarray:
    filtered = fourier * peaks * _circular_mask(fourier.shape, radius)
    inverse = np.abs(fftpack.ifft2(filtered)) ** 2
    inverse -= inverse.min()
    return inverse > 20_000_000


def _clear_border(values: np.ndarray, width: int) -> np.ndarray:
    output = values.copy()
    output[:, :width] = False
    output[:, -width:] = False
    output[:width, :] = False
    output[-width:, :] = False
    return output


def _isolate_bragg_peaks(
    filtered_fourier: np.ndarray,
    *,
    minimum_area: int = 50,
) -> list[np.ndarray]:
    half_spectrum = fftpack.fftshift(filtered_fourier).copy()
    half_spectrum[filtered_fourier.shape[0] // 2 :, :] = 0
    display = np.real(
        np.sqrt(fftpack.fftshift(half_spectrum.copy()) ** 2)
    ).astype(np.uint8)
    if np.unique(display).size < 2:
        return []
    binary = closing(display > threshold_otsu(display), square(3))
    components = label(_clear_border(binary, 20))
    spots: list[tuple[float, np.ndarray]] = []
    for region in regionprops(components):
        if region.area <= minimum_area:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        isolated = fftpack.fftshift(half_spectrum.copy())
        isolated[:min_row, :] = 0
        isolated[max_row:, :] = 0
        isolated[:, :min_col] = 0
        isolated[:, max_col:] = 0
        spots.append((float(region.area), isolated))
    return [spot for _, spot in sorted(spots, key=lambda item: item[0], reverse=True)]


def bragg_filter_symmetry(
    image: np.ndarray,
    *,
    number_of_peaks: int = 6,
    low_threshold: float = 0.05,
    sum_threshold: float = 0.5,
) -> np.ndarray:
    """Published Bragg-spot reconstruction branch for crystalline symmetry."""

    fourier, peaks, kernel = _peak_map(image)
    return _symmetry_from_peak_map(
        fourier,
        peaks,
        kernel,
        number_of_peaks=number_of_peaks,
        low_threshold=low_threshold,
        sum_threshold=sum_threshold,
    )


def _symmetry_from_peak_map(
    fourier: np.ndarray,
    peaks: np.ndarray,
    kernel: np.ndarray,
    *,
    number_of_peaks: int = 6,
    low_threshold: float = 0.05,
    sum_threshold: float = 0.5,
) -> np.ndarray:
    without_centre = peaks.copy()
    centre_y = peaks.shape[0] // 2
    centre_x = peaks.shape[1] // 2
    without_centre[
        centre_y - 10 : centre_y + 10,
        centre_x - 10 : centre_x + 10,
    ] = 0
    maximum = float(np.max(without_centre))
    if maximum <= 0:
        return np.zeros(fourier.shape, dtype=bool)
    binary_peaks = (peaks >= 0.5 * maximum).astype(np.float64)
    smoother_peaks = signal.convolve2d(binary_peaks, kernel, mode="full")
    smoother_peaks = _centered_crop(smoother_peaks, fourier.shape)
    isolated_peaks = _isolate_bragg_peaks(fourier * smoother_peaks)
    final = np.zeros(fourier.shape, dtype=np.float64)
    for isolated in isolated_peaks[:number_of_peaks]:
        reconstruction = np.abs(fftpack.ifft2(isolated)) ** 2
        reconstruction -= reconstruction.min()
        peak = float(reconstruction.max())
        if peak <= 0:
            continue
        reconstruction /= peak
        reconstruction[reconstruction < low_threshold] = 0
        final += reconstruction
    maximum = float(final.max())
    if maximum <= 0:
        return np.zeros(fourier.shape, dtype=bool)
    final /= maximum
    return final > sum_threshold


def predict_fourier(image: np.ndarray) -> np.ndarray:
    """Return the original three-class Fourier mask (0 normal, 1, or 2)."""

    fourier, peaks, kernel = _peak_map(image)
    class_2 = _amorphous_from_peak_map(fourier, peaks)
    class_1 = ~_symmetry_from_peak_map(fourier, peaks, kernel)
    prediction = np.zeros(image.shape, dtype=np.uint8)
    prediction[class_1] = 1
    prediction[class_2] = 2
    return prediction


def _predict_timed(sample: Sample) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    prediction = predict_fourier(sample.image)
    return prediction, time.perf_counter() - started


def _binary_counts(prediction: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    predicted = np.asarray(prediction, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    return {
        "true_positive_pixels": int(np.sum(predicted & target)),
        "false_positive_pixels": int(np.sum(predicted & ~target)),
        "false_negative_pixels": int(np.sum(~predicted & target)),
        "true_negative_pixels": int(np.sum(~predicted & ~target)),
    }


def _counts_to_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    true_positive = counts["true_positive_pixels"]
    false_positive = counts["false_positive_pixels"]
    false_negative = counts["false_negative_pixels"]
    true_negative = counts["true_negative_pixels"]
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    iou = true_positive / max(
        true_positive + false_positive + false_negative, 1
    )
    dice = 2 * true_positive / max(
        2 * true_positive + false_positive + false_negative, 1
    )
    specificity = true_negative / max(true_negative + false_positive, 1)
    return {
        **counts,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": dice,
        "specificity": specificity,
    }


def _add_counts(total: dict[str, int], values: dict[str, int]) -> None:
    for key, value in values.items():
        total[key] += value


def evaluate(
    archive_path: Path,
    *,
    paper_crop_pixels: int = 12,
    workers: int = 1,
) -> dict[str, Any]:
    target_specs = {
        "class_1_secondary_symmetry": ({1}, {1}),
        "class_2_blurred_or_amorphous": ({2}, {2}),
        "any_defect": ({1, 2}, {1, 2}),
    }
    protocols = {
        "full_field": 0,
        "published_evaluation_crop": paper_crop_pixels,
    }
    aggregate = {
        protocol: {
            target: {
                "true_positive_pixels": 0,
                "false_positive_pixels": 0,
                "false_negative_pixels": 0,
                "true_negative_pixels": 0,
            }
            for target in target_specs
        }
        for protocol in protocols
    }
    normal_fractions: dict[str, dict[str, list[float]]] = {
        protocol: {target: [] for target in target_specs}
        for protocol in protocols
    }
    per_image: list[dict[str, Any]] = []
    samples = list(_iter_samples(archive_path))
    started = time.perf_counter()
    if workers == 1:
        predictions = map(_predict_timed, samples)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        predictions = executor.map(_predict_timed, samples)
    for index, (sample, predicted) in enumerate(
        zip(samples, predictions), start=1
    ):
        prediction, sample_runtime = predicted
        image_record: dict[str, Any] = {
            "index": index,
            "filename": sample.filename,
            "category": sample.category,
            "runtime_seconds": sample_runtime,
            "positive_fractions": {},
        }
        for protocol, crop in protocols.items():
            if crop:
                selection = np.s_[crop:-crop, crop:-crop]
                evaluated_prediction = prediction[selection]
                evaluated_truth = sample.truth[selection]
            else:
                evaluated_prediction = prediction
                evaluated_truth = sample.truth
            image_record["positive_fractions"][protocol] = {}
            for target, (truth_classes, predicted_classes) in target_specs.items():
                truth_mask = np.isin(evaluated_truth, list(truth_classes))
                predicted_mask = np.isin(
                    evaluated_prediction, list(predicted_classes)
                )
                counts = _binary_counts(predicted_mask, truth_mask)
                _add_counts(aggregate[protocol][target], counts)
                positive_fraction = float(np.mean(predicted_mask))
                image_record["positive_fractions"][protocol][target] = (
                    positive_fraction
                )
                if sample.category == "normal":
                    normal_fractions[protocol][target].append(positive_fraction)
        per_image.append(image_record)
        print(
            f"[{index:02d}/61] {sample.category:9s} {sample.filename} "
            f"({image_record['runtime_seconds']:.2f} s)",
            flush=True,
        )
    if executor is not None:
        executor.shutdown()
    scored: dict[str, Any] = {}
    for protocol in protocols:
        scored[protocol] = {}
        for target in target_specs:
            controls = np.asarray(normal_fractions[protocol][target])
            scored[protocol][target] = {
                "micro_pixel_metrics": _counts_to_metrics(
                    aggregate[protocol][target]
                ),
                "normal_controls": {
                    "image_count": int(controls.size),
                    "mean_false_positive_area_fraction": float(
                        np.mean(controls)
                    ),
                    "p95_false_positive_area_fraction": float(
                        np.percentile(controls, 95)
                    ),
                    "maximum_false_positive_area_fraction": float(
                        np.max(controls)
                    ),
                },
            }
    return {
        "format": "phyevo-mic-reference-baseline-v1",
        "scientific_status": "same-data_unified-evaluator_reproduction",
        "baseline": {
            "name": "Dennler released Fourier/Bragg segmentation",
            "training_required": False,
            "weights_required": False,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_license": "MIT",
            "compatibility_changes": [
                "current SciPy Gaussian-window API",
                "numerically safe logarithm and normalization",
                "in-memory grayscale TIFF decoding",
                "silent Bragg-spot isolation",
            ],
        },
        "dataset": {
            "archive": str(archive_path.relative_to(PROJECT_ROOT)),
            "archive_sha256": _sha256_file(archive_path),
            "image_count": len(per_image),
            "defective_image_count": sum(
                record["category"] == "defective" for record in per_image
            ),
            "normal_image_count": sum(
                record["category"] == "normal" for record in per_image
            ),
            "label_semantics": {
                "0": "main crystal symmetry / non-defective",
                "1": "secondary crystal symmetry defect",
                "2": "blurred, amorphous, or beam-damaged region",
            },
        },
        "protocols": {
            "full_field": {"border_crop_pixels": 0},
            "published_evaluation_crop": {
                "border_crop_pixels": paper_crop_pixels
            },
        },
        "targets": scored,
        "runtime_seconds": time.perf_counter() - started,
        "worker_processes": workers,
        "per_image": per_image,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paper-crop-pixels", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    archive = args.archive.resolve()
    output = _inside_project(args.output)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if archive == DEFAULT_ARCHIVE.resolve():
        archive_sha256 = _sha256_file(archive)
        if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
            raise ValueError(
                "Dennler archive checksum changed: " + archive_sha256
            )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.paper_crop_pixels < 0:
        raise ValueError("paper crop must be non-negative")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    result = evaluate(
        archive,
        paper_crop_pixels=args.paper_crop_pixels,
        workers=args.workers,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
