"""Pre-registered, two-stage evaluation for prospectively locked datasets.

The prediction stage is deliberately unable to read label directories.  It
creates a checksum-sealed prediction set.  The scoring stage first records
which sealed prediction set is being unblinded, then reads labels.  Once an
asset has been unblinded, the ledger rejects predictions with any other hash.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml
from skimage.io import imread

from . import __version__
from .config import load_config
from .evaluation import boundary_metrics
from .framework import TrainingFreeDetectionFramework
from .governance import DataPartition, DataRegistry


PREDICTION_FORMAT = "phyevo-mic-locked-predictions-v1"
SCORE_FORMAT = "phyevo-mic-locked-score-v1"
PROTOCOL_FORMAT = "phyevo-mic-locked-protocol-v1"
SEAL_FORMAT = "phyevo-mic-protocol-seal-v1"
LEDGER_FORMAT = "phyevo-mic-unblinding-ledger-v1"


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _safe_project_path(
    project_root: Path,
    relative: str | Path,
    *,
    must_exist: bool = True,
) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Path must be project-relative: {relative!r}")
    target = (project_root / relative_path).resolve()
    try:
        target.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"Path escapes project root: {relative!r}") from error
    if must_exist and not target.exists():
        raise FileNotFoundError(target)
    return target


@dataclass(frozen=True)
class LockedProtocol:
    """Validated immutable protocol payload and its exact file hash."""

    path: Path
    payload: Mapping[str, Any]
    file_sha256: str

    @property
    def protocol_id(self) -> str:
        return str(self.payload["protocol_id"])

    @property
    def asset_id(self) -> str:
        return str(self.payload["asset_id"])

    @property
    def expected_image_count(self) -> int:
        layout = self.payload["dataset_layout"]
        return int(layout["expected_defective_images"]) + int(
            layout["expected_normal_images"]
        )

    @classmethod
    def load(
        cls, path: str | Path, project_root: str | Path
    ) -> "LockedProtocol":
        root = Path(project_root).resolve()
        protocol_path = Path(path).resolve()
        try:
            protocol_path.relative_to(root)
        except ValueError as error:
            raise ValueError("Protocol must be inside the project") from error
        payload = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("Locked protocol must be a mapping")
        required = {
            "protocol_format",
            "protocol_id",
            "asset_id",
            "registry_path",
            "archive_sha256",
            "dataset_layout",
            "detectors",
            "targets",
            "metrics",
            "frozen_files",
            "prohibited_actions",
            "interpretation_limits",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                "Locked protocol is missing: " + ", ".join(missing)
            )
        if payload["protocol_format"] != PROTOCOL_FORMAT:
            raise ValueError("Unsupported locked protocol format")
        if not payload["protocol_id"] or not payload["asset_id"]:
            raise ValueError("Protocol and asset ids must be non-empty")
        layout = payload["dataset_layout"]
        for key in (
            "defective_images",
            "defective_labels",
            "normal_images",
            "expected_defective_images",
            "expected_normal_images",
        ):
            if key not in layout:
                raise ValueError(f"dataset_layout.{key} is required")
        if int(layout["expected_defective_images"]) < 1:
            raise ValueError("At least one defective image is required")
        if int(layout["expected_normal_images"]) < 1:
            raise ValueError("At least one normal image is required")

        detectors = payload["detectors"]
        if not isinstance(detectors, Sequence) or not detectors:
            raise ValueError("At least one frozen detector is required")
        detector_ids: set[str] = set()
        for detector in detectors:
            detector_id = str(detector["id"])
            if detector_id in detector_ids:
                raise ValueError(f"Duplicate detector id: {detector_id}")
            detector_ids.add(detector_id)
            if not detector.get("config_path"):
                raise ValueError(f"Detector {detector_id} has no config")
        targets = payload["targets"]
        if not isinstance(targets, Sequence) or not targets:
            raise ValueError("At least one evaluation target is required")
        target_ids: set[str] = set()
        for target in targets:
            target_id = str(target["id"])
            if target_id in target_ids:
                raise ValueError(f"Duplicate target id: {target_id}")
            target_ids.add(target_id)
            operation = str(target["prediction"]["operation"])
            if operation not in {"single", "union"}:
                raise ValueError(
                    f"Unsupported prediction operation: {operation}"
                )
            members = tuple(str(item) for item in target["prediction"]["members"])
            if not members or not set(members).issubset(detector_ids):
                raise ValueError(
                    f"Target {target_id} references unknown detectors"
                )
            truth_classes = tuple(int(item) for item in target["truth_classes"])
            if not truth_classes or not set(truth_classes).issubset({1, 2}):
                raise ValueError(
                    f"Target {target_id} truth classes must be 1 and/or 2"
                )
        return cls(
            path=protocol_path,
            payload=payload,
            file_sha256=sha256_file(protocol_path),
        )


def _registry_asset(protocol: LockedProtocol, project_root: Path):
    registry_path = _safe_project_path(
        project_root, str(protocol.payload["registry_path"])
    )
    registry = DataRegistry.from_yaml(registry_path)
    matches = [
        asset for asset in registry.assets if asset.asset_id == protocol.asset_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Protocol asset {protocol.asset_id!r} is not unique in registry"
        )
    asset = matches[0]
    if asset.partition != DataPartition.LOCKED_EXTERNAL_TEST:
        raise ValueError("Protocol asset is not in locked_external_test")
    if asset.checksum_algorithm.lower() != "sha256":
        raise ValueError("Locked protocol currently requires SHA-256 assets")
    if asset.checksum.lower() != str(
        protocol.payload["archive_sha256"]
    ).lower():
        raise ValueError("Protocol and registry archive hashes disagree")
    archive = _safe_project_path(project_root, asset.relative_path)
    observed = sha256_file(archive)
    if observed.lower() != asset.checksum.lower():
        raise ValueError("Locked archive checksum mismatch")
    return registry, asset, archive


def create_protocol_seal(
    *,
    protocol_path: str | Path,
    seal_path: str | Path,
    project_root: str | Path,
    forbidden_unpacked_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Seal a protocol before labels are unpacked; never overwrite a seal."""

    root = Path(project_root).resolve()
    protocol = LockedProtocol.load(protocol_path, root)
    registry, asset, archive = _registry_asset(protocol, root)
    output = Path(seal_path).resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError("Protocol seal must stay inside project") from error
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite protocol seal: {output}")
    if forbidden_unpacked_path is not None:
        unpacked = _safe_project_path(
            root, forbidden_unpacked_path, must_exist=False
        )
        if unpacked.exists():
            raise RuntimeError(
                "Protocol must be sealed before the locked labels are unpacked"
            )

    frozen_files: dict[str, str] = {}
    for relative in protocol.payload["frozen_files"]:
        target = _safe_project_path(root, str(relative))
        if not target.is_file():
            raise ValueError(f"Frozen path is not a file: {relative}")
        frozen_files[str(relative)] = sha256_file(target)
    detector_configs = {}
    for detector in protocol.payload["detectors"]:
        config_path = str(detector["config_path"])
        target = _safe_project_path(root, config_path)
        config = load_config(target)
        if config.task != str(detector["id"]):
            raise ValueError(
                f"Detector {detector['id']} config declares {config.task}"
            )
        detector_configs[config_path] = sha256_file(target)

    seal = {
        "seal_format": SEAL_FORMAT,
        "sealed_at_utc": _utc_now(),
        "protocol_id": protocol.protocol_id,
        "asset_id": protocol.asset_id,
        "protocol_file_sha256": protocol.file_sha256,
        "registry_content_sha256": registry.content_hash(),
        "archive_relative_path": asset.relative_path,
        "archive_sha256": sha256_file(archive),
        "detector_config_sha256": detector_configs,
        "frozen_file_sha256": frozen_files,
        "state_at_seal": {
            "locked_labels_unpacked": False,
            "detector_metrics_observed": False,
        },
    }
    seal["seal_content_sha256"] = _canonical_hash(seal)
    _write_json_new(output, seal)
    return seal


def verify_protocol_seal(
    *,
    protocol_path: str | Path,
    seal_path: str | Path,
    project_root: str | Path,
) -> tuple[LockedProtocol, Mapping[str, Any]]:
    root = Path(project_root).resolve()
    protocol = LockedProtocol.load(protocol_path, root)
    seal_file = Path(seal_path).resolve()
    seal = json.loads(seal_file.read_text(encoding="utf-8"))
    expected_content_hash = seal.get("seal_content_sha256")
    content_without_hash = dict(seal)
    content_without_hash.pop("seal_content_sha256", None)
    if expected_content_hash != _canonical_hash(content_without_hash):
        raise ValueError("Protocol seal content hash mismatch")
    if seal.get("seal_format") != SEAL_FORMAT:
        raise ValueError("Unsupported protocol seal")
    if seal.get("protocol_id") != protocol.protocol_id:
        raise ValueError("Protocol id differs from seal")
    if seal.get("asset_id") != protocol.asset_id:
        raise ValueError("Asset id differs from seal")
    if seal.get("protocol_file_sha256") != protocol.file_sha256:
        raise ValueError("Protocol file changed after sealing")
    registry, asset, archive = _registry_asset(protocol, root)
    if seal.get("registry_content_sha256") != registry.content_hash():
        raise ValueError("Data registry changed after protocol sealing")
    if seal.get("archive_sha256") != sha256_file(archive):
        raise ValueError("Archive changed after protocol sealing")
    if seal.get("archive_relative_path") != asset.relative_path:
        raise ValueError("Archive path changed after protocol sealing")
    for relative, expected in seal["frozen_file_sha256"].items():
        if sha256_file(_safe_project_path(root, relative)) != expected:
            raise ValueError(f"Frozen implementation changed: {relative}")
    for relative, expected in seal["detector_config_sha256"].items():
        if sha256_file(_safe_project_path(root, relative)) != expected:
            raise ValueError(f"Frozen detector config changed: {relative}")
    return protocol, seal


def _enumerate_images(
    protocol: LockedProtocol, dataset_root: Path
) -> list[dict[str, Any]]:
    layout = protocol.payload["dataset_layout"]
    groups = (
        ("defective", str(layout["defective_images"])),
        ("normal", str(layout["normal_images"])),
    )
    records: list[dict[str, Any]] = []
    for category, relative_directory in groups:
        directory = _safe_project_path(
            dataset_root, relative_directory
        )
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        paths = sorted(
            path for path in directory.iterdir() if path.suffix.lower() == ".tif"
        )
        expected_key = (
            "expected_defective_images"
            if category == "defective"
            else "expected_normal_images"
        )
        if len(paths) != int(layout[expected_key]):
            raise ValueError(
                f"Expected {layout[expected_key]} {category} images, "
                f"found {len(paths)}"
            )
        for image_path in paths:
            relative = image_path.relative_to(dataset_root).as_posix()
            image_hash = sha256_file(image_path)
            image_id = hashlib.sha256(
                f"{category}\0{relative}\0{image_hash}".encode("utf-8")
            ).hexdigest()[:20]
            records.append(
                {
                    "image_id": image_id,
                    "category": category,
                    "relative_path": relative,
                    "filename": image_path.name,
                    "source_sha256": image_hash,
                    "source_bytes": image_path.stat().st_size,
                }
            )
    if len(records) != protocol.expected_image_count:
        raise AssertionError("Image enumeration count mismatch")
    return records


def generate_locked_predictions(
    *,
    protocol_path: str | Path,
    seal_path: str | Path,
    project_root: str | Path,
    dataset_root: str | Path,
    output_directory: str | Path,
    detector_factory: Callable[
        [str, Path], TrainingFreeDetectionFramework
    ]
    | None = None,
) -> Mapping[str, Any]:
    """Generate and seal predictions without traversing any label path."""

    root = Path(project_root).resolve()
    protocol, protocol_seal = verify_protocol_seal(
        protocol_path=protocol_path,
        seal_path=seal_path,
        project_root=root,
    )
    data_root = Path(dataset_root).resolve()
    try:
        data_root.relative_to(root)
    except ValueError as error:
        raise ValueError("Locked dataset root must be inside project") from error
    output = Path(output_directory).resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError("Prediction output must be inside project") from error
    output.mkdir(parents=True, exist_ok=False)

    images = _enumerate_images(protocol, data_root)
    detector_instances = {}
    for detector in protocol.payload["detectors"]:
        detector_id = str(detector["id"])
        config_path = _safe_project_path(root, str(detector["config_path"]))
        detector_instances[detector_id] = (
            detector_factory(detector_id, config_path)
            if detector_factory is not None
            else TrainingFreeDetectionFramework(load_config(config_path))
        )

    prediction_records: list[dict[str, Any]] = []
    total = len(images) * len(detector_instances)
    completed = 0
    for image in images:
        source = _safe_project_path(data_root, image["relative_path"])
        for detector_id, framework in detector_instances.items():
            completed += 1
            print(
                f"[locked-predict {completed}/{total}] "
                f"{image['category']} {image['filename']} :: {detector_id}",
                flush=True,
            )
            record = {
                "image_id": image["image_id"],
                "detector_id": detector_id,
                "status": "failed",
                "error_type": None,
                "error_message": None,
                "prediction_path": None,
                "prediction_sha256": None,
            }
            try:
                result = framework.detect(source)
                if tuple(result.mask.shape) != tuple(result.image_shape):
                    raise ValueError("Detector mask/image shape mismatch")
                reliability = (
                    np.asarray(result.reliability_map, dtype=np.float32)
                    if result.reliability_map is not None
                    else np.ones(result.image_shape, dtype=np.float32)
                )
                decision = (
                    np.asarray(result.decision_map, dtype=np.uint8)
                    if result.decision_map is not None
                    else np.zeros(result.image_shape, dtype=np.uint8)
                )
                if not np.all(np.isfinite(reliability)):
                    raise ValueError("Detector reliability contains non-finite values")
                relative_prediction = (
                    Path("predictions")
                    / image["image_id"]
                    / f"{detector_id}.npz"
                )
                prediction_path = output / relative_prediction
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                if prediction_path.exists():
                    raise FileExistsError(prediction_path)
                np.savez_compressed(
                    prediction_path,
                    mask=np.asarray(result.mask, dtype=np.uint8),
                    reliability=reliability,
                    decision=decision,
                )
                record.update(
                    {
                        "status": "success",
                        "prediction_path": relative_prediction.as_posix(),
                        "prediction_sha256": sha256_file(prediction_path),
                        "image_shape": list(result.image_shape),
                        "positive_pixels": int(np.sum(result.mask)),
                        "positive_fraction": float(np.mean(result.mask)),
                        "mean_reliability": float(np.mean(reliability)),
                        "invalid_fraction": float(np.mean(decision == 4)),
                    }
                )
            except Exception as error:  # preserve every frozen failure
                record["error_type"] = type(error).__name__
                record["error_message"] = str(error)
            prediction_records.append(record)

    manifest = {
        "prediction_format": PREDICTION_FORMAT,
        "created_at_utc": _utc_now(),
        "protocol_id": protocol.protocol_id,
        "protocol_file_sha256": protocol.file_sha256,
        "protocol_seal_content_sha256": protocol_seal[
            "seal_content_sha256"
        ],
        "asset_id": protocol.asset_id,
        "package_version": __version__,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "images": images,
        "predictions": prediction_records,
        "counts": {
            "images": len(images),
            "detectors": len(detector_instances),
            "expected_predictions": total,
            "successful_predictions": sum(
                item["status"] == "success" for item in prediction_records
            ),
            "failed_predictions": sum(
                item["status"] != "success" for item in prediction_records
            ),
        },
        "label_access": {
            "labels_enumerated": False,
            "label_pixels_read": False,
        },
    }
    manifest_path = output / "prediction_manifest.json"
    _write_json_new(manifest_path, manifest)
    prediction_seal = {
        "prediction_format": PREDICTION_FORMAT,
        "protocol_id": protocol.protocol_id,
        "asset_id": protocol.asset_id,
        "manifest_sha256": sha256_file(manifest_path),
        "prediction_set_sha256": _canonical_hash(
            {
                "manifest_sha256": sha256_file(manifest_path),
                "files": sorted(
                    (
                        item["prediction_path"],
                        item["prediction_sha256"],
                    )
                    for item in prediction_records
                    if item["status"] == "success"
                ),
            }
        ),
        "sealed_at_utc": _utc_now(),
    }
    prediction_seal["seal_content_sha256"] = _canonical_hash(prediction_seal)
    _write_json_new(output / "prediction_seal.json", prediction_seal)
    return {
        "output_directory": str(output),
        "manifest": manifest,
        "prediction_seal": prediction_seal,
    }


def _verify_prediction_set(
    prediction_directory: Path,
    protocol: LockedProtocol,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest_path = prediction_directory / "prediction_manifest.json"
    prediction_seal_path = prediction_directory / "prediction_seal.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(prediction_seal_path.read_text(encoding="utf-8"))
    seal_without_hash = dict(seal)
    observed_seal_hash = seal_without_hash.pop("seal_content_sha256", None)
    if observed_seal_hash != _canonical_hash(seal_without_hash):
        raise ValueError("Prediction seal content hash mismatch")
    if manifest.get("prediction_format") != PREDICTION_FORMAT:
        raise ValueError("Unsupported prediction manifest")
    if manifest.get("protocol_id") != protocol.protocol_id:
        raise ValueError("Prediction protocol id mismatch")
    if manifest.get("protocol_file_sha256") != protocol.file_sha256:
        raise ValueError("Prediction used a different protocol file")
    if manifest.get("asset_id") != protocol.asset_id:
        raise ValueError("Prediction asset id mismatch")
    if seal.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Prediction manifest changed after sealing")
    if seal.get("protocol_id") != protocol.protocol_id:
        raise ValueError("Prediction seal protocol id mismatch")
    if seal.get("asset_id") != protocol.asset_id:
        raise ValueError("Prediction seal asset id mismatch")
    files = []
    for record in manifest["predictions"]:
        if record["status"] != "success":
            continue
        prediction_path = _safe_project_path(
            prediction_directory, record["prediction_path"]
        )
        observed = sha256_file(prediction_path)
        if observed != record["prediction_sha256"]:
            raise ValueError(
                f"Prediction file changed: {record['prediction_path']}"
            )
        files.append((record["prediction_path"], observed))
    observed_set = _canonical_hash(
        {
            "manifest_sha256": sha256_file(manifest_path),
            "files": sorted(files),
        }
    )
    if seal.get("prediction_set_sha256") != observed_set:
        raise ValueError("Prediction set hash mismatch")
    return manifest, seal


def _append_unblinding_event(
    ledger_path: Path,
    event: Mapping[str, Any],
) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if ledger_path.exists() else "x"
    with ledger_path.open(mode, encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
        stream.write("\n")


def _register_unblinding(
    *,
    ledger_path: Path,
    protocol: LockedProtocol,
    prediction_seal: Mapping[str, Any],
) -> Mapping[str, Any]:
    prior: list[Mapping[str, Any]] = []
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prior.append(json.loads(line))
    for event in prior:
        if event.get("ledger_format") != LEDGER_FORMAT:
            raise ValueError("Unrecognized event in unblinding ledger")
        if (
            event.get("asset_id") == protocol.asset_id
            and event.get("prediction_set_sha256")
            != prediction_seal["prediction_set_sha256"]
        ):
            raise RuntimeError(
                "Locked asset was already unblinded against a different "
                "prediction set; post-unblinding replacement is forbidden"
            )
    event = {
        "ledger_format": LEDGER_FORMAT,
        "event": "score_started",
        "recorded_at_utc": _utc_now(),
        "protocol_id": protocol.protocol_id,
        "protocol_file_sha256": protocol.file_sha256,
        "asset_id": protocol.asset_id,
        "prediction_set_sha256": prediction_seal[
            "prediction_set_sha256"
        ],
    }
    event["event_sha256"] = _canonical_hash(event)
    _append_unblinding_event(ledger_path, event)
    return event


def _read_ranked_label_images(
    label_directory: Path,
    expected_names: Iterable[str],
) -> tuple[dict[str, np.ndarray], list[float]]:
    """Read black/gray/white labels and infer one global ordered value map."""

    label_paths = {
        path.name: path
        for path in label_directory.iterdir()
        if path.suffix.lower() == ".tif"
    }
    names = set(expected_names)
    if set(label_paths) != names:
        missing = sorted(names - set(label_paths))
        extra = sorted(set(label_paths) - names)
        raise ValueError(
            f"Label/image pairing mismatch; missing={missing}, extra={extra}"
        )
    raw_labels: dict[str, np.ndarray] = {}
    global_values: set[float] = set()
    for name in sorted(names):
        label = np.asarray(imread(str(label_paths[name])))
        if label.ndim == 3:
            if label.shape[-1] == 4:
                label = label[..., :3]
            if label.shape[-1] != 3:
                raise ValueError(f"Unsupported label shape for {name}")
            channels = label.astype(np.float64)
            if not (
                np.array_equal(channels[..., 0], channels[..., 1])
                and np.array_equal(channels[..., 1], channels[..., 2])
            ):
                raise ValueError(
                    f"Label {name} is not black/gray/white grayscale RGB"
                )
            label = channels[..., 0]
        if label.ndim != 2:
            raise ValueError(f"Label {name} is not 2-D")
        values = np.unique(label)
        global_values.update(float(item) for item in values)
        raw_labels[name] = label
    ordered = sorted(global_values)
    if len(ordered) != 3:
        raise ValueError(
            "Pre-registered decoder requires exactly three global "
            f"black/gray/white values; observed {ordered}"
        )
    decoded = {}
    for name, values in raw_labels.items():
        output = np.full(values.shape, 255, dtype=np.uint8)
        for class_id, raw_value in enumerate(ordered):
            output[np.asarray(values, dtype=float) == raw_value] = class_id
        if np.any(output == 255):
            raise ValueError(f"Could not decode every label pixel in {name}")
        decoded[name] = output
    return decoded, ordered


def _load_prediction_masks(
    manifest: Mapping[str, Any],
    prediction_directory: Path,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], str],
]:
    masks: dict[tuple[str, str], np.ndarray] = {}
    reliability: dict[tuple[str, str], np.ndarray] = {}
    status: dict[tuple[str, str], str] = {}
    image_shapes = {
        item["image_id"]: None for item in manifest["images"]
    }
    for record in manifest["predictions"]:
        key = (record["image_id"], record["detector_id"])
        status[key] = record["status"]
        if record["status"] != "success":
            continue
        path = _safe_project_path(
            prediction_directory, record["prediction_path"]
        )
        with np.load(path, allow_pickle=False) as payload:
            mask = np.asarray(payload["mask"], dtype=bool)
            q_map = np.asarray(payload["reliability"], dtype=float)
        if mask.shape != q_map.shape:
            raise ValueError("Prediction mask and reliability shape mismatch")
        masks[key] = mask
        reliability[key] = q_map
        image_shapes[record["image_id"]] = mask.shape
    return masks, reliability, status


def _aggregate_counts(
    predictions: Sequence[np.ndarray],
    truths: Sequence[np.ndarray],
) -> Mapping[str, float]:
    true_positive = sum(
        int(np.sum(predicted & truth))
        for predicted, truth in zip(predictions, truths)
    )
    false_positive = sum(
        int(np.sum(predicted & ~truth))
        for predicted, truth in zip(predictions, truths)
    )
    false_negative = sum(
        int(np.sum(~predicted & truth))
        for predicted, truth in zip(predictions, truths)
    )
    true_negative = sum(
        int(np.sum(~predicted & ~truth))
        for predicted, truth in zip(predictions, truths)
    )
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    union = true_positive + false_positive + false_negative
    return {
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "true_negative_pixels": true_negative,
        "iou": true_positive / max(union, 1),
        "dice": 2
        * true_positive
        / max(2 * true_positive + false_positive + false_negative, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "specificity": true_negative
        / max(true_negative + false_positive, 1),
    }


def _image_iou(prediction: np.ndarray, truth: np.ndarray) -> float:
    union = int(np.sum(prediction | truth))
    if union == 0:
        return 1.0
    return float(np.sum(prediction & truth) / union)


def _combine_for_target(
    *,
    image_id: str,
    members: Sequence[str],
    masks: Mapping[tuple[str, str], np.ndarray],
    reliability: Mapping[tuple[str, str], np.ndarray],
    status: Mapping[tuple[str, str], str],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, bool]:
    successful_masks = []
    successful_q = []
    failed = False
    for detector_id in members:
        key = (image_id, detector_id)
        if status.get(key) != "success":
            failed = True
            continue
        successful_masks.append(masks[key])
        successful_q.append(reliability[key])
    if failed or len(successful_masks) != len(members):
        return (
            np.zeros(shape, dtype=bool),
            np.zeros(shape, dtype=float),
            True,
        )
    return (
        np.logical_or.reduce(successful_masks),
        np.maximum.reduce(successful_q),
        False,
    )


def score_locked_predictions(
    *,
    protocol_path: str | Path,
    seal_path: str | Path,
    project_root: str | Path,
    dataset_root: str | Path,
    prediction_directory: str | Path,
    output_directory: str | Path,
    ledger_path: str | Path,
) -> Mapping[str, Any]:
    """Unblind one previously sealed prediction set and write frozen metrics."""

    root = Path(project_root).resolve()
    protocol, protocol_seal = verify_protocol_seal(
        protocol_path=protocol_path,
        seal_path=seal_path,
        project_root=root,
    )
    data_root = Path(dataset_root).resolve()
    prediction_root = Path(prediction_directory).resolve()
    output = Path(output_directory).resolve()
    ledger = Path(ledger_path).resolve()
    for candidate, name in (
        (data_root, "dataset"),
        (prediction_root, "prediction"),
        (output, "score output"),
        (ledger, "unblinding ledger"),
    ):
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{name} path must stay inside project") from error
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite score output: {output}")
    manifest, prediction_seal = _verify_prediction_set(
        prediction_root, protocol
    )
    unblinding_event = _register_unblinding(
        ledger_path=ledger,
        protocol=protocol,
        prediction_seal=prediction_seal,
    )

    layout = protocol.payload["dataset_layout"]
    image_records = manifest["images"]
    defective_names = [
        item["filename"]
        for item in image_records
        if item["category"] == "defective"
    ]
    label_directory = _safe_project_path(
        data_root, str(layout["defective_labels"])
    )
    decoded_labels, raw_label_values = _read_ranked_label_images(
        label_directory, defective_names
    )
    masks, reliability, status = _load_prediction_masks(
        manifest, prediction_root
    )

    target_results = {}
    for target in protocol.payload["targets"]:
        target_id = str(target["id"])
        members = tuple(str(item) for item in target["prediction"]["members"])
        truth_classes = tuple(int(item) for item in target["truth_classes"])
        predictions = []
        truths = []
        image_rows = []
        failed_images = 0
        for image in image_records:
            image_id = image["image_id"]
            if image["category"] == "defective":
                class_map = decoded_labels[image["filename"]]
                truth = np.isin(class_map, truth_classes)
                shape = tuple(class_map.shape)
            else:
                successful_shape = next(
                    (
                        predicted_mask.shape
                        for (record_image_id, _), predicted_mask in masks.items()
                        if record_image_id == image_id
                    ),
                    None,
                )
                if successful_shape is None:
                    source_image = np.asarray(
                        imread(
                            str(
                                _safe_project_path(
                                    data_root, image["relative_path"]
                                )
                            )
                        )
                    )
                    if source_image.ndim not in {2, 3}:
                        raise ValueError(
                            "Cannot infer normal-image shape after all "
                            "detectors failed"
                        )
                    successful_shape = tuple(source_image.shape[:2])
                shape = tuple(successful_shape)
                truth = np.zeros(shape, dtype=bool)
            predicted, q_map, failed = _combine_for_target(
                image_id=image_id,
                members=members,
                masks=masks,
                reliability=reliability,
                status=status,
                shape=shape,
            )
            if predicted.shape != truth.shape:
                raise ValueError(
                    f"Prediction/label shape mismatch for {image['filename']}"
                )
            failed_images += int(failed)
            predictions.append(predicted)
            truths.append(truth)
            boundary = (
                boundary_metrics(
                    predicted,
                    truth,
                    tolerance_pixels=float(
                        protocol.payload["metrics"][
                            "boundary_tolerance_pixels"
                        ]
                    ),
                )
                if np.any(truth)
                else None
            )
            assd = (
                float(boundary["average_symmetric_surface_distance"])
                if boundary is not None
                else None
            )
            if assd is not None and not np.isfinite(assd):
                assd = None
            image_rows.append(
                {
                    "image_id": image_id,
                    "category": image["category"],
                    "filename": image["filename"],
                    "detector_failure": failed,
                    "truth_positive_fraction": float(np.mean(truth)),
                    "predicted_positive_fraction": float(np.mean(predicted)),
                    "iou": _image_iou(predicted, truth),
                    "mean_reliability": float(np.mean(q_map)),
                    "boundary_f1": (
                        boundary["boundary_f1"]
                        if boundary is not None
                        else None
                    ),
                    "average_symmetric_surface_distance": (
                        assd
                    ),
                }
            )
        defective_rows = [
            row for row in image_rows if row["category"] == "defective"
        ]
        normal_rows = [
            row for row in image_rows if row["category"] == "normal"
        ]
        finite_assd = [
            float(row["average_symmetric_surface_distance"])
            for row in defective_rows
            if row["average_symmetric_surface_distance"] is not None
            and np.isfinite(row["average_symmetric_surface_distance"])
        ]
        truth_positive_rows = [
            row
            for row in defective_rows
            if row["truth_positive_fraction"] > 0
        ]
        boundary_f1_values = [
            float(row["boundary_f1"])
            for row in truth_positive_rows
            if row["boundary_f1"] is not None
        ]
        target_results[target_id] = {
            "role": str(target["role"]),
            "truth_classes": list(truth_classes),
            "prediction_members": list(members),
            "micro_pixel_metrics": _aggregate_counts(predictions, truths),
            "defective_image_macro": {
                "image_count": len(defective_rows),
                "truth_positive_image_count": len(truth_positive_rows),
                "mean_iou": float(
                    np.mean([row["iou"] for row in defective_rows])
                ),
                "median_iou": float(
                    np.median([row["iou"] for row in defective_rows])
                ),
                "mean_iou_on_truth_positive_images": (
                    float(
                        np.mean([row["iou"] for row in truth_positive_rows])
                    )
                    if truth_positive_rows
                    else None
                ),
                "mean_boundary_f1": (
                    float(np.mean(boundary_f1_values))
                    if boundary_f1_values
                    else None
                ),
                "mean_finite_assd_pixels": (
                    float(np.mean(finite_assd))
                    if finite_assd
                    else None
                ),
                "nonfinite_assd_image_count": sum(
                    row["truth_positive_fraction"] > 0
                    and row["average_symmetric_surface_distance"] is None
                    for row in defective_rows
                ),
            },
            "normal_controls": {
                "image_count": len(normal_rows),
                "mean_false_positive_area_fraction": float(
                    np.mean(
                        [
                            row["predicted_positive_fraction"]
                            for row in normal_rows
                        ]
                    )
                ),
                "p95_false_positive_area_fraction": float(
                    np.percentile(
                        [
                            row["predicted_positive_fraction"]
                            for row in normal_rows
                        ],
                        95,
                    )
                ),
                "maximum_false_positive_area_fraction": float(
                    np.max(
                        [
                            row["predicted_positive_fraction"]
                            for row in normal_rows
                        ]
                    )
                ),
            },
            "failed_image_count": failed_images,
            "per_image": image_rows,
        }

    score = {
        "score_format": SCORE_FORMAT,
        "scored_at_utc": _utc_now(),
        "protocol_id": protocol.protocol_id,
        "protocol_file_sha256": protocol.file_sha256,
        "protocol_seal_content_sha256": protocol_seal[
            "seal_content_sha256"
        ],
        "asset_id": protocol.asset_id,
        "prediction_set_sha256": prediction_seal[
            "prediction_set_sha256"
        ],
        "unblinding_event_sha256": unblinding_event["event_sha256"],
        "dataset": {
            "image_count": len(image_records),
            "defective_image_count": len(defective_names),
            "normal_image_count": len(image_records) - len(defective_names),
            "raw_label_values_sorted": raw_label_values,
            "label_semantics": {
                "0": "main_crystal_symmetry_non_defective",
                "1": "secondary_crystal_symmetry_defect",
                "2": "blurred_amorphous_or_beam_defect",
            },
        },
        "targets": target_results,
        "interpretation_limits": list(
            protocol.payload["interpretation_limits"]
        ),
    }
    score["score_content_sha256"] = _canonical_hash(score)
    output.mkdir(parents=True, exist_ok=False)
    score_path = output / "locked_metrics.json"
    _write_json_new(score_path, score)
    completed_event = {
        "ledger_format": LEDGER_FORMAT,
        "event": "score_completed",
        "recorded_at_utc": _utc_now(),
        "protocol_id": protocol.protocol_id,
        "protocol_file_sha256": protocol.file_sha256,
        "asset_id": protocol.asset_id,
        "prediction_set_sha256": prediction_seal[
            "prediction_set_sha256"
        ],
        "score_content_sha256": score["score_content_sha256"],
        "score_file_sha256": sha256_file(score_path),
    }
    completed_event["event_sha256"] = _canonical_hash(completed_event)
    _append_unblinding_event(ledger, completed_event)
    return score
