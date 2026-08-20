"""Fail-closed source checks for sealed benchmark prediction manifests."""

from __future__ import annotations

from typing import Any, Mapping


def verify_prediction_source_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_archive_sha256: str,
    archive_entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify every manifest source against checksum-indexed archive entries."""

    if manifest.get("archive_sha256") != expected_archive_sha256:
        raise ValueError("Prediction manifest archive checksum changed")
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Prediction manifest has no image records")
    by_source: dict[str, Mapping[str, Any]] = {}
    filenames = set()
    for record in images:
        source_path = str(record.get("source_relative_path", ""))
        filename = str(record.get("filename", ""))
        if not source_path or source_path in by_source:
            raise ValueError("Prediction source paths must be present and unique")
        if not filename or filename in filenames:
            raise ValueError("Prediction filenames must be present and unique")
        by_source[source_path] = record
        filenames.add(filename)
    if set(by_source) != set(archive_entries):
        raise ValueError("Prediction source paths differ from the fixed archive")
    for source_path, record in by_source.items():
        archive_record = archive_entries[source_path]
        if record.get("source_sha256") != archive_record["source_sha256"]:
            raise ValueError(f"Prediction source checksum changed: {source_path}")
        if list(record.get("image_shape", [])) != list(
            archive_record["image_shape"]
        ):
            raise ValueError(f"Prediction source shape changed: {source_path}")
        if record.get("filename") != archive_record["filename"]:
            raise ValueError(f"Prediction source filename changed: {source_path}")
        if record.get("category") != archive_record["category"]:
            raise ValueError(f"Prediction source category changed: {source_path}")
    return {
        "archive_sha256_verified": True,
        "source_records_verified": True,
        "source_record_count": len(by_source),
        "source_paths_unique": True,
        "source_hashes_verified": True,
        "source_shapes_verified": True,
    }
