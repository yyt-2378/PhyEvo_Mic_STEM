from copy import deepcopy

import pytest

from tfree_detection.benchmark_provenance import (
    verify_prediction_source_manifest,
)


ARCHIVE_SHA256 = "a" * 64
ARCHIVE_ENTRIES = {
    "dataset/defective/images/image-1.tif": {
        "filename": "image-1.tif",
        "category": "defective",
        "source_sha256": "b" * 64,
        "image_shape": [32, 48],
    }
}


def _manifest() -> dict:
    return {
        "archive_sha256": ARCHIVE_SHA256,
        "images": [
            {
                "filename": "image-1.tif",
                "category": "defective",
                "source_relative_path": (
                    "dataset/defective/images/image-1.tif"
                ),
                "source_sha256": "b" * 64,
                "image_shape": [32, 48],
            }
        ],
    }


def test_prediction_source_manifest_accepts_exact_archive_record() -> None:
    result = verify_prediction_source_manifest(
        _manifest(),
        expected_archive_sha256=ARCHIVE_SHA256,
        archive_entries=ARCHIVE_ENTRIES,
    )
    assert result["source_record_count"] == 1
    assert result["source_hashes_verified"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("archive_sha256", "c" * 64, "archive checksum"),
        (
            "source_relative_path",
            "dataset/defective/images/other.tif",
            "source paths",
        ),
        ("source_sha256", "d" * 64, "source checksum"),
        ("image_shape", [48, 32], "source shape"),
    ],
)
def test_prediction_source_manifest_rejects_tampering(
    field: str,
    value,
    message: str,
) -> None:
    manifest = deepcopy(_manifest())
    if field == "archive_sha256":
        manifest[field] = value
    else:
        manifest["images"][0][field] = value
    with pytest.raises(ValueError, match=message):
        verify_prediction_source_manifest(
            manifest,
            expected_archive_sha256=ARCHIVE_SHA256,
            archive_entries=ARCHIVE_ENTRIES,
        )
