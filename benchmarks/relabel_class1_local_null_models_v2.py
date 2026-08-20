#!/usr/bin/env python3
"""Create a semantics-corrected Class-1 null report without changing metrics.

The v1 control selected the nearest available ring pixels after matching the
number of pixels added by the stored hybrid.  It did not match the empirical
distance distribution of those pixels.  This migration verifies the sealed v1
bytes, renames that control accordingly, corrects the Class-1-negative versus
archive-normal counts, and writes a new immutable report and seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "benchmarks/paper_baselines/class1_local_null_models_v1.json"
DEFAULT_SOURCE_SEAL = ROOT / "benchmarks/paper_baselines/class1_local_null_models_v1.seal.json"
DEFAULT_PAIRED = ROOT / "benchmarks/paper_baselines/dennler_paired_statistics_v6.json"
DEFAULT_PAIRED_SEAL = ROOT / "benchmarks/paper_baselines/dennler_paired_statistics_v6.seal.json"
DEFAULT_OUTPUT = ROOT / "benchmarks/paper_baselines/class1_local_null_models_v2.json"
DEFAULT_OUTPUT_SEAL = ROOT / "benchmarks/paper_baselines/class1_local_null_models_v2.seal.json"

KEY_RENAMES = {
    "distance_matched_expansion": "area_matched_nearest_first_expansion",
    "p_minus_distance_matched": "p_minus_nearest_first",
    "p_vs_distance_matched": "p_vs_nearest_first",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_verified(path: Path, seal_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    recorded_raw_hash = seal.get("analysis_sha256", seal.get("analysis_file_sha256"))
    if recorded_raw_hash is None or _sha256(path) != recorded_raw_hash:
        raise RuntimeError(f"Raw analysis hash mismatch: {path}")
    if _content_sha256(payload) != seal["analysis_content_sha256"]:
        raise RuntimeError(f"Canonical analysis hash mismatch: {path}")
    analyzer = ROOT / seal["analyzer_path"]
    if _sha256(analyzer) != seal["analyzer_sha256"]:
        raise RuntimeError(f"Analyzer hash mismatch: {analyzer}")
    return payload, seal


def _rename_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            KEY_RENAMES.get(key, key): _rename_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_keys(item) for item in value]
    return value


def build(
    source: Path = DEFAULT_SOURCE,
    source_seal: Path = DEFAULT_SOURCE_SEAL,
    paired: Path = DEFAULT_PAIRED,
    paired_seal: Path = DEFAULT_PAIRED_SEAL,
    output: Path = DEFAULT_OUTPUT,
    output_seal: Path = DEFAULT_OUTPUT_SEAL,
) -> dict[str, Any]:
    for path in (source, source_seal, paired, paired_seal, output, output_seal):
        path.resolve().relative_to(ROOT.resolve())
    if output.exists() or output_seal.exists():
        raise FileExistsError("Refusing to overwrite a versioned v2 analysis or seal")

    original, original_seal = _load_verified(source, source_seal)
    paired_result, paired_result_seal = _load_verified(paired, paired_seal)
    corrected = _rename_keys(original)
    corrected["analysis_format"] = "phyevo-class1-local-null-models-v2"
    corrected["purpose"] = (
        "Test whether the stored H-seeded Class-1 increment exceeds local "
        "expansion controls on the already consumed archive; v2 corrects the "
        "control semantics without changing any stored metric."
    )
    corrected["dataset"].pop("nominal_normal_image_count")
    corrected["dataset"]["class1_negative_image_count"] = (
        corrected["dataset"]["image_count"]
        - corrected["dataset"]["class1_positive_image_count"]
    )
    corrected["dataset"]["archive_nominal_control_image_count"] = paired_result[
        "normal_image_count"
    ]
    corrected["method_definitions"]["area_matched_nearest_first_expansion"] = (
        "H plus the same per-image number of new pixels as P, selected nearest "
        "to H within the identical dilation ring with deterministic flat-index "
        "tie-breaking; added-pixel distance distributions are not matched"
    )
    corrected["semantic_correction"] = {
        "metrics_recomputed": False,
        "all_numeric_metrics_copied_exactly_from_v1": True,
        "legacy_term": "distance_matched_expansion",
        "corrected_term": "area_matched_nearest_first_expansion",
        "reason": (
            "The v1 control matched added area and allowed ring, then selected "
            "the nearest pixels. It did not match P's added-pixel distance distribution."
        ),
        "legacy_nominal_normal_field_corrected": True,
        "class1_negative_images_are_not_equivalent_to_archive_nominal_controls": True,
        "source_analysis_sha256": original_seal["analysis_sha256"],
        "source_analysis_content_sha256": original_seal["analysis_content_sha256"],
        "paired_statistics_sha256": paired_result_seal.get(
            "analysis_sha256", paired_result_seal["analysis_file_sha256"]
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(corrected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    analyzer = Path(__file__).resolve()
    seal = {
        "seal_format": "phyevo-analysis-seal-v2",
        "analysis_path": str(output.relative_to(ROOT)),
        "analysis_sha256": _sha256(output),
        "analysis_content_sha256": _content_sha256(corrected),
        "analyzer_path": str(analyzer.relative_to(ROOT)),
        "analyzer_sha256": _sha256(analyzer),
        "scientific_status": corrected["scientific_status"],
        "semantic_migration_only": True,
    }
    output_seal.write_text(
        json.dumps(seal, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-seal", type=Path, default=DEFAULT_SOURCE_SEAL)
    parser.add_argument("--paired", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--paired-seal", type=Path, default=DEFAULT_PAIRED_SEAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-seal", type=Path, default=DEFAULT_OUTPUT_SEAL)
    args = parser.parse_args()
    result = build(**vars(args))
    print(f"Wrote {args.output.relative_to(ROOT)} ({result['dataset']['image_count']} images)")


if __name__ == "__main__":
    main()
