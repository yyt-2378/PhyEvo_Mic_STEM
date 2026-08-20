#!/usr/bin/env python3
"""Verify the compatibility port against checksum-pinned upstream functions."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import types

import numpy as np

from run_dennler_reference_baseline import (
    PROJECT_ROOT,
    UPSTREAM_COMMIT,
    _circular_mask,
    bragg_filter_amorphous,
    bragg_filter_symmetry,
)


UPSTREAM_SOURCE = (
    PROJECT_ROOT
    / "benchmarks/upstream_dennler/bragg_filtering_72d37c3.py"
)
UPSTREAM_SOURCE_SHA256 = (
    "f01cbda29d4d24530618cb508e40e8d17828cef2128b3504e00a15305ec9c67c"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "benchmarks/paper_baselines/dennler_upstream_parity_v1.json"
)
EXPECTED = {
    "input_sha256": (
        "8403c4a3b512dce422f88cc5c5fdec87529073771492565c4f3ac7593e89f51e"
    ),
    "amorphous_positive_pixels": 270_516,
    "amorphous_mask_sha256": (
        "85cf20dfe8d9720aba72f28805fe4c743ab0e95a4d9363107f8f0a35218340fd"
    ),
    "symmetry_positive_pixels": 925_243,
    "symmetry_mask_sha256": (
        "e7b1157618fe2cb860795414406bc736a9583d4d8895a81f9bb5f2d6888252f3"
    ),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_upstream() -> types.ModuleType:
    source = UPSTREAM_SOURCE.read_text(encoding="utf-8")
    if _sha256_file(UPSTREAM_SOURCE) != UPSTREAM_SOURCE_SHA256:
        raise ValueError("Vendored upstream source checksum changed")
    stale_import = "from skimage.util import pad"
    stale_gaussian = "signal.gaussian(kernlen, std=std)"
    if source.count(stale_import) != 1 or source.count(stale_gaussian) != 1:
        raise ValueError("Unexpected upstream compatibility surface")
    executable = source.replace(stale_import, "# unused compatibility import")
    executable = executable.replace(
        stale_gaussian,
        "signal.windows.gaussian(kernlen, std=std)",
    )
    module = types.ModuleType("dennler_upstream_bragg_filtering")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(executable, str(UPSTREAM_SOURCE), "exec"), module.__dict__)
    return module


def _synthetic_input() -> np.ndarray:
    yy, xx = np.mgrid[:1024, :1024]
    rng = np.random.default_rng(721)
    image = (
        128
        + 50 * np.cos(2 * np.pi * xx / 16)
        + 45 * np.cos(2 * np.pi * yy / 18)
        + 25 * np.cos(2 * np.pi * (xx + yy) / 27)
        + 60 * np.cos(2 * np.pi * xx / 64)
        + 3 * rng.normal(size=xx.shape)
    )
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _mask_record(upstream: np.ndarray, current: np.ndarray) -> dict[str, object]:
    upstream_mask = np.asarray(upstream, dtype=bool)
    current_mask = np.asarray(current, dtype=bool)
    return {
        "pixel_equal": bool(np.array_equal(upstream_mask, current_mask)),
        "upstream_positive_pixels": int(upstream_mask.sum()),
        "current_positive_pixels": int(current_mask.sum()),
        "upstream_mask_sha256": _sha256_bytes(upstream_mask.tobytes()),
        "current_mask_sha256": _sha256_bytes(current_mask.tobytes()),
    }


def verify() -> dict[str, object]:
    upstream = _load_upstream()
    image = _synthetic_input()
    with contextlib.redirect_stdout(io.StringIO()):
        upstream_amorphous = upstream.bragg_filter_amorphous(image)
        upstream_symmetry = upstream.bragg_filter_symmetry(image)
    amorphous = _mask_record(
        upstream_amorphous,
        bragg_filter_amorphous(image),
    )
    symmetry = _mask_record(
        upstream_symmetry,
        bragg_filter_symmetry(image),
    )
    circular_equal = np.array_equal(
        np.asarray(upstream.circular_mask(), dtype=bool),
        _circular_mask((1024, 1024), 50),
    )
    observed = {
        "input_sha256": _sha256_bytes(image.tobytes()),
        "amorphous_positive_pixels": amorphous["current_positive_pixels"],
        "amorphous_mask_sha256": amorphous["current_mask_sha256"],
        "symmetry_positive_pixels": symmetry["current_positive_pixels"],
        "symmetry_mask_sha256": symmetry["current_mask_sha256"],
    }
    passed = bool(
        circular_equal
        and amorphous["pixel_equal"]
        and symmetry["pixel_equal"]
        and observed == EXPECTED
    )
    return {
        "format": "phyevo-dennler-upstream-parity-v1",
        "passed": passed,
        "upstream_repository": (
            "https://github.com/nkdnnlr/TEMDefectClassification"
        ),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source": UPSTREAM_SOURCE.relative_to(PROJECT_ROOT).as_posix(),
        "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
        "verifier_sha256": _sha256_file(Path(__file__).resolve()),
        "compatibility_transformations_applied_in_memory": [
            "omit unused removed skimage.util.pad import",
            "scipy.signal.gaussian to scipy.signal.windows.gaussian",
        ],
        "fixture": {
            "kind": "deterministic synthetic 1024x1024 periodic image",
            "random_seed": 721,
            **observed,
        },
        "circular_mask_pixel_equal": bool(circular_equal),
        "amorphous": amorphous,
        "symmetry": symmetry,
        "claim_boundary": (
            "This proves pixel parity for the released Fourier/Bragg functions "
            "at one non-degenerate fixed input and pins the exact upstream "
            "source. It is not parity with the unpublished trained VGG system."
        ),
    }


def main() -> None:
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {DEFAULT_OUTPUT}")
    result = verify()
    if not result["passed"]:
        raise RuntimeError("Dennler upstream parity failed")
    DEFAULT_OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
