from __future__ import annotations

from io import BytesIO

import numpy as np
import tifffile

from benchmarks.run_v04_physical_counterfactuals_v1 import (
    LEVELS,
    _decode_truth,
    _variant_specs,
    apply_variant,
)


def test_counterfactual_protocol_has_five_frozen_families_and_three_levels() -> None:
    assert set(LEVELS) == {
        "transfer_loss_proxy",
        "count_noise_proxy",
        "linear_drift_proxy",
        "scan_jitter_proxy",
        "field_edge_truncation",
    }
    assert all(set(levels) == {"low", "nominal", "high"} for levels in LEVELS.values())
    specs = _variant_specs()
    assert len(specs) == 15
    assert sum(spec["scope"] == "all_archive" for spec in specs.values()) == 5
    assert sum(spec["scope"] == "ladder_subset" for spec in specs.values()) == 10


def test_count_noise_is_deterministic_and_does_not_transform_reference_mask() -> None:
    image = np.linspace(0.0, 1.0, 64 * 80, dtype=float).reshape(64, 80)
    mask = image > 0.5
    first, valid, _ = apply_variant(
        image,
        family="count_noise_proxy",
        value=50.0,
        spacing=8.0,
        seed=17,
    )
    second, _, _ = apply_variant(
        image,
        family="count_noise_proxy",
        value=50.0,
        spacing=8.0,
        seed=17,
    )
    aligned, mask_valid, _ = apply_variant(
        mask,
        family="count_noise_proxy",
        value=50.0,
        spacing=8.0,
        seed=17,
        is_mask=True,
    )
    assert np.array_equal(first, second)
    assert np.array_equal(aligned, mask)
    assert np.all(valid) and np.array_equal(valid, mask_valid)


def test_geometric_proxies_transform_images_and_masks_on_identical_support() -> None:
    yy, xx = np.mgrid[:72, :96]
    image = np.sin(xx / 4.0) + np.cos(yy / 5.0)
    mask = (xx > 20) & (xx < 70) & (yy > 12)
    for family, value in (
        ("linear_drift_proxy", 0.10),
        ("scan_jitter_proxy", 0.10),
        ("field_edge_truncation", 1.00),
    ):
        transformed, valid, _ = apply_variant(
            image,
            family=family,
            value=value,
            spacing=9.0,
            seed=1,
        )
        aligned, aligned_valid, _ = apply_variant(
            mask,
            family=family,
            value=value,
            spacing=9.0,
            seed=1,
            is_mask=True,
        )
        assert transformed.shape == aligned.shape == valid.shape
        assert np.array_equal(valid, aligned_valid)
        assert aligned.dtype == bool


def test_label_decoder_accepts_an_allowed_per_image_value_subset() -> None:
    label = np.asarray([[0, 255], [255, 0]], dtype=np.uint8)
    stream = BytesIO()
    tifffile.imwrite(stream, label)

    class Archive:
        def read(self, _: str) -> bytes:
            return stream.getvalue()

    source = {
        "category": "defective",
        "source_relative_path": "root/defective/images/example.tif",
    }
    decoded = _decode_truth(Archive(), source, label.shape)
    assert decoded.tolist() == [[0, 2], [2, 0]]
