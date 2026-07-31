import numpy as np

from tfree_detection.review_canvas import (
    canvas_payload,
    display_point_to_source,
    normalize_canvas_geometry,
)


def test_canvas_payload_embeds_real_overlay_and_geometry_contract() -> None:
    image = np.arange(32 * 48, dtype=float).reshape(32, 48)
    prediction = np.zeros((32, 48), dtype=bool)
    prediction[8:16, 12:22] = True
    payload = canvas_payload(
        image,
        prediction_mask=prediction,
        geometry_hint="interface",
        existing_geometry=None,
    )
    assert payload["imageDataUrl"].startswith("data:image/png;base64,")
    assert payload["imageWidth"] == 48
    assert payload["imageHeight"] == 32
    assert payload["geometryType"] == "boundary_trace"
    assert payload["minimumPoints"] == 2


def test_canvas_geometry_is_source_pixel_clipped_and_validated() -> None:
    geometry = normalize_canvas_geometry(
        {"points_xy": [[-2, 3], [60, 40]]},
        image_shape=(32, 48),
        geometry_hint="line",
    )
    assert geometry is not None
    assert geometry["points_xy"] == [[0.0, 3.0], [47.0, 31.0]]
    assert geometry["geometry_type"] == "line_trace"
    assert geometry["valid_for_hint"]


def test_canvas_partial_or_cleared_geometry_is_not_submittable() -> None:
    partial = normalize_canvas_geometry(
        {"points_xy": [[4, 5]]},
        image_shape=(32, 48),
        geometry_hint="interface",
    )
    assert partial is not None and not partial["valid_for_hint"]
    assert normalize_canvas_geometry(
        None, image_shape=(32, 48), geometry_hint="interface"
    ) is None


def test_pointer_mapping_uses_canvas_content_box_without_progressive_drift() -> None:
    # A 1024x512 source is displayed at 800x400 inside a 2px border. The
    # surrounding Streamlit column may be wider, but it must not enter the
    # transform. Check separated points so a centre-only coincidence cannot
    # hide an aspect-ratio/letterbox bug.
    bounds = (100.0, 50.0, 804.0, 404.0)
    border = (2.0, 2.0, 2.0, 2.0)
    assert display_point_to_source(
        (102.0, 52.0),
        canvas_bounds_xywh=bounds,
        border_ltrb=border,
        image_shape=(512, 1024),
    ) == (0.0, 0.0)
    assert display_point_to_source(
        (502.0, 252.0),
        canvas_bounds_xywh=bounds,
        border_ltrb=border,
        image_shape=(512, 1024),
    ) == (512.0, 256.0)
    right_bottom = display_point_to_source(
        (902.0, 452.0),
        canvas_bounds_xywh=bounds,
        border_ltrb=border,
        image_shape=(512, 1024),
    )
    assert right_bottom == (1023.0, 511.0)
