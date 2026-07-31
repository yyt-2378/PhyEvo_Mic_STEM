"""Pure, testable logic used by the Streamlit expert-review surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .review_dataset import ReviewSample


class ReviewView(str, Enum):
    BLINDED = "blinded"
    ASSISTED = "assisted"
    ADJUDICATION = "adjudication"


@dataclass(frozen=True)
class ReviewQueueFilter:
    datasets: tuple[str, ...] = ()
    splits: tuple[str, ...] = ()
    priorities: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    review_eligible_only: bool = True


@dataclass(frozen=True)
class ViewPolicy:
    show_prediction: bool
    show_reference_label: bool
    allow_decision: bool
    reason: str


def filter_review_queue(
    samples: Iterable[ReviewSample],
    policy: ReviewQueueFilter,
) -> list[ReviewSample]:
    records = []
    for sample in samples:
        if policy.review_eligible_only and not sample.review_eligible:
            continue
        if policy.datasets and sample.dataset_id not in policy.datasets:
            continue
        if policy.splits and sample.review_split not in policy.splits:
            continue
        if (
            policy.priorities
            and sample.review_priority not in policy.priorities
        ):
            continue
        if policy.tasks and sample.primary_task not in policy.tasks:
            continue
        records.append(sample)
    return records


def view_policy(
    view: ReviewView | str,
    *,
    primary_review_exists: bool,
    reference_label_available: bool,
    prediction_available: bool,
) -> ViewPolicy:
    selected = ReviewView(view)
    if selected == ReviewView.BLINDED:
        return ViewPolicy(
            show_prediction=False,
            show_reference_label=False,
            allow_decision=True,
            reason="primary_judgement_without_model_or_reference",
        )
    if selected == ReviewView.ASSISTED:
        return ViewPolicy(
            show_prediction=prediction_available,
            show_reference_label=False,
            allow_decision=True,
            reason="model_assistance_reference_hidden",
        )
    if not primary_review_exists:
        return ViewPolicy(
            show_prediction=False,
            show_reference_label=False,
            allow_decision=False,
            reason="primary_review_required_before_adjudication",
        )
    return ViewPolicy(
        show_prediction=prediction_available,
        show_reference_label=reference_label_available,
        allow_decision=True,
        reason="post_commit_adjudication",
    )


def _selection_payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = event.get("selection", event)
    if not isinstance(selected, Mapping):
        return {}
    return selected


def _clip_point(
    x: Any,
    y: Any,
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    height, width = image_shape
    return (
        min(max(float(x), 0.0), float(width - 1)),
        min(max(float(y), 0.0), float(height - 1)),
    )


def _downsample_points(
    points: Sequence[tuple[float, float]],
    maximum: int,
) -> list[tuple[float, float]]:
    if len(points) <= maximum:
        return list(points)
    step = len(points) / maximum
    return [points[int(index * step)] for index in range(maximum)]


def selection_geometry(
    event: Mapping[str, Any],
    *,
    image_shape: tuple[int, int],
    geometry_hint: str,
    maximum_points: int = 128,
) -> dict[str, Any] | None:
    """Convert Streamlit Plotly selection state to source-pixel geometry."""

    if maximum_points < 3:
        raise ValueError("maximum_points must be at least 3")
    selection = _selection_payload(event)
    points: list[tuple[float, float]] = []
    selection_kind = "none"

    lasso_items = selection.get("lasso", [])
    if isinstance(lasso_items, list) and lasso_items:
        item = lasso_items[-1]
        if isinstance(item, Mapping):
            xs = item.get("x", [])
            ys = item.get("y", [])
            if isinstance(xs, list) and isinstance(ys, list):
                points = [
                    _clip_point(x, y, image_shape)
                    for x, y in zip(xs, ys)
                ]
                selection_kind = "lasso"

    if not points:
        box_items = selection.get("box", [])
        if isinstance(box_items, list) and box_items:
            item = box_items[-1]
            if isinstance(item, Mapping):
                x0, x1 = item.get("x", (None, None))
                y0, y1 = item.get("y", (None, None))
                if None not in (x0, x1, y0, y1):
                    points = [
                        _clip_point(x0, y0, image_shape),
                        _clip_point(x1, y0, image_shape),
                        _clip_point(x1, y1, image_shape),
                        _clip_point(x0, y1, image_shape),
                    ]
                    selection_kind = "box"

    if not points:
        selected_points = selection.get("points", [])
        if isinstance(selected_points, list):
            for item in selected_points:
                if not isinstance(item, Mapping):
                    continue
                if "x" in item and "y" in item:
                    points.append(
                        _clip_point(item["x"], item["y"], image_shape)
                    )
            if points:
                selection_kind = "points"

    if not points:
        return None
    points = _downsample_points(points, maximum_points)
    geometry_type = {
        "point": "point_set",
        "line": "line_trace",
        "interface": "boundary_trace",
        "region": "region_contour",
    }.get(geometry_hint, "open_set_geometry")
    minimum = 1 if geometry_hint == "point" else 2
    if geometry_hint == "region" and selection_kind != "box":
        minimum = 3
    return {
        "coordinate_space": "source_pixels_xy",
        "image_shape_yx": list(image_shape),
        "geometry_type": geometry_type,
        "selection_kind": selection_kind,
        "points_xy": [[x, y] for x, y in points],
        "valid_for_hint": len(points) >= minimum,
    }


def merge_selection_geometry(
    existing: Mapping[str, Any] | None,
    captured: Mapping[str, Any],
    *,
    geometry_hint: str,
    maximum_points: int = 512,
) -> dict[str, Any]:
    """Append click-selected control points; replace with box/lasso traces."""

    if maximum_points < 3:
        raise ValueError("maximum_points must be at least 3")
    merged = dict(captured)
    points = [list(item) for item in captured.get("points_xy", [])]
    if (
        existing is not None
        and captured.get("selection_kind") == "points"
        and existing.get("geometry_type") == captured.get("geometry_type")
        and existing.get("image_shape_yx") == captured.get("image_shape_yx")
    ):
        points = [list(item) for item in existing.get("points_xy", [])] + points
    unique = []
    seen = set()
    for point in points:
        key = (round(float(point[0]), 6), round(float(point[1]), 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append([float(point[0]), float(point[1])])
    if len(unique) > maximum_points:
        unique = unique[-maximum_points:]
    minimum = 1 if geometry_hint == "point" else 2
    if geometry_hint == "region":
        minimum = 3
    merged["points_xy"] = unique
    merged["valid_for_hint"] = len(unique) >= minimum
    return merged


def remove_last_geometry_point(
    geometry: Mapping[str, Any], *, geometry_hint: str
) -> dict[str, Any] | None:
    points = [list(item) for item in geometry.get("points_xy", [])]
    if not points:
        return None
    points.pop()
    if not points:
        return None
    updated = dict(geometry)
    updated["points_xy"] = points
    minimum = 1 if geometry_hint == "point" else 2
    if geometry_hint == "region":
        minimum = 3
    updated["valid_for_hint"] = len(points) >= minimum
    return updated
