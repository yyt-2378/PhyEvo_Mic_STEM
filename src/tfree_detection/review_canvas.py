"""Bidirectional browser canvas for expert geometry correction."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .review_prediction import comparison_overlay


_HTML = """
<div class="review-canvas-shell">
  <div class="review-canvas-toolbar">
    <span id="modeLabel" class="mode-label"></span>
    <span id="statusLabel" class="status-label"></span>
    <button id="undoButton" type="button">撤销一点</button>
    <button id="clearButton" type="button" class="danger">清空重画</button>
  </div>
  <div class="review-canvas-stage">
    <canvas id="annotationCanvas" aria-label="专家几何绘图区域"></canvas>
  </div>
  <div class="review-canvas-hint" id="interactionHint"></div>
</div>
"""

_CSS = """
.review-canvas-shell {
  width: 100%;
  color: var(--st-text-color);
  font-family: var(--st-font);
}
.review-canvas-toolbar {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  flex-wrap: wrap;
  margin-bottom: 0.55rem;
}
.mode-label {
  font-weight: 700;
  color: var(--st-text-color);
}
.status-label {
  margin-right: auto;
  color: var(--st-text-color);
  opacity: 0.82;
}
button {
  border: 1px solid color-mix(in srgb, var(--st-text-color) 28%, transparent);
  border-radius: 0.45rem;
  padding: 0.42rem 0.75rem;
  background: var(--st-secondary-background-color);
  color: var(--st-text-color);
  cursor: pointer;
  font-weight: 600;
}
button:hover { border-color: var(--st-primary-color); }
button:disabled { cursor: not-allowed; opacity: 0.42; }
button.danger { color: #ff7b7b; }
.review-canvas-stage {
  display: flex;
  width: 100%;
  justify-content: center;
  align-items: flex-start;
  overflow: auto;
}
#annotationCanvas {
  display: block;
  box-sizing: content-box;
  flex: 0 0 auto;
  border: 2px solid color-mix(in srgb, var(--st-text-color) 18%, transparent);
  border-radius: 0.5rem;
  cursor: crosshair;
  touch-action: none;
  image-rendering: auto;
  background: #05080d;
}
.review-canvas-hint {
  margin-top: 0.45rem;
  font-size: 0.88rem;
  opacity: 0.78;
}
"""

_JS = r"""
export default function(component) {
  const { data, parentElement, setStateValue } = component;
  const canvas = parentElement.querySelector('#annotationCanvas');
  const stage = parentElement.querySelector('.review-canvas-stage');
  const undoButton = parentElement.querySelector('#undoButton');
  const clearButton = parentElement.querySelector('#clearButton');
  const statusLabel = parentElement.querySelector('#statusLabel');
  const modeLabel = parentElement.querySelector('#modeLabel');
  const interactionHint = parentElement.querySelector('#interactionHint');
  const context = canvas.getContext('2d');
  const width = Number(data.imageWidth);
  const height = Number(data.imageHeight);
  const geometryHint = data.geometryHint;
  const geometryType = data.geometryType;
  const minimumPoints = Number(data.minimumPoints);
  const initial = data.existingGeometry;
  let points = initial && Array.isArray(initial.points_xy)
    ? initial.points_xy.map((point) => [Number(point[0]), Number(point[1])])
    : [];
  let drawingRegion = false;
  let imageReady = false;
  let destroyed = false;
  let resizeObserver = null;
  const image = new Image();

  canvas.width = width;
  canvas.height = height;
  modeLabel.textContent = geometryHint === 'region'
    ? '套索区域模式'
    : geometryHint === 'point'
      ? '逐点标注模式'
      : '顺序折线模式';
  interactionHint.textContent = geometryHint === 'region'
    ? '在图上按住并拖动，沿区域外缘绕一圈后松开。洋红轮廓是最终专家标注。'
    : geometryHint === 'point'
      ? '直接点击异常原子柱中心；每次点击增加一个洋红点。'
      : '从缺陷一端开始依次点击控制点；洋红折线会按点击顺序连接。';

  function valid() { return points.length >= minimumPoints; }

  function payload() {
    return {
      coordinate_space: 'source_pixels_xy',
      image_shape_yx: [height, width],
      geometry_type: geometryType,
      selection_kind: 'interactive_canvas',
      points_xy: points.map((point) => [point[0], point[1]]),
      valid_for_hint: valid(),
    };
  }

  function publish() {
    setStateValue('geometry', points.length ? payload() : null);
  }

  function updateStatus() {
    const state = valid() ? '可以提交' : `至少需要 ${minimumPoints} 个点`;
    statusLabel.textContent = `已记录 ${points.length} 个点 · ${state}`;
    statusLabel.style.color = valid() ? '#43d17d' : '';
    undoButton.disabled = points.length === 0;
    clearButton.disabled = points.length === 0;
  }

  function drawExpertGeometry() {
    if (!points.length) return;
    context.save();
    context.strokeStyle = '#ff3bd4';
    context.fillStyle = '#ff3bd4';
    context.lineWidth = Math.max(2, width / 220);
    context.lineJoin = 'round';
    context.lineCap = 'round';
    if (geometryHint === 'region' && points.length >= 2) {
      context.beginPath();
      context.moveTo(points[0][0], points[0][1]);
      points.slice(1).forEach((point) => context.lineTo(point[0], point[1]));
      if (!drawingRegion && points.length >= 3) context.closePath();
      context.stroke();
      if (!drawingRegion && points.length >= 3) {
        context.fillStyle = 'rgba(255, 59, 212, 0.16)';
        context.fill();
      }
    } else if (geometryHint !== 'point' && points.length >= 2) {
      context.beginPath();
      context.moveTo(points[0][0], points[0][1]);
      points.slice(1).forEach((point) => context.lineTo(point[0], point[1]));
      context.stroke();
    }
    const radius = Math.max(3.2, width / 170);
    points.forEach((point) => {
      context.beginPath();
      context.arc(point[0], point[1], radius, 0, Math.PI * 2);
      context.fillStyle = '#ff3bd4';
      context.fill();
      context.strokeStyle = '#ffffff';
      context.lineWidth = Math.max(1, width / 550);
      context.stroke();
    });
    context.restore();
  }

  function redraw() {
    if (!imageReady || destroyed) return;
    context.clearRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);
    drawExpertGeometry();
    updateStatus();
  }

  function numericStyle(value) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function canvasBorders() {
    const style = window.getComputedStyle(canvas);
    return {
      left: numericStyle(style.borderLeftWidth),
      top: numericStyle(style.borderTopWidth),
      right: numericStyle(style.borderRightWidth),
      bottom: numericStyle(style.borderBottomWidth),
    };
  }

  function fitCanvasToStage() {
    if (destroyed || !stage) return;
    const availableWidth = stage.clientWidth;
    if (!(availableWidth > 0)) return;
    const border = canvasBorders();
    const horizontalBorder = border.left + border.right;
    const availableContentWidth = Math.max(1, availableWidth - horizontalBorder);
    const availableContentHeight = Math.max(160, window.innerHeight * 0.72);
    const scale = Math.max(
      0.01,
      Math.min(availableContentWidth / width, availableContentHeight / height),
    );
    // Set the drawable content box explicitly. The bitmap, CSS content box,
    // and pointer transform therefore share exactly one aspect ratio.
    canvas.style.width = `${width * scale}px`;
    canvas.style.height = `${height * scale}px`;
    redraw();
  }

  function sourcePoint(event) {
    const bounds = canvas.getBoundingClientRect();
    const border = canvasBorders();
    const contentWidth = Math.max(1, bounds.width - border.left - border.right);
    const contentHeight = Math.max(1, bounds.height - border.top - border.bottom);
    const x = (event.clientX - bounds.left - border.left) * width / contentWidth;
    const y = (event.clientY - bounds.top - border.top) * height / contentHeight;
    return [
      Math.max(0, Math.min(width - 1, x)),
      Math.max(0, Math.min(height - 1, y)),
    ];
  }

  function pointerDown(event) {
    event.preventDefault();
    const point = sourcePoint(event);
    if (geometryHint === 'region') {
      drawingRegion = true;
      points = [point];
      canvas.setPointerCapture(event.pointerId);
    } else {
      points.push(point);
      publish();
    }
    redraw();
  }

  function pointerMove(event) {
    if (!drawingRegion) return;
    event.preventDefault();
    const point = sourcePoint(event);
    const previous = points[points.length - 1];
    const distance = Math.hypot(point[0] - previous[0], point[1] - previous[1]);
    if (distance >= Math.max(1.5, width / 350)) {
      points.push(point);
      if (points.length > 2048) points = points.slice(-2048);
      redraw();
    }
  }

  function pointerUp(event) {
    if (!drawingRegion) return;
    event.preventDefault();
    drawingRegion = false;
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    publish();
    redraw();
  }

  function undo(event) {
    event.preventDefault();
    points.pop();
    publish();
    redraw();
  }

  function clear(event) {
    event.preventDefault();
    points = [];
    publish();
    redraw();
  }

  image.onload = () => { imageReady = true; fitCanvasToStage(); };
  image.onerror = () => {
    statusLabel.textContent = '底图加载失败，请刷新页面';
    statusLabel.style.color = '#ff7b7b';
  };
  image.src = data.imageDataUrl;
  canvas.addEventListener('pointerdown', pointerDown);
  canvas.addEventListener('pointermove', pointerMove);
  canvas.addEventListener('pointerup', pointerUp);
  canvas.addEventListener('pointercancel', pointerUp);
  undoButton.addEventListener('click', undo);
  clearButton.addEventListener('click', clear);
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(fitCanvasToStage);
    resizeObserver.observe(stage);
  }
  window.addEventListener('resize', fitCanvasToStage);
  updateStatus();

  return () => {
    destroyed = true;
    canvas.removeEventListener('pointerdown', pointerDown);
    canvas.removeEventListener('pointermove', pointerMove);
    canvas.removeEventListener('pointerup', pointerUp);
    canvas.removeEventListener('pointercancel', pointerUp);
    undoButton.removeEventListener('click', undo);
    clearButton.removeEventListener('click', clear);
    window.removeEventListener('resize', fitCanvasToStage);
    if (resizeObserver) resizeObserver.disconnect();
  };
}
"""


_GEOMETRY_TYPE = {
    "point": "point_set",
    "line": "line_trace",
    "interface": "boundary_trace",
    "region": "region_contour",
}
_MINIMUM_POINTS = {"point": 1, "line": 2, "interface": 2, "region": 3}


def display_point_to_source(
    client_xy: tuple[float, float],
    *,
    canvas_bounds_xywh: tuple[float, float, float, float],
    border_ltrb: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    """Map a browser pointer position onto source pixels.

    This mirrors the browser implementation and deliberately uses the canvas
    content box, not its border box or a surrounding responsive container.
    """

    client_x, client_y = (float(value) for value in client_xy)
    left, top, display_width, display_height = (
        float(value) for value in canvas_bounds_xywh
    )
    border_left, border_top, border_right, border_bottom = (
        float(value) for value in border_ltrb
    )
    height, width = image_shape
    content_width = display_width - border_left - border_right
    content_height = display_height - border_top - border_bottom
    if content_width <= 0 or content_height <= 0:
        raise ValueError("Canvas content dimensions must be positive")
    source_x = (client_x - left - border_left) * width / content_width
    source_y = (client_y - top - border_top) * height / content_height
    return (
        min(max(source_x, 0.0), float(width - 1)),
        min(max(source_y, 0.0), float(height - 1)),
    )


def _image_data_url(image: np.ndarray) -> str:
    values = np.asarray(image)
    if values.dtype != np.uint8:
        raise TypeError("Review canvas image must be uint8")
    if values.ndim not in {2, 3}:
        raise ValueError("Review canvas image must be grayscale or RGB")
    buffer = BytesIO()
    Image.fromarray(values).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def canvas_payload(
    image: np.ndarray,
    *,
    prediction_mask: np.ndarray | None,
    geometry_hint: str,
    existing_geometry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if geometry_hint not in _GEOMETRY_TYPE:
        raise ValueError(f"Unsupported geometry hint {geometry_hint!r}")
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("Review canvas source image must be two-dimensional")
    overlay = comparison_overlay(
        values, label=None, prediction_mask=prediction_mask
    )
    return {
        "imageDataUrl": _image_data_url(overlay),
        "imageWidth": int(values.shape[1]),
        "imageHeight": int(values.shape[0]),
        "geometryHint": geometry_hint,
        "geometryType": _GEOMETRY_TYPE[geometry_hint],
        "minimumPoints": _MINIMUM_POINTS[geometry_hint],
        "existingGeometry": (
            None if existing_geometry is None else dict(existing_geometry)
        ),
    }


def normalize_canvas_geometry(
    value: Any,
    *,
    image_shape: tuple[int, int],
    geometry_hint: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("Canvas geometry must be an object or null")
    points = value.get("points_xy")
    if not isinstance(points, list) or not points:
        return None
    if len(points) > 2048:
        raise ValueError("Canvas geometry exceeds 2048 points")
    height, width = image_shape
    normalized = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise TypeError("Canvas points must be [x, y]")
        x, y = float(point[0]), float(point[1])
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("Canvas coordinates must be finite")
        normalized.append(
            [
                min(max(x, 0.0), float(width - 1)),
                min(max(y, 0.0), float(height - 1)),
            ]
        )
    minimum = _MINIMUM_POINTS[geometry_hint]
    return {
        "coordinate_space": "source_pixels_xy",
        "image_shape_yx": [height, width],
        "geometry_type": _GEOMETRY_TYPE[geometry_hint],
        "selection_kind": "interactive_canvas",
        "points_xy": normalized,
        "valid_for_hint": len(normalized) >= minimum,
    }


def review_geometry_canvas(
    image: np.ndarray,
    *,
    prediction_mask: np.ndarray | None,
    geometry_hint: str,
    existing_geometry: Mapping[str, Any] | None,
    key: str,
) -> dict[str, Any] | None:
    """Mount the interactive canvas and return source-pixel geometry."""

    try:
        import streamlit as st
    except ModuleNotFoundError as error:
        if error.name != "streamlit":
            raise
        raise RuntimeError(
            "The expert review canvas requires the optional review dependencies; "
            'install them with `python -m pip install -e ".[review]`.'
        ) from error

    data = canvas_payload(
        image,
        prediction_mask=prediction_mask,
        geometry_hint=geometry_hint,
        existing_geometry=existing_geometry,
    )
    # V2 registrations are scoped to the active Streamlit script context.
    # This module remains cached across reruns, so registering only at import
    # time leaves later contexts with an unregistered component.
    renderer = st.components.v2.component(
        "phyevo_review_geometry_canvas",
        html=_HTML,
        css=_CSS,
        js=_JS,
    )
    result = renderer(
        key=key,
        data=data,
        default={"geometry": existing_geometry},
        width="stretch",
        height="content",
        on_geometry_change=lambda: None,
    )
    return normalize_canvas_geometry(
        result.geometry,
        image_shape=tuple(int(item) for item in np.asarray(image).shape),
        geometry_hint=geometry_hint,
    )
