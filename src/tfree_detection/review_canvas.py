"""Bidirectional browser canvas for expert geometry correction."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from .review_prediction import normalize_for_display


_HTML = """
<div class="review-canvas-shell">
  <div class="review-canvas-toolbar" role="toolbar" aria-label="图像查看工具">
    <button id="fitButton" type="button">适合窗口</button>
    <button id="minusButton" type="button" aria-label="缩小">−</button>
    <span id="zoomLabel" class="zoom-label"></span>
    <button id="plusButton" type="button" aria-label="放大">＋</button>
    <button id="nativeButton" type="button">原始像素 1:1</button>
    <button id="fullButton" type="button">全屏看图</button>
    <button id="themeButton" type="button" aria-pressed="false">柔和深色</button>
    <label class="layer-control"><input id="predictionToggle" type="checkbox">显示预测轮廓</label>
    <label class="slider-label">轮廓浓度<input id="opacitySlider" type="range" min="15" max="100" value="55" aria-label="预测轮廓浓度"></label>
    <label class="slider-label">屏幕亮度<input id="brightnessSlider" type="range" min="50" max="110" value="90" aria-label="图像屏幕亮度"></label>
  </div>
  <div class="review-canvas-toolbar editing-tools">
    <span id="modeLabel" class="mode-label"></span>
    <button id="drawButton" type="button" aria-pressed="true">标注</button>
    <button id="panButton" type="button" aria-pressed="false">拖动看图</button>
    <button id="eraseButton" type="button" aria-pressed="false">删除单点</button>
    <button id="undoButton" type="button">撤销</button>
    <button id="redoButton" type="button">恢复</button>
    <button id="clearButton" type="button">清空标注</button>
    <span id="statusLabel" role="status" aria-live="polite"></span>
  </div>
  <div class="review-canvas-stage">
    <div class="canvas-center"><canvas id="annotationCanvas" tabindex="0" aria-label="专家图像查看和标注区域"></canvas></div>
  </div>
  <div class="review-canvas-hint" id="interactionHint"></div>
  <div class="review-canvas-hint">琥珀细线＝模型预测 · 青绿空心标记＝您的标注。亮度调节只影响屏幕显示；放大后仍按原图像素保存。按住 B 暂看原图。</div>
</div>
"""

_CSS = """
.review-canvas-shell {
  width:100%; box-sizing:border-box; padding:12px; border-radius:12px;
  background:#eef1ef; color:#293b36; font-family:var(--st-font, sans-serif);
  border:1px solid #c8d0cd;
}
.review-canvas-toolbar { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
.review-canvas-toolbar button {
  min-height:40px; border:1px solid #b4c1ba; border-radius:7px; padding:7px 12px;
  background:#f8faf8; color:#253d35; cursor:pointer; font-size:15px; font-weight:500;
}
.review-canvas-toolbar button:hover { background:#e0e8e3; }
.review-canvas-toolbar button[aria-pressed="true"] { background:#365e52; border-color:#365e52; color:#fff; }
.review-canvas-toolbar button:focus-visible, #annotationCanvas:focus-visible { outline:3px solid #557fba; outline-offset:2px; }
.review-canvas-toolbar button:disabled { cursor:not-allowed; opacity:.45; }
.mode-label { font-weight:600; }
#statusLabel { font-size:14px; margin-left:auto; }
.zoom-label { min-width:46px; text-align:center; font-variant-numeric:tabular-nums; }
.layer-control,.slider-label { display:flex; align-items:center; gap:6px; font-size:14px; min-height:40px; }
.layer-control { margin-left:auto; }
input[type="checkbox"] { width:19px; height:19px; accent-color:#466c5e; }
input[type="range"] { width:82px; accent-color:#466c5e; }
.review-canvas-stage { width:100%; height:clamp(300px, 60vh, 720px); overflow:auto; background:#353b39; border-radius:8px; overscroll-behavior:contain; }
.canvas-center { display:flex; min-width:100%; min-height:100%; width:max-content; align-items:center; justify-content:center; }
#annotationCanvas { display:block; box-sizing:content-box; flex:none; border:0; cursor:crosshair; touch-action:none; }
.review-canvas-hint { margin-top:8px; font-size:14px; line-height:1.65; color:#435a51; }
.review-canvas-shell:fullscreen { padding:16px; overflow:auto; }
.review-canvas-shell:fullscreen .review-canvas-stage { height:calc(100vh - 220px); }
.review-canvas-shell.dim { background:#242e29; color:#e2e9e4; border-color:#54665b; }
.review-canvas-shell.dim .review-canvas-toolbar button { background:#35433a; color:#e2e9e4; border-color:#667b6c; }
.review-canvas-shell.dim .review-canvas-toolbar button[aria-pressed="true"] { background:#b5d5c3; color:#183326; }
.review-canvas-shell.dim .review-canvas-hint { color:#c0d0c5; }
@media(max-width:640px) {
 .review-canvas-shell { padding:8px; }
 .review-canvas-toolbar { gap:6px; }
 .review-canvas-toolbar button { min-height:44px; padding:7px 10px; }
 .layer-control { margin-left:0; }
 .review-canvas-stage { height:52vh; min-height:280px; }
 #statusLabel { width:100%; margin-left:0; }
}
"""

_JS = r"""
export default function(component) {
  const { data, parentElement, setStateValue } = component;
  const q = (selector) => parentElement.querySelector(selector);
  const shell = q('.review-canvas-shell'), canvas = q('#annotationCanvas');
  const stage = q('.review-canvas-stage'), context = canvas.getContext('2d');
  const width = Number(data.imageWidth), height = Number(data.imageHeight);
  const editable = !data.readOnly, minimum = Number(data.minimumPoints);
  const hint = data.geometryHint;
  const initial = data.existingGeometry;
  let points = initial && Array.isArray(initial.points_xy) ? initial.points_xy.map(p => p.slice()) : [];
  let undoStack = [], redoStack = [], gesture = null, imageReady = false, predictionReady = false;
  let destroyed = false, scale = 1, fitMode = true, mode = editable ? 'draw' : 'pan', bare = false;
  let assistanceShown = !!data.predictionShown;
  const listeners = [];
  const image = new Image(), prediction = new Image();
  const preferenceKey = 'phyevo-review-display-v1';
  function theme(dark) {
    shell.classList.toggle('dim', dark);
    q('#themeButton').textContent = dark ? '柔和浅色' : '柔和深色';
    q('#themeButton').setAttribute('aria-pressed', String(dark));
  }
  // Only cosmetic preferences persist; scientific pixels, drafts and labels do not.
  try {
    const prefs=JSON.parse(localStorage.getItem(preferenceKey)||'{}');
    if (Number.isFinite(prefs.brightness)) q('#brightnessSlider').value=Math.max(50,Math.min(110,prefs.brightness));
    if (Number.isFinite(prefs.opacity)) q('#opacitySlider').value=Math.max(15,Math.min(100,prefs.opacity));
    theme(prefs.dark === true);
  } catch (_) { /* Storage can be unavailable in restricted browsers. */ }
  function savePreferences() {
    try { localStorage.setItem(preferenceKey, JSON.stringify({
      brightness:Number(q('#brightnessSlider').value), opacity:Number(q('#opacitySlider').value), dark:shell.classList.contains('dim')
    })); } catch (_) { /* No review data depend on localStorage. */ }
  }
  canvas.width = width; canvas.height = height;
  const copy = (list) => list.map(p => p.slice());
  const on = (element, event, callback, options) => {
    element.addEventListener(event, callback, options);
    listeners.push(() => element.removeEventListener(event, callback, options));
  };
  q('#modeLabel').textContent = editable ? (hint === 'point' ? '点位标注' : hint === 'region' ? '区域套索' : '折线标注') : '查看图像';
  ['drawButton','eraseButton','undoButton','redoButton','clearButton'].forEach(id => q('#'+id).hidden = !editable);
  q('#predictionToggle').disabled = !data.predictionDataUrl;
  q('#opacitySlider').disabled = !data.predictionDataUrl;
  q('#interactionHint').textContent = !editable
    ? '先看原图，再按需打开预测。要画标注，请将上方“专家结论”设为“修正几何”。'
    : hint === 'region'
      ? '按住沿区域边缘画一圈，松开结束；再次绘制会替换轮廓，可撤销。放大后选“拖动看图”移动视野。'
      : '轻点添加空心标记；“删除单点”后点击错点。“拖动看图”只移动视野，不会增加标注。';
  function remember() { undoStack.push(copy(points)); if (undoStack.length > 80) undoStack.shift(); redoStack = []; }
  function publish() {
    if (!editable) return;
    setStateValue('geometry', points.length ? {
      coordinate_space:'source_pixels_xy', image_shape_yx:[height,width],
      geometry_type:data.geometryType, selection_kind:'interactive_canvas',
      points_xy:copy(points), valid_for_hint:points.length >= minimum
    } : null);
  }
  function status() {
    q('#statusLabel').textContent = editable
      ? points.length + ' 个点 · ' + (points.length >= minimum ? '待点击页面保存' : '至少需要 '+minimum+' 个点')
      : (q('#predictionToggle').checked ? '预测仅供对照' : '原图 · 预测已隐藏');
    q('#undoButton').disabled = undoStack.length === 0;
    q('#redoButton').disabled = redoStack.length === 0;
    q('#clearButton').disabled = points.length === 0;
    q('#zoomLabel').textContent = Math.round(scale*100) + '%';
    q('#minusButton').disabled = scale <= .05001;
    q('#plusButton').disabled = scale >= 8;
    for (const [id, m] of [['drawButton','draw'],['panButton','pan'],['eraseButton','erase']]) {
      q('#'+id).setAttribute('aria-pressed', String(mode === m));
    }
    canvas.style.cursor = mode === 'pan' ? 'grab' : mode === 'erase' ? 'pointer' : 'crosshair';
  }
  function expertGeometry() {
    if (!points.length || bare) return;
    context.save();
    context.lineJoin = 'round'; context.lineCap = 'round';
    if (hint !== 'point' && points.length > 1) {
      context.beginPath(); context.moveTo(...points[0]);
      points.slice(1).forEach(p => context.lineTo(...p));
      if (hint === 'region' && (!gesture || gesture.kind !== 'region') && points.length >= 3) context.closePath();
      context.strokeStyle = 'rgba(12,24,21,.7)'; context.lineWidth = 3.5/scale; context.stroke();
      context.strokeStyle = '#a1d8c8'; context.lineWidth = 1.7/scale; context.stroke();
    }
    const stride = hint === 'region' ? Math.max(1,Math.ceil(points.length/12)) : 1;
    points.forEach((p,i) => {
      if (i % stride !== 0 && i !== points.length-1) return;
      context.beginPath(); context.arc(p[0],p[1],5/scale,0,Math.PI*2);
      context.strokeStyle='rgba(12,24,21,.8)'; context.lineWidth=3.5/scale; context.stroke();
      context.strokeStyle='#a1d8c8'; context.lineWidth=1.7/scale; context.stroke();
    });
    context.restore();
  }
  function redraw() {
    if (!imageReady || destroyed) return;
    context.clearRect(0,0,width,height);
    context.save(); context.filter='brightness('+q('#brightnessSlider').value+'%)';
    context.drawImage(image,0,0,width,height); context.restore();
    if (!bare && predictionReady && q('#predictionToggle').checked) {
      context.save(); context.globalAlpha=Number(q('#opacitySlider').value)/100;
      context.drawImage(prediction,0,0,width,height); context.restore();
      if (!assistanceShown) { assistanceShown=true; setStateValue('prediction_shown',true); }
    }
    expertGeometry(); status();
  }
  function setScale(next, fit=false) {
    const bounds=canvas.getBoundingClientRect(), stageBounds=stage.getBoundingClientRect();
    const centerX=(stageBounds.left+stage.clientWidth/2-bounds.left)/scale;
    const centerY=(stageBounds.top+stage.clientHeight/2-bounds.top)/scale;
    scale=Math.max(.05,Math.min(8,next)); fitMode=fit;
    canvas.style.width=(width*scale)+'px'; canvas.style.height=(height*scale)+'px';
    stage.scrollLeft=Math.max(0,centerX*scale-stage.clientWidth/2);
    stage.scrollTop=Math.max(0,centerY*scale-stage.clientHeight/2);
    redraw();
  }
  function fit() {
    if (!stage.clientWidth || !stage.clientHeight) return;
    setScale(Math.min((stage.clientWidth-4)/width,(stage.clientHeight-4)/height),true);
    stage.scrollLeft=0; stage.scrollTop=0;
  }
  function sourcePoint(e) {
    const b=canvas.getBoundingClientRect();
    return [Math.max(0,Math.min(width-1,(e.clientX-b.left)*width/b.width)),
            Math.max(0,Math.min(height-1,(e.clientY-b.top)*height/b.height))];
  }
  function down(e) {
    if (!imageReady || !e.isPrimary || e.button !== 0 || gesture) return;
    e.preventDefault(); canvas.focus({preventScroll:true}); canvas.setPointerCapture(e.pointerId);
    gesture={kind:mode,x:e.clientX,y:e.clientY,left:stage.scrollLeft,top:stage.scrollTop,before:copy(points),id:e.pointerId};
    if (editable && mode === 'draw' && hint === 'region') {
      gesture.kind='region'; points=[sourcePoint(e)];
    }
    redraw();
  }
  function move(e) {
    if (!gesture || e.pointerId !== gesture.id) return;
    e.preventDefault();
    if (gesture.kind === 'pan') {
      stage.scrollLeft=gesture.left-(e.clientX-gesture.x); stage.scrollTop=gesture.top-(e.clientY-gesture.y);
    } else if (gesture.kind === 'region') {
      const p=sourcePoint(e), last=points[points.length-1];
      if (Math.hypot(p[0]-last[0],p[1]-last[1])*scale >= 3 && points.length < 2048) { points.push(p); redraw(); }
    }
  }
  function up(e) {
    if (!gesture || e.pointerId !== gesture.id) return;
    e.preventDefault();
    const g=gesture; gesture=null;
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    if (e.type === 'pointercancel') { points=g.before; redraw(); return; }
    const isClick=Math.hypot(e.clientX-g.x,e.clientY-g.y)<7;
    if (g.kind === 'region') {
      if (points.length >= minimum) { undoStack.push(g.before); redoStack=[]; publish(); }
      else points=g.before;
    } else if (editable && isClick && g.kind === 'draw') {
      if (points.length < 2048) { remember(); points.push(sourcePoint(e)); publish(); }
    } else if (editable && isClick && g.kind === 'erase' && points.length) {
      const p=sourcePoint(e);
      let nearest=-1, distance=14/scale;
      points.forEach((v,i)=>{const d=Math.hypot(p[0]-v[0],p[1]-v[1]); if(d<distance){nearest=i;distance=d;}});
      if(nearest>=0){remember();points.splice(nearest,1);publish();}
    }
    redraw();
  }
  function undo() { if(undoStack.length){redoStack.push(copy(points));points=undoStack.pop();publish();redraw();} }
  function redo() { if(redoStack.length){undoStack.push(copy(points));points=redoStack.pop();publish();redraw();} }
  on(canvas,'pointerdown',down); on(canvas,'pointermove',move);
  on(canvas,'pointerup',up); on(canvas,'pointercancel',up);
  on(q('#fitButton'),'click',fit);
  on(q('#nativeButton'),'click',()=>setScale(1));
  on(q('#plusButton'),'click',()=>setScale(scale*1.5));
  on(q('#minusButton'),'click',()=>setScale(scale/1.5));
  on(q('#predictionToggle'),'change',redraw);
  on(q('#opacitySlider'),'input',redraw); on(q('#brightnessSlider'),'input',redraw);
  on(q('#opacitySlider'),'change',savePreferences); on(q('#brightnessSlider'),'change',savePreferences);
  on(q('#themeButton'),'click',()=>{theme(!shell.classList.contains('dim'));savePreferences();});
  for(const [id,m] of [['drawButton','draw'],['panButton','pan'],['eraseButton','erase']]){
    on(q('#'+id),'click',()=>{mode=m;status();});
  }
  on(q('#undoButton'),'click',undo); on(q('#redoButton'),'click',redo);
  on(q('#clearButton'),'click',()=>{
    if(points.length && window.confirm('清空当前未保存的标注？清空后仍可点击“撤销”。')){remember();points=[];publish();redraw();}
  });
  on(q('#fullButton'),'click',async()=>{
    try {
      if(document.fullscreenElement) await document.exitFullscreen();
      else await shell.requestFullscreen();
      fit();
    } catch (_) { q('#statusLabel').textContent='此浏览器不支持全屏，可使用放大按钮'; }
  });
  on(canvas,'keydown',e=>{
    if(e.key.toLowerCase()==='b'){e.preventDefault();bare=true;redraw();}
    if(editable && (e.ctrlKey||e.metaKey) && e.key.toLowerCase()==='z'){
      e.preventDefault(); if(e.shiftKey)redo();else undo();
    }
  });
  on(canvas,'keyup',e=>{if(e.key.toLowerCase()==='b'){bare=false;redraw();}});
  on(canvas,'blur',()=>{bare=false;redraw();});
  image.onload=()=>{imageReady=true;fit();};
  image.onerror=()=>{q('#statusLabel').textContent='图像加载失败，请刷新页面';};
  prediction.onload=()=>{predictionReady=true;redraw();};
  if(data.predictionDataUrl) prediction.src=data.predictionDataUrl;
  image.src=data.imageDataUrl;
  const observer=new ResizeObserver(()=>{if(fitMode)fit();});
  observer.observe(stage);
  status();
  return ()=>{destroyed=true;observer.disconnect();listeners.forEach(remove=>remove());};
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
    read_only: bool = False,
    prediction_shown: bool = False,
) -> dict[str, Any]:
    if geometry_hint not in _GEOMETRY_TYPE:
        raise ValueError(f"Unsupported geometry hint {geometry_hint!r}")
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("Review canvas source image must be two-dimensional")
    prediction_url = None
    if prediction_mask is not None:
        mask = np.asarray(prediction_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("Prediction mask shape does not match image")
        boundary = mask & ~binary_erosion(mask)
        rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
        rgba[boundary] = (223, 183, 115, 255)
        prediction_url = _image_data_url(rgba)
    return {
        "imageDataUrl": _image_data_url(normalize_for_display(values)),
        "predictionDataUrl": prediction_url,
        "imageWidth": int(values.shape[1]),
        "imageHeight": int(values.shape[0]),
        "geometryHint": geometry_hint,
        "geometryType": _GEOMETRY_TYPE[geometry_hint],
        "minimumPoints": _MINIMUM_POINTS[geometry_hint],
        "readOnly": read_only,
        "predictionShown": prediction_shown,
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


def review_image_canvas(
    image: np.ndarray,
    *,
    prediction_mask: np.ndarray | None,
    geometry_hint: str,
    existing_geometry: Mapping[str, Any] | None,
    key: str,
    read_only: bool = False,
    prediction_shown: bool = False,
) -> tuple[dict[str, Any] | None, bool]:
    """Return geometry and whether the model layer was actually displayed."""

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
        read_only=read_only,
        prediction_shown=prediction_shown,
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
        default={"geometry": existing_geometry, "prediction_shown": prediction_shown},
        width="stretch",
        height="content",
        on_geometry_change=lambda: None,
        on_prediction_shown_change=lambda: None,
    )
    geometry = normalize_canvas_geometry(
        result.geometry,
        image_shape=tuple(int(item) for item in np.asarray(image).shape),
        geometry_hint=geometry_hint,
    )
    return geometry, bool(prediction_shown or result.prediction_shown)


def review_geometry_canvas(
    image: np.ndarray,
    *,
    prediction_mask: np.ndarray | None,
    geometry_hint: str,
    existing_geometry: Mapping[str, Any] | None,
    key: str,
) -> dict[str, Any] | None:
    """Compatibility wrapper for callers that only need editable geometry."""
    geometry, _ = review_image_canvas(
        image, prediction_mask=prediction_mask, geometry_hint=geometry_hint,
        existing_geometry=existing_geometry, key=key,
    )
    return geometry
