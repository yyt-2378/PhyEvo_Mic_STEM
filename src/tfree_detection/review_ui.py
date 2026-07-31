"""Offline expert-review bundles with blinded/assisted provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.io import imsave

from .governance import (
    CandidateRecord,
    DataPartition,
    candidate_records_from_result,
)
from .models import DetectionResult
from .review import (
    ReviewAction,
    ReviewCandidatePacket,
    ReviewResponse,
    make_review_packet,
)


@dataclass(frozen=True)
class ReviewBundleItem:
    """One candidate plus its own source image for a multi-image bundle."""

    candidate: CandidateRecord
    normalized_image: np.ndarray
    mask: np.ndarray
    source_provenance: Mapping[str, Any]

    def validate(self) -> None:
        image = np.asarray(self.normalized_image)
        mask = np.asarray(self.mask)
        if image.ndim != 2 or mask.shape != image.shape:
            raise ValueError(
                "Review item image must be 2-D and mask must match its shape"
            )
        if self.candidate.partition == DataPartition.LOCKED_EXTERNAL_TEST.value:
            raise ValueError(
                "Locked external-test candidates cannot enter a review bundle"
            )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _write_json_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _candidate_crop(
    result: DetectionResult,
    candidate: CandidateRecord,
    *,
    padding_pixels: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    height, width = result.image_shape
    x1, y1, x2, y2 = (
        candidate.geometry_bbox_xyxy or candidate.bbox_xyxy
    )
    crop_bbox = (
        max(0, x1 - padding_pixels),
        max(0, y1 - padding_pixels),
        min(width, x2 + padding_pixels),
        min(height, y2 + padding_pixels),
    )
    cx1, cy1, cx2, cy2 = crop_bbox
    gray = np.clip(
        result.normalized_image[cy1:cy2, cx1:cx2], 0, 1
    )
    raw = np.round(gray * 255).astype(np.uint8)
    rgb = np.repeat(raw[..., None], 3, axis=-1).astype(np.float32)
    matched_region = next(
        (
            region
            for region in result.regions
            if region.candidate_id == candidate.candidate_id
        ),
        None,
    )
    if matched_region is None:
        candidate_mask = np.zeros_like(result.mask, dtype=bool)
        candidate_mask[y1:y2, x1:x2] = result.final_geometry_mask[
            y1:y2, x1:x2
        ]
    else:
        candidate_mask = result.final_geometry_for_region(matched_region)
    local_mask = candidate_mask[cy1:cy2, cx1:cx2]
    if np.any(local_mask):
        orange = np.asarray([255, 132, 0], dtype=np.float32)
        rgb[local_mask] = 0.52 * rgb[local_mask] + 0.48 * orange
        boundary = local_mask & ~binary_erosion(local_mask)
        rgb[boundary] = np.asarray([0, 229, 255], dtype=np.float32)
    return raw, np.round(np.clip(rgb, 0, 255)).astype(np.uint8), crop_bbox


def _candidate_crop_item(
    item: ReviewBundleItem,
    *,
    padding_pixels: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    item.validate()
    image = np.asarray(item.normalized_image, dtype=float)
    mask = np.asarray(item.mask, dtype=bool)
    candidate = item.candidate
    height, width = image.shape
    x1, y1, x2, y2 = (
        candidate.geometry_bbox_xyxy or candidate.bbox_xyxy
    )
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Candidate bbox lies outside its source image")
    crop_bbox = (
        max(0, x1 - padding_pixels),
        max(0, y1 - padding_pixels),
        min(width, x2 + padding_pixels),
        min(height, y2 + padding_pixels),
    )
    cx1, cy1, cx2, cy2 = crop_bbox
    raw = np.round(
        np.clip(image[cy1:cy2, cx1:cx2], 0, 1) * 255
    ).astype(np.uint8)
    rgb = np.repeat(raw[..., None], 3, axis=-1).astype(np.float32)
    local_mask = mask[cy1:cy2, cx1:cx2]
    if np.any(local_mask):
        orange = np.asarray([255, 132, 0], dtype=np.float32)
        rgb[local_mask] = 0.52 * rgb[local_mask] + 0.48 * orange
        boundary = local_mask & ~binary_erosion(local_mask)
        rgb[boundary] = np.asarray([0, 229, 255], dtype=np.float32)
    return raw, np.round(np.clip(rgb, 0, 255)).astype(np.uint8), crop_bbox


def _bundle_html(
    packets: Sequence[ReviewCandidatePacket],
    crop_bboxes: Mapping[str, Sequence[int]],
    *,
    title: str,
) -> str:
    packet_json = json.dumps(
        [packet.to_dict() for packet in packets],
        ensure_ascii=False,
    ).replace("</", "<\\/")
    bbox_json = json.dumps(crop_bboxes).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: #0b1020; color: #e8edf7; }}
header {{ padding: 14px 22px; background: #121a30; display: flex;
  align-items: center; gap: 18px; border-bottom: 1px solid #26304a; }}
header h1 {{ font-size: 18px; margin: 0; }}
header input {{ background: #080d1a; border: 1px solid #3b4664;
  color: white; padding: 8px; border-radius: 6px; }}
#layout {{ display: grid; grid-template-columns: minmax(460px, 1fr) 390px;
  min-height: calc(100vh - 64px); }}
#viewer {{ padding: 18px; display: flex; justify-content: center;
  align-items: flex-start; }}
#canvasWrap {{ position: relative; max-width: 100%; }}
#image {{ display: block; max-width: min(100%, 950px); max-height: 78vh;
  image-rendering: pixelated; border: 1px solid #34405e; }}
#canvas {{ position: absolute; inset: 0; width: 100%; height: 100%;
  cursor: crosshair; }}
#panel {{ padding: 18px; background: #10182b; border-left: 1px solid #26304a;
  overflow-y: auto; }}
.muted {{ color: #94a3b8; font-size: 13px; }}
.badge {{ display: inline-block; border: 1px solid #52617f; border-radius: 999px;
  padding: 4px 9px; margin: 2px; font-size: 12px; }}
label {{ display: block; margin-top: 13px; margin-bottom: 5px; }}
select, textarea, #labels {{ width: 100%; box-sizing: border-box;
  background: #080d1a; color: white; border: 1px solid #3b4664;
  padding: 9px; border-radius: 6px; }}
button {{ border: 0; border-radius: 6px; padding: 9px 12px; cursor: pointer;
  background: #2563eb; color: white; margin: 8px 5px 0 0; }}
button.secondary {{ background: #334155; }}
button.export {{ background: #059669; }}
#progress {{ margin-left: auto; color: #a5b4fc; }}
#meta {{ white-space: pre-wrap; font-size: 12px; background: #080d1a;
  padding: 10px; border-radius: 6px; max-height: 210px; overflow: auto; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <label style="margin:0">Reviewer ID
    <input id="reviewer" autocomplete="off" placeholder="expert-a">
  </label>
  <span id="condition" class="badge"></span>
  <span id="progress"></span>
</header>
<div id="layout">
  <main id="viewer"><div id="canvasWrap">
    <img id="image" alt="candidate crop">
    <canvas id="canvas"></canvas>
  </div></main>
  <aside id="panel">
    <h2 id="candidateTitle"></h2>
    <div id="actions"></div>
    <label for="action">Decision / correction</label>
    <select id="action"></select>
    <label for="labels">Defect labels, comma separated</label>
    <input id="labels" placeholder="coherence-loss, phase-slip">
    <label for="notes">Notes</label>
    <textarea id="notes" rows="3"></textarea>
    <p class="muted">For geometry adjustment, click the image: one point for
    point, two or more for line, three or more for contour. Coordinates are
    stored in original source pixels.</p>
    <button class="secondary" id="clear">Clear geometry</button>
    <pre id="meta"></pre>
    <button class="secondary" id="prev">Previous</button>
    <button id="saveNext">Save + next</button>
    <button class="export" id="export">Export review_responses.json</button>
  </aside>
</div>
<script>
const packets = {packet_json};
const cropBboxes = {bbox_json};
let index = 0;
let startedAt = performance.now();
let points = [];
const responses = new Map();
const image = document.getElementById("image");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
function packet() {{ return packets[index]; }}
function assisted(p) {{ return p.model_assistance_policy === "assisted"; }}
function resizeCanvas() {{
  canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
  draw();
}}
function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#00e5ff"; ctx.strokeStyle = "#00e5ff"; ctx.lineWidth = 2;
  points.forEach((p, i) => {{
    ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, Math.PI * 2); ctx.fill();
    if (i) {{ ctx.beginPath(); ctx.moveTo(points[i-1][0], points[i-1][1]);
      ctx.lineTo(p[0], p[1]); ctx.stroke(); }}
  }});
}}
canvas.addEventListener("click", event => {{
  const rect = canvas.getBoundingClientRect();
  points.push([
    (event.clientX - rect.left) * canvas.width / rect.width,
    (event.clientY - rect.top) * canvas.height / rect.height
  ]); draw();
}});
function show() {{
  const p = packet(); const c = p.candidate;
  document.getElementById("candidateTitle").textContent = c.candidate_id;
  document.getElementById("condition").textContent =
    assisted(p) ? "ASSISTED" : "BLINDED";
  document.getElementById("progress").textContent =
    `${{index + 1}} / ${{packets.length}}`;
  document.getElementById("actions").innerHTML =
    p.allowed_actions.map(x => `<span class="badge">${{x}}</span>`).join("");
  const action = document.getElementById("action");
  action.innerHTML = p.allowed_actions.map(x =>
    `<option value="${{x}}">${{x}}</option>`).join("");
  image.src = assisted(p) ? p.overlay_artifact : p.image_artifact;
  image.onload = resizeCanvas;
  document.getElementById("meta").textContent = assisted(p)
    ? JSON.stringify(c, null, 2)
    : JSON.stringify({{candidate_id: c.candidate_id, geometry: c.geometry,
        bbox_xyxy: c.bbox_xyxy}}, null, 2);
  const old = responses.get(c.candidate_id);
  document.getElementById("labels").value =
    old ? old.defect_labels.join(", ") : "";
  document.getElementById("notes").value = old ? old.notes : "";
  action.value = old ? old.action : p.allowed_actions[0];
  points = old && old.geometry_payload ? old.geometry_payload.local_points : [];
  startedAt = performance.now(); draw();
}}
function geometryPayload(action) {{
  if (!action.startsWith("adjust_")) return null;
  const p = packet(); const bbox = cropBboxes[p.candidate.candidate_id];
  const sx = image.naturalWidth / canvas.width;
  const sy = image.naturalHeight / canvas.height;
  const source = points.map(q => [bbox[0] + q[0] * sx, bbox[1] + q[1] * sy]);
  return {{coordinate_space: "source_pixels", crop_bbox_xyxy: bbox,
    local_points: points, points_xy: source}};
}}
function save() {{
  const p = packet(); const reviewer = document.getElementById("reviewer").value.trim();
  if (!reviewer) {{ alert("Reviewer ID is required."); return false; }}
  const action = document.getElementById("action").value;
  const labels = document.getElementById("labels").value.split(",")
    .map(x => x.trim()).filter(Boolean);
  if ((action === "accept" || action === "change_type" ||
      action.startsWith("adjust_")) && !labels.length) {{
    alert("Accepted or adjusted candidates need at least one label.");
    return false;
  }}
  const minimum = action === "adjust_point" ? 1 :
    action === "adjust_line" ? 2 : action === "adjust_contour" ? 3 : 0;
  if (points.length < minimum) {{
    alert(`This action needs at least ${{minimum}} geometry points.`); return false;
  }}
  const candidateId = p.candidate.candidate_id;
  const id = crypto.randomUUID ? crypto.randomUUID() :
    `${{Date.now()}}-${{Math.random().toString(16).slice(2)}}`;
  responses.set(candidateId, {{
    response_id: oldResponseId(candidateId) || id,
    candidate_id: candidateId, reviewer_id: reviewer, action,
    defect_labels: labels,
    annotation_duration_seconds: Math.max(0.01,
      (performance.now() - startedAt) / 1000),
    model_assistance_shown: assisted(p),
    geometry_payload: geometryPayload(action),
    notes: document.getElementById("notes").value
  }});
  return true;
}}
function oldResponseId(id) {{
  const old = responses.get(id); return old ? old.response_id : null;
}}
document.getElementById("clear").onclick = () => {{ points = []; draw(); }};
document.getElementById("prev").onclick = () => {{
  if (index > 0) {{ index -= 1; show(); }}
}};
document.getElementById("saveNext").onclick = () => {{
  if (!save()) return;
  if (index < packets.length - 1) {{ index += 1; show(); }}
}};
document.getElementById("export").onclick = () => {{
  if (!save()) return;
  const blob = new Blob([JSON.stringify([...responses.values()], null, 2)],
    {{type: "application/json"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = "review_responses.json";
  link.click(); URL.revokeObjectURL(link.href);
}};
show();
</script>
</body>
</html>
"""


def build_review_bundle(
    result: DetectionResult,
    output_directory: str | Path,
    *,
    show_model_assistance: bool,
    dataset_id: str | None = None,
    partition: DataPartition | None = None,
    padding_pixels: int = 32,
    title: str = "PhyEvo-Mic expert review",
) -> Mapping[str, Any]:
    """Create a new self-contained static review bundle."""

    if padding_pixels < 0:
        raise ValueError("padding_pixels cannot be negative")
    if partition == DataPartition.LOCKED_EXTERNAL_TEST:
        raise ValueError(
            "Locked external-test candidates cannot enter a review bundle"
        )
    records = candidate_records_from_result(
        result,
        dataset_id=dataset_id,
        partition=partition,
    )
    if not records:
        raise ValueError("Review bundle requires at least one candidate")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    packets = []
    crop_bboxes: dict[str, list[int]] = {}
    artifact_hashes = {}
    for index, candidate in enumerate(records, start=1):
        raw, overlay, crop_bbox = _candidate_crop(
            result,
            candidate,
            padding_pixels=padding_pixels,
        )
        stable_suffix = hashlib.sha256(
            candidate.candidate_id.encode("utf-8")
        ).hexdigest()[:12]
        stem = f"candidate_{index:04d}_{stable_suffix}"
        raw_path = assets / f"{stem}_raw.png"
        overlay_path = assets / f"{stem}_assisted.png"
        imsave(raw_path, raw, check_contrast=False)
        raw_relative = raw_path.relative_to(output).as_posix()
        if show_model_assistance:
            imsave(overlay_path, overlay, check_contrast=False)
            overlay_relative = overlay_path.relative_to(output).as_posix()
            artifact_hashes[overlay_relative] = _sha256(overlay_path)
        else:
            overlay_relative = raw_relative
        packets.append(
            make_review_packet(
                candidate,
                image_artifact=raw_relative,
                overlay_artifact=overlay_relative,
                show_model_assistance=show_model_assistance,
            )
        )
        crop_bboxes[candidate.candidate_id] = list(crop_bbox)
        artifact_hashes[raw_relative] = _sha256(raw_path)
    packet_path = output / "review_packets.json"
    _write_json_new(
        packet_path, [packet.to_dict() for packet in packets]
    )
    html_path = output / "index.html"
    with html_path.open("x", encoding="utf-8") as stream:
        stream.write(
            _bundle_html(
                packets,
                crop_bboxes,
                title=title,
            )
        )
    manifest = {
        "bundle_version": "1.0",
        "task": result.task,
        "candidate_count": len(records),
        "model_assistance_shown": show_model_assistance,
        "condition": "assisted" if show_model_assistance else "blinded",
        "source_provenance": result.metadata.get("input_provenance", {}),
        "crop_bboxes_xyxy": crop_bboxes,
        "artifact_sha256": artifact_hashes,
        "review_packets_sha256": _sha256(packet_path),
        "index_html_sha256": _sha256(html_path),
    }
    manifest["bundle_content_sha256"] = hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    _write_json_new(output / "bundle_manifest.json", manifest)
    return manifest


def build_multisource_review_bundle(
    items: Sequence[ReviewBundleItem],
    output_directory: str | Path,
    *,
    show_model_assistance: bool,
    padding_pixels: int = 48,
    title: str = "PhyEvo-Mic prospective expert review",
) -> Mapping[str, Any]:
    """Create one blinded/assisted bundle spanning multiple source images."""

    if padding_pixels < 0:
        raise ValueError("padding_pixels cannot be negative")
    if not items:
        raise ValueError("Review bundle requires at least one candidate")
    candidate_ids: set[str] = set()
    for item in items:
        item.validate()
        if item.candidate.candidate_id in candidate_ids:
            raise ValueError("Candidate ids must be unique within a bundle")
        candidate_ids.add(item.candidate.candidate_id)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    packets = []
    crop_bboxes: dict[str, list[int]] = {}
    artifact_hashes = {}
    source_provenance: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(items, start=1):
        candidate = item.candidate
        raw, overlay, crop_bbox = _candidate_crop_item(
            item, padding_pixels=padding_pixels
        )
        stable_suffix = hashlib.sha256(
            candidate.candidate_id.encode("utf-8")
        ).hexdigest()[:12]
        stem = f"candidate_{index:04d}_{stable_suffix}"
        raw_path = assets / f"{stem}_raw.png"
        overlay_path = assets / f"{stem}_assisted.png"
        imsave(raw_path, raw, check_contrast=False)
        raw_relative = raw_path.relative_to(output).as_posix()
        if show_model_assistance:
            imsave(overlay_path, overlay, check_contrast=False)
            overlay_relative = overlay_path.relative_to(output).as_posix()
            artifact_hashes[overlay_relative] = _sha256(overlay_path)
        else:
            overlay_relative = raw_relative
        packets.append(
            make_review_packet(
                candidate,
                image_artifact=raw_relative,
                overlay_artifact=overlay_relative,
                show_model_assistance=show_model_assistance,
            )
        )
        crop_bboxes[candidate.candidate_id] = list(crop_bbox)
        artifact_hashes[raw_relative] = _sha256(raw_path)
        source_provenance.setdefault(
            candidate.source_id, item.source_provenance
        )
    packet_path = output / "review_packets.json"
    _write_json_new(
        packet_path, [packet.to_dict() for packet in packets]
    )
    html_path = output / "index.html"
    with html_path.open("x", encoding="utf-8") as stream:
        stream.write(
            _bundle_html(packets, crop_bboxes, title=title)
        )
    manifest = {
        "bundle_version": "1.1-multisource",
        "task": "atomic_point_defect",
        "candidate_count": len(items),
        "source_count": len(source_provenance),
        "model_assistance_shown": show_model_assistance,
        "condition": (
            "assisted" if show_model_assistance else "blinded"
        ),
        "source_provenance": source_provenance,
        "crop_bboxes_xyxy": crop_bboxes,
        "artifact_sha256": artifact_hashes,
        "review_packets_sha256": _sha256(packet_path),
        "index_html_sha256": _sha256(html_path),
    }
    manifest["bundle_content_sha256"] = hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    _write_json_new(output / "bundle_manifest.json", manifest)
    return manifest


def load_and_validate_review_responses(
    response_path: str | Path,
    packet_path: str | Path,
) -> list[ReviewResponse]:
    """Validate a browser export against its exact packet assistance policy."""

    packet_payload = json.loads(
        Path(packet_path).read_text(encoding="utf-8")
    )
    response_payload = json.loads(
        Path(response_path).read_text(encoding="utf-8")
    )
    if not isinstance(packet_payload, list) or not isinstance(
        response_payload, list
    ):
        raise TypeError("Packets and responses must both be JSON arrays")
    packets = {
        str(item["candidate"]["candidate_id"]): item
        for item in packet_payload
    }
    responses = [
        ReviewResponse.from_dict(item) for item in response_payload
    ]
    response_ids: set[str] = set()
    candidate_ids: set[str] = set()
    for response in responses:
        if response.response_id in response_ids:
            raise ValueError("Duplicate review response id")
        if response.candidate_id in candidate_ids:
            raise ValueError("One export may contain one response per candidate")
        response_ids.add(response.response_id)
        candidate_ids.add(response.candidate_id)
        packet = packets.get(response.candidate_id)
        if packet is None:
            raise ValueError("Response references a candidate outside the bundle")
        if response.action.value not in packet["allowed_actions"]:
            raise ValueError("Response action is not allowed for this geometry")
        expected_assistance = (
            packet["model_assistance_policy"] == "assisted"
        )
        if response.model_assistance_shown != expected_assistance:
            raise ValueError("Response assistance condition does not match packet")
        if response.geometry_payload is not None:
            points = response.geometry_payload.get("points_xy", [])
            minimum = {
                ReviewAction.ADJUST_POINT: 1,
                ReviewAction.ADJUST_LINE: 2,
                ReviewAction.ADJUST_CONTOUR: 3,
            }.get(response.action, 0)
            if len(points) < minimum:
                raise ValueError("Geometry response has too few points")
    return responses
