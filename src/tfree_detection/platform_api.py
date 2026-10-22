"""HTTP adapter for private microscopy workspaces; never mounts the legacy review DB."""
from __future__ import annotations

import hashlib
import io
import json
import math
import secrets
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .platform_store import PlatformStore, digest, new_id

TASKS = {
    "atomic_point_defect": "原子点缺陷", "phase_boundary": "相界与界面",
    "stacking_fault": "堆垛层错", "amorphous": "非晶区域",
    "symmetry_relation": "对称关系异常",
}
MAX_UPLOAD = 20 * 1024 * 1024
MAX_PIXELS = 2048 * 2048
COOKIE = "phyevo_session"


class Login(BaseModel):
    code: str = Field(min_length=1, max_length=256)


class Name(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class UserCreate(Name):
    role: str = "researcher"


class JobCreate(BaseModel):
    task: str = "atomic_point_defect"


class Share(BaseModel):
    expert_id: str = Field(min_length=32, max_length=32)
    active: bool


class Feedback(BaseModel):
    action: str
    note: str = Field(default="", max_length=3000)
    geometry: list[dict] = Field(default_factory=list, max_length=500)
    prediction_seen: bool = False
    request_id: str = Field(min_length=16, max_length=80)


class ResearchRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=3000)
    consent: bool = False


def decode_image(data: bytes):
    """No resampling of analysis input. Reject stacks instead of silently selecting a plane."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"PNG", "TIFF", "JPEG"}:
                raise ValueError("仅支持 PNG、TIFF、JPEG")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("这是多层图像，请先导出要分析的单层；不会自动丢弃其他层")
            w, h = image.size
            if min(w, h) < 64 or max(w, h) > 4096 or w * h > MAX_PIXELS:
                raise ValueError("图像至少 64×64，最多 419 万像素，单边不超过 4096 像素")
            fmt = image.format
            if image.mode in {"L", "I", "F", "I;16", "I;16B", "I;16L"}:
                arr = np.array(image)
            else:
                # Accept grayscale content stored in RGB, but never silently mix colors.
                rgba = np.array(image.convert("RGBA"))
                if not (np.all(rgba[..., 0] == rgba[..., 1]) and np.all(rgba[..., 1] == rgba[..., 2]) and np.all(rgba[..., 3] == 255)):
                    raise ValueError("请上传二维灰度原图；彩色标注图或透明图不适合定量分析")
                arr = rgba[..., 0]
            if not np.isfinite(arr).all():
                raise ValueError("图像含有无效数值")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法解码图像，请检查文件格式") from exc
    lo, hi = np.percentile(arr, [0.5, 99.5])
    preview = np.uint8(np.clip((arr.astype(float) - lo) / max(float(hi - lo), 1e-9), 0, 1) * 255)
    return arr, preview, fmt


def create_app(root: Path, *, origin: str, secure: bool = True, repo: Path | None = None):
    repo = repo or Path(__file__).resolve().parents[2]
    store = PlatformStore(root)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    cookie = "__Host-phyevo_session" if secure else COOKIE
    attempts: dict[str, list[float]] = {}
    rate_lock = threading.Lock()
    upload_lock = threading.Lock()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "source-checkout"

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        # Limit JSON bodies too; uploads have a streamed hard limit below.
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"detail": "无效请求长度"}, status_code=400)
        is_upload = request.url.path.endswith("/samples") and request.method == "POST"
        limit = MAX_UPLOAD if is_upload else 256 * 1024
        if length > limit:
            return JSONResponse({"detail": "请求内容过大"}, status_code=413)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            if request.headers.get("origin") != origin.rstrip("/"):
                return JSONResponse({"detail": "请从本站页面操作"}, status_code=403)
            if not is_upload:
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > limit:
                        return JSONResponse({"detail": "请求内容过大"}, status_code=413)
                request._body = bytes(body)
        response = await call_next(request)
        response.headers.update({
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'; form-action 'self'",
        })
        return response

    def identity(request: Request):
        user = store.session(request.cookies.get(cookie, ""))
        if not user:
            raise HTTPException(401, "请使用您的个人激活码登录")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if not secrets.compare_digest(request.headers.get("x-csrf-token", ""), user["csrf"]):
                raise HTTPException(403, "会话校验失败，请刷新页面后重试")
        return user

    def project(pid, user, owner=False):
        row = store.project(pid, user["id"], owner=owner)
        if not row:
            raise HTTPException(404, "项目不存在或尚未授权")
        return row

    def sample(sid, user, owner=False):
        row = store.one("SELECT * FROM samples WHERE id=?", (sid,))
        if not row:
            raise HTTPException(404, "图像不存在或尚未授权")
        project(row["project_id"], user, owner)
        return row

    def job(jid, user, owner=False):
        row = store.one("SELECT * FROM jobs WHERE id=?", (jid,))
        if not row:
            raise HTTPException(404, "任务不存在或尚未授权")
        sample(row["sample_id"], user, owner)
        return row

    def admin(user):
        if user["role"] != "admin":
            raise HTTPException(403, "只有平台管理员可管理激活码")

    @app.get("/healthz")
    def health():
        heartbeat = store.root / "worker.heartbeat"
        healthy = heartbeat.exists() and time.time() - heartbeat.stat().st_mtime < 30
        return {"status": "ok", "worker": "online" if healthy else "offline"}

    @app.post("/api/login")
    def login(body: Login, response: Response):
        # Global + code-prefix buckets do not trust client-supplied X-Forwarded-For.
        key = digest(body.code.split(".")[1] if "." in body.code else "invalid")
        with rate_lock:
            now = time.monotonic()
            for old in list(attempts):
                attempts[old] = [t for t in attempts[old] if now - t < 60]
                if not attempts[old]:
                    del attempts[old]
            if len(attempts.get("global", [])) >= 100 or len(attempts.get(key, [])) >= 10:
                raise HTTPException(429, "尝试过于频繁，请一分钟后再试")
            attempts.setdefault("global", []).append(now)
            attempts.setdefault(key, []).append(now)
        user = store.authenticate(body.code.strip())
        if not user:
            raise HTTPException(401, "激活码无效或已停用，请联系邀请人")
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
        store.execute("INSERT INTO sessions VALUES(?,?,?,?)", (digest(token), user["id"], csrf, time.time() + 43200))
        response.set_cookie(cookie, token, max_age=43200, httponly=True, secure=secure, samesite="strict", path="/")
        return {**user, "csrf": csrf}

    @app.post("/api/logout")
    def logout(request: Request, response: Response, user=Depends(identity)):
        store.execute("UPDATE sessions SET expires=0 WHERE token_hash=?", (digest(request.cookies.get(cookie, "")),))
        response.delete_cookie(cookie, path="/", secure=secure, httponly=True, samesite="strict")
        return {"ok": True}

    @app.get("/api/me")
    def me(user=Depends(identity)):
        return user

    @app.get("/api/capabilities")
    def capabilities(user=Depends(identity)):
        return {"tasks": TASKS, "source_commit": revision, "platform_version": "1.0-private-workspaces",
                "engine": "TrainingFreeDetectionFramework", "rsi_execution": "request_only",
                "student_training": "not_enabled_in_public_service", "max_upload_mib": 20}

    @app.get("/api/users")
    def users(user=Depends(identity)):
        admin(user)
        return store.all("SELECT id,name,role,disabled,created FROM users ORDER BY created")

    @app.post("/api/users")
    def create_user(body: UserCreate, user=Depends(identity)):
        admin(user)
        if body.role not in {"researcher", "expert"} or not body.name.strip():
            raise HTTPException(422, "请选择研究用户或专家，并填写名称")
        return store.create_user(body.name, body.role)

    @app.post("/api/users/{uid}/disable")
    def disable_user(uid: str, user=Depends(identity)):
        admin(user)
        if uid == user["id"]:
            raise HTTPException(400, "不能停用当前管理员")
        store.execute("UPDATE users SET disabled=1 WHERE id=?", (uid,))
        return {"ok": True}

    @app.get("/api/experts")
    def experts(user=Depends(identity)):
        return store.all("SELECT id,name FROM users WHERE role='expert' AND disabled=0 ORDER BY name")

    @app.post("/api/users/{uid}/renew")
    def renew_user(uid: str, user=Depends(identity)):
        admin(user)
        if uid == user["id"]:
            raise HTTPException(400, "当前管理员激活码需在服务器维护流程中更新")
        try:
            return store.renew_code(uid)
        except ValueError as exc:
            raise HTTPException(404, "账号不存在") from exc

    @app.get("/api/projects")
    def projects(user=Depends(identity)):
        return store.projects(user["id"])

    @app.post("/api/projects")
    def new_project(body: Name, user=Depends(identity)):
        if not body.name.strip():
            raise HTTPException(422, "请输入项目名称")
        pid = new_id()
        store.execute("INSERT INTO projects VALUES(?,?,?,?)", (pid, user["id"], body.name.strip(), time.time()))
        return project(pid, user)

    @app.get("/api/projects/{pid}")
    def project_detail(pid: str, user=Depends(identity)):
        row = project(pid, user)
        samples = store.all("SELECT * FROM samples WHERE project_id=? ORDER BY created DESC", (pid,))
        for s in samples:
            s["jobs"] = store.all("SELECT * FROM jobs WHERE sample_id=? ORDER BY created DESC", (s["id"],))
        return {**row, "samples": samples,
                "shares": store.all("SELECT s.expert_id,s.active,u.name FROM shares s JOIN users u ON u.id=s.expert_id WHERE project_id=?", (pid,)) if row["owner_id"] == user["id"] else []}

    @app.post("/api/projects/{pid}/share")
    def share_project(pid: str, body: Share, user=Depends(identity)):
        project(pid, user, owner=True)
        if not store.one("SELECT id FROM users WHERE id=? AND role='expert' AND disabled=0", (body.expert_id,)):
            raise HTTPException(404, "请选择有效专家账号")
        store.execute("""INSERT INTO shares VALUES(?,?,?,?) ON CONFLICT(project_id,expert_id)
          DO UPDATE SET active=excluded.active,updated=excluded.updated""", (pid, body.expert_id, int(body.active), time.time()))
        return {"ok": True}

    @app.post("/api/projects/{pid}/samples")
    async def upload(pid: str, request: Request, name: str, user=Depends(identity)):
        project(pid, user, owner=True)
        if not name or len(name) > 180 or "/" in name or "\\" in name or any(ord(c) < 32 for c in name):
            raise HTTPException(422, "请使用普通文件名")
        buf = bytearray()
        async for chunk in request.stream():
            buf.extend(chunk)
            if len(buf) > MAX_UPLOAD:
                raise HTTPException(413, "单张文件不能超过 20 MiB")
        try:
            arr, preview, fmt = decode_image(bytes(buf))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        with upload_lock:
            total = store.one("SELECT COALESCE(SUM(bytes),0) AS n FROM samples")["n"]
            personal = store.one("SELECT COALESCE(SUM(s.bytes),0) AS n FROM samples s JOIN projects p ON p.id=s.project_id WHERE p.owner_id=?", (user["id"],))["n"]
            if total + len(buf) > 1024**3 or personal + len(buf) > 200 * 1024**2:
                raise HTTPException(409, "上传存储额度已满，请联系管理员；现有数据不会自动删除")
            sid = new_id()
            folder = store.sample_path(sid)
            folder.mkdir(parents=True, mode=0o700)
            with (folder / "original").open("xb") as stream:
                stream.write(buf)
            Image.fromarray(preview).save(folder / "preview.png")
            store.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?)", (sid, pid, name, arr.shape[1], arr.shape[0], fmt, hashlib.sha256(buf).hexdigest(), len(buf), time.time()))
        return sample(sid, user)

    @app.get("/api/samples/{sid}/preview")
    def preview(sid: str, user=Depends(identity)):
        sample(sid, user)
        return FileResponse(store.sample_path(sid) / "preview.png", media_type="image/png")

    @app.post("/api/samples/{sid}/jobs")
    def enqueue(sid: str, body: JobCreate, user=Depends(identity)):
        sample(sid, user, owner=True)
        if body.task not in TASKS:
            raise HTTPException(422, "不支持此任务")
        with store.db() as con:
            con.execute("BEGIN IMMEDIATE")
            count = con.execute("""SELECT COUNT(*) FROM jobs j JOIN samples s ON s.id=j.sample_id
              JOIN projects p ON p.id=s.project_id WHERE p.owner_id=? AND j.status IN ('queued','running')""", (user["id"],)).fetchone()[0]
            if count >= 3:
                raise HTTPException(409, "您已有 3 个等待或运行中的任务，请稍后提交")
            # Bounded result storage: each job keeps only display maps + a JSON report.
            if con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] >= 1000:
                raise HTTPException(409, "平台任务额度已满，请联系管理员")
            jid = new_id()
            con.execute("INSERT INTO jobs(id,sample_id,task,status,created) VALUES(?,?,?,'queued',?)", (jid, sid, body.task, time.time()))
        return job(jid, user)

    @app.get("/api/jobs/{jid}")
    def job_detail(jid: str, user=Depends(identity)):
        row = job(jid, user)
        if row["status"] == "completed":
            row["summary"] = json.loads((store.job_path(jid) / "summary.json").read_text())
        row["feedback"] = store.all("""SELECT f.*,u.name AS reviewer_name FROM feedback f JOIN users u ON u.id=f.user_id
           WHERE f.job_id=? ORDER BY created""", (jid,))
        for f in row["feedback"]:
            f["geometry"] = json.loads(f["geometry"])
        return row

    @app.post("/api/jobs/{jid}/cancel")
    def cancel(jid: str, user=Depends(identity)):
        job(jid, user, owner=True)
        store.execute("""UPDATE jobs SET cancel_requested=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END
          WHERE id=? AND status IN ('queued','running')""", (jid,))
        return {"ok": True}

    @app.get("/api/jobs/{jid}/artifacts/{name}")
    def artifact(jid: str, name: str, user=Depends(identity)):
        row = job(jid, user)
        if row["status"] != "completed" or name not in {"outline.png", "heatmap.png", "summary.json"}:
            raise HTTPException(404, "结果尚不可用")
        return FileResponse(store.job_path(jid) / name, media_type="application/json" if name.endswith("json") else "image/png")

    @app.post("/api/jobs/{jid}/feedback")
    def save_feedback(jid: str, body: Feedback, user=Depends(identity)):
        row = job(jid, user)
        s = sample(row["sample_id"], user)
        if row["status"] != "completed":
            raise HTTPException(409, "请等待分析完成")
        if body.action not in {"accept", "correct", "reject", "no_defect", "artifact", "unknown"}:
            raise HTTPException(422, "无效审核结论")
        if body.action == "accept" and not body.prediction_seen:
            raise HTTPException(422, "请先查看模型轮廓，再选择接受预测")
        if body.action == "correct" and not body.geometry:
            raise HTTPException(422, "修正几何时请至少添加一个点、线或区域")
        if body.action != "correct" and body.geometry:
            raise HTTPException(422, "仅修正几何结论可提交几何标注")
        total_points = 0
        for g in body.geometry:
            pts = g.get("points", [])
            kind = g.get("type")
            if not isinstance(kind, str) or not isinstance(pts, list):
                raise HTTPException(422, "无效标注格式")
            required = {"point": 1, "line": 2, "polygon": 3}.get(kind)
            if not required or len(pts) < required or (g["type"] != "polygon" and len(pts) != required):
                raise HTTPException(422, "标注几何不完整")
            total_points += len(pts)
            for p in pts:
                if not isinstance(p, list) or len(p) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p) or not (0 <= p[0] < s["width"] and 0 <= p[1] < s["height"]):
                    raise HTTPException(422, "标注坐标超出原图")
        if total_points > 2000:
            raise HTTPException(422, "标注点过多")
        geometry = json.dumps(body.geometry, ensure_ascii=False, sort_keys=True)
        with store.db() as con:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute("SELECT * FROM feedback WHERE job_id=? AND user_id=? AND request_id=?", (jid, user["id"], body.request_id)).fetchone()
            if existing:
                if (existing["action"], existing["note"], existing["geometry"], existing["prediction_seen"]) != (body.action, body.note, geometry, int(body.prediction_seen)):
                    raise HTTPException(409, "重复请求内容不同，请重新提交")
                return {"revision": existing["revision"], "persisted": True}
            rev = con.execute("SELECT COALESCE(MAX(revision),0)+1 FROM feedback WHERE job_id=? AND user_id=?", (jid, user["id"])).fetchone()[0]
            con.execute("INSERT INTO feedback VALUES(?,?,?,?,?,?,?,?,?,?)", (new_id(), jid, user["id"], rev, body.action, body.note, geometry, int(body.prediction_seen), body.request_id, time.time()))
        return {"revision": rev, "persisted": True}

    @app.get("/api/projects/{pid}/export")
    def export_project(pid: str, user=Depends(identity)):
        p = project(pid, user, owner=True)
        rows = store.all("""SELECT f.*,s.name AS image_name,s.sha256 AS image_sha256,j.task
          FROM feedback f JOIN jobs j ON j.id=f.job_id JOIN samples s ON s.id=j.sample_id
          WHERE s.project_id=? ORDER BY f.created""", (pid,))
        for row in rows:
            row["geometry"] = json.loads(row["geometry"])
        return {"schema": "phyevo-private-review-v1", "project": p, "coordinate_system": "original image pixels, x right, y down", "revisions": rows}

    @app.get("/api/projects/{pid}/research")
    def research(pid: str, user=Depends(identity)):
        project(pid, user)
        return store.all("SELECT * FROM research_requests WHERE project_id=? ORDER BY created DESC", (pid,))

    @app.post("/api/projects/{pid}/research")
    def request_research(pid: str, body: ResearchRequest, user=Depends(identity)):
        project(pid, user, owner=True)
        if not body.consent:
            raise HTTPException(422, "请明确授权将本项目反馈用于后续改进研究")
        rid = new_id()
        store.execute("INSERT INTO research_requests(id,project_id,user_id,reason,created) VALUES(?,?,?,?,?)", (rid, pid, user["id"], body.reason, time.time()))
        return {"id": rid, "status": "requested", "executed": False}

    @app.get("/")
    def index():
        return FileResponse(repo / "apps/platform/index.html")

    @app.get("/assets/{name}")
    def asset(name: str):
        if name not in {"app.js", "style.css"}:
            raise HTTPException(404)
        return FileResponse(repo / "apps/platform" / name)

    return app
