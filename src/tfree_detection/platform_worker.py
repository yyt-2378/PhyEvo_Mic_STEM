"""Bounded, serial, restartable worker. Only new platform jobs are touched."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .platform_store import PlatformStore


def execute_job(root: Path, jid: str):
    # Heavy scientific imports happen after authenticated, persisted job submission.
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_erosion
    from matplotlib import colormaps
    from .config import load_config
    from .framework import TrainingFreeDetectionFramework
    from .platform_api import TASKS, decode_image

    store = PlatformStore(root)
    row = store.one("SELECT * FROM jobs WHERE id=?", (jid,))
    if not row or row["status"] != "running" or row["task"] not in TASKS:
        raise ValueError("Job is not executable")
    sample = store.one("SELECT * FROM samples WHERE id=?", (row["sample_id"],))
    data = (store.sample_path(sample["id"]) / "original").read_bytes()
    image, _, _ = decode_image(data)
    repo = Path(__file__).resolve().parents[2]
    config = load_config(repo / "configs" / f"{row['task']}.yaml")
    result = TrainingFreeDetectionFramework(config).detect(image)
    folder = store.job_path(jid)
    folder.mkdir(parents=True, exist_ok=False, mode=0o700)
    mask = np.asarray(result.final_geometry_mask, dtype=bool)
    outline = mask & ~binary_erosion(mask)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[outline] = [238, 174, 65, 255]
    Image.fromarray(rgba).save(folder / "outline.png")
    score = np.nan_to_num(np.asarray(result.score_map, dtype=float))
    lo, hi = float(score.min()), float(score.max())
    scaled = (score - lo) / max(hi - lo, 1e-12)
    heat = Image.fromarray(np.uint8(scaled * 255), mode="P")
    palette = np.uint8(colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255)
    heat.putpalette(palette.ravel().tolist())
    heat.save(folder / "heatmap.png")
    summary = result.summary()
    summary["platform_provenance"] = {
        "sample_id": sample["id"], "original_filename": sample["name"],
        "original_sha256": sample["sha256"], "job_id": jid,
        "coordinate_system": "original image pixels, x right, y down",
        "analysis_input": "original grayscale values, not the display contrast preview",
        "score_display_range": [lo, hi],
        "interpretation": "Physical anomaly candidates, not element identification or calibrated probability",
    }
    with (folder / "summary.json").open("x") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)


def worker_loop(root: Path, *, once=False):
    store = PlatformStore(root)
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    if not once:
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
    with (store.root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store.execute("UPDATE jobs SET status='failed',finished=?,error=? WHERE status='running'",
                      (time.time(), "上次服务运行中断。原图已保存，可重新分析。"))
        while not stopping:
            (store.root / "worker.heartbeat").touch()
            job = store.claim_job()
            if not job:
                if once:
                    return
                time.sleep(2)
                continue
            artifact_root = store.root / "jobs"
            artifact_bytes = sum(p.stat().st_size for p in artifact_root.rglob("*") if p.is_file()) if artifact_root.exists() else 0
            if artifact_bytes >= 512 * 1024**2:
                store.execute("UPDATE jobs SET status='failed',finished=?,error=? WHERE id=?", (time.time(), "分析结果存储已达 512 MiB 额度，请联系管理员。现有数据不会删除。", job["id"]))
                if once:
                    return
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
            env["MPLCONFIGDIR"] = str(store.root / "mpl-cache")
            log_path = store.root / f"job-{job['id']}.log"
            status, error = "failed", "分析未完成，请检查原图或更换分析任务。"
            with log_path.open("x") as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/serve_platform.py"), "execute", "--root", str(root), "--job", job["id"]], stdout=log, stderr=log, env=env)
                started = time.monotonic()
                while child.poll() is None:
                    (store.root / "worker.heartbeat").touch()
                    cancel = store.one("SELECT cancel_requested FROM jobs WHERE id=?", (job["id"],))["cancel_requested"]
                    if cancel or stopping or time.monotonic() - started > 240:
                        child.terminate()
                        try:
                            child.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        status = "cancelled" if cancel else "failed"
                        error = "用户取消分析；原图保留。" if cancel else ("服务重启中断分析；原图保留，可重新提交。" if stopping else "分析超出 4 分钟额度；原图保留，可尝试较小区域。")
                        break
                    time.sleep(1)
                else:
                    if child.returncode == 0:
                        status, error = "completed", None
            store.execute("UPDATE jobs SET status=?,finished=?,error=? WHERE id=?", (status, time.time(), error, job["id"]))
            if once:
                return
