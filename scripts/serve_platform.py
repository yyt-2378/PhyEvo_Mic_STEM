#!/usr/bin/env python3
"""Project-local private platform CLI. No system-service or global configuration writes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "serve", "worker", "execute", "supervise", "start", "status"])
    parser.add_argument("--root", type=Path, default=REPO / "local_platform/v1")
    parser.add_argument("--origin", default="https://phyevo-review.tailf74731.ts.net:8443")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--local", action="store_true", help="Non-secure cookies for local validation only")
    parser.add_argument("--job")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(REPO) or root == REPO:
        parser.error("Storage must be a dedicated real subdirectory of this repository")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    from tfree_detection.platform_store import PlatformStore

    if args.command == "init":
        store = PlatformStore(root)
        if store.one("SELECT id FROM users LIMIT 1"):
            parser.error("Already initialized; activation codes cannot be recovered. Use the admin UI.")
        print(json.dumps(store.create_user("平台负责人", "admin"), ensure_ascii=False))
    elif args.command == "serve":
        import uvicorn
        from tfree_detection.platform_api import create_app
        uvicorn.run(create_app(root, origin=args.origin, secure=not args.local), host="127.0.0.1", port=args.port, access_log=False)
    elif args.command in {"execute", "worker"}:
        from tfree_detection.platform_worker import execute_job, worker_loop
        if args.command == "execute":
            execute_job(root, args.job)
        else:
            worker_loop(root)
    elif args.command == "status":
        status = root / "service.json"
        print(status.read_text() if status.exists() else '{"status":"not started"}')
    elif args.command == "start":
        command = [sys.executable, str(Path(__file__).resolve()), "supervise", "--root", str(root), "--origin", args.origin, "--port", str(args.port)]
        if args.local:
            command.append("--local")
        with (root / "service.log").open("a") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"supervisor_pid": child.pid, "root": str(root)}))
    else:
        with (root / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            children = {}
            stopping = False

            def stop(signum, frame):
                nonlocal stopping
                stopping = True

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            while not stopping:
                for kind in ("serve", "worker"):
                    if kind not in children or children[kind].poll() is not None:
                        command = [sys.executable, str(Path(__file__).resolve()), kind, "--root", str(root), "--origin", args.origin, "--port", str(args.port)]
                        if args.local:
                            command.append("--local")
                        children[kind] = subprocess.Popen(command, stdin=subprocess.DEVNULL)
                (root / "service.json").write_text(json.dumps({"supervisor_pid": os.getpid(), "api_pid": children["serve"].pid, "worker_pid": children["worker"].pid, "heartbeat": time.time(), "port": args.port, "origin": args.origin}))
                time.sleep(3)
            for child in children.values():
                if child.poll() is None:
                    child.terminate()
            for child in children.values():
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()


if __name__ == "__main__":
    main()
