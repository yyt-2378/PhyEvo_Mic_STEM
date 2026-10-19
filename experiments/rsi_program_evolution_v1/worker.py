"""Trusted isolated adapter. Imported candidate code is not trusted."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import resource
import sys


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, f"/program/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, allow_nan=False)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    request = json.loads(Path("/input/request.json").read_text())
    if request["mode"] == "probe":
        # No candidate imports: exercise the real mounts, namespace and environment.
        import socket
        checks = {"credentials_absent": not any(k in os.environ for k in
                  ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")),
                  "host_home_absent": not Path("/home/aiprogram/.codex").exists(),
                  "production_absent": not Path(request["production_path"]).exists(),
                  "network_interfaces": sorted(p.name for p in Path("/sys/class/net").glob("*"))
                  if Path("/sys/class/net").exists() else []}
        try:
            with Path("/input/request.json").open("a") as f:
                f.write("unexpected-write")
            checks["input_readonly"] = False
        except OSError:
            checks["input_readonly"] = True
        try:
            with socket.create_connection(("1.1.1.1", 443), timeout=.5):
                checks["network_blocked"] = False
        except OSError:
            checks["network_blocked"] = True
        if request.get("cuda"):
            import ctypes
            import ctypes.util
            import torch
            torch.set_num_threads(2)
            checks["cuda_library"] = ctypes.util.find_library("cuda")
            try:
                driver = ctypes.CDLL("libcuda.so.1")
                checks["cuda_driver_init"] = driver.cuInit(0)
            except OSError as e:
                checks["cuda_library_error"] = str(e)
            checks["cuda_available"] = torch.cuda.is_available()
            if checks["cuda_available"]:
                checks["cuda_result"] = float((torch.ones(8, device="cuda") * 2).sum().cpu())
                checks["cuda_name"] = torch.cuda.get_device_name(0)
        new_json("/work/result.json", checks)
        return
    if request["mode"] == "policy":
        policy = load_module("policy")
        answer = policy.plan(request["context"])
        new_json("/work/result.json", answer)
        return
    import numpy as np
    seed = int(request["seed"])
    np.random.seed(seed)
    if request.get("device") == "cuda":
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        torch.cuda.set_per_process_memory_fraction(.5)
    pipeline = load_module("pipeline")
    samples = []
    for item in request["samples"]:
        # Only basename files in the explicitly staged input bundle are allowed.
        name = item["file"]
        if Path(name).name != name:
            raise ValueError("Invalid staged sample path")
        with np.load("/input/" + name, allow_pickle=False) as z:
            samples.append({key: z[key] for key in z.files})
    context = {"seed": seed, "device": request.get("device", "cpu"),
               "training_steps": request.get("training_steps", 0)}
    if request.get("state_file"):
        with np.load("/input/state.npz", allow_pickle=False) as z:
            context["parent_state"] = {key: z[key] for key in z.files}
    if request["mode"] == "fit":
        if not hasattr(pipeline, "fit"):
            raise ValueError("Training requested but fit(samples, context) is absent")
        state = pipeline.fit(samples, context)
        if not isinstance(state, dict) or any(not isinstance(k, str) for k in state):
            raise ValueError("Checkpoint must be a dictionary of numeric arrays")
        state = {k: np.asarray(v) for k, v in state.items()}
        if any(v.dtype.hasobject or not np.isfinite(v).all() for v in state.values()):
            raise ValueError("Nonfinite or nonnumeric checkpoint")
        with Path("/work/state.npz").open("xb") as f:
            np.savez_compressed(f, **state)
        new_json("/work/result.json", {"trained": True, "state_keys": sorted(state)})
        return
    if request["mode"] != "predict":
        raise ValueError("Unknown execution mode")
    state = {}
    if request.get("state_file"):
        with np.load("/input/state.npz", allow_pickle=False) as z:
            state = {key: z[key] for key in z.files}
    rows = []
    for index, sample in enumerate(samples):
        if any(k in sample for k in ("human", "known", "points_xy", "label")):
            raise ValueError("Supervision in inference inputs")
        result = pipeline.predict(sample, state, context)
        if not isinstance(result, dict) or "score" not in result or "support" not in result:
            raise ValueError("predict must return score and support")
        score, support = np.asarray(result["score"]), np.asarray(result["support"])
        shape = sample["image"].shape
        if score.shape != shape or support.shape != shape or not np.isfinite(score).all():
            raise ValueError("Invalid prediction shape or score")
        if support.dtype != np.bool_:
            raise ValueError("Support must be boolean")
        name = f"prediction-{index:04d}.npz"
        with Path("/work", name).open("xb") as f:
            np.savez_compressed(f, score=score.astype(np.float32), support=support)
        rows.append({"index": index, "file": name})
    new_json("/work/result.json", {"predictions": rows})


if __name__ == "__main__":
    main()
