"""OS isolation for generated programs, with no production-tree mount."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time

from .contracts import file_hash, inside, write_new


@dataclass(frozen=True)
class Sandbox:
    workspace: Path
    runtime: Path
    timeout: int = 180
    gpu: int | None = None

    def __post_init__(self):
        root = self.workspace.resolve(strict=True)
        runtime = self.runtime.resolve(strict=True)
        runtime.relative_to(root)
        if not (runtime / "bin/python").is_file() or self.timeout <= 0:
            raise ValueError("A project-local Python runtime and positive timeout are required")
        if self.gpu is not None and (not isinstance(self.gpu, int) or self.gpu < 0):
            raise ValueError("Explicit nonnegative GPU index required")

    def command(self, program, inputs, output, worker):
        paths = [inside(self.workspace, p) for p in (program, inputs, output, worker)]
        program, inputs, output, worker = paths
        # The repository, archive, evaluator labels and credentials are absent.
        cmd = ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--clearenv", "--ro-bind", "/usr", "/usr",
               "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
               "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
               "--ro-bind", str(self.runtime.resolve()), "/runtime",
               "--ro-bind", str(worker), "/worker.py",
               "--ro-bind", str(program), "/program", "--ro-bind", str(inputs), "/input",
               "--bind", str(output), "/work", "--chdir", "/work",
               "--setenv", "PATH", "/runtime/bin:/usr/bin", "--setenv", "TMPDIR", "/work/tmp",
               "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "PYTHONNOUSERSITE", "1",
               "--setenv", "OMP_NUM_THREADS", "2", "--setenv", "OPENBLAS_NUM_THREADS", "2",
               "--setenv", "MKL_NUM_THREADS", "2", "--setenv", "CUDA_CACHE_DISABLE", "1",
               "--setenv", "XDG_CACHE_HOME", "/work/cache"]
        if self.gpu is not None:
            for path in (f"/dev/nvidia{self.gpu}", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"):
                if Path(path).exists():
                    cmd += ["--dev-bind", path, path]
            # NVIDIA runtime discovery needs this read-only kernel metadata.
            for path in ("/sys", "/proc/driver/nvidia", "/etc/ld.so.cache"):
                if Path(path).exists():
                    cmd += ["--ro-bind", path, path]
            # With only one device node mounted, CUDA enumeration can differ
            # from the host. UUID binding prevents selecting the wrong index.
            device = subprocess.run(["/usr/bin/nvidia-smi", f"--id={self.gpu}",
                    "--query-gpu=uuid", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10, check=True).stdout.strip()
            if not device.startswith("GPU-") or "\n" in device:
                raise ValueError("Could not bind the requested physical GPU")
            cmd += ["--setenv", "CUDA_VISIBLE_DEVICES", device]
        else:
            cmd += ["--setenv", "CUDA_VISIBLE_DEVICES", ""]
        return cmd + ["/runtime/bin/python", "-I", "-B", "/worker.py"]

    def run(self, program, inputs, output):
        output = inside(self.workspace, output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "tmp").mkdir()
        (output / "cache").mkdir()
        worker = Path(__file__).with_name("worker.py").resolve()
        command = self.command(program, inputs, output, worker)
        started = time.monotonic()
        # Don't pass host API keys, proxy credentials or user runtime variables.
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            try:
                p = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   env=env, timeout=self.timeout, check=False)
                status, returncode = ("ok" if p.returncode == 0 else "failed"), p.returncode
            except subprocess.TimeoutExpired:
                status, returncode = "timeout", None
        receipt = {"status": status, "returncode": returncode,
                   "seconds": time.monotonic() - started, "gpu": self.gpu,
                   "worker_sha256": file_hash(worker), "isolation": "bubblewrap-new-user-pid-net-mount-namespaces",
                   "network": False, "host_credentials": False, "repository_mounted": False,
                   "write_mounts": [str(output)], "stdout_sha256": file_hash(output / "stdout.log"),
                   "stderr_sha256": file_hash(output / "stderr.log")}
        write_new(output / "execution.json", receipt)
        return receipt
