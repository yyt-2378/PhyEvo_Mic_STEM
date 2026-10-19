"""Verify source-only releases without mounting outputs, labels or model weights.

Linux/bubblewrap; no install, downloads, cleanup, GPU or scientific-data replay.
Each invocation keeps a new, small workspace-local verification directory.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RSI = "experiments/rsi_program_evolution_v1"
RSI_TESTS = (
    "core", "loop", "provider", "meta", "meta_study", "learning_study",
    "active_study", "frontier_study", "system_state", "stability_state",
    "stability_diagnostics", "expansion", "field_storage", "prediction_storage",
    "storage_budget", "composition", "composition_continuation",
    "composition_final_audit", "composition_retention_audit", "retained_portfolio",
)
DESELECT = {
    "tests/test_author_pretrained.py::test_native_postprocessing_matches_published_functions":
        "Requires separately downloaded official source and OpenCV; not a source-only test.",
    "tests/test_emseek_author_pretrained.py::test_official_preprocessing_identical_non_square_input":
        "Requires separately downloaded official EMSeek source; not a source-only test.",
}


def git_paths(*args):
    result = subprocess.run(["git", "ls-files", "-z", *args], cwd=ROOT,
                            capture_output=True, check=True)
    return {p.decode() for p in result.stdout.split(b"\0") if p}


def checked(relative):
    path = ROOT / relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(ROOT):
        raise ValueError(f"Source must be a real workspace file: {relative}")
    if not path.is_file():
        raise ValueError(f"Not a source file: {relative}")
    return path


def source_files(staged=False):
    tracked = git_paths("--cached")
    available = tracked if staged else tracked | git_paths("--others", "--exclude-standard")
    prefixes = ("src/", "scripts/", "tests/", "apps/", "configs/", "schemas/", RSI + "/")
    result = {p for p in available if p.startswith(prefixes)
              and Path(p).suffix in {".py", ".yaml", ".yml", ".json", ".html", ".js", ".css", ".md"}}
    result |= {p for p in available if p.startswith("benchmarks/") and p.endswith(".py")}
    # Already-published fixture inventories only: no new labels, raw data or arrays.
    result |= {p for p in tracked if (
        p.startswith("datasets/") and Path(p).suffix in {".yaml", ".json", ".jsonl"}
        and not p.startswith(("datasets/public/", "datasets/expert_annotations/")))}
    result |= {"pyproject.toml", "benchmarks/fau_stack_v20_frozen_baseline.yaml",
               "benchmarks/baseline_v0_2_manifest.json", "benchmarks/version_iteration_evidence_v1.json",
               "benchmarks/latest_student_ablation.json",
               "reference_scale_radial_wiener.png"}
    return sorted(result)


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def read_source(name, staged):
    checked(name)
    if staged:
        return subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
    return checked(name).read_bytes()


def verify(profile, staged=False):
    files = source_files(staged)
    syntax_count = 0
    hashes = {}
    for name in files:
        payload = read_source(name, staged)
        hashes[name] = hashlib.sha256(payload).hexdigest()
        if name.endswith(".py"):
            ast.parse(payload, filename=name)
            syntax_count += 1
    print(f"Syntax OK: {syntax_count} Python files", flush=True)
    if profile == "syntax":
        return 0
    if not Path("/usr/bin/bwrap").is_file():
        raise RuntimeError("bubblewrap required; no unrestricted fallback")
    scratch_parent = ROOT / ".codex-tmp"
    if scratch_parent.is_symlink() or not scratch_parent.resolve().is_relative_to(ROOT):
        raise ValueError("Unsafe scratch directory")
    scratch_parent.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="source-release-", dir=scratch_parent))
    stage = scratch / "source"
    stage.mkdir()
    for name in files:
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as dest:
            dest.write(read_source(name, staged))
        if hashlib.sha256(target.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"Source changed during snapshot: {name}")
    work = scratch / "work"
    work.mkdir()
    (stage / ".codex-tmp").mkdir()
    runtime = Path(sys.prefix).resolve()
    if runtime.is_relative_to(ROOT):
        (stage / runtime.relative_to(ROOT)).mkdir(parents=True, exist_ok=True)
    if profile == "rsi":
        (stage / ".venvs/sam2").mkdir(parents=True, exist_ok=True)
    for name in ("tmp", "cache", "config", "matplotlib", "sandbox-home"):
        (work / name).mkdir()
    root_tests = [p for p in files if p.startswith("tests/test_") and p.endswith(".py")]
    if profile == "review":
        tests = [p for p in root_tests if Path(p).name.startswith(("test_review", "test_expert_review"))]
    elif profile == "rsi":
        tests = [f"{RSI}/test_{name}.py" for name in RSI_TESTS]
    else:
        tests = root_tests
    if not tests or not set(tests) <= set(files):
        raise ValueError("Missing declared release tests")
    logical_work = ROOT / ".codex-tmp"
    # A synthetic passwd entry lets Path.home() work without reading host account
    # configuration or redefining HOME. All library state remains in this run.
    with (work / "passwd").open("x") as stream:
        stream.write(f"sandbox:x:{os.getuid()}:{os.getgid()}::{logical_work / 'sandbox-home'}:/bin/false\n")
    command = ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--clearenv", "--ro-bind", "/usr", "/usr",
               "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
               "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
               "--bind", str(work / "tmp"), "/tmp",
               "--ro-bind", str(work / "passwd"), "/etc/passwd",
               "--ro-bind", str(stage), str(ROOT),
               "--bind", str(work), str(logical_work), "--chdir", str(ROOT)]
    # Only interpreter runtimes, never user configuration or the host home tree.
    command += ["--ro-bind", str(runtime), str(runtime)]
    local_runtime = ROOT / ".venvs/sam2"
    if profile == "rsi":
        if not local_runtime.is_dir() or not local_runtime.resolve().is_relative_to(ROOT):
            raise ValueError("RSI tests require the existing project-local .venvs/sam2 runtime")
        if local_runtime.resolve() != runtime:
            command += ["--ro-bind", str(local_runtime), str(local_runtime)]
    environment = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin", "LANG": "C.UTF-8",
        "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "TMPDIR": str(logical_work / "tmp"), "XDG_CACHE_HOME": str(logical_work / "cache"),
        "XDG_CONFIG_HOME": str(logical_work / "config"),
        "MPLCONFIGDIR": str(logical_work / "matplotlib"), "MPLBACKEND": "Agg",
    }
    for key, value in environment.items():
        command += ["--setenv", key, value]
    excluded = {node: reason for node, reason in DESELECT.items() if node.split("::")[0] in tests}
    command += [sys.executable, "-B", "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider",
                "--basetemp", str(logical_work / "pytest-release"),
                "--junitxml", str(logical_work / "junit.xml"), *tests]
    command += ["--deselect=" + node for node in excluded]
    save_new(scratch / "manifest.json", {"profile": profile, "source_sha256": hashes,
             "source": "git_index" if staged else "working_tree",
             "tests": tests, "explicit_deselections": excluded,
             "outputs_mounted": False, "real_annotations_mounted": False,
             "network": False, "scientific_results_revalidated": False})
    print(f"Verification directory: {scratch.relative_to(ROOT)}", flush=True)
    with (scratch / "pytest.log").open("x") as log:
        completed = subprocess.run(command, cwd=ROOT, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    changed = [p for p, h in hashes.items() if hashlib.sha256(read_source(p, staged)).hexdigest() != h]
    save_new(scratch / "result.json", {"returncode": completed.returncode,
             "source_changed_during_tests": changed, "profile": profile,
             "outputs_mounted": False, "test_log": "pytest.log", "junit": "work/junit.xml"})
    print((scratch / "pytest.log").read_text(), flush=True)
    return completed.returncode or bool(changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("syntax", "core", "rsi", "review"), default="syntax")
    parser.add_argument("--staged", action="store_true", help="Verify exact Git index bytes, excluding unstaged edits")
    args = parser.parse_args()
    raise SystemExit(verify(args.profile, args.staged))


if __name__ == "__main__":
    main()
