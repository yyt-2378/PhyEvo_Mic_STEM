"""Immutable experiment contracts and program proposals (not a solution menu)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
import hashlib
import json
import math
from pathlib import Path
import re


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inside(root, path):
    root = Path(root).resolve(strict=True)
    path = Path(path)
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    resolved.relative_to(root)
    if resolved == root:
        raise ValueError("An explicit child path is required")
    return resolved


def write_new(path, value):
    """Create a new artifact, never replace existing scientific evidence."""
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(canonical(value) + "\n")


@dataclass(frozen=True)
class Protocol:
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    audit_groups: tuple[str, ...]
    seeds: tuple[int, ...] = (23, 41, 67)
    area_fraction: float = .10
    point_budget_per_512: float = 50.
    point_tolerance_px: float = 6.
    maximum_rounds: int = 4
    maximum_model_calls: int = 16
    maximum_output_tokens: int = 6000
    maximum_training_steps: int = 400
    timeout_seconds: int = 180
    minimum_gain: float = .002
    maximum_group_regression: float = .01
    label_semantics: str = "positive_unknown"
    data_version: str = ""
    engine_version: str = ""

    def __post_init__(self):
        sets = [set(self.train_groups), set(self.validation_groups), set(self.audit_groups)]
        if any(not s for s in sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Nonempty source-disjoint train, validation and audit groups required")
        if any(len(v) != len(set(v)) for v in (self.train_groups, self.validation_groups,
                                             self.audit_groups, self.seeds)):
            raise ValueError("Duplicate source or seed")
        if not self.seeds or not 0 < self.area_fraction <= 1:
            raise ValueError("Invalid seeds or area budget")
        numbers = [self.point_budget_per_512, self.point_tolerance_px, self.minimum_gain,
                   self.maximum_group_regression]
        if any(not math.isfinite(x) or x < 0 for x in numbers) or self.point_budget_per_512 == 0:
            raise ValueError("Invalid evaluation contract")
        if any(v <= 0 for v in (self.maximum_rounds, self.maximum_model_calls,
                               self.maximum_output_tokens, self.maximum_training_steps,
                               self.timeout_seconds)):
            raise ValueError("Positive finite budgets required")
        if self.label_semantics != "positive_unknown":
            raise ValueError("Unknown pixels must not become negative labels")

    @property
    def identity(self):
        return digest(asdict(self))

    def split(self, group):
        for name in ("train", "validation", "audit"):
            if group in getattr(self, name + "_groups"):
                return name
        raise ValueError("Unknown acquisition group")


@dataclass(frozen=True)
class Proposal:
    action: str
    parents: tuple[str, ...]
    observation: str
    hypothesis: str
    intervention: str
    expected_effect: str
    falsifier: str
    pipeline_code: str
    policy_code: str
    evidence_ids: tuple[str, ...] = ()
    training_steps: int = 0
    target: str = "analysis"

    def validate(self, protocol):
        counts = {"generate": 0, "tune": 1, "evolve": 2, "repair": 1}
        if self.action not in counts or len(self.parents) != counts[self.action]:
            raise ValueError("Action and ancestry disagree")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("Crossover requires distinct parents")
        if not all(re.fullmatch(r"[a-f0-9]{64}", p) for p in self.parents):
            raise ValueError("Invalid parent identity")
        if not all(isinstance(x, str) and x.strip() for x in
                   (self.observation, self.hypothesis, self.intervention,
                    self.expected_effect, self.falsifier)):
            raise ValueError("A concrete experiment and falsifier are required")
        if self.target not in {"analysis", "learning", "routing", "research_policy"}:
            raise ValueError("Unknown modification target")
        if not 0 <= self.training_steps <= protocol.maximum_training_steps:
            raise ValueError("Training budget exceeded")
        for code, required in ((self.pipeline_code, {"predict"}), (self.policy_code, {"plan"})):
            if len(code.encode()) > 80000:
                raise ValueError("Candidate code too large")
            tree = ast.parse(code)
            defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            if not required <= defined:
                raise ValueError("Missing executable entry point")
        # AST checks are an interface check, NOT the security boundary. Programs
        # execute only in a networkless, minimally mounted bubblewrap process.
        return self

    @property
    def identity(self):
        return digest(asdict(self))

    @classmethod
    def from_dict(cls, value):
        v = dict(value)
        for key in ("parents", "evidence_ids"):
            v[key] = tuple(v.get(key, ()))
        return cls(**v)


def protected_snapshot(root):
    root = Path(root).resolve()
    names = []
    for directory in ("src", "scripts", "configs", "tests", "benchmarks"):
        names.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".toml"}
                     and not p.is_symlink())
    names.extend(root / n for n in ("RSI_iterate.md", "pyproject.toml",
        "outputs/expert_cycle_001/learning_dataset_v1/manifest.json",
        "outputs/interface_improvement_20261015_v1/release_v2/results.json",
        "outputs/best_release_20261014_v1/manifest.json") if (root / n).is_file())
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(set(names))}


def verify_snapshot(root, before):
    changed = [name for name, h in before.items()
               if not inside(root, name).is_file() or file_hash(inside(root, name)) != h]
    if changed:
        raise ValueError("Protected files changed: " + ", ".join(changed))
    return {"checked": len(before), "changed": []}


def engine_fingerprint():
    folder = Path(__file__).resolve().parent
    # Bind the execution, proposal and measurement contract, not unrelated
    # reporting/CLI additions. New report files cannot invalidate a sealed run.
    modules = ("archive.py", "contracts.py", "controller.py", "evaluation.py",
               "executor.py", "provider.py", "sandbox.py", "worker.py")
    return digest({name: file_hash(folder / name) for name in modules})
