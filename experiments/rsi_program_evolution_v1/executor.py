"""Proposal -> real fit/predict jobs -> frozen evaluation, outside model control."""
from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
import time

import numpy as np

from .contracts import digest, engine_fingerprint, file_hash, inside, write_new
from .evaluation import evaluate, summarize


@dataclass
class Sample:
    key: str
    group: str
    geometry: str
    fields: dict
    human: np.ndarray
    known: np.ndarray
    points_xy: np.ndarray

    def validation_target(self):
        return {**self.fields, "human": self.human, "known": self.known,
                "points_xy": self.points_xy, "geometry": self.geometry}


def sample_fingerprint(samples):
    records = []
    for sample in samples:
        arrays = {**sample.fields, "human": sample.human, "known": sample.known,
                  "points_xy": sample.points_xy}
        hashes = {}
        for name, array in sorted(arrays.items()):
            array = np.ascontiguousarray(array)
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise ValueError("Inputs must be finite numeric arrays")
            hashes[name] = {"shape": array.shape, "dtype": str(array.dtype),
                            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        records.append({"key": sample.key, "group": sample.group, "geometry": sample.geometry,
                        "arrays": hashes})
    return digest(records)


class Executor:
    def __init__(self, archive, sandbox, samples):
        self.archive, self.sandbox, self.samples = archive, sandbox, tuple(samples)
        self.protocol = archive.protocol
        self._data_fingerprint = sample_fingerprint(samples)
        if self.protocol.data_version != self._data_fingerprint:
            raise ValueError("Input arrays do not match the frozen protocol")
        self.assert_contract()
        if len({s.key for s in samples}) != len(samples):
            raise ValueError("Duplicate input sample")
        for sample in samples:
            self.protocol.split(sample.group)
            if any(k in sample.fields for k in ("human", "known", "points_xy", "label")):
                raise ValueError("Labels belong in a separate evaluator view")
        self.jobs = archive.root / "jobs"
        self.jobs.mkdir(exist_ok=True)

    def assert_contract(self):
        if self.protocol.engine_version != engine_fingerprint():
            raise ValueError("Frozen evaluation/execution engine changed; new experiment required")

    def _input(self, destination, samples, mode, seed, steps, *, state=None):
        destination.mkdir(exist_ok=False)
        records = []
        for i, sample in enumerate(samples):
            # Candidate inference sees numeric fields, never source/case names.
            fields = dict(sample.fields)
            if mode == "fit":
                fields.update(human=sample.human, known=sample.known)
            name = f"sample-{i:04d}.npz"
            with (destination / name).open("xb") as f:
                np.savez_compressed(f, **fields)
            records.append({"file": name})
        if state is not None:
            with np.load(state, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            with (destination / "state.npz").open("xb") as f:
                np.savez_compressed(f, **arrays)
        write_new(destination / "request.json", {
            "mode": mode, "samples": records, "seed": seed, "training_steps": steps,
            "state_file": state is not None,
            "device": "cuda" if self.sandbox.gpu is not None else "cpu"})

    def _trial_folder(self, candidate, split):
        prefix = f"{candidate[:16]}-{split}"
        counter = 1
        target = self.jobs / f"{prefix}-{counter:03d}"
        while target.exists():
            counter += 1
            target = self.jobs / f"{prefix}-{counter:03d}"
        target.mkdir(exist_ok=False)
        return target

    def policy(self, identity, context):
        self.assert_contract()
        candidate = self.archive.candidate(identity)
        folder = self._trial_folder(identity, "policy")
        inputs = folder / "input"; inputs.mkdir()
        write_new(inputs / "request.json", {"mode": "policy", "context": context})
        receipt = self.sandbox.run(candidate["directory"], inputs, folder / "execution")
        if receipt["status"] != "ok":
            return None, receipt
        value = json.loads((folder / "execution/result.json").read_text())
        if not isinstance(value, dict) or not isinstance(value.get("focus"), str):
            raise ValueError("Research policy must return action, parents and focus")
        action, parents = value.get("action"), value.get("parents")
        counts = {"generate": 0, "tune": 1, "evolve": 2, "repair": 1}
        if action not in counts or not isinstance(parents, list) or len(parents) != counts[action]:
            raise ValueError("Policy ancestry/action contract violated")
        if len(set(parents)) != len(parents):
            raise ValueError("Policy crossover selected the same parent twice")
        for p in parents:
            self.archive.candidate(p)
        receipt.update(input_sha256=digest(context), plan_sha256=digest(value),
                       policy_sha256=candidate["policy_hash"], artifact=str(folder.relative_to(self.archive.root)))
        return value, receipt

    def execute(self, identity, *, split="validation", audit_seal=None):
        self.assert_contract()
        if sample_fingerprint(self.samples) != self._data_fingerprint:
            raise ValueError("Frozen sample arrays changed")
        if split not in {"validation", "audit"}:
            raise ValueError("Unsupported execution split")
        if split == "audit":
            seals = self.archive.events("search_sealed")
            if not seals or audit_seal != seals[-1]["hash"]:
                raise ValueError("Audit is available only after search is sealed")
        candidate = self.archive.candidate(identity)
        proposal = candidate["proposal"]
        test = [s for s in self.samples if self.protocol.split(s.group) == split]
        train = [s for s in self.samples if self.protocol.split(s.group) == "train"]
        if not test:
            raise ValueError("No samples for requested split")
        prior = [t for t in self.archive.trials(identity, visible_only=False)
                 if t["split"] == split and t["outcome"] == "ok"]
        if prior:
            receipt = prior[-1]["receipt"]
            for name, expected in receipt["artifacts"].items():
                if file_hash(inside(self.archive.root, name)) != expected:
                    raise ValueError("Cached execution artifacts changed")
            return prior[-1]["id"], receipt  # Content-bound, immutable reuse only.
        folder = self._trial_folder(identity, split)
        rows, artifacts, runs, checkpoints = [], {}, [], {}
        started = time.monotonic()
        for seed in self.protocol.seeds:
            seed_folder = folder / f"seed-{seed}"; seed_folder.mkdir()
            state = None
            for parent in proposal.parents:
                for t in reversed(self.archive.trials(parent)):
                    checkpoint = t["receipt"].get("checkpoints", {}).get(str(seed))
                    if checkpoint:
                        path = inside(self.archive.root, checkpoint["path"])
                        if file_hash(path) != checkpoint["sha256"]:
                            raise ValueError("Inherited checkpoint changed")
                        state = path
                        break
                if state is not None:
                    break
            if proposal.training_steps:
                fit_input = seed_folder / "fit_input"
                self._input(fit_input, train, "fit", seed, proposal.training_steps, state=state)
                fit_out = seed_folder / "fit_output"
                fit_receipt = self.sandbox.run(candidate["directory"], fit_input, fit_out)
                runs.append(fit_receipt)
                if fit_receipt["status"] != "ok":
                    return self._failure(identity, split, folder, runs, fit_out)
                state = fit_out / "state.npz"
            if state is not None:
                checkpoints[str(seed)] = {"path": str(state.relative_to(self.archive.root)),
                                           "sha256": file_hash(state)}
            infer_input = seed_folder / "predict_input"
            self._input(infer_input, test, "predict", seed, 0, state=state)
            infer_out = seed_folder / "predict_output"
            receipt = self.sandbox.run(candidate["directory"], infer_input, infer_out)
            runs.append(receipt)
            if receipt["status"] != "ok":
                return self._failure(identity, split, folder, runs, infer_out)
            try:
                for index, sample in enumerate(test):
                    file = infer_out / f"prediction-{index:04d}.npz"
                    with np.load(file, allow_pickle=False) as z:
                        prediction = {k: z[k] for k in z.files}
                    row = evaluate(sample.validation_target(), prediction, self.protocol)
                    rows.append({"case": sample.key, "group": sample.group, "seed": seed,
                                 "prediction": str(file.relative_to(self.archive.root)), **row})
                    artifacts[str(file.relative_to(self.archive.root))] = file_hash(file)
            except (ValueError, OSError, KeyError) as e:
                return self._failure(identity, split, folder, runs, infer_out, str(e))
        receipt = {"rows": rows, "summary": summarize(rows), "artifacts": artifacts,
                   "executions": runs, "checkpoints": checkpoints,
                   "seconds": time.monotonic() - started,
                   "protocol_sha256": self.protocol.identity,
                   "pipeline_sha256": candidate["code_hash"], "actual_execution": True,
                   "requested_training_steps_per_seed": proposal.training_steps,
                   "provenance": "new sandbox execution, not historical-result attribution"}
        write_new(folder / "evaluation.json", receipt)
        trial = self.archive.record_trial(identity, split, "ok", receipt)
        return trial, receipt

    def _failure(self, identity, split, folder, runs, output, evaluation_error=None):
        # Only candidate-generated errors are exposed, not host paths or labels.
        error = evaluation_error or (output / "stderr.log").read_text()[-5000:]
        receipt = {"executions": runs, "error": error, "actual_execution": True,
                   "protocol_sha256": self.protocol.identity, "artifact": str(folder.relative_to(self.archive.root))}
        write_new(folder / "failure.json", receipt)
        outcome = "timeout" if runs[-1]["status"] == "timeout" else "failed"
        return self.archive.record_trial(identity, split, outcome, receipt), receipt

    def diagnose(self, identity):
        trials = [t for t in self.archive.trials(identity) if t["split"] == "validation" and t["outcome"] == "ok"]
        if not trials:
            return {"status": "no_successful_validation", "candidate": identity}
        receipt = trials[-1]["receipt"]
        summary = receipt["summary"]
        diagnostics = []
        for sample in self.samples:
            if self.protocol.split(sample.group) != "validation":
                continue
            human = np.asarray(sample.human, bool)
            total = max(1, int(human.sum()))
            support = np.asarray(sample.fields.get("support", sample.fields["valid"]), bool)
            physics = np.asarray(sample.fields.get("physics", sample.fields["image"]))
            student = np.asarray(sample.fields.get("student", sample.fields["image"]))
            current_row = next(r for r in receipt["rows"] if r["case"] == sample.key)
            with np.load(inside(self.archive.root, current_row["prediction"]), allow_pickle=False) as z:
                current_support = z["support"].astype(bool)
            diagnostics.append({"group": sample.group, "geometry": sample.geometry,
                "current_known_positive_coverage": float(current_row["coverage"]),
                "positive_mass_outside_current_support": float((human & ~current_support).sum() / total),
                "current_support_fraction": float(current_support.mean()),
                "positive_mass_outside_initial_support": float((human & ~support).sum() / total),
                "positive_mass_outside_physical_validity": float((human & ~sample.fields["valid"]).sum() / total),
                "physics_student_rank_correlation": _correlation(physics, student),
                "initial_support_fraction": float(support.mean()),
                "image_std": float(np.std(sample.fields["image"]))})
        grouped = {}
        for group in sorted({d["group"] for d in diagnostics}):
            items = [d for d in diagnostics if d["group"] == group]
            grouped[group] = {k: float(np.mean([d[k] for d in items])) for k in items[0]
                              if k not in {"group", "geometry"}}
        return {"candidate": identity, "trial": trials[-1]["id"], "validation": summary,
                "physical_diagnostics": grouped, "label_semantics": "positive_unknown",
                "raw_images_or_annotations_in_prompt": False}


def _correlation(a, b):
    x, y = a.ravel()[::max(1, a.size // 4096)], b.ravel()[::max(1, b.size // 4096)]
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.
    return float(np.corrcoef(x, y)[0, 1])
