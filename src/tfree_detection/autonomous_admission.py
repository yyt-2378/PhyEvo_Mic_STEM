"""Enforced family-wide admission for a sealed autonomous research update.

The scientific study stays immutable. Recompute its ORIGINAL gate; do not pick
successful sources from a failed family. Rejected programs are never constructed
in active mode. Explicit shadow mode always retains incumbent primary outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .autonomous_rsi import promotion
from .expert_supervision import contained_path, sha256_file, write_new_json
from .interface_release import InterfaceResearchRuntime

FORMAT = "phyevo-autonomous-admission-v1"


def read(path):
    return json.loads(path.read_text())


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def reference(root, path):
    path = contained_path(root, path)
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


def read_reference(root, ref):
    path = contained_path(root, ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError(f"Admission dependency changed: {ref['path']}")
    return read(path)


def validate_rows(rows, jobs):
    expected = {(j["outer_group"], n, j["seed"]) for j in jobs for n in j["test_cases"]}
    keys = [(r["group"], r["case_number"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Admission needs every expected source, case and seed exactly once")
    for r in rows:
        if (not np.isfinite(r["coverage"]) or not 0 <= r["coverage"] <= 1 or
                not isinstance(r["budget"], int) or r["budget"] < 1 or
                not isinstance(r["selected_pixels"], int) or not 0 <= r["selected_pixels"] <= r["budget"]):
            raise ValueError("Invalid admission metric or pixel budget")


def family_decision(protocol, audits):
    jobs = [j for j in protocol["jobs"] if j["stage"] == "outer"]
    groups = [a["search"]["outer_group"] for a in audits]
    if len(groups) != len(set(groups)) or set(groups) != set(protocol["groups"]):
        raise ValueError("Cannot admit a cherry-picked source subset")
    candidates = [row for a in audits for row in a["rows"]]
    incumbent = [row for a in audits for row in a["baseline_rows"]]
    validate_rows(candidates, jobs); validate_rows(incumbent, jobs)
    return canonical(promotion(candidates, incumbent, incumbent,
        minimum_gain=protocol["minimum_macro_gain"],
        source_regression=protocol["source_regression_tolerance"],
        seed_regression=protocol["seed_regression_tolerance"]))


def collect_evidence(root, study, release, arm):
    """Read/verify closed search, audit, independent replay and model identities."""
    root = Path(root).resolve(); study = contained_path(root, study); release = contained_path(root, release)
    p = read(study / "protocol.json"); result = read(release / "results.json")
    if p["format"] != "phyevo-autonomous-program-rsi-v1" or arm not in result["final_audit_gates"]:
        raise ValueError("Unsupported study or arm")
    protocol_hash = sha256_file(study / "protocol.json")
    if contained_path(root, result["protocol"]) != study / "protocol.json":
        raise ValueError("Release belongs to another study")
    for name, digest in p["source_hashes"].items():
        if sha256_file(contained_path(root, name)) != digest:
            raise ValueError("Frozen controller source changed")
    refs = {"protocol": reference(root, study / "protocol.json"),
            "results": reference(root, release / "results.json"),
            "search_seal": reference(root, study / "search_seal.json"),
            "release_seal": reference(root, release / "seal.json")}
    search_seal = read_reference(root, refs["search_seal"])
    if search_seal["protocol_sha256"] != protocol_hash or not search_seal["all_searches_closed_before_any_outer_audit"]:
        raise ValueError("Outer audit was not isolated from search")
    if result["search_seal_sha256"] != refs["search_seal"]["sha256"]:
        raise ValueError("Release/search seal mismatch")
    release_seal = read_reference(root, refs["release_seal"])
    for name, digest in release_seal["files"].items():
        if sha256_file(contained_path(release, name)) != digest:
            raise ValueError("Research release changed")
    audits = []; receipts = []; bindings = []
    for search in p["searches"]:
        if search["arm"] != arm:
            continue
        closed_path = study / "searches" / f'search-{search["index"]:02d}' / "closed.json"
        closed_ref = reference(root, closed_path)
        if search_seal["files"].get(str(closed_path.relative_to(study))) != closed_ref["sha256"]:
            raise ValueError("Search was modified or not sealed")
        closed = read(closed_path)
        verification_path = study / "verification" / f'search-{search["index"]:02d}.json'
        verification = read(verification_path)
        if (verification["closed_sha256"] != closed_ref["sha256"] or verification["search"] != search or
                not all(verification.get(k) is True for k in ("source_exclusion_verified", "governor_replayed", "parent_chain_verified", "all_masks_match_numpy_stable_topk"))):
            raise ValueError("Independent candidate verification missing or mismatched")
        audit_path = study / "audit" / f'search-{search["index"]:02d}' / "result.json"
        audit_ref = reference(root, audit_path); audit = read(audit_path)
        completion = read(audit_path.parent / "complete.json")
        if (audit_ref["sha256"] != completion["result_sha256"] or audit["search"] != search or
                audit["outer_results_returned_to_proposer"] is not False or
                audit["search_seal_sha256"] != refs["search_seal"]["sha256"] or
                audit["lineage"] != closed["lineage"]):
            raise ValueError("Audit provenance mismatch")
        audits.append(audit)
        receipts.append({"closed": closed_ref, "verification": reference(root, verification_path), "audit": audit_ref})
    gate = family_decision(p, audits)
    if gate != result["final_audit_gates"][arm]:
        raise ValueError("Stored gate differs from frozen-rule recomputation")
    for job in [j for j in p["jobs"] if j["stage"] == "outer"]:
        rows = [r for r in result["bindings"] if r["arm"] == arm and r["group"] == job["outer_group"] and r["seed"] == job["seed"]]
        if len(rows) != 1:
            raise ValueError("Missing or duplicate configured model binding")
        candidate_ref = reference(root, rows[0]["path"]); b = read_reference(root, candidate_ref)
        base_ref = reference(root, b["base_binding"]); base = read_reference(root, base_ref)
        checkpoint = contained_path(root, base["checkpoint"])
        meta = read(checkpoint / "checkpoint.json")
        if (b["job_index"] != job["index"] or b["global_research_admission"] != gate["accepted"] or
                contained_path(root, job["checkpoint"]) != checkpoint or
                sha256_file(checkpoint / "checkpoint.json") != job["checkpoint_sha256"] or
                sha256_file(checkpoint / "state.npz") != job["state_sha256"] or
                meta["outer_group"] != job["outer_group"] or job["outer_group"] in meta["train_groups"]):
            raise ValueError("Configured base/model identity mismatch")
        matching = next(a for a in receipts if a["audit"]["path"] == b["audit"])
        if b["audit_sha256"] != matching["audit"]["sha256"]:
            raise ValueError("Binding refers to another audited candidate")
        bindings.append({"job_index": job["index"], "seed": job["seed"], "outer_group": job["outer_group"],
                         "candidate": candidate_ref, "incumbent": base_ref})
    return {"study": str(study.relative_to(root)), "release": str(release.relative_to(root)), "arm": arm,
            "refs": refs, "receipts": receipts, "gate": gate, "bindings": bindings}


def publish_admission(root, study, release, output, *, arm="feedback"):
    root = Path(root).resolve(); output = contained_path(root, output)
    if output.exists():
        raise FileExistsError("Admission records are immutable; use a new path")
    evidence = collect_evidence(root, study, release, arm)
    evidence.update(format=FORMAT, production_approved=False,
        selected_family="candidate" if evidence["gate"]["accepted"] else "incumbent",
        policy="family-wide frozen gate; failed family cannot be selected source by source",
        shadow_policy="explicit opt-in; incumbent primary even when shadow candidate is successful",
        admission_source_sha256=sha256_file(Path(__file__).resolve()))
    output.parent.resolve().relative_to(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(output, evidence)
    return evidence


def load_admission(root, admission):
    root = Path(root).resolve(); a = read(contained_path(root, admission))
    if a.get("format") != FORMAT or a.get("production_approved") is not False:
        raise ValueError("Unsupported admission record")
    if a.get("admission_source_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("Admission implementation changed")
    # Recompute the gate from hash-bound full-family audit rows, not a boolean
    # in a candidate binding. An edited selected_family/accepted flag is useless.
    p = read_reference(root, a["refs"]["protocol"])
    result = read_reference(root, a["refs"]["results"])
    for name, digest in p["source_hashes"].items():
        if sha256_file(contained_path(root, name)) != digest:
            raise ValueError("Frozen controller source changed")
    for key in ("search_seal", "release_seal"):
        read_reference(root, a["refs"][key])
    audits = []
    for r in a["receipts"]:
        read_reference(root, r["closed"]); read_reference(root, r["verification"])
        audits.append(read_reference(root, r["audit"]))
    gate = family_decision(p, audits)
    expected = "candidate" if gate["accepted"] else "incumbent"
    if gate != a["gate"] or gate != result["final_audit_gates"][a["arm"]] or a["selected_family"] != expected:
        raise ValueError("Admission decision tampered or stale")
    expected_jobs = {j["index"] for j in p["jobs"] if j["stage"] == "outer"}
    if len(a["bindings"]) != len(expected_jobs) or {b["job_index"] for b in a["bindings"]} != expected_jobs:
        raise ValueError("Admission model family incomplete")
    for b in a["bindings"]:
        candidate = read_reference(root, b["candidate"])
        incumbent = read_reference(root, b["incumbent"])
        job = next(j for j in p["jobs"] if j["index"] == b["job_index"])
        expected_bindings = [r for r in result["bindings"] if r["arm"] == a["arm"] and
                             r["group"] == job["outer_group"] and r["seed"] == job["seed"]]
        if (len(expected_bindings) != 1 or b["candidate"]["path"] != expected_bindings[0]["path"] or
                candidate["job_index"] != b["job_index"] or b["incumbent"]["path"] != candidate["base_binding"] or
                contained_path(root, incumbent["checkpoint"]) != contained_path(root, job["checkpoint"])):
            raise ValueError("Admission configured model identity changed")
    return a


def _candidate_runtime(root, binding, device):
    # Lazy construction is a correctness property for rejected active updates.
    from .autonomous_runtime import AutonomousResearchRuntime
    return AutonomousResearchRuntime(root, binding, device=device, allow_experimental=True)


def _incumbent_runtime(root, binding, device):
    return InterfaceResearchRuntime(root, binding, device=device, allow_experimental=True)


def validate_prediction(result, image):
    mask = np.asarray(result["mask"])
    budget = int(np.ceil(.1 * image.size))
    if mask.dtype != np.bool_ or mask.shape != image.shape or int(mask.sum()) > budget or result["budget"] != budget:
        raise ValueError("Prediction violates frozen output geometry or budget")
    for key, value in result.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite prediction field: {key}")


class AdmittedRSIRuntime:
    """Configured model with enforced active gate; no image-name routing."""

    def __init__(self, root, admission, job_index, *, mode="active", device="cpu", allow_experimental=False):
        if not allow_experimental:
            raise ValueError("Explicit research-runtime opt-in required")
        if mode not in {"active", "shadow"}:
            raise ValueError("Mode must be active or explicit shadow")
        self.root = Path(root).resolve(); self.admission = load_admission(self.root, admission)
        self.job = next((b for b in self.admission["bindings"] if b["job_index"] == job_index), None)
        if self.job is None:
            raise ValueError("Unknown configured model; no source-name dispatch")
        self.mode = mode; self.device = device; self.candidate = None
        self.base = _incumbent_runtime(self.root, self.job["incumbent"]["path"], device)
        self.gate = self.admission["gate"]

    def predict(self, normalized_image):
        image = np.asarray(normalized_image, np.float32)
        if image.ndim != 2 or min(image.shape) < 4 or not np.isfinite(image).all():
            raise ValueError("Expected finite normalized 2-D image")
        incumbent = self.base.predict(image); validate_prediction(incumbent, image)
        primary = dict(incumbent); evaluated = False; error = None; candidate = None
        selected = "incumbent"; state = "rejected_update_incumbent_retained"
        if self.gate["accepted"] or self.mode == "shadow":
            evaluated = True
            try:
                if self.candidate is None:
                    self.candidate = _candidate_runtime(self.root, self.job["candidate"]["path"], self.device)
                candidate = self.candidate.predict(image); validate_prediction(candidate, image)
                if candidate.get("status") == "invalid_synthesized_score_returned_baseline":
                    raise ValueError("Candidate reported an invalid synthesized score")
                if self.mode == "active":
                    primary = dict(candidate); selected = "candidate"; state = "admitted_candidate_active"
                else:
                    # A shadow candidate can NEVER replace the displayed primary.
                    primary["shadow_mask"] = candidate["mask"].copy()
                    state = "explicit_shadow_incumbent_primary"
            except (ValueError, RuntimeError, FloatingPointError, OSError) as exc:
                # Existing verified incumbent remains available on candidate
                # loading/execution failure. Do not reinterpret a failure as gain.
                error = f"{type(exc).__name__}: {exc}"; state = "candidate_failure_incumbent_retained"
        primary["rsi_admission"] = {"state": state, "mode": self.mode, "selected": selected,
            "family_accepted": self.gate["accepted"], "gate_reason": self.gate["reason"],
            "candidate_evaluated": evaluated, "candidate_error": error,
            "uses_expert_labels_at_inference": False, "uses_source_filename": False,
            "production_approved": False, "new_blind_external_test": False}
        return primary
