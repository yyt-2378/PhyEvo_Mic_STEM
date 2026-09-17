from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "benchmarks/fau_stack_v20_frozen_baseline.yaml"


def test_fau_v20_source_is_byte_exact_and_keeps_public_entrypoints() -> None:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    baseline = payload["baseline"]
    source = (PROJECT_ROOT / baseline["source"]).resolve()

    source.relative_to(PROJECT_ROOT.resolve())
    assert baseline["immutable"] is True
    assert source.is_file() and not source.is_symlink()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == baseline["sha256"]

    tree = ast.parse(source.read_text(encoding="utf-8"))
    symbols = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert set(baseline["public_entrypoints"]) <= symbols


def test_fau_v20_manifest_locks_current_candidate_policy() -> None:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    contract = payload["behavior_contract"]

    assert contract["candidate_owner"] == "fixed_delta_bic_structure_first"
    assert contract["minimum_delta_bic"] == 6.0
    assert contract["within_image_top_quantile_gate"] is False
    assert contract["local_brightness_gate"] is False
    assert contract["brightness_role"] == "audit_only"
    assert contract["gpa_role"] == "posthoc_spatial_audit_only"
    assert contract["reference_interpreted_candidate_count"] == 0

    parity = payload["generic_adapter_parity"]
    assert parity["target_possible_samples"]["RA"]["generic_candidate_count"] == 32
    assert parity["target_possible_samples"]["H2"]["generic_candidate_count"] == 71
    reference = parity["target_free_reference"]
    assert reference["retained_model_score_count"] == 0
    assert reference["generic_candidate_count"] == 0
    assert reference["false_positive_audit_retained"] is False
