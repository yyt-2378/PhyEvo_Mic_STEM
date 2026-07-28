from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_version_iteration_comparison.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_version_iteration_comparison",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_uses_frozen_project_evidence() -> None:
    payload = _load_script_module().collect_comparison()

    registries = payload["registries"]
    assert [item["asset_count"] for item in registries] == [
        5,
        220,
        226,
        291,
    ]
    assert [item["dataset_count"] for item in registries] == [3, 4, 6, 7]
    assert [item["split_group_count"] for item in registries] == [
        5,
        6,
        12,
        29,
    ]
    assert [
        item["active_learning_asset_count"] for item in registries
    ] == [0, 107, 107, 139]
    assert all(
        item["partition_counts"]["calibration"] == 0
        for item in registries
    )

    atomic = payload["atomic_wse2_train_domain"]["metrics"]
    assert atomic["Legacy v0.2"]["precision"] == pytest.approx(
        0.48949127858868696
    )
    assert atomic["Consensus v2"]["precision"] == pytest.approx(
        0.8914257145237897
    )
    assert atomic["Legacy v0.2"]["recall"] == pytest.approx(
        0.9225826575171553
    )
    assert atomic["Consensus v2"]["recall"] == pytest.approx(
        0.6673736743605739
    )
    assert atomic["Consensus v2"]["f1"] == pytest.approx(
        0.7632977774606685
    )

    locked = payload["dennler_locked_external"]["targets"]
    assert locked["any_expert_marked_defect"]["iou"] == pytest.approx(
        0.1395962754010481
    )

    student = payload["student_synthetic_ablation"]
    assert student["iou_difference"] == pytest.approx(0.0611656)
    assert student["release_claim_allowed"] is False
    assert payload["claim_boundary"] == {
        "calibration_asset_count": 0,
        "nature_level_method_claim_ready": False,
        "data_growth_is_accuracy_growth": False,
    }
