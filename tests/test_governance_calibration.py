from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tfree_detection.calibration import (
    CalibrationExample,
    IsotonicCalibrator,
    SplitConformalBinary,
    TemperatureCalibrator,
    calibration_report,
)
from tfree_detection.evidence import LabelTier
from tfree_detection.governance import (
    AnnotationLedger,
    AnnotationRecord,
    AnnotationSource,
    DataAsset,
    DataPartition,
    DataRegistry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _asset(
    asset_id: str,
    partition: DataPartition,
    group: str,
    *,
    active: bool = False,
) -> DataAsset:
    return DataAsset(
        asset_id=asset_id,
        dataset_id="dataset",
        relative_path=f"datasets/{asset_id}.tif",
        partition=partition,
        split_group_id=group,
        acquisition_id=group,
        sample_id=group,
        lab_id="lab",
        tasks=("amorphous_region",),
        checksum="0" * 64,
        eligible_for_active_learning=active,
    )


def test_checked_in_registry_is_honest_and_deterministic() -> None:
    registry = DataRegistry.from_yaml(
        PROJECT_ROOT / "datasets" / "partition_registry.yaml"
    )
    assert len(registry.assets) == 5
    assert not registry.active_learning_assets()
    development = registry.assets_for_partition(
        DataPartition.DEVELOPMENT_TEST
    )
    locked = registry.assets_for_partition(
        DataPartition.LOCKED_EXTERNAL_TEST
    )
    assert len(development) == 4
    assert len(locked) == 1
    assert all(
        asset.partition == DataPartition.DEVELOPMENT_TEST
        for asset in development
    )
    assert locked[0].eligible_for_active_learning is False
    assert len(registry.content_hash()) == 64

    # The public repository intentionally excludes the four third-party raw
    # assets under datasets/public/.  Verify the checked-in reference asset
    # unconditionally; full-registry verification remains an integration
    # command for workspaces where the registered datasets are materialized.
    checked_in = [
        asset
        for asset in registry.assets
        if asset.relative_path == "reference_scale_radial_wiener.png"
    ]
    assert len(checked_in) == 1
    assert sum(
        asset.relative_path.startswith("datasets/public/")
        for asset in registry.assets
    ) == 4
    tracked_registry = DataRegistry(
        registry_version=registry.registry_version,
        split_unit=registry.split_unit,
        assets=checked_in,
    )
    verification = tracked_registry.verify_files(PROJECT_ROOT)
    assert verification["verified"] is True
    assert verification["asset_count"] == 1


def test_registry_rejects_group_leakage_and_test_selection() -> None:
    with pytest.raises(ValueError, match="Split leakage"):
        DataRegistry(
            registry_version="1",
            split_unit="explicit_group",
            assets=[
                _asset("a", DataPartition.TRAIN, "same"),
                _asset("b", DataPartition.CALIBRATION, "same"),
            ],
        ).validate()
    with pytest.raises(ValueError, match="Only train assets"):
        DataRegistry(
            registry_version="1",
            split_unit="explicit_group",
            assets=[
                _asset(
                    "locked",
                    DataPartition.LOCKED_EXTERNAL_TEST,
                    "locked",
                    active=True,
                )
            ],
        ).validate()


def test_gold_cannot_be_created_by_teacher_or_overwritten() -> None:
    ledger = AnnotationLedger()
    candidate_id = "image:task:0001"
    with pytest.raises(ValueError, match="Gold labels require"):
        ledger.append(
            AnnotationRecord(
                annotation_id="bad-gold",
                candidate_id=candidate_id,
                label_tier=LabelTier.GOLD,
                source=AnnotationSource.PHYSICS_PSEUDO,
                annotation_version="1",
                defect_labels=("occupancy",),
                accepted=True,
            )
        )
    gold = AnnotationRecord(
        annotation_id="gold-1",
        candidate_id=candidate_id,
        label_tier=LabelTier.GOLD,
        source=AnnotationSource.EXPERT_CONSENSUS,
        annotation_version="1",
        defect_labels=("occupancy",),
        accepted=True,
        reviewers=("expert-a", "expert-b"),
    )
    ledger.append(gold)
    with pytest.raises(ValueError, match="cannot be overwritten"):
        ledger.append(
            AnnotationRecord(
                annotation_id="silver-after-gold",
                candidate_id=candidate_id,
                label_tier=LabelTier.SILVER,
                source=AnnotationSource.PHYSICS_PSEUDO,
                annotation_version="2",
                defect_labels=("intensity",),
                accepted=True,
                parent_annotation_id=gold.annotation_id,
            )
        )


def _calibration_examples() -> list[CalibrationExample]:
    scores = [-3, -2, -1, -0.4, 0.2, 0.8, 1.4, 2.0, 3.0]
    labels = [False, False, False, True, False, True, True, True, True]
    return [
        CalibrationExample(
            candidate_id=f"candidate-{index}",
            raw_score=score,
            is_true_defect=label,
            partition=DataPartition.CALIBRATION,
            label_tier=LabelTier.GOLD,
            split_group_id=f"acquisition-{index}",
        )
        for index, (score, label) in enumerate(zip(scores, labels))
    ]


def test_gold_only_calibrators_and_risk_coverage_contract() -> None:
    examples = _calibration_examples()
    isotonic = IsotonicCalibrator().fit(examples)
    grid = np.linspace(-4, 4, 81)
    probabilities = isotonic.predict(grid)
    assert np.all(np.diff(probabilities) >= 0)
    assert np.min(probabilities) >= 0
    assert np.max(probabilities) <= 1

    temperature = TemperatureCalibrator().fit(examples)
    temperature_probabilities = temperature.predict(
        [item.raw_score for item in examples]
    )
    report = calibration_report(
        temperature_probabilities,
        [item.is_true_defect for item in examples],
        bins=4,
    )
    assert 0 <= report["brier_score"] <= 1
    assert 0 <= report["expected_calibration_error"] <= 1
    assert report["risk_coverage"]["coverage"][-1] == 1.0

    conformal = SplitConformalBinary(alpha=0.2).fit(
        temperature_probabilities,
        [item.is_true_defect for item in examples],
    )
    assert len(conformal.predict_sets([0.1, 0.5, 0.9])) == 3
    assert "exchangeable" in conformal.to_dict()["assumption"]


def test_calibration_rejects_non_gold_or_wrong_partition() -> None:
    examples = _calibration_examples()
    with pytest.raises(ValueError, match="only the calibration partition"):
        IsotonicCalibrator().fit(
            [
                replace(
                    examples[0],
                    partition=DataPartition.TRAIN,
                ),
                *examples[1:],
            ]
        )
    with pytest.raises(ValueError, match="only Gold"):
        TemperatureCalibrator().fit(
            [
                replace(examples[0], label_tier=LabelTier.SILVER),
                *examples[1:],
            ]
        )
