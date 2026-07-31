import json
from pathlib import Path

import pytest

from tfree_detection.cli import main
from tfree_detection.dataset_label_adapter import (
    EpistemicState,
    ExternalLabel,
    SemanticRole,
    adapter_manifest,
    default_label_adapter_registry,
)
from tfree_detection.evidence import LabelTier


def _external(
    dataset_id: str,
    source_label: str,
    *,
    qualification: str = "author_curated",
) -> ExternalLabel:
    return ExternalLabel(
        dataset_id=dataset_id,
        asset_id="asset-1",
        annotation_id="annotation-1",
        source_label=source_label,
        spatial_support="region",
        annotation_qualification=qualification,
        source_payload={"palette_value": 2, "original_note": "preserve me"},
    )


def test_emseek_atomic_mask_is_not_silently_promoted_to_vacancy() -> None:
    adapted = default_label_adapter_registry().adapt(
        _external("emseek_wse2_atomic_defects", "WSe2 point mask")
    )
    assert adapted.ontology_family == "atomic_site_anomaly"
    assert adapted.ontology_label == "atomic_column_intensity_anomaly"
    assert adapted.label_tier == LabelTier.SILVER
    assert "vacancy_like" in adapted.alternative_hypotheses
    assert adapted.eligible_for_hard_label_training is False
    serialized = adapted.to_dict()
    assert serialized["original"]["source_payload"]["original_note"] == (
        "preserve me"
    )


def test_dennler_composite_label_remains_unresolved() -> None:
    adapted = default_label_adapter_registry().adapt(
        _external("dennler_2021_hrstem_defects", "class_2")
    )
    assert adapted.epistemic_state == EpistemicState.COMPOSITE
    assert adapted.label_tier == LabelTier.UNRESOLVED
    assert {
        "local_amorphous_region",
        "defocus_or_blur",
        "beam_damage_consistent_region",
    }.issubset(set(adapted.alternative_hypotheses))
    assert adapted.eligible_for_hard_label_training is False


def test_simulation_gold_is_explicitly_separate_and_trainable() -> None:
    adapted = default_label_adapter_registry().adapt(
        _external("stemgym_v1", "oxygen vacancy")
    )
    assert adapted.ontology_label == "vacancy_like"
    assert adapted.label_tier == LabelTier.GOLD
    assert adapted.annotation_qualification == "simulation_truth"
    assert adapted.eligible_for_hard_label_training is True


def test_unknown_source_label_uses_open_set_exit() -> None:
    adapted = default_label_adapter_registry().adapt(
        _external("guo_2021_atomic_defects", "new-unmapped-author-label")
    )
    assert adapted.semantic_role == SemanticRole.UNKNOWN
    assert adapted.epistemic_state == EpistemicState.UNRESOLVED
    assert adapted.ontology_family == "open_set_resolution"


def test_wse2_3d_vacancy_status_remains_author_silver() -> None:
    adapted = default_label_adapter_registry().adapt(
        _external(
            "zhang_2026_wse2_3d_defects_coordinates",
            "Vacancy",
        )
    )
    assert adapted.ontology_label == (
        "author_reconstructed_vacancy_candidate"
    )
    assert adapted.epistemic_state == EpistemicState.PROBABLE
    assert adapted.label_tier == LabelTier.SILVER
    assert adapted.eligible_for_hard_label_training is False
    assert "coordinate_refinement_failure" in (
        adapted.alternative_hypotheses
    )


def test_gold_rejects_unqualified_source() -> None:
    adapted = default_label_adapter_registry().adapt(
        _external("stemgym_v1", "oxygen vacancy", qualification="none")
    )
    assert adapted.annotation_qualification == "simulation_truth"
    assert adapted.label_tier == LabelTier.GOLD


def test_adapter_manifest_and_cli_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = adapter_manifest()
    assert set(manifest["adapters"]) >= {
        "emseek-v1",
        "guo-2021-v1",
        "al-sigma3-v1",
        "dennler-2021-v1",
        "stemgym-v1",
        "wse2-3d-coordinates-v1",
    }
    source = tmp_path / "labels.json"
    output = tmp_path / "adapted.jsonl"
    source.write_text(
        json.dumps(
            [
                _external(
                    "al_sigma3_twin_boundaries", "Sigma3_ORI STEM"
                ).to_dict()
            ]
        ),
        encoding="utf-8",
    )
    main(
        [
            "adapt-labels",
            "--input",
            str(source),
            "--output",
            str(output),
        ]
    )
    captured = json.loads(capsys.readouterr().out)
    assert captured["adapted"] == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["original"]["source_label"] == "Sigma3_ORI STEM"
    assert payload["adapted"]["ontology_label"] == "twin_boundary"

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(
            [
                "adapt-labels",
                "--input",
                str(source),
                "--output",
                str(output),
            ]
        )
