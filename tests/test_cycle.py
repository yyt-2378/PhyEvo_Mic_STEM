import pytest

from tfree_detection.cycle import CycleStage, SelfImprovementCycle


def _cycle() -> SelfImprovementCycle:
    return SelfImprovementCycle(
        cycle_id="cycle-001",
        registry_hash="registry-v1",
        teacher_version="teacher-1.0",
        parent_student_version=None,
        locked_external_test_hash="locked-empty-v1",
    )


def test_cycle_enforces_order_and_required_artifacts() -> None:
    cycle = _cycle()
    with pytest.raises(ValueError, match="Invalid cycle transition"):
        cycle.advance(
            CycleStage.STUDENT_TRAINED,
            artifact_hashes={
                "student_checkpoint": "checkpoint",
                "training_manifest": "training",
            },
        )
    with pytest.raises(ValueError, match="missing artifacts"):
        cycle.advance(
            CycleStage.TEACHER_COMPLETED,
            artifact_hashes={"candidate_table": "candidates"},
        )
    cycle.advance(
        CycleStage.TEACHER_COMPLETED,
        artifact_hashes={
            "candidate_table": "candidates",
            "teacher_manifest": "teacher",
        },
    )
    assert cycle.stage == CycleStage.TEACHER_COMPLETED
    assert len(cycle.content_hash()) == 64


def test_locked_test_cannot_change_during_evaluation() -> None:
    cycle = _cycle()
    transitions = [
        (
            CycleStage.TEACHER_COMPLETED,
            {"candidate_table": "a", "teacher_manifest": "b"},
        ),
        (
            CycleStage.ACQUISITION_FROZEN,
            {"acquisition_plan": "c", "train_partition_hash": "d"},
        ),
        (
            CycleStage.EXPERT_REVIEW_COMPLETED,
            {"annotation_ledger": "e", "review_time_report": "f"},
        ),
        (
            CycleStage.STUDENT_TRAINED,
            {"student_checkpoint": "g", "training_manifest": "h"},
        ),
        (
            CycleStage.CALIBRATED,
            {"calibrator": "i", "calibration_report": "j"},
        ),
    ]
    for stage, artifacts in transitions:
        cycle.advance(stage, artifact_hashes=artifacts)
    with pytest.raises(ValueError, match="changed during the cycle"):
        cycle.advance(
            CycleStage.EVALUATED,
            artifact_hashes={
                "evaluation_report": "k",
                "locked_test_hash": "changed",
            },
        )


def test_exploratory_silver_cycle_cannot_promote_production_release():
    cycle = _cycle()
    for stage, artifacts in [
        (CycleStage.TEACHER_COMPLETED, {"candidate_table": "a", "teacher_manifest": "b"}),
        (CycleStage.ACQUISITION_FROZEN, {"acquisition_plan": "c", "train_partition_hash": "d"}),
        (CycleStage.EXPERT_REVIEW_COMPLETED, {"annotation_ledger": "e", "review_time_report": "f"}),
        (CycleStage.STUDENT_TRAINED, {"student_checkpoint": "g", "training_manifest": "h"}),
        (CycleStage.EXPLORATORY_EVALUATED, {"evaluation_report": "i", "held_out_groups_manifest": "j"}),
    ]:
        cycle.advance(stage, artifact_hashes=artifacts)
    with pytest.raises(ValueError, match="Invalid cycle transition"):
        cycle.advance(CycleStage.RELEASE_ACCEPTED, artifact_hashes={"release_decision": "a", "model_lineage": "b"})
    cycle.advance(CycleStage.RELEASE_REJECTED, artifact_hashes={"release_decision": "a", "model_lineage": "b"})
    assert cycle.stage == CycleStage.RELEASE_REJECTED
