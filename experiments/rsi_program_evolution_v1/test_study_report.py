import pytest
from experiments.rsi_program_evolution_v1.study_report import resolved_rounds


def test_late_transport_response_does_not_count_as_extra_scientific_trial():
    events = [{"kind": "round_finished", "body": {"round": 1, "status": "provider_failed"}},
              {"kind": "late_response_executed", "body": {"round": 1, "status": "accepted"}},
              {"kind": "round_finished", "body": {"round": 2, "status": "retained_for_research"}}]
    rows = resolved_rounds(events)
    assert len(rows) == 2 and rows[0]["status"] == "accepted"
    assert rows[0]["original_transport_outcome"] == "provider_failed"


def test_report_rejects_retrospective_replacement_of_valid_trial():
    with pytest.raises(ValueError, match="original terminal"):
        resolved_rounds([{"kind": "round_finished", "body": {"round": 1, "status": "accepted"}},
                         {"kind": "late_response_executed", "body": {"round": 1, "status": "accepted"}}])
