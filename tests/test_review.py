import pytest

from tfree_detection.evidence import LabelTier
from tfree_detection.review import (
    ReviewAction,
    ReviewResponse,
    consensus_outcomes,
)


def _response(
    response_id: str,
    reviewer: str,
    action: ReviewAction,
    labels=("occupancy",),
    assisted=False,
) -> ReviewResponse:
    return ReviewResponse(
        response_id=response_id,
        candidate_id="image:atomic:0001",
        reviewer_id=reviewer,
        action=action,
        defect_labels=labels,
        annotation_duration_seconds=18.0,
        model_assistance_shown=assisted,
    )


def test_two_independent_agreeing_reviewers_create_gold() -> None:
    outcomes = consensus_outcomes(
        [
            _response("r1", "expert-a", ReviewAction.ACCEPT),
            _response(
                "r2", "expert-b", ReviewAction.ACCEPT, assisted=True
            ),
        ],
        annotation_version="pilot-1",
    )
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.status == LabelTier.GOLD.value
    assert outcome.annotation is not None
    assert outcome.annotation.label_tier == LabelTier.GOLD
    assert outcome.annotation.reviewers == ("expert-a", "expert-b")
    assert outcome.assistance_conditions == ("assisted", "blinded")


def test_disagreement_and_single_review_remain_unresolved() -> None:
    disagreement = consensus_outcomes(
        [
            _response("r1", "expert-a", ReviewAction.ACCEPT),
            _response("r2", "expert-b", ReviewAction.REJECT, labels=()),
        ],
        annotation_version="pilot-1",
    )[0]
    assert disagreement.status == LabelTier.UNRESOLVED.value
    assert disagreement.annotation is None
    assert "disagreement" in disagreement.disagreement_reason

    single = consensus_outcomes(
        [_response("r3", "expert-a", ReviewAction.ACCEPT)],
        annotation_version="pilot-1",
    )[0]
    assert single.status == LabelTier.UNRESOLVED.value
    assert single.disagreement_reason == "fewer_than_two_independent_reviewers"


def test_same_reviewer_cannot_vote_twice_for_one_candidate() -> None:
    with pytest.raises(ValueError, match="at most one response"):
        consensus_outcomes(
            [
                _response("r1", "expert-a", ReviewAction.ACCEPT),
                _response("r2", "expert-a", ReviewAction.ACCEPT),
            ],
            annotation_version="pilot-1",
        )
