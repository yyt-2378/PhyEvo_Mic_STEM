from tfree_detection.review_cases import (
    build_expert_review_cases,
    summarize_review_cases,
)
from tfree_detection.review_dataset import ReviewSample


def _sample(
    sample_id: str,
    *,
    dataset_id: str = "al_sigma3_twin_boundaries",
    channel: str | None = "HAADF",
) -> ReviewSample:
    return ReviewSample(
        sample_id=sample_id,
        dataset_id=dataset_id,
        source_path="source.emd" if channel else "source.png",
        source_format="emd" if channel else "png",
        source_partition="development_test",
        review_split="train",
        split_group_id="group",
        acquisition_id="acquisition",
        material="Al",
        primary_task="phase_boundary",
        prediction_tasks=("phase_boundary",),
        geometry_hint="interface",
        frame_index=0 if channel else None,
        dataset_path="Data/Image/channel/Data" if channel else None,
        channel_name=channel,
        label_kind="source_prior_only",
    )


def test_five_sigma3_channels_become_one_shared_boundary_case() -> None:
    channels = ["BF", "HAADF", "DF2", "HAADF", "DF4"]
    samples = [
        _sample(f"sigma-channel-{index}", channel=channel)
        for index, channel in enumerate(channels)
    ]
    cases = build_expert_review_cases(samples)
    assert len(cases) == 1
    assert cases[0].case_type == "multichannel_frame"
    assert cases[0].primary_sample_id == "sigma-channel-1"
    assert cases[0].annotation_scope == (
        "shared_across_five_detector_channels"
    )
    summary = summarize_review_cases(cases)
    assert summary["image_plane_count"] == 5
    assert summary["duplicate_annotation_planes_avoided"] == 4


def test_non_sigma_sample_remains_single_plane_case() -> None:
    sample = _sample(
        "guo-zro2", dataset_id="guo_2021_atomic_defects", channel=None
    )
    cases = build_expert_review_cases([sample])
    assert len(cases) == 1
    assert cases[0].sample_ids == ("guo-zro2",)
