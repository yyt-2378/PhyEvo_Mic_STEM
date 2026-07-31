from tfree_detection.review_cases import build_expert_review_cases
from tfree_detection.review_dataset import ReviewSample
from tfree_detection.review_platform import audit_review_platform
from tfree_detection.review_prediction import PREDICTION_FORMAT, ReviewPrediction


def _sample(label_kind="none", label_path=None, split="train") -> ReviewSample:
    return ReviewSample(
        sample_id="sample",
        dataset_id="synthetic",
        source_path="image.png",
        source_format="png",
        source_partition="train",
        review_split=split,
        split_group_id="group",
        acquisition_id="acquisition",
        material="synthetic",
        primary_task="amorphous_region",
        prediction_tasks=("amorphous_region",),
        geometry_hint="region",
        label_kind=label_kind,
        label_path=label_path,
    )


def _prediction() -> ReviewPrediction:
    return ReviewPrediction(
        prediction_format=PREDICTION_FORMAT,
        sample_id="sample",
        task="amorphous_region",
        sample_sha256="a" * 64,
        config_path="configs/amorphous.yaml",
        config_sha256="b" * 64,
        pixel_sha256="c" * 64,
        mask_path="mask.png",
        mask_sha256="d" * 64,
        image_shape=(32, 32),
        detected_area_pixels=0,
        detected_area_fraction=0,
        candidates=(),
    )


def test_audit_proves_unlabeled_case_and_prediction_coverage() -> None:
    sample = _sample()
    case = build_expert_review_cases([sample])
    report = audit_review_platform(
        [sample], case, {("sample", "amorphous_region"): _prediction()},
        require_v1_inventory=False,
    )
    assert report["passed"]
    assert report["existing_pixel_label_count"] == 0
    assert report["primary_prediction_count"] == 1


def test_audit_rejects_labeled_or_prediction_missing_samples() -> None:
    sample = _sample(label_kind="pixel_mask", label_path="label.png")
    case = build_expert_review_cases([sample])
    report = audit_review_platform(
        [sample], case, {}, require_v1_inventory=False
    )
    assert not report["passed"]
    assert report["existing_pixel_label_count"] == 1
    assert "expert_plane_prediction_missing" in report["issues"]
