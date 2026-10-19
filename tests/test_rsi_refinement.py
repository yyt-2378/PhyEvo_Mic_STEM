from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tfree_detection.expert_student import (
    expert_adaptation_loss, make_expert_student, predict_scores, sample_batch, torch,
)
from tfree_detection.rsi_refinement import (
    POLICIES, RefinedSampler, predict_with_embedding, refinement_loss, source_probabilities,
)

pytestmark = pytest.mark.skipif(torch is None, reason="optional torch unavailable")


def synthetic():
    torch.set_num_threads(1)
    entries, arrays = [], {}
    rng = np.random.default_rng(5)
    for n, (group, geometry, shape) in enumerate([
        ("a", 1, (19, 37)), ("b", 1, (8, 11)), ("c", 4, (33, 40)),
        ("c", 4, (25, 29)), ("d", 4, (40, 40)),
    ], 1):
        human = np.zeros(shape, np.float32)
        human[2:4, 3:5] = 1
        arrays[n] = {"image": rng.random(shape).astype(np.float32), "teacher": human.copy(),
                     "teacher_weight": np.ones(shape, np.float32), "human": human,
                     "human_weight": human.copy()}
        entries.append({"case_number": n, "group": group, "geometry_index": geometry,
                        "teacher_positive_pixels": 4, "action": "correct_geometry",
                        "semantic_kind": "interface_band_roi_not_centerline" if geometry == 4 else "sparse_positive_points"})
    return entries, arrays


def sample(sampler, rng, **kwargs):
    return sampler.sample(rng, batch_size=24, patch_size=16, device="cpu", augmentation_seed=99, **kwargs)


def test_control_reproduces_old_pixels_labels_and_next_rng_exactly():
    entries, arrays = synthetic()
    old_rng, new_rng = np.random.default_rng(19), np.random.default_rng(19)
    old = sample_batch(entries, arrays, old_rng, batch_size=24, patch_size=16, device="cpu")
    new, _ = sample(RefinedSampler(entries, arrays, POLICIES["control"]), new_rng)
    for key, value in old.items():
        assert torch.equal(value, new[key]), key
    assert old_rng.bit_generator.state == new_rng.bit_generator.state
    model = make_expert_student(4)
    output = model(old["image"])
    assert torch.equal(expert_adaptation_loss(output, old)[0], refinement_loss(output, new, POLICIES["control"])[0])


def test_photo_does_not_change_crop_schedule_or_labels():
    entries, arrays = synthetic()
    a, records_a = sample(RefinedSampler(entries, arrays, POLICIES["positive_crops"]), np.random.default_rng(23))
    b, records_b = sample(RefinedSampler(entries, arrays, POLICIES["photometric"]), np.random.default_rng(23))
    assert records_a == records_b
    for key in a:
        if key != "image":
            assert torch.equal(a[key], b[key]), key
    assert not torch.equal(a["image"], b["image"])
    assert b["image"].min() >= 0 and b["image"].max() <= 1


def test_guided_crops_retain_training_positives_and_account_for_padding():
    entries, arrays = synthetic()
    sampler = RefinedSampler(entries, arrays, replace(POLICIES["positive_crops"], positive_crop_probability=1))
    data, records = sample(sampler, np.random.default_rng(7))
    assert all(r["guided"] and r["has_human_supervision"] for r in records)
    assert sampler.audit["positive_patches"] == sampler.audit["patches"] == 24
    assert sum(sampler.audit["by_case"].values()) == 24
    assert torch.all(data["human_weight"][data["valid_pixels"] == 0] == 0)
    assert torch.all(data["teacher_weight"][data["valid_pixels"] == 0] == 0)


def test_source_weight_and_holdout_guard():
    entries, arrays = synthetic()
    groups, weights = source_probabilities(entries, 4, .5)
    assert groups == ["c", "d"]
    np.testing.assert_allclose(weights, np.array([np.sqrt(2), 1]) / (np.sqrt(2)+1))
    with pytest.raises(ValueError, match="held-out"):
        RefinedSampler(entries, arrays, POLICIES["weak_roi"], excluded_groups=["a"])


def test_weak_bag_does_not_turn_unknown_into_expert_negative():
    logits = torch.zeros((1, 5, 16, 16), requires_grad=True)
    human = torch.zeros((1, 16, 16))
    human[:, 4:8, 4:8] = 1
    data = {"geometry_index": torch.tensor([4]), "teacher": torch.zeros_like(human),
            "teacher_weight": torch.zeros_like(human), "human": human, "human_weight": human.clone(),
            "weak_band": torch.tensor([True]), "valid_pixels": torch.ones_like(human)}
    loss, _ = refinement_loss(SimpleNamespace(geometry_logits=logits), data, POLICIES["weak_bag"])
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 4]) == 4  # top quarter of 16 known pixels
    assert torch.all(logits.grad[0, 4][human[0] == 0] == 0)


def test_occupancy_penalty_is_explicit_not_a_label_mutation_and_ignores_padding():
    logits = torch.zeros((1, 5, 16, 16), requires_grad=True)
    human = torch.zeros((1, 16, 16))
    valid = human.clone()
    valid[:, :8, :8] = 1
    data = {"geometry_index": torch.tensor([1]), "teacher": human.clone(),
            "teacher_weight": human.clone(), "human": human, "human_weight": human.clone(),
            "weak_band": torch.tensor([False]), "valid_pixels": valid}
    loss, parts = refinement_loss(SimpleNamespace(geometry_logits=logits), data, POLICIES["weak_roi"])
    loss.backward()
    assert float(parts["occupancy_prior"]) > 0 and float(parts["human"]) == 0
    assert logits.grad[0, 1, 2, 2] > 0
    assert logits.grad[0, 1, 12, 12] == 0
    assert torch.count_nonzero(data["human_weight"]) == 0


def test_embedding_extraction_preserves_original_scores_exactly():
    synthetic()
    model = make_expert_student(4)
    image = np.random.default_rng(8).random((19, 37)).astype(np.float32)
    old = predict_scores(model, image, 1, normalized=True, tile_size=16, overlap=4)
    new, embedding = predict_with_embedding(model, image, 1, tile_size=16, overlap=4)
    np.testing.assert_array_equal(old, new)
    assert embedding.ndim == 1 and np.isfinite(embedding).all()
    np.testing.assert_allclose(np.linalg.norm(embedding), 1, atol=1e-6)


@pytest.mark.parametrize("kwargs", [{"source_power": float("nan")}, {"weak_band_fraction": 0}, {"occupancy_weight": -1}])
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(POLICIES["control"], **kwargs)
