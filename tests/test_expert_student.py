import numpy as np
import pytest

from tfree_detection.expert_student import (
    evaluate_case, expert_adaptation_loss, make_expert_student, matched_points,
    predict_scores, restore_checkpoint, save_checkpoint, torch,
)


pytestmark = pytest.mark.skipif(torch is None, reason="optional torch unavailable")


def batch():
    torch.set_num_threads(1)
    zeros = torch.zeros((1, 16, 16))
    human = zeros.clone()
    human[0, 4, 3] = 1
    return {"image": zeros[:, None], "teacher": zeros.clone(), "human": human,
            "human_weight": human.clone(), "teacher_weight": zeros.clone(),
            "teacher_available": torch.zeros(1), "geometry_index": torch.ones(1, dtype=torch.long)}


def test_empty_teacher_does_not_block_human_gradient_and_unknown_has_no_human_gradient():
    data = batch()
    model = make_expert_student(4)
    output = model(data["image"])
    output.geometry_logits.retain_grad()
    loss, parts = expert_adaptation_loss(output, data)
    loss.backward()
    gradient = output.geometry_logits.grad
    assert float(parts["human"]) > 0
    assert gradient[0, 1, 4, 3] < 0
    assert gradient[0, 1, 10, 10] == 0
    assert float(parts["teacher_positive"]) == 0


def test_teacher_only_ablation_ignores_expert_labels():
    data = batch()
    model = make_expert_student(4)
    loss, _ = expert_adaptation_loss(model(data["image"]), data, use_expert=False)
    assert float(loss) == 0


def test_unknown_auxiliary_heads_are_not_trained_or_reported_as_evidence():
    model = make_expert_student(4)
    for head in (model.evidence_head, model.violation_head, model.reliability_ood_head):
        assert all(not parameter.requires_grad for parameter in head.parameters())
        assert all(torch.count_nonzero(parameter) == 0 for parameter in head.parameters())


def test_pickle_free_checkpoint_resumes_exact_optimizer_and_rng(tmp_path):
    data = batch()
    torch.manual_seed(12)
    model = make_expert_student(4)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    rng = np.random.default_rng(2)

    def step(m, opt):
        opt.zero_grad()
        loss, _ = expert_adaptation_loss(m(data["image"]), data)
        loss.backward()
        opt.step()

    step(model, optimizer)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, optimizer, rng, {"base_channels": 4, "learning_rate": 0.001})
    random_expected = rng.integers(100000)
    step(model, optimizer)
    restored, optimizer2, rng2, meta = restore_checkpoint(checkpoint, resume=True)
    assert rng2.integers(100000) == random_expected
    step(restored, optimizer2)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    assert meta["calibrated"] is False
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, model, optimizer, rng, {})


def test_tiled_prediction_keeps_non_square_source_coordinates():
    data = batch()
    model = make_expert_student(4)
    score = predict_scores(model, np.zeros((19, 37), np.float32), 1, tile_size=16, overlap=4)
    assert score.shape == (19, 37)
    assert np.isfinite(score).all() and score.min() >= 0 and score.max() <= 1


def test_matches_are_one_to_one():
    assert matched_points(np.array([[5, 5]]), np.array([[5, 5], [6, 5]])) == 1
    assert matched_points(np.array([[5, 5]]), np.array([[25, 25]])) == 0


def test_sparse_evaluation_does_not_invent_precision_or_dense_accuracy():
    entry = {"case_number": 1, "geometry": "point", "group": "a", "action": "correct_geometry",
             "semantic_kind": "sparse_positive_points", "points_xy": [[5, 5]]}
    human = np.zeros((32, 32), bool)
    human[5, 5] = True
    score = human.astype(float)
    result = evaluate_case(entry, {"human": human, "teacher": np.zeros_like(human)}, score)
    assert result["precision"] is None
    assert result["point_recall_at_budget"]["50"]["matches"] == 1
    assert set(result["null_controls"]) == {"uniform_spatial_null", "border_only_null"}


def test_teacher_background_loss_ignores_padded_pixels():
    data = batch()
    data["teacher_available"][:] = 1
    data["valid_pixels"] = torch.zeros_like(data["teacher"])
    data["valid_pixels"][:, :8, :8] = 1
    model = make_expert_student(4)
    output = model(data["image"])
    output.geometry_logits.retain_grad()
    loss, _ = expert_adaptation_loss(output, data, use_expert=False)
    loss.backward()
    assert output.geometry_logits.grad[0, 1, 12, 12] == 0
    assert output.geometry_logits.grad[0, 1, 5, 5] != 0


def test_wide_interface_band_does_not_dilute_sparse_point_supervision():
    from types import SimpleNamespace
    logits = torch.zeros((2, 5, 16, 16), requires_grad=True)
    human = torch.zeros((2, 16, 16))
    human[0, 4, 3] = 1
    human[1, 4:12, :] = 1
    data = {"geometry_index": torch.tensor([1, 4]), "teacher": torch.zeros_like(human),
            "teacher_weight": torch.zeros_like(human), "teacher_available": torch.zeros(2),
            "human": human, "human_weight": human}
    loss, _ = expert_adaptation_loss(SimpleNamespace(geometry_logits=logits), data)
    loss.backward()
    point = logits.grad[0, 1].abs().sum()
    band = logits.grad[1, 4].abs().sum()
    assert torch.isclose(point, band)
