from argparse import Namespace
import json

import numpy as np
import pytest

from scripts.run_expert_learning_cycle import final_training_fold, launch_gpu_workers, train_model
from tfree_detection.expert_student import torch


def test_gpu_scheduler_refuses_occupied_or_duplicate_devices(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.run_expert_learning_cycle.subprocess.check_output",
                        lambda *a, **k: "2, 2048, 90\n3, 17, 0\n")
    with pytest.raises(ValueError, match="not idle"):
        launch_gpu_workers(tmp_path, {}, [2], resume=False)
    with pytest.raises(ValueError, match="unique"):
        launch_gpu_workers(tmp_path, {}, [3, 3], resume=False)
    assert not list(tmp_path.iterdir())


def test_final_fit_does_not_include_rejected_or_unknown_geometry():
    fold = final_training_fold([{"case_number": 1, "human_known_pixels": 0},
                                {"case_number": 2, "human_known_pixels": 5}])
    assert fold["train_case_numbers"] == [2]
    assert fold["validation_case_numbers"] == []


@pytest.mark.skipif(torch is None, reason="optional torch")
def test_training_checkpoint_can_resume_without_restarting_or_overwriting(tmp_path):
    torch.set_num_threads(1)
    image = np.zeros((16, 16), np.float32)
    human = image.copy()
    human[4, 3] = 1
    entries = [{"case_number": 1, "group": "train", "geometry": "point", "geometry_index": 1,
                "teacher_positive_pixels": 0, "action": "correct_geometry"}]
    arrays = {1: {"image": image, "teacher": image.copy(), "human": human,
                  "human_weight": human.copy(), "teacher_weight": image.copy()}}
    protocol = {"device": "cpu", "seed": 2, "base_channels": 4, "learning_rate": 0.001,
                "steps": 2, "batch_size": 1, "patch_size": 16, "checkpoint_every": 1,
                "learning_manifest_sha256": "synthetic"}
    fold = {"fold_id": "f", "held_out_group": "test", "train_case_numbers": [1]}
    run = tmp_path / "run"
    _, checkpoint = train_model(entries, arrays, run, protocol, "protocol", fold, "teacher_plus_expert", resume=False)
    before = (checkpoint / "state.npz").read_bytes()
    _, same = train_model(entries, arrays, run, protocol, "protocol", fold, "teacher_plus_expert", resume=True)
    assert checkpoint == same
    assert (same / "state.npz").read_bytes() == before
    with pytest.raises(ValueError, match="Resume mismatch"):
        train_model(entries, arrays, run, protocol, "changed", fold, "teacher_plus_expert", resume=True)
    with pytest.raises(ValueError, match="leaked"):
        train_model(entries, arrays, tmp_path / "leak", protocol, "protocol",
                    {**fold, "held_out_group": "train"}, "teacher_only", resume=False)
