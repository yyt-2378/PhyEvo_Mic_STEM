from scripts.run_rsi_refinement import experiment_folds, representatives, train
from tfree_detection.expert_student import restore_checkpoint, torch


def manifest():
    groups = ["point", "zro", "sigma3-orientation-i", "sigma3-orientation-ii"]
    entries = [{"case_number": i+1, "group": g, "human_known_pixels": 1} for i, g in enumerate(groups)]
    entries += [{"case_number": 5, "group": "region", "human_known_pixels": 1}]
    return {"entries": entries, "folds": [{"held_out_group": g} for g in groups]}


def test_directed_and_nested_folds_do_not_mix_heldout_sources():
    data = manifest()
    folds = experiment_folds(data)
    assert len(folds) == 4+6+2
    for fold in folds:
        train_entries = [e for e in data["entries"] if e["case_number"] in fold["train_case_numbers"]]
        assert not {e["group"] for e in train_entries} & set(fold["excluded_groups"])
        assert not set(fold["train_case_numbers"]) & set(fold["validation_case_numbers"])
        assert {e["case_number"] for e in representatives(train_entries)} <= set(fold["train_case_numbers"])
    directed = next(f for f in folds if f["fold_id"] == "ORI-to-ORII")
    assert directed["train_case_numbers"] == [3] and directed["validation_case_numbers"] == [4]


def test_checkpoint_resume_preserves_augmentation_optimizer_weights_and_sampling(tmp_path):
    from test_rsi_refinement import synthetic
    entries, arrays = synthetic()
    for e in entries:
        e["geometry"] = "point" if e["geometry_index"] == 1 else "interface"
    fold = {"fold_id": "test", "excluded_groups": ["heldout"], "train_case_numbers": list(arrays), "validation_case_numbers": [99]}
    protocol = {"base_channels": 4, "device": "cpu", "learning_rate": .001, "learning_manifest_sha256": "synthetic",
                "batch_size": 4, "patch_size": 16, "steps": 4, "checkpoints": [2,4], "gradient_clip": 5}
    full, resumed = tmp_path/"full", tmp_path/"resumed"
    train(entries, arrays, full, protocol, "same-binding", fold, "weak_roi", 23, resume=False)
    train(entries, arrays, resumed, {**protocol, "steps": 2}, "same-binding", fold, "weak_roi", 23, resume=False)
    train(entries, arrays, resumed, protocol, "same-binding", fold, "weak_roi", 23, resume=True)
    a, _, rng_a, meta_a = restore_checkpoint(full/"step-000004", resume=True)
    b, _, rng_b, meta_b = restore_checkpoint(resumed/"step-000004", resume=True)
    assert meta_a["sampler_audit"] == meta_b["sampler_audit"]
    assert rng_a.bit_generator.state == rng_b.bit_generator.state
    for name, tensor in a.state_dict().items():
        assert torch.equal(tensor, b.state_dict()[name]), name
    before = (resumed/"step-000004"/"state.npz").read_bytes()
    train(entries, arrays, resumed, protocol, "same-binding", fold, "weak_roi", 23, resume=True)
    assert (resumed/"step-000004"/"state.npz").read_bytes() == before
