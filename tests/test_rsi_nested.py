import pytest

from scripts.run_rsi_nested import idle_gpus, pair_folds


def test_pair_folds_exclude_both_sources_without_dropping_other_geometry():
    manifest = {"folds": [{"held_out_group": g} for g in "abc"],
                "entries": [{"case_number": n, "group": g, "human_known_pixels": known}
                            for n, g, known in [(1, "a", 1), (2, "b", 1), (3, "c", 1),
                                                (4, "region", 1), (5, "reject", 0)]]}
    folds = pair_folds(manifest)
    assert len(folds) == 3
    for fold in folds:
        train = set(fold["train_case_numbers"])
        valid = set(fold["validation_case_numbers"])
        assert not train & valid
        assert 4 in train and 5 not in train
        assert len(valid) == 2
        assert not {e["group"] for e in manifest["entries"] if e["case_number"] in train} & set(fold["excluded_groups"])


def test_gpu_guard_rejects_occupied_and_duplicate_devices(monkeypatch):
    monkeypatch.setattr("scripts.run_rsi_nested.subprocess.check_output",
                        lambda *a, **kw: "0, 50000, 40\n2, 17, 0\n")
    idle_gpus([2])
    with pytest.raises(ValueError, match="occupied"):
        idle_gpus([0])
    with pytest.raises(ValueError, match="unique"):
        idle_gpus([2, 2])
