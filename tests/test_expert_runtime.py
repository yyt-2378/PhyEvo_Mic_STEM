import hashlib
import json

import numpy as np
import pytest

from tfree_detection.expert_runtime import (
    ExpertStudentRuntime, detect_with_expert_student, save_shadow_prediction,
)
from tfree_detection.expert_student import make_expert_student, save_checkpoint, torch
from tfree_detection.models import DetectionResult


pytestmark = pytest.mark.skipif(torch is None, reason="optional torch unavailable")


def checkpoint(root):
    torch.set_num_threads(1)
    model = make_expert_student(4)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    path = root / "checkpoint"
    save_checkpoint(path, model, optimizer, np.random.default_rng(1), {
        "base_channels": 4, "learning_rate": 0.001, "geometry_support": ["point", "interface"],
        "learning_manifest_sha256": "synthetic", "unvalidated_geometries": ["region", "line"],
    })
    return path


def result(image, task="atomic_point_defect"):
    raw = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(raw.dtype).encode())
    digest.update(np.asarray(raw.shape, dtype=np.int64).tobytes())
    digest.update(raw.view(np.uint8).tobytes())
    empty = np.zeros_like(image, dtype=bool)
    empty.setflags(write=False)
    return DetectionResult(task, image.shape, image, image.copy(), image.copy(), empty,
                           [], {}, {}, {"input_provenance": {"pixel_sha256": digest.hexdigest()}})


def test_experimental_opt_in_required_before_reading_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="allow_experimental"):
        ExpertStudentRuntime(tmp_path / "absent", project_root=tmp_path)


def test_runtime_refuses_untrained_task_and_outputs_source_pixel_candidates(tmp_path):
    path = checkpoint(tmp_path)
    runtime = ExpertStudentRuntime(path, project_root=tmp_path, allow_experimental=True)
    image = np.random.default_rng(1).random((24, 40)).astype(np.float32)
    prediction = runtime.predict(image, "point")
    assert prediction.score.shape == image.shape
    assert prediction.metadata["coordinate_space"] == "source_pixels_xy"
    assert prediction.metadata["teacher_mask_overwritten"] is False
    assert prediction.metadata["mode"] == "experimental_shadow_not_confirmed_defect"
    with pytest.raises(ValueError, match="No expert training"):
        runtime.predict(image, "line")
    output = tmp_path / "prediction"
    save_shadow_prediction(prediction, output, project_root=tmp_path)
    with pytest.raises(FileExistsError):
        save_shadow_prediction(prediction, output, project_root=tmp_path)


def test_physics_output_and_empty_mask_remain_untouched(tmp_path):
    path = checkpoint(tmp_path)
    image = np.random.default_rng(1).random((24, 40)).astype(np.float32)
    teacher = result(image)

    class Framework:
        def detect(self, source, **kwargs):
            return teacher

    combined = detect_with_expert_student(Framework(), image, checkpoint=path, project_root=tmp_path,
                                         allow_experimental=True)
    assert combined.teacher is teacher
    assert not teacher.mask.any()
    assert teacher.mask.flags.writeable is False
    assert combined.student is not None
    assert combined.student_status == "experimental_shadow"
    teacher.task = "stacking_fault"
    fallback = detect_with_expert_student(Framework(), image, checkpoint=path, project_root=tmp_path,
                                        allow_experimental=True)
    assert fallback.student is None
    assert fallback.student_status == "unsupported_task_physics_retained"


def test_changed_input_between_branches_is_rejected(tmp_path):
    path = checkpoint(tmp_path)
    image = np.zeros((16, 16), np.float32)
    teacher = result(image)

    class Framework:
        def detect(self, source, **kwargs):
            source[0, 0] = 1
            return teacher

    with pytest.raises(ValueError, match="Input changed"):
        detect_with_expert_student(Framework(), image, checkpoint=path, project_root=tmp_path,
                                  allow_experimental=True)


def test_interface_uses_exact_area_budget_not_uncalibrated_half_threshold(tmp_path):
    path = checkpoint(tmp_path)
    runtime = ExpertStudentRuntime(path, project_root=tmp_path, allow_experimental=True)
    image = np.zeros((24, 40), np.float32)
    prediction = runtime.predict(image, "interface")
    assert prediction.candidate_mask.sum() == 96
    assert prediction.metadata["candidate_selection"] == "fixed_area_budget_ranking"
    assert prediction.metadata["area_budget_fraction"] == 0.1
    assert prediction.metadata["interface_semantics"] == "affected_band_not_exact_centerline"
    # A constant input still has a budgeted ranking; it is NOT a defect decision.
    assert "normal image" in prediction.metadata["warnings"][-1]
    for fraction in (0, -0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="Area budget"):
            runtime.predict(image, "interface", area_budget_fraction=fraction)


@pytest.mark.parametrize("supported", [True, False])
def test_cli_persists_combined_branch_status_without_relabeling_teacher(tmp_path, monkeypatch, capsys, supported):
    from tfree_detection import cli, expert_runtime
    from tfree_detection.expert_supervision import write_new_json

    teacher = result(np.zeros((16, 16), np.float32))
    status = "experimental_shadow" if supported else "unsupported_task_physics_retained"
    prediction = (expert_runtime.ExpertShadowPrediction(
        np.zeros((16, 16), np.float32), np.zeros((16, 16), bool), np.empty((0, 2)),
        {"mode": "experimental_shadow_not_confirmed_defect", "teacher_mask_overwritten": False},
    ) if supported else None)
    calls = []

    def detect(framework, source, **options):
        calls.append(options)
        assert options["allow_experimental"] is True
        return expert_runtime.ExpertAssistedDetection(teacher, prediction, status)

    def save_original(original, config, output):
        assert original is teacher
        directory = tmp_path / "result"
        directory.mkdir()
        write_new_json(directory / "summary.json", original.summary())
        return {"summary": str(directory / "summary.json")}

    monkeypatch.setattr(expert_runtime, "detect_with_expert_student", detect)
    monkeypatch.setattr(cli, "save_result", save_original)
    argv = ["detect", "--input", str(tmp_path / "synthetic.npy"), "--output", str(tmp_path / "result"),
            "--student-checkpoint", str(tmp_path / "checkpoint"), "--project-root", str(tmp_path),
            "--allow-experimental-student"]
    cli.main(argv)
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((tmp_path / "result/expert_assisted_summary.json").read_text())
    assert saved == printed
    assert saved["expert_student_status"] == status
    assert saved["neural_training_used_in_student_branch"] is supported
    assert (tmp_path / "result/expert_student_shadow/student_shadow.json").is_file() is supported
    assert "expert_student_status" not in json.loads((tmp_path / "result/summary.json").read_text())
    assert not teacher.mask.any()
    with pytest.raises(FileExistsError, match="new output"):
        cli.main(argv)
    assert len(calls) == 1
