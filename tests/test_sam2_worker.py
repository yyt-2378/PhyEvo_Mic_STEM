import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tfree_detection.config import (
    DetectorConfig,
    SAM3RefinementConfig,
    load_config,
)
from tfree_detection.stage2.sam3_refine import (
    build_physics_prompt_plan,
    load_sam3_response,
    prepare_sam3_exchange,
)


PROJECT_ROOT = Path(__file__).parents[1]
WORKER_PATH = PROJECT_ROOT / "scripts" / "run_sam2_boundary_worker.py"


def _load_worker_module():
    spec = importlib.util.spec_from_file_location(
        "test_run_sam2_boundary_worker", WORKER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_sam2_runtime(monkeypatch, shape):
    calls = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def reset_peak_memory_stats():
            calls["reset_peak_memory_stats"] = True

        @staticmethod
        def max_memory_allocated():
            return 4096

    torch_module = ModuleType("torch")
    torch_module.__version__ = "test-torch"
    torch_module.version = SimpleNamespace(cuda="test-cuda")
    torch_module.cuda = FakeCuda()
    torch_module.bfloat16 = object()
    torch_module.autocast = lambda *args, **kwargs: nullcontext()
    torch_module.inference_mode = lambda: nullcontext()

    def build_sam2(
        config_file,
        checkpoint,
        *,
        device,
        mode,
        apply_postprocessing,
    ):
        calls["build"] = {
            "config_file": config_file,
            "checkpoint": checkpoint,
            "device": device,
            "mode": mode,
            "apply_postprocessing": apply_postprocessing,
        }
        return object()

    class FakePredictor:
        def __init__(self, model):
            calls["model"] = model

        def set_image(self, image):
            calls["image_shape"] = image.shape

        def predict(self, **kwargs):
            calls["predict"] = kwargs
            mask = np.zeros(shape, dtype=bool)
            mask[2:8, 3:9] = True
            smaller = np.zeros(shape, dtype=bool)
            smaller[3:7, 4:8] = True
            return (
                np.stack([mask, smaller]),
                np.asarray([0.93, 0.80]),
                np.zeros((2, 256, 256), dtype=np.float32),
            )

    sam2_module = ModuleType("sam2")
    sam2_module.__version__ = "test-sam2"
    build_module = ModuleType("sam2.build_sam")
    build_module.build_sam2 = build_sam2
    predictor_module = ModuleType("sam2.sam2_image_predictor")
    predictor_module.SAM2ImagePredictor = FakePredictor
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "sam2", sam2_module)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_module)
    monkeypatch.setitem(
        sys.modules, "sam2.sam2_image_predictor", predictor_module
    )
    return calls


def _write_protocol_array(worker, directory, filename, values):
    path = directory / filename
    np.save(path, values, allow_pickle=False)
    return {
        "file": filename,
        "sha256": worker._array_sha256(values),
        "dtype": str(values.dtype),
        "shape": list(values.shape),
    }


def test_sam2_profiles_load_and_keep_candidate_local_fallback():
    amorphous = load_config(PROJECT_ROOT / "configs" / "amorphous_sam2.yaml")
    interface = load_config(
        PROJECT_ROOT / "configs" / "phase_boundary_sam2.yaml"
    )

    for config in (amorphous, interface):
        refinement = config.stage2.sam3
        assert refinement.enabled
        assert refinement.backend == "sam2"
        assert refinement.device == "cuda"
        assert refinement.fallback_to_coarse
        assert (
            Path(refinement.worker_script).name
            == "run_sam2_boundary_worker.py"
        )
    assert interface.stage2.sam3.interface_geometry_mode == "paired_domains"


def test_configuration_rejects_unknown_sam_backend():
    config = DetectorConfig()
    config.stage2.sam3.enabled = True
    config.stage2.sam3.backend = "unknown"

    with pytest.raises(ValueError, match="sam2.*sam3"):
        config.validate()


def test_exchange_request_preserves_sam2_worker_identity(tmp_path: Path):
    shape = (24, 24)
    image = np.zeros(shape, dtype=np.float32)
    coarse = np.zeros(shape, dtype=bool)
    coarse[6:18, 7:17] = True
    score = coarse.astype(np.float32)
    config = SAM3RefinementConfig(
        backend="sam2",
        model_variant="sam2.1_hiera_large",
        checkpoint_path="/models/sam2.1_hiera_large.pt",
        device="cuda",
        ensemble_size=2,
    )
    plan = build_physics_prompt_plan(coarse, score, {}, config)

    exchange_path = (
        PROJECT_ROOT
        / ".codex-tmp/pytest"
        / tmp_path.parent.name
        / tmp_path.name
    )
    exchange_path.mkdir(parents=True, exist_ok=False)
    _, request = prepare_sam3_exchange(
        exchange_path,
        image,
        coarse,
        score,
        plan,
        config,
    )

    assert request["protocol_version"] == "tfree-sam3-boundary-v2"
    assert request["worker"] == {
        "backend": "sam2",
        "checkpoint_path": "/models/sam2.1_hiera_large.pt",
        "model_variant": "sam2.1_hiera_large",
        "device": "cuda",
    }
    assert request["geometry_kind"] == "region"
    assert len(request["prompts"]) == 2


def test_exchange_request_rejects_path_outside_project():
    shape = (8, 8)
    image = np.zeros(shape, dtype=np.float32)
    coarse = np.zeros(shape, dtype=bool)
    coarse[2:6, 2:6] = True
    score = coarse.astype(np.float32)
    config = SAM3RefinementConfig(ensemble_size=1)
    plan = build_physics_prompt_plan(coarse, score, {}, config)

    with pytest.raises(ValueError, match="inside the project root"):
        prepare_sam3_exchange(
            PROJECT_ROOT.parent / "sam-exchange-outside-project-must-not-be-created",
            image,
            coarse,
            score,
            plan,
            config,
        )


def test_sam2_worker_emits_v2_protocol_consumed_by_existing_gate(
    tmp_path: Path, monkeypatch
):
    worker = _load_worker_module()
    shape = (12, 14)
    calls = _install_fake_sam2_runtime(monkeypatch, shape)
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    coarse = np.zeros(shape, dtype=bool)
    coarse[2:8, 3:9] = True
    core = np.zeros(shape, dtype=bool)
    core[3:7, 4:8] = True
    negative = ~coarse
    labels = coarse.astype(np.int32)
    score = coarse.astype(np.float32)
    checkpoint = tmp_path / "sam2.1_hiera_large.pt"
    checkpoint.write_bytes(b"test checkpoint placeholder")

    arrays = {
        name: _write_protocol_array(worker, tmp_path, filename, values)
        for name, filename, values in (
            ("image", "image.npy", image),
            ("coarse_mask", "coarse_mask.npy", coarse),
            ("score_map", "score_map.npy", score),
            ("physics_core", "physics_core.npy", core),
            ("negative_reference", "negative_reference.npy", negative),
            ("component_labels", "component_labels.npy", labels),
        )
    }
    prompt = {
        "candidate_id": "candidate-0001",
        "component_label": 1,
        "ensemble_index": 0,
        "prompt_role": "candidate",
        "box_xyxy": [1, 1, 11, 10],
        "positive_points_xy": [[5, 5]],
        "negative_points_xy": [[1, 1]],
        "use_coarse_mask_prompt": True,
    }
    request = {
        "protocol_version": "tfree-sam3-boundary-v2",
        "request_id": "sam2-protocol-test",
        "response_file": "sam_response.json",
        "arrays": arrays,
        "prompts": [prompt],
        "ensemble_size": 1,
        "use_coarse_mask_prompt": True,
        "model_variant": "sam2.1_hiera_large",
        "geometry_kind": "region",
        "worker": {
            "backend": "sam2",
            "checkpoint_path": str(checkpoint),
            "model_variant": "sam2.1_hiera_large",
            "device": "cuda",
        },
    }
    request_path = tmp_path / "sam_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    result = worker.run(request_path)

    assert result["request_id"] == request["request_id"]
    assert result["proposal_count"] == 1
    response = json.loads(
        (tmp_path / "sam_response.json").read_text(encoding="utf-8")
    )
    assert response["protocol_version"] == "tfree-sam3-boundary-v2"
    assert response["worker"]["backend"] == "sam2"
    assert response["worker"]["device"] == "cuda"
    assert response["worker"]["model_variant"] == "sam2.1_hiera_large"
    assert (
        response["worker"]["interactive_api"]
        == "SAM2ImagePredictor.predict"
    )
    assert response["worker"]["training_or_fitting_used"] is False
    assert response["worker"]["torch_cuda_version"] == "test-cuda"
    assert response["proposals"][0]["candidate_id"] == "candidate-0001"
    assert response["proposals"][0]["prompt_role"] == "candidate"
    assert calls["build"]["device"] == "cuda"
    assert calls["image_shape"] == (*shape, 3)
    assert calls["predict"]["mask_input"].shape == (1, 256, 256)

    proposals, loaded_response = load_sam3_response(request_path, request)
    assert len(proposals) == 1
    assert proposals[0].candidate_id == "candidate-0001"
    assert proposals[0].mask.shape == shape
    assert loaded_response["worker"]["backend"] == "sam2"
