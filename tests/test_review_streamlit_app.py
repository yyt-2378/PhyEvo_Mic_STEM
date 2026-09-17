from pathlib import Path

import numpy as np
import pytest

from tfree_detection import review_cases, review_dataset, review_prediction
from tfree_detection.review_store import CaseReviewAction, ReviewStore


streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "apps/review_dataset_viewer.py"
SAFE_TEST_ROOT = PROJECT_ROOT / ".codex-tmp/pytest"


@pytest.fixture(autouse=True)
def synthetic_review_assets(monkeypatch):
    """Exercise the real app/ledger without distributing microscopy assets.

    Keep the checked-in case inventory and all real platform gates. Only the
    image/prediction I/O is synthetic, so a clean checkout runs these tests.
    """

    streamlit.cache_data.clear()
    monkeypatch.delenv("TFREE_REVIEW_ACCESS_CODE", raising=False)
    calls = []
    manifest_loader = review_dataset.load_review_manifest
    cases_loader = review_cases.load_review_cases
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    mask = np.zeros(image.shape, dtype=bool)
    mask[24:40, 24:40] = True

    def load_manifest(path):
        calls.append("manifest")
        return manifest_loader(path)

    def load_cases(path):
        calls.append("cases")
        return cases_loader(path)

    def load_plane(sample, root):
        calls.append("image")
        return image.copy()

    def load_mask(prediction, root):
        calls.append("mask")
        return mask.copy()

    def predictions(path, *, project_root):
        calls.append("predictions")
        samples = manifest_loader(
            PROJECT_ROOT / "datasets/standardized/expert_review_queue_v1.jsonl"
        )
        return {
            (sample.sample_id, sample.primary_task): review_prediction.ReviewPrediction(
                prediction_format=review_prediction.PREDICTION_FORMAT,
                sample_id=sample.sample_id,
                task=sample.primary_task,
                sample_sha256="a" * 64,
                config_path=f"configs/{sample.primary_task}.yaml",
                config_sha256="b" * 64,
                pixel_sha256="c" * 64,
                mask_path="synthetic-test-mask.png",
                mask_sha256="d" * 64,
                image_shape=image.shape,
                detected_area_pixels=int(mask.sum()),
                detected_area_fraction=float(mask.mean()),
                candidates=(),
            )
            for sample in samples
        }

    monkeypatch.setattr(review_dataset, "load_review_manifest", load_manifest)
    monkeypatch.setattr(review_cases, "load_review_cases", load_cases)
    monkeypatch.setattr(review_dataset, "load_review_plane", load_plane)
    monkeypatch.setattr(review_prediction, "load_prediction_mask", load_mask)
    monkeypatch.setattr(review_prediction, "discover_review_predictions", predictions)
    yield calls
    streamlit.cache_data.clear()


def _project_tmp_path(tmp_path: Path) -> Path:
    """Mirror pytest's unique name without writing outside the project."""

    destination = SAFE_TEST_ROOT / tmp_path.parent.name / tmp_path.name
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _widget(elements, label):
    return next(item for item in elements if item.label == label)


def test_access_code_gate_blocks_assets_before_authentication(
    tmp_path, monkeypatch, synthetic_review_assets
) -> None:
    store_path = _project_tmp_path(tmp_path) / "gated-review-store.sqlite3"
    monkeypatch.setenv("TFREE_REVIEWER_ID", "gated-app-test-expert")
    monkeypatch.setenv("TFREE_REVIEW_STORE_PATH", str(store_path))
    monkeypatch.setenv("TFREE_REVIEW_ACCESS_CODE", "correct-test-code")
    app = AppTest.from_file(str(APP), default_timeout=60)

    app.run()
    assert not app.exception
    assert any("专家审核访问验证" in item.value for item in app.title)
    assert not app.metric
    assert not store_path.exists()
    assert synthetic_review_assets == []

    _widget(app.text_input, "审核访问码").input("wrong-test-code")
    _widget(app.button, "进入审核").click()
    app.run()
    assert not app.exception
    assert any("访问码错误" in item.value for item in app.error)
    assert not app.metric
    assert not store_path.exists()
    assert synthetic_review_assets == []

    _widget(app.text_input, "审核访问码").input("correct-test-code")
    _widget(app.button, "进入审核").click()
    app.run()
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["总病例"] == "154"
    assert store_path.exists()
    assert {"manifest", "cases", "predictions", "image", "mask"} <= set(
        synthetic_review_assets
    )


def test_app_loads_154_cases_with_synthetic_images_and_persists_acceptance(
    tmp_path, monkeypatch
) -> None:
    store_path = _project_tmp_path(tmp_path) / "review-store.sqlite3"
    monkeypatch.setenv("TFREE_REVIEWER_ID", "app-test-expert")
    monkeypatch.setenv("TFREE_REVIEW_STORE_PATH", str(store_path))
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["总病例"] == "154"
    assert metrics["已完成"] == "0"
    assert any("source:" in item.value for item in app.caption)
    assert any("prediction status:" in item.value for item in app.caption)
    action = _widget(app.selectbox, "专家结论")
    assert action.value is None
    action.select(CaseReviewAction.ACCEPT_PREDICTION)
    app.run()
    assert not app.exception
    labels = _widget(app.multiselect, "缺陷类型")
    labels.select(labels.options[0])
    _widget(app.button, "保存当前").click()
    app.run()
    assert not app.exception
    assert any("显示预测轮廓" in item.value for item in app.error)
    assert ReviewStore(store_path).list_current(reviewer_id="app-test-expert") == []
    # AppTest has no JS runtime: emulate the component's actual layer-display event.
    canvas_key = next(k for k in app.session_state.filtered_state if k.startswith("geometry-editor-"))
    app.session_state[canvas_key] = {"geometry": None, "prediction_shown": True}
    _widget(app.button, "保存当前").click()
    app.run()
    assert not app.exception
    assert any("已持久化第1版" in item.value for item in app.success)

    store = ReviewStore(store_path)
    current = store.list_current(reviewer_id="app-test-expert")
    assert len(current) == 1
    assert current[0].action == "accept_prediction"
    assert len(store.list_history(reviewer_id="app-test-expert")) == 1

    reloaded = AppTest.from_file(str(APP), default_timeout=60)
    reloaded.run()
    assert not reloaded.exception
    metrics = {item.label: item.value for item in reloaded.metric}
    assert metrics["已完成"] == "1"
    assert any("已保存：第1版" in item.value for item in reloaded.success)


def test_canvas_and_submit_buttons_share_form_transaction_boundary(
    tmp_path, monkeypatch
) -> None:
    store_path = _project_tmp_path(tmp_path) / "canvas-store.sqlite3"
    monkeypatch.setenv("TFREE_REVIEWER_ID", "canvas-test-expert")
    monkeypatch.setenv("TFREE_REVIEW_STORE_PATH", str(store_path))
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    assert not app.exception
    action = _widget(app.selectbox, "专家结论")
    action.select(CaseReviewAction.CORRECT_GEOMETRY)
    app.run()
    assert not app.exception
    assert any(item.value == "专家几何编辑" for item in app.subheader)
    assert len(app.get("form")) == 1
    component = app.get("bidi_component")[0]
    form_id = component.proto.form_id
    assert form_id.startswith("review-form-canvas-test-expert-")
    submit_buttons = [
        item
        for item in app.button
        if item.label in {"保存当前", "保存并进入下一例"}
    ]
    assert len(submit_buttons) == 2
    assert all(item.proto.is_form_submitter for item in submit_buttons)
    assert all(item.proto.form_id == form_id for item in submit_buttons)
    assert ReviewStore(store_path).list_history(
        reviewer_id="canvas-test-expert"
    ) == []


def test_save_keeps_selected_case_and_save_next_keeps_confirmation(tmp_path, monkeypatch) -> None:
    store_path = _project_tmp_path(tmp_path) / "navigation.sqlite3"
    monkeypatch.setenv("TFREE_REVIEWER_ID", "navigation-expert")
    monkeypatch.setenv("TFREE_REVIEW_STORE_PATH", str(store_path))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    selector = _widget(app.selectbox, "审核病例")
    selector.select("case:guo-mowte2-large-fov")
    app.run()
    selected_id = _widget(app.selectbox, "审核病例").value
    _widget(app.selectbox, "专家结论").select(CaseReviewAction.NO_DEFECT)
    app.run()
    _widget(app.button, "保存当前").click()
    app.run()
    app.run()  # completed checkmark changes the formatted option label
    assert _widget(app.selectbox, "审核病例").value == selected_id
    _widget(app.button, "保存并进入下一例").click()
    app.run()
    assert _widget(app.selectbox, "审核病例").value != selected_id
    assert any("上一例已持久化" in item.value for item in app.success)
    app.run()
    assert any("上一例已持久化" in item.value for item in app.success)
