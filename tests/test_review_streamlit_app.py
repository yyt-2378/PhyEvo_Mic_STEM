from pathlib import Path

import pytest

from tfree_detection.review_store import CaseReviewAction, ReviewStore


streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "apps/review_dataset_viewer.py"


def _widget(elements, label):
    return next(item for item in elements if item.label == label)


def test_real_app_loads_154_cases_and_persists_acceptance(
    tmp_path, monkeypatch
) -> None:
    store_path = tmp_path / "review-store.sqlite3"
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
    store_path = tmp_path / "canvas-store.sqlite3"
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
