"""Durable Streamlit application for the 154 unlabeled review cases."""

from __future__ import annotations

import hmac
import os
import re
import time
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np
import streamlit as st

from tfree_detection.review_canvas import review_image_canvas
from tfree_detection.review_cases import ExpertReviewCase, load_review_cases
from tfree_detection.review_dataset import (
    ReviewSample,
    load_review_manifest,
    load_review_plane,
)
from tfree_detection.review_platform import (
    audit_review_platform,
    file_sha256,
    require_review_platform,
)
from tfree_detection.review_prediction import (
    ReviewPrediction,
    discover_review_predictions,
    load_prediction_mask,
)
from tfree_detection.review_store import (
    REVIEW_PROJECT_ID,
    CaseReviewAction,
    ReviewStore,
    ReviewSubmission,
    revisions_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "datasets/standardized/expert_review_queue_v1.jsonl"
)
DEFAULT_CASES = (
    PROJECT_ROOT / "datasets/standardized/expert_review_cases_v1.jsonl"
)
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs/expert_review_2d_predictions_v1"
DEFAULT_STORE = (
    PROJECT_ROOT / "datasets/expert_annotations/review_store_v1.sqlite3"
)
_REVIEWER_ID = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")
_ACCESS_GRANTED_KEY = "expert-review-access-granted"

ACTION_LABELS = {
    CaseReviewAction.ACCEPT_PREDICTION: "接受预测",
    CaseReviewAction.CORRECT_GEOMETRY: "修正几何",
    CaseReviewAction.REJECT_PREDICTION: "拒绝预测",
    CaseReviewAction.NO_DEFECT: "无缺陷",
    CaseReviewAction.ARTIFACT: "Artifact / 成像伪影",
    CaseReviewAction.UNKNOWN: "Unknown / 暂不能判断",
}
ACTION_GUIDANCE = {
    CaseReviewAction.ACCEPT_PREDICTION: (
        "请打开“显示预测轮廓”，确认琥珀色轮廓的位置、范围和数量基本正确。"
        "允许极少量不影响结论的像素误差。"
    ),
    CaseReviewAction.CORRECT_GEOMETRY: (
        "目标缺陷存在，但模型多圈、漏圈、位置或范围明显不对。选择后在下方重新给出完整"
        "专家几何；不是在模型轮廓上擦除。"
    ),
    CaseReviewAction.REJECT_PREDICTION: (
        "模型结果整体无物理意义，而且本图暂时不能形成可靠的替代标注。若能够指出正确"
        "位置，请优先选择“修正几何”。"
    ),
    CaseReviewAction.NO_DEFECT: (
        "专家判断该图不存在当前任务所问的目标缺陷。模型圈出的内容全部属于正常结构。"
    ),
    CaseReviewAction.ARTIFACT: (
        "主要异常来自漂移、失焦、扫描跳变、饱和、污染、边缘或其他成像伪影，不应作为"
        "真实材料缺陷训练。"
    ),
    CaseReviewAction.UNKNOWN: (
        "图像证据不足或物理类型无法可靠判断。宁可标unknown，也不要猜测。"
    ),
}
GEOMETRY_HELP = {
    "point": (
        "逐个轻点异常原子柱中心，至少1个点。青绿空心圈记录中心，不遮盖原子细节。"
    ),
    "line": (
        "从缺陷线一端开始，沿中心线依次点击控制点，到另一端结束；至少2个点。"
    ),
    "interface": (
        "从界面一端开始，沿两侧晶格真正分界依次点击控制点；至少2个点。"
    ),
    "region": (
        "按住鼠标用套索沿非晶区域外缘画一个闭合轮廓；至少3个点。"
    ),
}
GEOMETRY_LABELS = {
    "point": "点：异常原子/原子柱",
    "line": "线：错层/位错",
    "interface": "界面轨迹：相界面/晶界",
    "region": "区域轮廓：非晶/损伤区",
}
TASK_LABELS = {
    "atomic_point_defect": (
        "atomic_vacancy",
        "substitution_or_intensity_anomaly",
        "atomic_column_distortion",
        "dislocation_core",
    ),
    "stacking_fault": (
        "stacking_fault",
        "partial_dislocation",
        "line_defect",
    ),
    "phase_boundary": (
        "twin_boundary",
        "grain_boundary",
        "phase_interface",
    ),
    "amorphous_region": (
        "amorphous_region",
        "coherence_loss",
        "beam_damage_region",
    ),
}
LABEL_DISPLAY = {
    "atomic_vacancy": "原子空位 / atomic vacancy",
    "substitution_or_intensity_anomaly": "替位或强度异常",
    "atomic_column_distortion": "原子柱模糊、拉伸或扭曲",
    "dislocation_core": "位错核心 / dislocation core",
    "stacking_fault": "错层 / stacking fault",
    "partial_dislocation": "不全位错 / partial dislocation",
    "line_defect": "其他线缺陷",
    "twin_boundary": "孪晶界 / twin boundary",
    "grain_boundary": "晶界 / grain boundary",
    "phase_interface": "两相界面 / phase interface",
    "amorphous_region": "非晶区域 / amorphous region",
    "coherence_loss": "晶格相干性丧失",
    "beam_damage_region": "束流损伤区域",
}

TASK_DISPLAY = {
    "atomic_point_defect": "原子点缺陷",
    "stacking_fault": "错层与线缺陷",
    "phase_boundary": "相界面与晶界",
    "amorphous_region": "非晶与损伤区域",
}


@st.cache_data(show_spinner=False)
def _samples(path: str) -> list[ReviewSample]:
    return load_review_manifest(path)


@st.cache_data(show_spinner=False)
def _cases(path: str) -> list[ExpertReviewCase]:
    return load_review_cases(path)


@st.cache_data(show_spinner=False)
def _plane(sample: ReviewSample, root: str) -> np.ndarray:
    return load_review_plane(sample, root)


@st.cache_data(show_spinner=False)
def _prediction_index(path: str):
    return discover_review_predictions(path, project_root=PROJECT_ROOT)


def _configured_store_path() -> Path:
    configured = os.environ.get("TFREE_REVIEW_STORE_PATH")
    path = DEFAULT_STORE if not configured else Path(configured)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    path.relative_to(PROJECT_ROOT)
    return path


def _require_access_code() -> None:
    """Stop before loading review assets unless the configured code matches."""

    expected = os.environ.get("TFREE_REVIEW_ACCESS_CODE")
    if expected is None:
        return
    if not expected:
        st.error("服务器审核访问码配置为空，已拒绝加载审核数据。")
        st.stop()
    if st.session_state.get(_ACCESS_GRANTED_KEY) is True:
        return

    st.title("PhyEvo-Mic · 专家审核访问验证")
    st.caption("请输入项目负责人提供的共享访问码，再进入正式审核队列。")
    with st.form("expert-review-access-form", enter_to_submit=True):
        supplied = st.text_input("审核访问码", type="password")
        submitted = st.form_submit_button("进入审核", type="primary")
    if submitted:
        if hmac.compare_digest(
            supplied.encode("utf-8"), expected.encode("utf-8")
        ):
            st.session_state[_ACCESS_GRANTED_KEY] = True
            st.rerun()
        st.error("访问码错误；病例、预测和审核数据库均未加载。")
    st.stop()


def _reviewer_id() -> str:
    default = os.environ.get("TFREE_REVIEWER_ID", "")
    reviewer = st.sidebar.text_input(
        "Reviewer ID",
        value=default,
        placeholder="例如 expert-01",
        help="只允许英文字母、数字、点、下划线、@和短横线。",
    ).strip()
    if not reviewer:
        st.info("请先在左侧输入唯一 Reviewer ID，再开始正式审核。")
        st.stop()
    if not _REVIEWER_ID.fullmatch(reviewer):
        st.error("Reviewer ID 格式不合法。")
        st.stop()
    return reviewer


def _prediction_for(
    sample: ReviewSample,
    predictions: Mapping[tuple[str, str], ReviewPrediction],
) -> ReviewPrediction | None:
    return predictions.get((sample.sample_id, sample.primary_task))


def _filtered_cases(
    cases: list[ExpertReviewCase],
    *,
    reviewed_case_ids: set[str],
) -> list[ExpertReviewCase]:
    st.sidebar.header("审核队列")
    datasets = sorted({item.dataset_id for item in cases})
    splits = sorted({item.review_split for item in cases})
    geometries = sorted({item.geometry_hint for item in cases})
    only_remaining = st.sidebar.toggle("只看未完成", value=False)
    with st.sidebar.expander("筛选病例", expanded=False):
        selected_datasets = st.multiselect("数据集", datasets, default=datasets)
        selected_splits = st.multiselect("划分", splits, default=splits)
        selected_geometries = st.multiselect(
            "标注几何", geometries, default=geometries,
            format_func=lambda value: GEOMETRY_LABELS[value],
        )
    return [
        item
        for item in cases
        if item.dataset_id in selected_datasets
        and item.review_split in selected_splits
        and item.geometry_hint in selected_geometries
        and (not only_remaining or item.case_id not in reviewed_case_ids)
    ]


def _next_case_id(
    cases: list[ExpertReviewCase],
    current_case_id: str,
    completed: set[str],
) -> str:
    current_index = next(
        index for index, item in enumerate(cases) if item.case_id == current_case_id
    )
    ordered = cases[current_index + 1 :] + cases[: current_index + 1]
    return next(
        (item.case_id for item in ordered if item.case_id not in completed),
        current_case_id,
    )


def _current_defaults(
    current,
) -> tuple[CaseReviewAction | None, tuple[str, ...], str]:
    if current is None:
        return None, (), ""
    return (
        CaseReviewAction(current.action),
        tuple(str(item) for item in current.submission.get("defect_labels", [])),
        str(current.submission.get("notes", "")),
    )


def main() -> None:
    st.set_page_config(
        page_title="PhyEvo-Mic 专家审核",
        page_icon="🔬",
        layout="wide",
    )
    _require_access_code()
    st.html("""<style>
      .stMainBlockContainer {max-width:1250px; padding-top:4rem;}
      h1 {font-size:1.8rem !important;}
      h2 {font-size:1.2rem !important;}
      [data-testid="stSidebar"] h2 {font-size:1.15rem;}
      [data-testid="stForm"] {border-radius:12px;}
      [data-testid="stCaptionContainer"] {line-height:1.65;}
      button, input, textarea {font-size:16px;}
    </style>""")
    st.title("PhyEvo-Mic · 专家审核")
    st.caption("看原图 → 按需对照预测或修正 → 保存。可随时放大和全屏查看。")
    with st.expander("操作说明与结论区别", expanded=False):
        st.markdown(
            """
1. 默认显示原图。点击“显示预测轮廓”后，琥珀细线表示预测，可调淡。
2. 用“放大”“原始像素 1:1”“全屏看图”观察细节；拖动模式不增加标注。
3. 选择“修正几何”后，在同一张图上点击或画轮廓，青绿空心标记是您的答案。
4. 错点可单独删除，误操作可撤销、恢复。选好结论再绘制，保存后再换病例。
5. 补充缺陷类型和备注，点击保存；绿色“已持久化第 X 版”表示成功。

**关键区别：**“无缺陷”表示图里没有目标缺陷；“拒绝预测”表示模型整体错误且当前无法给出
可靠替代标注；如果缺陷存在并且你能画出正确位置，应选择“修正几何”。
"""
        )

    samples = _samples(str(DEFAULT_MANIFEST))
    cases = _cases(str(DEFAULT_CASES))
    predictions = _prediction_index(str(DEFAULT_PREDICTIONS))
    try:
        audit = audit_review_platform(samples, cases, predictions)
        require_review_platform(audit)
    except Exception as error:
        st.error(f"审核资产硬门禁未通过：{error}")
        st.stop()
    sample_by_id = {item.sample_id: item for item in samples}
    reviewer = _reviewer_id()
    store = ReviewStore(_configured_store_path())
    integrity = store.integrity_report()
    if not integrity["valid"]:
        st.error("审核数据库完整性检查失败，已停止写入。")
        st.stop()
    current_revisions = store.list_current(reviewer_id=reviewer)
    current_by_case = {item.case_id: item for item in current_revisions}
    reviewed_case_ids = set(current_by_case)
    queue = _filtered_cases(cases, reviewed_case_ids=reviewed_case_ids)
    if not queue:
        st.success("当前筛选范围已经全部完成。")
        st.stop()

    progress = store.progress(
        [item.case_id for item in cases], reviewer_id=reviewer
    )
    top = st.sidebar.columns(2)
    top[0].metric("总病例", len(cases))
    top[1].metric("已完成", progress["completed_count"])
    st.sidebar.progress(progress["completion_fraction"])
    st.sidebar.caption(f"还剩 {progress['remaining_count']} 例 · 疲劳时请暂停，已保存结果会保留。")

    exports = st.sidebar.expander("导出我的审核记录")
    exports.download_button(
        "下载当前有效审核 JSONL",
        data=revisions_jsonl(current_revisions),
        file_name=f"{reviewer}-current-reviews.jsonl",
        mime="application/x-ndjson",
        disabled=not current_revisions,
    )
    exports.download_button(
        "下载完整修订历史 JSONL",
        data=revisions_jsonl(store.list_history(reviewer_id=reviewer)),
        file_name=f"{reviewer}-review-history.jsonl",
        mime="application/x-ndjson",
        disabled=not current_revisions,
    )

    if "pending_case_id" in st.session_state:
        pending = st.session_state.pop("pending_case_id")
        if pending in {item.case_id for item in queue}:
            st.session_state["case_selector"] = pending
    case_ids = [item.case_id for item in queue]
    case_names = {
        item.case_id: (
            f"{index + 1:03d}/{len(queue):03d} · "
            f"{'✓ ' if item.case_id in reviewed_case_ids else ''}"
            f"{TASK_DISPLAY.get(sample_by_id[item.primary_sample_id].primary_task, item.annotation_target)}"
            f" · {sample_by_id[item.primary_sample_id].material}"
        )
        for index, item in enumerate(queue)
    }
    if st.session_state.get("case_selector") not in case_ids:
        st.session_state["case_selector"] = case_ids[0]
    selected_id = st.selectbox(
        "审核病例",
        case_ids,
        index=case_ids.index(st.session_state["case_selector"]),
        key="case_selector",
        format_func=lambda value: case_names[value],
    )
    case = queue[case_ids.index(selected_id)]
    current = current_by_case.get(case.case_id)
    notice = st.session_state.get("review-save-message")
    if notice is not None and notice["target_case_id"] == selected_id:
        st.success(notice["message"])

    preferred_sample = (
        case.primary_sample_id
        if current is None
        else str(current.submission.get("selected_sample_id", case.primary_sample_id))
    )
    if preferred_sample not in case.sample_ids:
        preferred_sample = case.primary_sample_id
    if case.case_type == "multichannel_frame":
        selected_sample_id = st.radio(
            "同一物理帧的成像通道（一次审核对5个通道共同生效）",
            case.sample_ids,
            index=case.sample_ids.index(preferred_sample),
            horizontal=True,
            key=f"channel-{reviewer}-{case.case_id}",
            format_func=lambda value: (
                f"{sample_by_id[value].channel_name or 'unknown'} · "
                f"{value.split('-channel-')[-1].split('-frame-')[0]}"
            ),
        )
        st.caption(
            "默认HAADF；BF/DF通道只作为同视野辅助证据。坐标按当前显示通道的"
            "原始像素记录，并在导出中保留5个sample ID。"
        )
    else:
        selected_sample_id = case.primary_sample_id
    sample = sample_by_id[selected_sample_id]
    image = _plane(sample, str(PROJECT_ROOT))
    prediction = _prediction_for(sample, predictions)
    prediction_mask = (
        None
        if prediction is None
        else load_prediction_mask(prediction, PROJECT_ROOT)
    )

    st.markdown(f"**本例：{TASK_DISPLAY.get(sample.primary_task, sample.primary_task)}**"
                f" · {GEOMETRY_LABELS[case.geometry_hint]}")
    if current is not None:
        st.success(
            f"已保存：第{current.revision_number}版 · "
            f"{ACTION_LABELS[CaseReviewAction(current.action)]} · "
            f"{current.submitted_at_utc}"
        )

    default_action, default_labels, default_notes = _current_defaults(current)
    available_actions = list(CaseReviewAction)
    if prediction is None or prediction.status != "completed":
        available_actions = [
            item
            for item in available_actions
            if item
            not in {
                CaseReviewAction.ACCEPT_PREDICTION,
                CaseReviewAction.REJECT_PREDICTION,
            }
        ]
    if default_action is not None and default_action not in available_actions:
        default_action = CaseReviewAction.CORRECT_GEOMETRY
    action = st.selectbox(
        "专家结论",
        available_actions,
        index=(
            None
            if default_action is None
            else available_actions.index(default_action)
        ),
        key=f"action-{reviewer}-{case.case_id}",
        format_func=lambda value: ACTION_LABELS[value],
        placeholder="先查看下方原图，再选择结论",
    )
    if action is None:
        st.caption("可先查看原图；选择结论后再标注和保存。")
    else:
        st.caption(ACTION_GUIDANCE[action])
    started_key = f"review-started:{reviewer}:{case.case_id}"
    st.session_state.setdefault(started_key, time.monotonic())
    with st.form(
        key=f"review-form-{reviewer}-{case.case_id}",
        clear_on_submit=False,
        enter_to_submit=False,
        border=False,
    ):
        draft_key = f"geometry-draft:{reviewer}:{case.case_id}"
        shown_key = f"prediction-shown:{reviewer}:{case.case_id}:{selected_sample_id}"
        if draft_key not in st.session_state and current is not None:
            stored_geometry = current.submission.get("geometry")
            if stored_geometry is not None:
                st.session_state[draft_key] = stored_geometry
        draft_geometry = st.session_state.get(draft_key)
        editing = action == CaseReviewAction.CORRECT_GEOMETRY
        if editing:
            st.subheader("专家几何编辑")
            st.caption(GEOMETRY_HELP[case.geometry_hint])
        if prediction_mask is None:
            st.caption("此通道没有可用预测，可独立查看和标注。")
        draft_geometry, assistance_shown = review_image_canvas(
            image,
            prediction_mask=prediction_mask,
            geometry_hint=case.geometry_hint,
            existing_geometry=draft_geometry,
            key=f"geometry-editor-{reviewer}-{case.case_id}-{selected_sample_id}",
            read_only=not editing,
            prediction_shown=st.session_state.get(shown_key, False),
        )
        st.session_state[draft_key] = draft_geometry
        st.session_state[shown_key] = assistance_shown

        if action in {
            CaseReviewAction.ACCEPT_PREDICTION,
            CaseReviewAction.CORRECT_GEOMETRY,
        }:
            label_options = list(TASK_LABELS.get(sample.primary_task, ()))
            for label in default_labels:
                if label not in label_options:
                    label_options.append(label)
            defect_labels = tuple(
                st.multiselect(
                    "缺陷类型",
                    label_options,
                    default=list(default_labels),
                    key=f"labels-{reviewer}-{case.case_id}",
                    format_func=lambda value: LABEL_DISPLAY.get(value, value),
                    help=(
                        "可以多选；原子相模糊或扭曲可选"
                        "atomic_column_distortion。"
                    ),
                )
            )
        else:
            defect_labels = ()

        notes = st.text_area(
            "备注（拒绝、artifact或unknown时建议说明原因）",
            value=default_notes,
            key=f"notes-{reviewer}-{case.case_id}",
            max_chars=8000,
        )
        save_only, save_next = st.columns(2)
        submit_only = save_only.form_submit_button(
            "保存当前", type="secondary", width="stretch"
        )
        submit_next = save_next.form_submit_button(
            "保存并进入下一例", type="primary", width="stretch"
        )
    if submit_only or submit_next:
        geometry = (
            draft_geometry
            if action == CaseReviewAction.CORRECT_GEOMETRY
            else None
        )
        try:
            if action is None:
                raise ValueError("请先选择专家结论")
            if action in {
                CaseReviewAction.ACCEPT_PREDICTION,
                CaseReviewAction.REJECT_PREDICTION,
            } and not assistance_shown:
                raise ValueError("请先打开图上方的“显示预测轮廓”进行对照，再接受或拒绝预测。")
            submission = ReviewSubmission(
                project_id=REVIEW_PROJECT_ID,
                reviewer_id=reviewer,
                case_id=case.case_id,
                dataset_id=case.dataset_id,
                review_split=case.review_split,
                selected_sample_id=selected_sample_id,
                applies_to_sample_ids=case.sample_ids,
                geometry_hint=case.geometry_hint,
                annotation_scope=case.annotation_scope,
                image_shape=tuple(int(item) for item in image.shape),
                action=action,
                defect_labels=defect_labels,
                geometry=geometry,
                model_assistance_shown=assistance_shown,
                prediction_sha256=(
                    None if prediction is None else prediction.mask_sha256
                ),
                case_manifest_sha256=file_sha256(DEFAULT_CASES),
                annotation_duration_seconds=max(
                    0.01, time.monotonic() - st.session_state[started_key]
                ),
                notes=notes,
                client_submission_id=f"click:{uuid.uuid4()}",
            )
            saved = store.save(submission)
        except (TypeError, ValueError, RuntimeError) as error:
            st.error(f"未保存：{error}")
        else:
            st.session_state[started_key] = time.monotonic()
            st.success(
                f"已持久化第{saved.revision_number}版；刷新页面不会丢失。"
            )
            if submit_next:
                completed = reviewed_case_ids | {case.case_id}
                next_id = _next_case_id(queue, case.case_id, completed)
                st.session_state["review-save-message"] = {
                    "target_case_id": next_id,
                    "message": f"上一例已持久化第{saved.revision_number}版；"
                    + ("当前已进入下一例。" if next_id != case.case_id else "当前筛选范围已完成。"),
                }
                st.session_state["pending_case_id"] = next_id
                st.rerun()

    with st.expander("病例来源与技术信息", expanded=False):
        st.caption(f"source: `{sample.source_path}`")
        st.caption(f"病例：{case.case_id} · 数据集：{sample.dataset_id} · {sample.review_split}")
        if prediction is not None:
            st.caption(f"prediction status: `{prediction.status}` · mask SHA-256: `{prediction.mask_sha256}`")
        if draft_geometry:
            st.json(draft_geometry)
    history = store.history(reviewer, case.case_id)
    if history:
        with st.expander(f"本病例完整修订历史（{len(history)}版）"):
            st.json([item.to_dict() for item in history])


if __name__ == "__main__":
    main()
