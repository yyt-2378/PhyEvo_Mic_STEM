"""Access-gated FAU site review. Original analysis stays immutable.

One append-only ledger, independent of the legacy 154-case expert database.
The browser supplies opinions, never model facts or filesystem paths.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import sqlite3

import numpy as np
from PIL import Image
from .fau_paths import SELECTED, resolve_input, result_dir

ROOT = Path(__file__).resolve().parents[2]
RESULTS = result_dir('same_plane_skeleton_three_class_20261021_v1')
MODEL = RESULTS / 'joint_three_state_sites_centered.json'
STORE = ROOT / 'local_fau_review/v1/reviews.sqlite3'
ASSETS = ROOT / 'apps/fau_site_review'
FEEDBACK = RESULTS.parent / 'review_feedback_three_color_20261022_v1'
FEEDBACK_EDITIONS = {
    'feedback-20261022-v3': (RESULTS.parent / 'review_feedback_three_color_20261022_v3',
        '10月22日 · 第三版：减少不确定', 'FAU_RA_H2联合反馈学习_三色标注迭代.pdf'),
    'feedback-20261022-v2': (RESULTS.parent / 'review_feedback_three_color_20261022_v2',
        '10月22日 · RA + H2 联合学习第二版', 'FAU_RA_H2联合反馈学习_三色标注迭代.pdf'),
    'feedback-20261022': (FEEDBACK, '10月22日 · RA反馈第一版', 'FAU_RA人工反馈学习_三色标注迭代.pdf'),
}
COLOR_ONLY_REVIEW_VERSION = 5
STATES = {'occupied', 'empty_like', 'uncertain'}
IDENTIFIER = re.compile(r'^[A-Za-z0-9_.@-]{1,64}$')
MANUAL = re.compile(r'^(RA|H2)-M[0-9a-f]{12}$')


def scoped(path: Path) -> Path:
    path = resolve_input(path)
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('路径必须在本项目内的专用目录中')
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(scoped(path).read_bytes()).hexdigest()


def image_url(values: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(values).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def load_bundle(edition='original') -> dict:
    """Called only after the original access gate; sends no disk paths to clients."""
    rows = json.loads(scoped(MODEL).read_text())
    hashes = json.loads(scoped(RESULTS / 'input_hashes.json').read_text())
    samples = {}
    with np.load(scoped(RESULTS / 'atom_centres.npz'), allow_pickle=False) as atoms:
        for sample in ('RA', 'H2'):
            path = ROOT / f'real_data_test/4D_data/{sample}.png'
            checksum = digest(path)
            if hashes[str(path.relative_to(ROOT))] != checksum:
                raise ValueError(f'{sample} 原图已变化，不能混用旧坐标')
            with Image.open(scoped(path)) as im:
                gray = np.asarray(im.convert('L'))
            observed = atoms[sample + '_observed']
            full = atoms[sample + '_full_expected']
            matrix = atoms[sample + '_global_affine']
            theoretical = (full - matrix[:, 2]) @ np.linalg.inv(matrix[:, :2]).T
            samples[sample] = dict(width=gray.shape[1], height=gray.shape[0], image=image_url(gray),
                                   sha256=checksum, measured=np.round(observed, 3).tolist(),
                                   ideal=np.round(theoretical, 3).tolist(),
                                   plane=6 if sample == 'RA' else 7)
    sites = []
    with np.load(scoped(RESULTS / 'local_comparisons.npz'), allow_pickle=False) as patches:
        for row in rows:
            item = {k: row[k] for k in ('id', 'sample', 'x', 'y', 'status', 'reason', 'local_D_pct', 'fit_RMSE_px', 'endpoints_measured')}
            item['bic'] = row['profile']['bics'][1] if row['profile']['valid'] else None
            item['ncc'] = row['shape'].get('outer_NCC')
            for name in ('target', 'expected'):
                key = row['id'] + '_' + name
                item[name] = image_url(np.uint8(np.clip(patches[key], 0, 1) * 255)) if key in patches.files else None
            sites.append(item)
    if len(sites) != 101 or len({s['id'] for s in sites}) != 101:
        raise ValueError('FAU 搜索位点数量或ID不一致')
    bundle = dict(release=digest(MODEL), samples=samples, sites=sites)
    if edition in FEEDBACK_EDITIONS:
        return apply_feedback(bundle, edition)
    if edition != 'original':
        raise ValueError('未知显示版本')
    return dict(bundle, model_release=bundle['release'], edition='original')


def apply_feedback(bundle, edition='feedback-20261022'):
    """Keep annotation identity stable; separately version what the model shows.

    The human ledger remains authoritative. Learned colors never become human
    reviews and never count toward reviewed progress.
    """
    import copy
    bundle = copy.deepcopy(bundle)
    directory = FEEDBACK_EDITIONS[edition][0]
    snapshot = json.loads(scoped(directory / 'review_snapshot.json').read_text())
    reference = json.loads(scoped(directory / 'morphology_reference_model.json').read_text())
    if snapshot['parent_release'] != bundle['release']:
        raise ValueError('学习版本与原始位点版本不对应')
    rows = json.loads(scoped(directory / 'three_color_sites.json').read_text())
    mapping = {r['id']: r for r in rows}
    if len(mapping) != len(rows) or set(mapping) != {s['id'] for s in bundle['sites']}:
        raise ValueError('学习版本位点清单不一致')
    for path, checksum in json.loads(scoped(directory / 'input_hashes.json').read_text()).items():
        if digest(ROOT / path) != checksum:
            raise ValueError('学习版本输入已变化')
    training_samples = sorted({sid.split('-')[0] for sid in reference['training_ids']})
    with np.load(scoped(directory / 'morphology_patches.npz'), allow_pickle=False) as patches:
        for site in bundle['sites']:
            r = mapping[site['id']]
            if site['sample'] != r['sample'] or not np.allclose([site['x'],site['y']],r['model_center']):
                raise ValueError('学习版本坐标来源不对应')
            site['previous_status'] = site['status']
            # Do not copy r['status']: that field already incorporates human opinions.
            site['status'] = r['model_prediction']['state']
            if site['status'] not in STATES:
                raise ValueError('学习版本分类无效')
            site['reason'] = f'{" + ".join(training_samples)} {len(reference["training_ids"])}个明确审核样例的同构形态比较；这是模型推断，不是新增人工审核'
            if edition=='feedback-20261022-v3':
                site['reason'] += f'；校准后的相对距离差门槛={reference["selected_decision_margin"]:.2f}（不是置信概率）'
                if r['model_prediction']['state']=='uncertain':
                    site['reason'] += '；' + ('原图/外围骨架支持不足' if r['model_prediction']['reason']=='unreliable_peripheral_frame' else '有无形态仍接近或参考分歧')
            if site['sample'] not in training_samples:
                site['reason'] += '；跨图迁移尚无本图人工验证'
            site['model_references'] = [x['id'] for x in r['model_prediction'].get('nearest_references',[])]
            site['ncc'] = r['outer_NCC']
            site['bic'] = None  # This release uses morphology, not a newly fitted two-O profile.
            if r['moved_px'] > .001:
                site['local_D_pct'] = None  # Never present old-center strain as newly measured strain.
            for name in ('target','expected'):
                site[name] = image_url(np.uint8(np.clip(patches[site['id']+'_'+name],0,1)*255))
    bundle.update(model_release=digest(directory/'three_color_sites.json'), edition=edition)
    return bundle


class ConflictError(ValueError):
    pass


class SiteLedger:
    def __init__(self, path: Path):
        self.path = scoped(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        for suffix in ('-wal', '-shm'):
            scoped(Path(str(self.path) + suffix))
        with self.connect() as con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS fau_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    reviewer TEXT NOT NULL, release TEXT NOT NULL,
                    site_id TEXT NOT NULL, sample TEXT NOT NULL,
                    revision INTEGER NOT NULL, request_id TEXT NOT NULL,
                    payload TEXT NOT NULL, created TEXT NOT NULL,
                    UNIQUE(reviewer, release, site_id, revision),
                    UNIQUE(reviewer, request_id)
                );
                CREATE INDEX IF NOT EXISTS fau_events_owner ON fau_events(reviewer,release,seq);
            ''')

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=15)
        con.execute('PRAGMA busy_timeout=15000')
        con.execute('PRAGMA synchronous=FULL')
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def history(self, reviewer, release):
        if not IDENTIFIER.fullmatch(reviewer):
            raise ValueError('审核人 ID 格式无效')
        with self.connect() as con:
            return [json.loads(row['payload']) for row in con.execute(
                'SELECT payload FROM fau_events WHERE reviewer=? AND release=? ORDER BY seq', (reviewer, release))]

    def current(self, reviewer, release):
        return {row['site_id']: row for row in self.history(reviewer, release)}

    def submit(self, reviewer, action, bundle):
        if not IDENTIFIER.fullmatch(reviewer):
            raise ValueError('审核人 ID 格式无效')
        if not isinstance(action, dict) or len(json.dumps(action)) > 16000:
            raise ValueError('提交格式或大小无效')
        if action.get('release') != bundle['release']:
            raise ConflictError('结果版本已变化，请刷新并重新核对')
        model_release = bundle.get('model_release',bundle['release'])
        if action.get('model_release',bundle['release']) != model_release:
            raise ConflictError('显示模型已切换，请刷新后再保存；已有审核不会丢失')
        request_id = action.get('request_id', '')
        if not re.fullmatch(r'[a-f0-9-]{20,64}', request_id):
            raise ValueError('提交编号无效')
        site_id = action.get('site_id', '')
        sample = action.get('sample')
        if sample not in bundle['samples']:
            raise ValueError('样品无效')
        originals = {s['id']: s for s in bundle['sites']}
        site = originals.get(site_id)
        manual = MANUAL.fullmatch(site_id)
        if not site and not (manual and manual.group(1) == sample):
            raise ValueError('位点 ID 无效')
        if site and site['sample'] != sample:
            raise ValueError('位点和样品不对应')
        decision = action.get('decision')
        if decision not in STATES:
            raise ValueError('请选择三种判断之一')
        alignment = action.get('alignment')
        if alignment not in {'good', 'bad', 'uncertain', 'not_reviewed'}:
            raise ValueError('骨架评价无效；只改颜色时请选择不评价对齐')
        note = action.get('note', '')
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError('备注最多2000字')
        xy = action.get('center')
        if not isinstance(xy, list) or len(xy) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy):
            raise ValueError('中心必须是有限原图坐标')
        image = bundle['samples'][sample]
        if not (0 <= xy[0] < image['width'] and 0 <= xy[1] < image['height']):
            raise ValueError('中心超出原图范围')
        base = action.get('base_revision')
        if type(base) is not int or base < 0:
            raise ValueError('修订版本无效')
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            repeated = con.execute('SELECT payload FROM fau_events WHERE reviewer=? AND request_id=?', (reviewer, request_id)).fetchone()
            if repeated:
                saved = json.loads(repeated['payload'])
                if any(saved[k] != v for k, v in dict(release=bundle['release'], site_id=site_id, sample=sample, decision=decision, alignment=alignment, center=xy, note=note).items()):
                    raise ConflictError('同一提交编号的内容发生变化，请重新提交')
                if saved.get('model_release',saved['release']) != model_release:
                    raise ConflictError('同一提交编号不能用于另一个模型版本')
                return saved
            previous = con.execute('SELECT MAX(revision) FROM fau_events WHERE reviewer=? AND release=? AND site_id=?',
                                   (reviewer, bundle['release'], site_id)).fetchone()[0] or 0
            if previous != base:
                raise ConflictError('另一标签页已更新此点。请刷新，避免覆盖新记录')
            saved = dict(reviewer=reviewer, release=bundle['release'], site_id=site_id, sample=sample,
                         model_release=model_release, model_edition=bundle.get('edition','original'),
                         revision=previous + 1, request_id=request_id, decision=decision, alignment=alignment,
                         center=xy, note=note, manual=site is None, image_sha256=image['sha256'],
                         plane_one_based=image['plane'], source_shape_yx=[image['height'], image['width']],
                         model_status=site['status'] if site else None,
                         model_center=[site['x'], site['y']] if site else None,
                         created=datetime.now(timezone.utc).isoformat(), coordinate_space='source_pixels_xy',
                         interpretation='human_image_opinion_not_chemical_ground_truth')
            con.execute('INSERT INTO fau_events(reviewer,release,site_id,sample,revision,request_id,payload,created) VALUES(?,?,?,?,?,?,?,?)',
                        (reviewer, bundle['release'], site_id, sample, saved['revision'], request_id,
                         json.dumps(saved, ensure_ascii=False, allow_nan=False), saved['created']))
            return saved


def render():
    import streamlit as st
    # This route must never inherit the legacy optional-code behavior.
    if not os.environ.get('TFREE_REVIEW_ACCESS_CODE') or not st.session_state.get('expert-review-access-granted'):
        st.error('请先通过专家审核访问验证。'); st.stop()
    st.html('''<style>.stMainBlockContainer{max-width:1700px;padding-top:2.5rem}
        h1{font-size:1.7rem!important}[data-testid="stCaptionContainer"]{line-height:1.6}</style>''')
    st.title('FAU · 已有标记改色审核')
    st.caption('直接点选现有红 / 蓝 / 黄圆圈，改成你的判断并保存；不需要重新画点，也不必逐项填写骨架评价。')
    with st.sidebar:
        editions = [key for key,(directory,_,pdf) in FEEDBACK_EDITIONS.items()
                    if scoped(directory/'three_color_sites.json').exists() and scoped(directory/'morphology_reference_model.json').exists()] + ['original']
        requested = st.query_params.get('edition',editions[0])
        edition = st.selectbox('标注显示版本',editions,index=editions.index(requested) if requested in editions else 0,
            format_func=lambda v:FEEDBACK_EDITIONS[v][1] if v!='original' else '10月21日 · 原始三色模型')
        st.query_params['edition'] = edition
        if edition != 'original':
            directory=FEEDBACK_EDITIONS[edition][0]
            summary=json.loads(scoped(directory/'summary.json').read_text())
            counts=summary['reviewed'];total=sum(counts.values());uncertain=counts.get('uncertain',0)
            samples='、'.join(f'{s} {sum(v["human"].values())}处' for s,v in summary['samples'].items())
            st.caption(f'本版冻结审核：{samples}；{total-uncertain}处明确判断用于形态参考，{uncertain}处不确定保留。已保存人工判断始终优先；其余为尚未确认的模型推断。')
            if 'selected_margin' in summary:
                counts_by_sample='；'.join(f'{s} 黄{v["final"].get("uncertain",0)}个' for s,v in summary['samples'].items())
                st.caption(f'发布时：{counts_by_sample}。本版通过分区验证校准判断门槛，提高自动判读覆盖；红/蓝仍是可审核的倾向，不是元素确证。')
        if edition != editions[0]:
            st.info('已有更新的学习版，可在上方切换；各版共享你的审核进度，不会丢失记录。')
        st.subheader('审核身份')
        default = st.query_params.get('reviewer', '')
        reviewer = st.text_input('本次 Reviewer ID', value=default if IDENTIFIER.fullmatch(default) else '',
                                 placeholder='例如 fau-owner', max_chars=64).strip()
        st.caption('每次使用同一个 ID 才能恢复自己的进度。共享访问码下的 ID 用于署名，不是独立账号权限。')
    if not IDENTIFIER.fullmatch(reviewer):
        st.info('请先在左侧填写审核人 ID，例如 fau-owner。'); st.stop()
    st.query_params['reviewer'] = reviewer
    try:
        bundle = st.cache_data(show_spinner=False)(load_bundle)(edition)
        path = Path(os.environ.get('TFREE_FAU_REVIEW_STORE_PATH', str(STORE)))
        store = SiteLedger(path)
    except (OSError, ValueError, sqlite3.Error) as exc:
        st.error('审核数据暂不可用：' + str(exc)); st.stop()
    current = store.current(reviewer, bundle['release'])
    with st.sidebar:
        completed = sum(s['id'] in current for s in bundle['sites'])
        st.metric('已审核模型位点', f'{completed} / {len(bundle["sites"])}')
        st.progress(completed / len(bundle['sites']))
        st.caption(f'另有 {sum(r["manual"] for r in current.values())} 个手工补点。选择“不确定”并保存，也算已审核。')
        history = store.history(reviewer, bundle['release'])
        st.download_button('导出我的审核记录', json.dumps(dict(release=bundle['release'], reviewer=reviewer, history=history), ensure_ascii=False, indent=2),
                           file_name=f'fau_review_{reviewer}.json', mime='application/json', on_click='ignore')
        st.caption('保存不会自动重训练或修改算法；下一轮按你的纠错记录比较与迭代。')
        with st.expander('已有报告与说明'):
            report = FEEDBACK_EDITIONS[edition][0] / FEEDBACK_EDITIONS[edition][2] if edition != 'original' else RESULTS / 'FAU_同层原子中心拟合_三色位点与局部应变.pdf'
            if not scoped(report).is_file():
                report = SELECTED / 'with_latest_three_color_20261022/FAU_RA_H2_指定点位骨架与畸变对比_含最新三色标注.pdf'
                label = '下载最终结构对比报告（含最新三色总览）'
            else:
                label = '下载当前分析 PDF'
            if scoped(report).is_file():
                st.download_button(label, scoped(report).read_bytes(), file_name=report.name, on_click='ignore')
            st.write('默认显示模型三色标记，人工保存后在同一圆圈上替换显示颜色；模型原判断仍保留。红=有占位倾向，蓝=无占位倾向，黄=不确定，均非元素确证。')
    resume = st.session_state.get('fau-resume-' + reviewer)
    if not resume:
        resume = next((s['id'] for s in bundle['sites'] if s['id'] not in current), bundle['sites'][0]['id'])
    data = dict(bundle, reviews=current, reviewer=reviewer, resume=resume, receipt=st.session_state.get('fau-receipt-' + reviewer))
    component = st.components.v2.component('phyevo_fau_site_review',
        html=scoped(ASSETS / 'index.html').read_text(), css=scoped(ASSETS / 'style.css').read_text(),
        js=scoped(ASSETS / 'app.js').read_text(), isolate_styles=False)
    # A trigger must not also have a persistent default with the same key:
    # Streamlit merges persistent values last, which would mask the save event.
    result = component(key='fau-workbench-' + reviewer + '-' + edition, data=data,
                       width='stretch', height='content', on_action_change=lambda: None)
    if result.action:
        try:
            saved = store.submit(reviewer, result.action, bundle)
        except (ValueError, sqlite3.Error) as exc:
            st.error('没有保存成功：' + str(exc)); return
        ids = {s['id'] for s in bundle['sites']} | set(store.current(reviewer, bundle['release']))
        target = result.action.get('next_id')
        st.session_state['fau-resume-' + reviewer] = target if target in ids else saved['site_id']
        st.session_state['fau-receipt-' + reviewer] = f'已保存 {saved["site_id"]} · 第 {saved["revision"]} 版 · {saved["created"]}'
        st.rerun()
