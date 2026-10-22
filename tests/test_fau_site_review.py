"""Synthetic review-ledger tests. No mutation of microscopy assets or old reviews."""
import json
from pathlib import Path
import uuid

import pytest

from tfree_detection.fau_site_review import ConflictError, ROOT, SiteLedger


@pytest.fixture
def bundle():
    return dict(release='a'*64, samples={'RA':dict(width=100,height=90,sha256='b'*64,plane=6),
                                      'H2':dict(width=80,height=70,sha256='c'*64,plane=7)},
                sites=[dict(id='RA-P001',sample='RA',status='occupied',x=30.,y=40.)])


@pytest.fixture
def ledger(tmp_path):
    assert tmp_path.resolve().is_relative_to(ROOT)
    return SiteLedger(tmp_path / 'reviews.sqlite3')


def action(bundle, **changes):
    return dict(release=bundle['release'], request_id=str(uuid.uuid4()), site_id='RA-P001',
                sample='RA', decision='empty_like', alignment='good', center=[32.,43.],
                note='人工修正', base_revision=0, **{}) | changes


def test_save_refresh_history_and_user_isolation(ledger,bundle):
    a=action(bundle);saved=ledger.submit('owner',a,bundle)
    assert saved['revision']==1 and saved['model_status']=='occupied' and saved['decision']=='empty_like'
    assert SiteLedger(ledger.path).current('owner',bundle['release'])['RA-P001']['center']==[32.,43.]
    assert ledger.current('other',bundle['release'])=={}
    ledger.submit('owner',action(bundle,base_revision=1,decision='uncertain'),bundle)
    assert len(ledger.history('owner',bundle['release']))==2
    assert ledger.current('owner',bundle['release'])['RA-P001']['revision']==2


def test_retry_is_idempotent_and_stale_tabs_cannot_overwrite(ledger,bundle):
    a=action(bundle);first=ledger.submit('owner',a,bundle)
    assert ledger.submit('owner',a,bundle)==first
    assert len(ledger.history('owner',bundle['release']))==1
    with pytest.raises(ConflictError):ledger.submit('owner',action(bundle),bundle)
    with pytest.raises(ConflictError):ledger.submit('owner',a|{'note':'changed'},bundle)


@pytest.mark.parametrize('change',[
    {'sample':'H2'}, {'decision':'confirmed_element'}, {'alignment':''}, {'center':[float('nan'),1.]},
    {'center':[-1.,2.]}, {'center':[100.,10.]}, {'site_id':'../../data'}, {'site_id':'H2-M012345abcdef'},
    {'note':'x'*2001}, {'release':'new-version'}, {'base_revision':True}, {'request_id':'no'}])
def test_invalid_writes_leave_empty_ledger(ledger,bundle,change):
    with pytest.raises(ValueError):ledger.submit('owner',action(bundle,**change),bundle)
    assert ledger.history('owner',bundle['release'])==[]


def test_manual_point_is_separate_and_preserves_source_identity(ledger,bundle):
    row=ledger.submit('owner',action(bundle,site_id='RA-M012345abcdef',center=[10,20]),bundle)
    assert row['manual'] and row['model_status'] is None
    assert row['image_sha256']=='b'*64 and row['plane_one_based']==6
    assert bundle['sites']==[dict(id='RA-P001',sample='RA',status='occupied',x=30.,y=40.)]


def test_color_only_correction_preserves_center_and_separates_alignment(ledger,bundle):
    bundle['sites'][0]['status']='uncertain'
    before=json.dumps(bundle,sort_keys=True)
    saved=ledger.submit('owner',action(bundle,decision='occupied',alignment='not_reviewed',center=[30.,40.]),bundle)
    assert saved['model_status']=='uncertain' and saved['decision']=='occupied'
    assert saved['alignment']=='not_reviewed'  # Not an assessment of bad or uncertain alignment.
    assert saved['center']==saved['model_center']==[30.,40.]
    updated=ledger.submit('owner',action(bundle,decision='empty_like',alignment='not_reviewed',center=[30.,40.],base_revision=1),bundle)
    assert updated['revision']==2 and updated['model_status']=='uncertain'
    assert [r['decision'] for r in ledger.history('owner',bundle['release'])]==['occupied','empty_like']
    assert json.dumps(bundle,sort_keys=True)==before


def test_reading_model_and_reviews_never_confirms_unsaved_labels(ledger,bundle):
    assert ledger.current('owner',bundle['release'])=={}
    assert ledger.history('owner',bundle['release'])==[]


def test_learning_version_preserves_progress_and_tracks_displayed_model(ledger,bundle):
    ledger.submit('owner',action(bundle,center=[30.,40.]),bundle)
    updated=dict(bundle,model_release='e'*64,edition='feedback-20261022')
    assert len(ledger.current('owner',updated['release']))==1
    with pytest.raises(ConflictError):
        ledger.submit('owner',action(bundle,base_revision=1),updated)
    saved=ledger.submit('owner',action(bundle,base_revision=1,model_release='e'*64),updated)
    assert saved['model_release']=='e'*64 and saved['model_edition']=='feedback-20261022'
    assert saved['revision']==2 and len(ledger.current('owner',bundle['release']))==1
    assert [r['model_release'] for r in ledger.history('owner',bundle['release'])]==[bundle['release'],'e'*64]


def test_path_and_reviewer_validation(ledger,bundle):
    with pytest.raises(ValueError):SiteLedger(ROOT.parent/'outside.sqlite3')
    with pytest.raises(ValueError):ledger.submit('../owner',action(bundle),bundle)


def test_gate_precedes_fau_assets(monkeypatch,tmp_path):
    st=pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    import tfree_detection.fau_site_review as review
    touched=[]
    monkeypatch.setenv('TFREE_REVIEW_ACCESS_CODE','synthetic-test-code')
    monkeypatch.setenv('TFREE_FAU_REVIEW_STORE_PATH',str(tmp_path/'gated.sqlite3'))
    monkeypatch.setattr(review,'load_bundle',lambda:touched.append(True))
    app=AppTest.from_file(str(ROOT/'apps/review_dataset_viewer.py'),default_timeout=30)
    app.query_params['view']='fau-sites'
    app.run();assert not app.exception and not touched
    app.text_input[0].input('wrong');app.button[0].click().run()
    assert not app.exception and not touched and not (tmp_path/'gated.sqlite3').exists()


def test_authorized_fau_route_does_not_open_legacy_queue(monkeypatch):
    st=pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    import tfree_detection.fau_site_review as review
    from tfree_detection import review_dataset
    called=[]
    monkeypatch.setenv('TFREE_REVIEW_ACCESS_CODE','synthetic-test-code')
    monkeypatch.setattr(review,'render',lambda:called.append('fau'))
    monkeypatch.setattr(review_dataset,'load_review_manifest',lambda *_:pytest.fail('Legacy data opened'))
    app=AppTest.from_file(str(ROOT/'apps/review_dataset_viewer.py'),default_timeout=30)
    app.query_params['view']='fau-sites';app.session_state['expert-review-access-granted']=True
    app.run();assert not app.exception and called==['fau']


def test_fau_route_fails_closed_without_configured_access_code(monkeypatch):
    st=pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    import tfree_detection.fau_site_review as review
    monkeypatch.delenv('TFREE_REVIEW_ACCESS_CODE',raising=False)
    monkeypatch.setattr(review,'load_bundle',lambda:pytest.fail('Data loaded without gate'))
    app=AppTest.from_file(str(ROOT/'apps/review_dataset_viewer.py'),default_timeout=30)
    app.query_params['view']='fau-sites';app.run()
    assert not app.exception and any('请先通过' in e.value for e in app.error)


@pytest.mark.parametrize('old_version', [None, 4])
def test_old_live_fau_module_is_upgraded_once_after_authentication(monkeypatch, old_version):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    import importlib
    import tfree_detection.fau_site_review as review
    calls=[]
    monkeypatch.setenv('TFREE_REVIEW_ACCESS_CODE','synthetic-test-code')
    if old_version is None:
        monkeypatch.delattr(review,'COLOR_ONLY_REVIEW_VERSION')
    else:
        monkeypatch.setattr(review,'COLOR_ONLY_REVIEW_VERSION',old_version)
    monkeypatch.setattr(review,'render',lambda:calls.append('render'))
    real_reload=importlib.reload
    def reload(module):
        if module is not review:return real_reload(module)
        calls.append('upgrade')
        monkeypatch.setattr(module,'COLOR_ONLY_REVIEW_VERSION',5,raising=False)
        return module
    monkeypatch.setattr(importlib,'reload',reload)
    app=AppTest.from_file(str(ROOT/'apps/review_dataset_viewer.py'),default_timeout=30)
    app.query_params['view']='fau-sites'
    app.run();assert not app.exception and calls==[]
    app.session_state['expert-review-access-granted']=True
    app.run();assert not app.exception and calls==['upgrade','render']
    app.run();assert not app.exception and calls==['upgrade','render','render']
