"""Small storage tests; synthetic outcomes are not scientific RSI evidence."""
import json
import os

import numpy as np
import pytest

from .contracts import file_hash
from .field_storage import ArrayPool, allocated_bytes
from .member_storage import MemberBundle, adapt_runner, input_loader, stage_members
from .member_continuation import estimate
from . import test_stable_study as scientific_tests
from .test_storage_continuation import small_policy
from .test_system_state import fixture, prepare_initial, proposal


def inputs(tmp_path, extra=False):
    pool = ArrayPool(tmp_path, tmp_path/'pool', max_pool_bytes=16*1024**2, minimum_free_bytes=0)
    image = np.arange(64,dtype=np.float32).reshape(8,8)
    source = tmp_path/'prediction.npz'
    fields = {'score': image, 'support': np.ones((8,8),bool)}
    if extra:
        fields['human'] = np.ones((8,8),bool)
    with source.open('xb') as stream:
        np.savez_compressed(stream, **fields)
    folder = tmp_path/'input';folder.mkdir()
    return pool, source, folder, image


def test_members_decode_exactly_and_hardlink_existing_bytes(tmp_path):
    pool, source, folder, image = inputs(tmp_path)
    digest = file_hash(source)
    stage_members(pool,folder,'sample-0000.fields.json',{'image':image},{'b1':source})
    loader=input_loader(np.load,folder)
    with loader(folder/'sample-0000.fields.json',allow_pickle=False) as arrays:
        assert np.array_equal(arrays['image'],image)
        assert np.array_equal(arrays['branch_b1_score'],image)
        assert arrays['branch_b1_support'].dtype == np.bool_
        assert set(arrays.files)=={'image','branch_b1_score','branch_b1_support'}
    assert os.path.samefile(source,folder/'arrays'/f'{digest}.npz')
    assert file_hash(source)==digest
    # Only the raw image, not a separately packed prediction pair, enters pool.
    assert len(list(pool.root.glob('*.npz')))==1


def test_labels_in_borrowed_npz_are_rejected_before_exposure(tmp_path):
    pool, source, folder, image = inputs(tmp_path,extra=True)
    with pytest.raises(ValueError,match='Only score/support'):
        stage_members(pool,folder,'sample-0000.fields.json',{'image':image},{'b1':source})
    assert not (folder/'arrays').exists()
    assert source.stat().st_nlink==1


def test_changed_reference_metadata_is_rejected(tmp_path):
    pool, source, folder, image=inputs(tmp_path)
    stage_members(pool,folder,'sample-0000.fields.json',{'image':image},{'b1':source})
    manifest=folder/'sample-0000.fields.json';obj=json.loads(manifest.read_text())
    obj['fields'][1]['identity']='0'*64
    bad=folder/'sample-0001.fields.json'
    with bad.open('x') as stream:json.dump(obj,stream)
    with pytest.raises(ValueError,match='array changed'):
        MemberBundle(bad,np.load)['branch_b1_score']


def test_traversal_and_overwrite_are_rejected(tmp_path):
    pool, source, folder, image=inputs(tmp_path)
    with pytest.raises(ValueError,match='basename'):
        stage_members(pool,folder,'../bad.json',{'image':image},{'b1':source})
    stage_members(pool,folder,'sample-0000.fields.json',{'image':image},{'b1':source})
    with pytest.raises(FileExistsError):
        stage_members(pool,folder,'sample-0000.fields.json',{'image':image},{'b1':source})


def test_estimate_has_no_branch_repacking_or_state_change(tmp_path):
    archive, original=fixture(tmp_path);original.development_transforms=('rot90',)
    try:
        initial,_=prepare_initial(original)
        runner=adapt_runner(original,tmp_path/'storage',small_policy())
        before=archive.verify();size=allocated_bytes(tmp_path/'storage')
        plan=estimate(runner,proposal(initial),runner.frozen_state(initial)['library'],small_policy())
        assert plan['borrowable_existing_prediction_pairs']>0
        assert plan['uncomputed_prediction_members_repacked'] is False
        assert archive.verify()==before and allocated_bytes(tmp_path/'storage')==size
        assert not list(runner.array_pool.root.glob('*.npz'))
    finally:archive.close()


def patch_science(monkeypatch):
    previous=scientific_tests.experiment
    def experiment(tmp_path,patch):
        initial,base_load=previous(tmp_path,patch)
        def load(*args,**kwargs):
            values=base_load(*args,**kwargs)
            return (*values[:-1],adapt_runner(values[-1],tmp_path/'storage',small_policy()))
        patch.setattr(scientific_tests.study,'load',load)
        return initial,load
    monkeypatch.setattr(scientific_tests,'experiment',experiment)


def test_actual_two_round_acceptance_and_inheritance_unchanged(tmp_path,monkeypatch):
    patch_science(monkeypatch)
    scientific_tests.test_executed_improvements_are_accepted_and_actually_inherited_twice(tmp_path,monkeypatch)


def test_actual_rotation_failure_still_rejected(tmp_path,monkeypatch):
    patch_science(monkeypatch)
    scientific_tests.test_original_gain_with_rotation_failure_is_banked_but_never_deployed(tmp_path,monkeypatch)


def test_invalid_reply_still_counts_and_is_not_repaired(tmp_path,monkeypatch):
    patch_science(monkeypatch)
    scientific_tests.test_invalid_reply_keeps_raw_bytes_and_consumes_round(tmp_path,monkeypatch)
