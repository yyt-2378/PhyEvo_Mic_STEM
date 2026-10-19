"""Tiny synthetic storage equivalence, isolation, integrity and capacity tests."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from .contracts import file_hash
from .field_storage import ArrayBundle, ArrayPool, allocated_bytes, input_loader
from .field_storage_runtime import FieldStorageExecutor, FieldStorageSandbox, storage_binding
from .stability_state import StabilitySystemExecutor
from .test_system_state import fixture, prepare_initial, proposal


def pool(tmp_path, budget=16 * 1024**2, reserve=0):
    return ArrayPool(tmp_path, tmp_path / 'objects', max_pool_bytes=budget, minimum_free_bytes=reserve)


@pytest.mark.parametrize('value', [np.array(7, dtype=np.int64), np.array([True, False]),
    np.arange(12, dtype='>f4').reshape(3, 4), np.arange(12, dtype=np.float32).reshape(3, 4).T,
    np.array([1+2j], dtype=np.complex64)])
def test_roundtrip_preserves_shape_dtype_bytes_order_and_links(tmp_path, value):
    store = pool(tmp_path)
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(); b.mkdir()
    fields = {'second': value, 'first': np.ones((2, 2), bool)}
    store.stage(a, 'sample-0000.fields.json', fields)
    store.stage(b, 'sample-0000.fields.json', fields)
    for folder in (a, b):
        with input_loader(np.load, folder)(folder / 'sample-0000.fields.json', allow_pickle=False) as data:
            assert data.files == ['second', 'first']
            actual = data['second']
            assert actual.shape == value.shape and actual.dtype == value.dtype
            assert actual.tobytes(order='C') == value.tobytes(order='C')
            assert actual.flags.f_contiguous == value.flags.f_contiguous
            assert actual.flags.c_contiguous == value.flags.c_contiguous
    for item in (a / 'arrays').iterdir():
        assert os.path.samefile(item, b / 'arrays' / item.name)
        assert os.path.samefile(item, store.root / item.name)


def test_add_branch_stores_only_new_array_not_full_bundle(tmp_path):
    rng = np.random.default_rng(19)
    store = pool(tmp_path)
    fields = {f'base_{i}': rng.normal(size=(128, 128)).astype('f4') for i in range(12)}
    old = tmp_path / 'legacy'; old.mkdir()
    for i in range(6):
        fields[f'branch_{i}'] = rng.normal(size=(128, 128)).astype('f4')
        folder = tmp_path / f'run{i}'; folder.mkdir()
        store.stage(folder, 'sample-0000.fields.json', fields)
        with (old / f'bundle{i}.npz').open('xb') as stream:
            np.savez_compressed(stream, **fields)
    legacy = allocated_bytes(old)
    actual = allocated_bytes(store.root)
    assert len(list(store.root.glob('*.npz'))) == 18
    assert actual < legacy * .30
    print(json.dumps({'synthetic_whole_bundle_bytes': legacy, 'field_pool_bytes': actual,
                      'pool_saving_fraction': 1 - actual / legacy, 'rounds': 6}))


def test_concurrent_identical_fields_create_one_object(tmp_path):
    store = pool(tmp_path)
    value = np.arange(100, dtype='f4')
    with ThreadPoolExecutor(max_workers=4) as executor:
        refs = list(executor.map(store.put, [value] * 12))
    assert all(r == refs[0] for r in refs)
    assert len(list(store.root.glob('*.npz'))) == 1


def test_dtype_shape_are_part_of_identity_and_normal_npz_still_loads(tmp_path):
    store = pool(tmp_path)
    original = np.array([1, 2], dtype='i2')
    variants = [original, original.reshape(1, 2), original.view('u1')]
    assert len({store.put(v)['identity'] for v in variants}) == 3
    normal = tmp_path / 'ordinary.npz'
    with normal.open('xb') as stream:
        np.savez_compressed(stream, values=original)
    with input_loader(np.load, tmp_path)(normal, allow_pickle=False) as data:
        assert np.array_equal(data['values'], original)


def test_cannot_adopt_existing_artifacts_or_stage_outside_workspace(tmp_path):
    existing = tmp_path / 'previous_assets'; existing.mkdir()
    with (existing / 'existing.txt').open('x') as stream:
        stream.write('Preserve existing input')
    before = {p.name: file_hash(p) for p in existing.iterdir()}
    with pytest.raises(ValueError, match='nonempty'):
        ArrayPool(tmp_path, existing, max_pool_bytes=65536, minimum_free_bytes=0)
    assert before == {p.name: file_hash(p) for p in existing.iterdir()}
    store = pool(tmp_path)
    with pytest.raises(ValueError):
        store.stage(tmp_path.parent, 'sample-0000.fields.json', {'image': np.ones(2)})


def test_pool_policy_change_is_rejected(tmp_path):
    store = pool(tmp_path)
    before = (store.root / 'policy.json').read_bytes()
    with pytest.raises(ValueError, match='cannot silently change'):
        pool(tmp_path, budget=32 * 1024**2)
    assert (store.root / 'policy.json').read_bytes() == before


@pytest.mark.parametrize('reserve', [0, 2**63])
def test_budget_and_disk_reserve_fail_before_array_write(tmp_path, reserve):
    store = pool(tmp_path, budget=65536 if not reserve else 16*1024**2, reserve=reserve)
    with pytest.raises(OSError, match='before payload write'):
        store.put(np.random.default_rng(3).normal(size=(256, 256)).astype('f4'))
    assert not list(store.root.glob('*.npz'))


@pytest.mark.parametrize('value', [np.array([np.nan]), np.array([object()]), np.array(['text'])])
def test_invalid_arrays_are_rejected(tmp_path, value):
    with pytest.raises(ValueError, match='numeric'):
        pool(tmp_path).put(value)


def test_refuses_overwrite_and_detects_tampered_reference(tmp_path):
    store = pool(tmp_path)
    folder = tmp_path / 'input'; folder.mkdir()
    store.stage(folder, 'sample-0000.fields.json', {'image': np.ones((2, 2))})
    with pytest.raises(FileExistsError):
        store.stage(folder, 'sample-0000.fields.json', {'image': np.zeros((2, 2))})
    original = folder / 'sample-0000.fields.json'
    value = json.loads(original.read_text())
    value['fields'][0]['sha256'] = '0'*64
    bad = folder / 'sample-0001.fields.json'
    with bad.open('x') as stream:
        json.dump(value, stream)
    with pytest.raises(ValueError, match='hash mismatch'):
        ArrayBundle(bad, np.load)['image']
    value['fields'][0]['file'] = '../policy.json'
    another = folder / 'sample-0002.fields.json'
    with another.open('x') as stream:
        json.dump(value, stream)
    with pytest.raises(ValueError, match='object reference'):
        ArrayBundle(another, np.load)['image']


def runner(tmp_path, layered):
    archive, original = fixture(tmp_path)
    args = (original.workspace, archive)
    if layered:
        sandbox = FieldStorageSandbox(original.workspace, original.workspace / '.venvs/sam2', timeout=45)
        result = FieldStorageExecutor(*args, sandbox, original.samples, tmp_path / 'pool', ('rot90',),
                                      max_pool_bytes=16*1024**2, minimum_free_bytes=0)
    else:
        result = StabilitySystemExecutor(*args, original.sandbox, original.samples,
                                         tmp_path / 'pool', ('rot90',))
    return archive, result


def test_training_labels_and_source_membership_remain_separate(tmp_path):
    archive, r = runner(tmp_path, True)
    try:
        with pytest.raises(ValueError, match='nontraining'):
            r.stage(tmp_path / 'badfit', [r.samples[1]], mode='fit')
        assert not (tmp_path / 'badfit').exists()
        poisoned = replace(r.samples[0], fields=dict(r.samples[0].fields, human=r.samples[0].human))
        with pytest.raises(ValueError, match='Labels'):
            r.stage(tmp_path / 'badpredict', [poisoned])
        r.stage(tmp_path / 'fit', [r.samples[0]], mode='fit')
        r.stage(tmp_path / 'predict', [r.samples[0]])
        fit = ArrayBundle(tmp_path / 'fit/sample-0000.fields.json', np.load)
        predicted = ArrayBundle(tmp_path / 'predict/sample-0000.fields.json', np.load)
        assert 'human' in fit and 'known' in fit
        assert 'human' not in predicted and 'known' not in predicted
    finally:
        archive.close()


def test_real_whole_system_replay_and_rotation_match_legacy_bytes(tmp_path):
    measured = []
    for flag in (False, True):
        archive, r = runner(tmp_path / ('layered' if flag else 'legacy'), flag)
        try:
            initial, _ = prepare_initial(r)
            child = r.register_system(proposal(initial), r.frozen_state(initial))
            _, normal = r.execute(child)
            _, replay = r.execute(child, replay=True)
            _, rotated = r.probe_system(child, 'rot90')
            for receipt in (normal, replay, rotated):
                assert 'rows' in receipt, receipt
                if flag:
                    assert receipt['execution']['field_storage']['implementation_hashes'] == storage_binding()
            measured.append([{'rows': result['rows'], 'summary': result['summary'],
                              'prediction_hashes': {k:file_hash(r.checked_ref(v)) for k,v in result['predictions'].items()}}
                             for result in (normal, replay, rotated)])
            archive.verify()
        finally:
            archive.close()
    assert measured[0] == measured[1]


def test_actual_fit_and_frozen_state_match_legacy(tmp_path):
    code = '''import numpy as np
def fit(samples,context):
    assert len(samples)==1 and 'human' in samples[0] and 'known' in samples[0]
    w=context.get('parent_state',{}).get('parameter',np.array([1.])).copy()
    initial=w.copy(); loss=[]
    for _ in range(context['training_steps']):
        loss.append(float(w[0]**2)); w-=.1*2*w
    return {'parameter':w,'audit_initial_parameters':initial,'audit_final_parameters':w.copy(),
            'audit_loss_trace':np.array(loss),'audit_steps':np.array([context['training_steps']])}
def route(sample,branches,state,context):
    assert 'human' not in sample and 'known' not in sample
    return {'branch':'b0','reason':'synthetic fitted state'}
'''
    hashes = []
    for flag in (False, True):
        archive, r = runner(tmp_path / ('layered' if flag else 'legacy'), flag)
        try:
            initial, _ = prepare_initial(r)
            child = r.register_system(proposal(initial, router_code=code, router_training_steps=2),
                                      r.frozen_state(initial))
            trial, result = r.execute(child)
            assert 'rows' in result, result
            ref = r.frozen_state(child)['router_checkpoint']
            with np.load(r.checked_ref(ref), allow_pickle=False) as values:
                assert np.allclose(values['parameter'], [.64])
            _, rotated = r.probe_system(child, 'rot90')
            assert 'rows' in rotated and rotated['router_training'] is None, rotated
            assert r.frozen_state(child)['router_checkpoint'] == ref
            first_hash = file_hash(r.checked_ref(ref))
            # Synthetic fixture state only; this is NOT evidence of quality admission.
            archive.select(child, True, 'synthetic storage continuation setup', trial)
            next_child = r.register_system(proposal(child, router_code=code, router_training_steps=2),
                                           r.frozen_state(child))
            _, next_result = r.execute(next_child)
            assert 'rows' in next_result, next_result
            inherited = r.frozen_state(next_child)['router_checkpoint']
            with np.load(r.checked_ref(inherited), allow_pickle=False) as values:
                assert np.allclose(values['audit_initial_parameters'], [.64])
                assert np.allclose(values['parameter'], [.4096])
            assert file_hash(r.checked_ref(ref)) == first_hash
            hashes.append((first_hash, file_hash(r.checked_ref(inherited))))
        finally:
            archive.close()
    assert hashes[0] == hashes[1]


def test_candidate_only_sees_readonly_job_arrays_not_global_store(tmp_path):
    archive, r = runner(tmp_path, True)
    try:
        r.stage(tmp_path / 'fit', [r.samples[0]], mode='fit')
        r.stage(tmp_path / 'predict', [r.samples[1]])
        program = tmp_path / 'program'; program.mkdir()
        code = """from pathlib import Path
def predict(sample,state,context):
    assert not Path(GLOBAL_POOL).exists()
    assert 'human' not in sample and 'known' not in sample
    path=next(Path('/input/arrays').glob('*.npz'))
    try:
        with path.open('ab') as f: f.write(b'not permitted')
    except OSError:
        pass
    else:
        raise AssertionError('Input array writable')
    return {'score':sample['base'],'support':sample['valid']}
""".replace('GLOBAL_POOL', repr(str(r.array_pool.root)))
        with (program / 'pipeline.py').open('x') as stream:
            stream.write(code)
        receipt = r.sandbox.run(program, tmp_path / 'predict', tmp_path / 'output')
        assert receipt['status'] == 'ok', (tmp_path / 'output/stderr.log').read_text()
        assert receipt['field_storage']['predictions_reused'] is False
    finally:
        archive.close()


@pytest.mark.skipif(os.environ.get('PHYEVO_FIELD_GPU_TESTS') != '1',
                    reason='Requires explicit isolated GPU 3/4 test launch')
@pytest.mark.parametrize('gpu', [3, 4])
def test_gpu_input_decoding_matches_legacy_prediction_bytes(tmp_path, gpu):
    from .runtime_compat import ConfiguredSandbox
    from .system_state import SystemExecutor
    archive, original = fixture(tmp_path)
    try:
        legacy = SystemExecutor(original.workspace, archive,
            ConfiguredSandbox(original.workspace, original.workspace / '.venvs/sam2', timeout=90, gpu=gpu),
            original.samples)
        layered = FieldStorageExecutor(original.workspace, archive,
            FieldStorageSandbox(original.workspace, original.workspace / '.venvs/sam2', timeout=90, gpu=gpu),
            original.samples, tmp_path / 'pool', ('rot90',),
            max_pool_bytes=16*1024**2, minimum_free_bytes=0)
        program = tmp_path / 'program'; program.mkdir()
        code = '''import torch
def predict(sample,state,context):
    assert context['device']=='cuda'
    values=torch.as_tensor(sample['image'],device='cuda')
    assert values.is_cuda
    return {'score':(values*2).cpu().numpy(),'support':sample['valid']}
'''
        with (program / 'pipeline.py').open('x') as stream:
            stream.write(code)
        hashes = []
        for name, r in [('legacy', legacy), ('layered', layered)]:
            r.stage(tmp_path / (name + '_input'), r.samples[:2])
            output = tmp_path / (name + '_output')
            receipt = r.sandbox.run(program, tmp_path / (name + '_input'), output)
            assert receipt['status'] == 'ok', (output / 'stderr.log').read_text()
            hashes.append([file_hash(output / f'prediction-{i:04d}.npz') for i in range(2)])
        assert hashes[0] == hashes[1]
    finally:
        archive.close()
