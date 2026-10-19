"""Small synthetic tests: real compute remains, identical bytes share storage."""
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from .contracts import file_hash
from .field_storage import allocated_bytes, new_json
from .field_storage_runtime import FieldStorageExecutor
from .prediction_storage import PredictionPool, PredictionWriter
from .prediction_storage_runtime import SharedPredictionSandbox
from .test_system_state import fixture
from . import test_field_storage as field_tests


def pool(tmp_path, budget=16*1024**2, reserve=0):
    return PredictionPool(tmp_path, tmp_path / 'predictions', max_pool_bytes=budget, minimum_free_bytes=reserve)


def emit(tmp_path, payloads, known=None, budget=1024**2):
    tmp_path.mkdir()
    writer = PredictionWriter({'enabled': True, 'known_hashes': known or {},
                              'max_prediction_bytes': 1024**2, 'max_written_prediction_bytes': budget},
                              tmp_path, Path.open)
    for i, payload in enumerate(payloads):
        with writer.open(tmp_path / f'prediction-{i:04d}.npz', 'xb') as stream:
            stream.write(payload)
    writer.finish()
    new_json(tmp_path / 'result.json', {'predictions': [{'index': i, 'file': f'prediction-{i:04d}.npz'}
                                                      for i in range(len(payloads))]})


def test_first_local_and_later_cached_predictions_are_hardlinks(tmp_path):
    store = pool(tmp_path)
    key, payload = 'a'*64, b'synthetic content; no production data'
    first, second = tmp_path / 'first', tmp_path / 'second'
    emit(first, [payload, payload])
    one = store.finalize(first, [key, key], {})
    assert one['new_canonical_objects'] == 1 and one['hardlinked_new_paths'] == 1
    known = store.snapshot([key])
    assert store.snapshot(['b'*64]) == {}
    emit(second, [payload, payload], known)
    two = store.finalize(second, [key, key], known)
    assert two['new_prediction_payload_bytes'] == 0 and two['hardlinked_new_paths'] == 2
    assert os.path.samefile(first / 'prediction-0000.npz', second / 'prediction-0001.npz')
    assert len(list((store.root / 'objects').glob('*.npz'))) == 1
    assert one['existing_files_replaced'] == two['existing_files_replaced'] == 0


def test_changed_prediction_adds_object_and_preserves_previous_bytes(tmp_path):
    store = pool(tmp_path); key = 'a'*64
    first, second = tmp_path / 'first', tmp_path / 'second'
    emit(first, [b'first']); store.finalize(first, [key], {})
    known = store.snapshot([key]); before = file_hash(first / 'prediction-0000.npz')
    emit(second, [b'second'], known); result = store.finalize(second, [key], known)
    assert result['new_canonical_objects'] == 1 and result['hardlinked_new_paths'] == 0
    assert file_hash(first / 'prediction-0000.npz') == before


def test_unexposed_duplicate_is_retained_not_replaced(tmp_path):
    store = pool(tmp_path); key = 'a'*64
    for i in range(2):
        output = tmp_path / str(i); emit(output, [b'concurrent first write'])
        result = store.finalize(output, [key], {})
    assert result['concurrent_or_unexposed_duplicates_retained'] == 1
    assert not os.path.samefile(tmp_path / '0/prediction-0000.npz', tmp_path / '1/prediction-0000.npz')


def test_unknown_cached_reference_fails_closed(tmp_path):
    store = pool(tmp_path); payload = b'unknown'; digest = hashlib.sha256(payload).hexdigest()
    output = tmp_path / 'out'; emit(output, [payload], {digest: len(payload)})
    with pytest.raises(ValueError, match='not visible'):
        store.finalize(output, ['a'*64], {})
    assert not (output / 'prediction-0000.npz').exists()


def test_missing_canonical_is_not_treated_as_success(tmp_path):
    store = pool(tmp_path); payload = b'missing'; digest = hashlib.sha256(payload).hexdigest()
    known = {digest: len(payload)}; output = tmp_path / 'out'; emit(output, [payload], known)
    with pytest.raises(FileNotFoundError):
        store.finalize(output, ['a'*64], known)


@pytest.mark.parametrize('reserve,budget', [(0, 65536), (2**63, 1024**2)])
def test_pool_capacity_or_reserve_rejects_catalog_growth(tmp_path, reserve, budget):
    store = pool(tmp_path, budget=budget, reserve=reserve)
    output = tmp_path / 'out'; emit(output, [b'x'*70000])
    with pytest.raises(OSError):
        store.finalize(output, ['a'*64], {})
    assert (output / 'prediction-0000.npz').exists()  # Failed new work remains auditable.
    assert not list((store.root / 'objects').glob('*.npz'))


def test_payload_budget_and_memory_bound_prevent_prediction_write(tmp_path):
    with pytest.raises(OSError, match='before write'):
        emit(tmp_path / 'small', [b'12345'], budget=4)
    assert not list((tmp_path / 'small').glob('prediction-*.npz'))
    config = {'enabled': True, 'known_hashes': {}, 'max_prediction_bytes': 4, 'max_written_prediction_bytes': 10}
    writer = PredictionWriter(config, tmp_path, Path.open)
    with pytest.raises(OSError, match='memory'):
        with writer.open(tmp_path / 'prediction-0000.npz', 'xb') as stream:
            stream.write(b'12345')
    assert not (tmp_path / 'prediction-0000.npz').exists()


def test_npz_serialization_is_byte_identical_to_legacy(tmp_path):
    arrays = {'score': np.arange(64, dtype='f4').reshape(8, 8), 'support': np.ones((8, 8), bool)}
    legacy = io.BytesIO(); np.savez_compressed(legacy, **arrays)
    config = {'enabled': True, 'known_hashes': {}, 'max_prediction_bytes': 65536,
              'max_written_prediction_bytes': 65536}
    writer = PredictionWriter(config, tmp_path, Path.open)
    path = tmp_path / 'prediction-0000.npz'
    with writer.open(path, 'xb') as stream:
        np.savez_compressed(stream, **arrays)
    assert path.read_bytes() == legacy.getvalue()
    with pytest.raises(FileExistsError):
        writer.open(path, 'xb')


def shared_runner(tmp_path, gpu=None, write_budget=16*1024**2):
    archive, original = fixture(tmp_path)
    store = PredictionPool(original.workspace, tmp_path / 'predictions',
                           max_pool_bytes=16*1024**2, minimum_free_bytes=0)
    sandbox = SharedPredictionSandbox(original.workspace, original.workspace / '.venvs/sam2',
                                      timeout=90, gpu=gpu, prediction_pool=store,
                                      max_written_prediction_bytes=write_budget)
    result = FieldStorageExecutor(original.workspace, archive, sandbox, original.samples,
                                  tmp_path / 'inputs_pool', ('rot90',), max_pool_bytes=16*1024**2,
                                  minimum_free_bytes=0)
    return archive, result


CODE = '''from pathlib import Path
def predict(sample,state,context):
    assert 'human' not in sample and 'known' not in sample
    assert not Path(GLOBAL_POOL).exists()
    with Path('/work/actually_computed.log').open('a') as stream: stream.write('computed\\n')
    return {'score':sample['base'],'support':sample['valid']}
'''


def test_real_inference_reexecutes_but_second_payload_is_not_written(tmp_path):
    archive, r = shared_runner(tmp_path)
    try:
        program = tmp_path / 'program'; program.mkdir()
        with (program / 'pipeline.py').open('x') as stream:
            stream.write(CODE.replace('GLOBAL_POOL', repr(str(r.sandbox.prediction_pool.root))))
        receipts = []
        for name in ('first', 'second'):
            inputs, output = tmp_path / (name + '_input'), tmp_path / (name + '_output')
            r.stage(inputs, r.samples[:2])
            result = r.sandbox.run(program, inputs, output)
            assert result['status'] == 'ok', (output / 'stderr.log').read_text()
            assert (output / 'actually_computed.log').read_text().splitlines() == ['computed']*2
            receipts.append(result['prediction_storage']['sharing'])
        assert receipts[0]['new_prediction_payload_bytes'] > 0
        assert receipts[1]['new_prediction_payload_bytes'] == 0
        assert os.path.samefile(tmp_path / 'first_output/prediction-0000.npz',
                               tmp_path / 'second_output/prediction-0001.npz')
    finally:
        archive.close()


def test_real_worker_output_budget_failure_does_not_produce_fake_prediction(tmp_path):
    archive, r = shared_runner(tmp_path, write_budget=1)
    try:
        program = tmp_path / 'program'; program.mkdir()
        with (program / 'pipeline.py').open('x') as stream:
            stream.write(CODE.replace('GLOBAL_POOL', repr(str(r.sandbox.prediction_pool.root))))
        r.stage(tmp_path / 'input', r.samples[:1])
        result = r.sandbox.run(program, tmp_path / 'input', tmp_path / 'output')
        assert result['status'] == 'failed'
        assert 'budget exceeded before write' in (tmp_path / 'output/stderr.log').read_text()
        assert not list((tmp_path / 'output').glob('prediction-*.npz'))
        assert result['prediction_storage']['sharing'] is None
    finally:
        archive.close()


def test_shared_output_system_replay_rotation_matches_legacy(tmp_path, monkeypatch):
    original = field_tests.runner
    monkeypatch.setattr(field_tests, 'runner', lambda folder, flag: shared_runner(folder) if flag else original(folder, flag))
    field_tests.test_real_whole_system_replay_and_rotation_match_legacy_bytes(tmp_path)


def test_shared_output_fit_and_actual_parent_continuation_match_legacy(tmp_path, monkeypatch):
    original = field_tests.runner
    monkeypatch.setattr(field_tests, 'runner', lambda folder, flag: shared_runner(folder) if flag else original(folder, flag))
    field_tests.test_actual_fit_and_frozen_state_match_legacy(tmp_path)


@pytest.mark.skipif(os.environ.get('PHYEVO_FIELD_GPU_TESTS') != '1', reason='Explicit isolated GPU 3/4 launch required')
@pytest.mark.parametrize('gpu', [3, 4])
def test_real_gpu_shared_output_equals_original_worker(tmp_path, gpu):
    from .runtime_compat import ConfiguredSandbox
    from .system_state import SystemExecutor
    archive, r = shared_runner(tmp_path, gpu=gpu)
    try:
        legacy = SystemExecutor(r.workspace, archive, ConfiguredSandbox(r.workspace, r.workspace / '.venvs/sam2',
                                                                       timeout=90, gpu=gpu), r.samples)
        program = tmp_path / 'program'; program.mkdir()
        with (program / 'pipeline.py').open('x') as stream:
            stream.write("import torch\ndef predict(sample,state,context):\n"
                         "    x=torch.as_tensor(sample['image'],device='cuda')\n"
                         "    assert x.is_cuda and context['device']=='cuda'\n"
                         "    return {'score':(x*2).cpu().numpy(),'support':sample['valid']}\n")
        hashes = []
        for name, executor in [('legacy', legacy), ('shared1', r), ('shared2', r)]:
            executor.stage(tmp_path / (name + '_input'), executor.samples[:2])
            output = tmp_path / (name + '_output')
            result = executor.sandbox.run(program, tmp_path / (name + '_input'), output)
            assert result['status'] == 'ok', (output / 'stderr.log').read_text()
            hashes.append([file_hash(output / f'prediction-{i:04d}.npz') for i in range(2)])
        assert hashes[0] == hashes[1] == hashes[2]
        assert result['prediction_storage']['sharing']['new_prediction_payload_bytes'] == 0
    finally:
        archive.close()
