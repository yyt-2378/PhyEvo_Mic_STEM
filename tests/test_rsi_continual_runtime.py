import json

import numpy as np
import pytest

from test_rsi_runtime import Framework, physics, put
from test_rsi_continual import control_rows
from tfree_detection import rsi_continual_runtime as runtime
from tfree_detection.expert_supervision import normalize_image, sha256_file
from tfree_detection.rsi_continual import retention_gate


def bundle(root):
    training = root/'training'; training.mkdir()
    run = training/'fixed_replay'/'seed-23'/'outer-00'; run.mkdir(parents=True)
    part = {'train': [{'case_number': 1, 'rows': [0, 128]}], 'outer_group': 'heldout'}
    protocol = put(training/'protocol.json', {'format': 'phyevo-bounded-continual-development-v1',
        'folds': [{'fold_id': 'outer-00', 'partition': part}], 'arms': ['fixed_replay'], 'seeds': [23],
        'rounds': 1, 'gate': {'minimum_gain': .005, 'maximum_loss': .01}, 'source_hashes': {}})
    def seal(path, value):
        put(path, value); put(path.with_suffix('.seal.json'), {'sha256': sha256_file(path), 'inputs': {}})
    identity = {'arm': 'fixed_replay', 'seed': 23, 'fold_id': 'outer-00', 'protocol_sha256': sha256_file(protocol)}
    seal(training/'all_training_sealed.json', {'ready': True})
    seal(run/'training_complete.json', {**identity, 'rounds': 1})
    rd = run/'round-01'; rd.mkdir()
    routes, decisions = retention_gate(control_rows(), {}, 1)
    seal(rd/'feedback.json', {'incumbent': routes, 'decisions': decisions,
                             'control_rows': control_rows(), 'uses_outer_results': False})
    checkpoint = rd/'checkpoint'; checkpoint.mkdir()
    with (checkpoint/'state.npz').open('xb') as stream:
        np.savez(stream, placeholder=np.zeros(1))
    meta = {**identity, 'round': 1, 'train_partition': part['train'], 'train_groups': ['a'],
            'geometry_support': ['point'], 'state_sha256': sha256_file(checkpoint/'state.npz')}
    put(checkpoint/'checkpoint.json', meta)
    path = root/'binding.json'
    runtime.create_binding(root, training, run, path)
    return path, meta


def test_default_never_loads_optional_binding(tmp_path, monkeypatch):
    image = np.ones((32, 32))
    native = physics(image)
    monkeypatch.setattr(runtime, 'load_binding', lambda *a: pytest.fail('loaded optional model'))
    result = runtime.detect_with_continual(Framework(native), image, project_root=tmp_path, binding='missing')
    assert result.primary is native


def test_selected_route_uses_source_pixels_and_preserves_physics(tmp_path, monkeypatch):
    binding, meta = bundle(tmp_path)
    monkeypatch.setattr(runtime, 'restore_checkpoint', lambda *a, **k: (object(), None, None, meta))
    image = np.random.default_rng(5).integers(0, 65535, (32, 48), dtype=np.uint16)
    calls = []
    def predict(model, image, geometry, **kwargs):
        calls.append(image.copy())
        assert runtime.torch.are_deterministic_algorithms_enabled()
        assert not runtime.torch.backends.cuda.matmul.allow_tf32
        return image.copy()
    monkeypatch.setattr(runtime, 'predict_scores', predict)
    native = physics(image, nonempty=True)
    before = native.summary()
    result = runtime.detect_with_continual(Framework(native), image, project_root=tmp_path,
                                         binding=binding, experimental_research=True)
    assert result.metadata['status'] == 'continual_research', result.metadata
    assert result.primary_branch == 'student_expert'
    assert result.physics is native and native.summary() == before
    np.testing.assert_array_equal(calls[0], normalize_image(image))
    assert result.metadata['production_approved'] is False
    from tfree_detection.rsi_runtime import _save_comparison
    _save_comparison(result, tmp_path/'comparison.png')
    assert (tmp_path/'comparison.png').stat().st_size > 0


def test_changed_route_is_rejected_and_falls_back(tmp_path):
    binding, _ = bundle(tmp_path)
    value = json.loads(binding.read_text())
    value['routes']['point']['round'] = 0
    changed = put(tmp_path/'changed.json', value)
    with pytest.raises(ValueError, match='Routes differ'):
        runtime.load_binding(tmp_path, changed)
    image = np.ones((32, 32)); native = physics(image)
    result = runtime.detect_with_continual(Framework(native), image, project_root=tmp_path,
                                         binding=changed, experimental_research=True)
    assert result.primary is native and result.metadata['status'] == 'physics_fallback'


def test_changed_checkpoint_is_rejected(tmp_path):
    binding, _ = bundle(tmp_path)
    value = json.loads(binding.read_text())
    state = tmp_path/value['models']['point']/'state.npz'
    with state.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Changed continual'):
        runtime.load_binding(tmp_path, binding)
