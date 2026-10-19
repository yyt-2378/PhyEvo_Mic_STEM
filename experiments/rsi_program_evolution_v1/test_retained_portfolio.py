"""Synthetic export integrity and actual isolated portfolio execution."""
import json
from pathlib import Path

import numpy as np
import pytest

from .contracts import digest, file_hash, write_new
from .retained_portfolio import export
from .runtime_compat import ConfiguredSandbox
from .replay_retained_portfolio import historical_route

ROOT = Path(__file__).resolve().parents[2]


def test_stress_route_reason_is_read_from_record_and_branch_cannot_disagree(tmp_path):
    write_new(tmp_path / 'route.json', {'branch': 'b1', 'reason': 'actual recorded reason'})
    observed = {}
    assert historical_route(tmp_path, 'route.json', 'b1', observed)['reason'] == 'actual recorded reason'
    assert observed['route.json'] == file_hash(tmp_path / 'route.json')
    with pytest.raises(ValueError, match='sealed selected branch'):
        historical_route(tmp_path, 'route.json', 'b0', {})


def fixture(tmp_path):
    episode = tmp_path / 'old'; episode.mkdir()
    candidate = 'a' * 64
    program = episode / 'archive/programs' / candidate
    program.mkdir(parents=True)
    capabilities = episode / 'capabilities'; capabilities.mkdir()
    library = {}
    for name, code in {
        'b0': "def predict(sample,state,context):\n return {'score':sample['base'],'support':sample['support']}\n",
        'b1': "def predict(sample,state,context):\n return {'score':sample['image']*state['factor'],'support':sample['valid']}\n",
    }.items():
        folder = capabilities / name; folder.mkdir()
        (folder / 'pipeline.py').write_text(code)
        checkpoint = None
        if name == 'b1':
            with (folder / 'state.npz').open('xb') as f:
                np.savez_compressed(f, factor=np.array(2., dtype='f4'))
            checkpoint = {'path': f'capabilities/{name}/state.npz', 'sha256': file_hash(folder / 'state.npz')}
        write_new(folder / 'capability.json', {
            'program': {'path': f'capabilities/{name}/pipeline.py', 'sha256': file_hash(folder / 'pipeline.py')},
            'checkpoint': checkpoint, 'weight_training_groups': ['synthetic-train'] if checkpoint else [],
        })
        library[name] = {'path': f'capabilities/{name}/capability.json', 'sha256': file_hash(folder / 'capability.json')}
    library['b2'] = {'path': 'discarded/missing.json', 'sha256': '0' * 64}
    router = "def route(sample,branches,state,context):\n return {'branch':'b1','reason':'retained synthetic weighted branch'}\n"
    (program / 'router.py').write_text(router)
    write_new(program / 'system_spec.json', {'active_branches': ['b0', 'b1'], 'library': library,
                                           'router_sha256': digest(router)})
    write_new(program / 'system_state.json', {'candidate': candidate, 'router_checkpoint': None,
              'specification_sha256': file_hash(program / 'system_spec.json')})
    write_new(episode / 'accepted.json', {'candidate': candidate, 'status': 'accepted',
              'actual_execution': True, 'decision': {'accepted': True}})
    return episode, file_hash(episode / 'accepted.json'), candidate


def test_export_preserves_code_weights_and_skips_discarded_inactive_programs(tmp_path):
    episode, accepted_hash, _ = fixture(tmp_path)
    before = {str(p.relative_to(episode)): file_hash(p) for p in episode.rglob('*') if p.is_file()}
    target = tmp_path / 'export'
    result = export(tmp_path, episode, 'accepted.json', accepted_hash, target)
    assert result['inactive_branch_records_not_resurrected'] == ['b2']
    assert result['status'] == 'exported_pending_numerical_replay'
    assert not result['new_scientific_gain']
    assert file_hash(target / 'program/b1.npz') == file_hash(episode / 'capabilities/b1/state.npz')
    assert before == {str(p.relative_to(episode)): file_hash(p) for p in episode.rglob('*') if p.is_file()}
    assert not (target / 'program/b2.py').exists()


def test_wrong_accepted_hash_and_copy_limit_fail_before_creating_destination(tmp_path):
    episode, h, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='hash mismatch'):
        export(tmp_path, episode, 'accepted.json', '0' * 64, tmp_path / 'bad_hash')
    with pytest.raises(ValueError, match='storage ceiling'):
        export(tmp_path, episode, 'accepted.json', h, tmp_path / 'too_large', maximum_copy_bytes=1)
    assert not (tmp_path / 'bad_hash').exists() and not (tmp_path / 'too_large').exists()


def test_source_escape_and_unauthorized_source_are_rejected(tmp_path):
    episode, h, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        export(tmp_path, episode, '../../outside.json', h, tmp_path / 'escape')
    another = tmp_path / 'other_workspace'; another.mkdir()
    with pytest.raises(ValueError, match='authorized'):
        export(another, episode, 'accepted.json', h, another / 'escape')


def test_existing_destination_is_not_overwritten(tmp_path):
    episode, h, _ = fixture(tmp_path)
    destination = tmp_path / 'existing'; destination.mkdir()
    with pytest.raises(FileExistsError):
        export(tmp_path, episode, 'accepted.json', h, destination)
    assert list(destination.iterdir()) == []


def test_changed_active_program_is_rejected_before_export(tmp_path):
    episode, h, _ = fixture(tmp_path)
    (episode / 'capabilities/b1/pipeline.py').write_text('raise RuntimeError("changed fixture")\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        export(tmp_path, episode, 'accepted.json', h, tmp_path / 'bad_active')
    assert not (tmp_path / 'bad_active').exists()


def test_real_bubblewrap_lazy_execution_preserves_values_routes_and_order(tmp_path):
    episode, h, _ = fixture(tmp_path)
    target = tmp_path / 'export'
    export(tmp_path, episode, 'accepted.json', h, target)
    expected = {i: np.arange(64, dtype='f4').reshape(8, 8) + i for i in range(3)}
    for mode, order in [('normal', [0, 1, 2]), ('reverse', [2, 1, 0])]:
        inputs = tmp_path / (mode + '_input'); inputs.mkdir()
        records = []
        for j, i in enumerate(order):
            path = inputs / f'sample-{j:04d}.npz'
            with path.open('xb') as f:
                np.savez_compressed(f, image=expected[i], base=expected[i] * 0,
                                    valid=np.ones((8, 8), bool), support=np.ones((8, 8), bool))
            records.append({'file': path.name})
        write_new(inputs / 'request.json', {'mode': 'predict', 'seed': 41, 'training_steps': 0,
                  'samples': records, 'state_file': False, 'device': 'cpu'})
        output = tmp_path / (mode + '_output')
        result = ConfiguredSandbox(ROOT, ROOT / '.venvs/sam2', timeout=30).run(target / 'program', inputs, output)
        assert result['status'] == 'ok', (output / 'stderr.log').read_text()
        for j, i in enumerate(order):
            with np.load(output / f'prediction-{j:04d}.npz', allow_pickle=False) as z:
                assert np.array_equal(z['score'], expected[i] * 2)
                assert z['support'].all()
            assert json.loads((output / f'route-{j:04d}.json').read_text())['branch'] == 'b1'
            assert json.loads((output / f'computed-{j:04d}.json').read_text())['computed'] == ['b1']
