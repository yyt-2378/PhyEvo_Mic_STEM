"""Shared modules and explicit-run configuration, using only synthetic inputs."""
from dataclasses import asdict

import numpy as np
import pytest

from . import composition_study as study
from .contracts import Protocol, engine_fingerprint, file_hash, write_new
from .executor import Sample, sample_fingerprint
from .evaluation import evaluate
from .study_io import load_study
from .training_trace import trace_check


def settings():
    return {'scope': 'new_run', 'source': 'inputs/parent',
            'source_diagnostics': 'inputs/parent/development.json',
            'physical_source': 'inputs/physical', 'runtime': 'runtime',
            'source_aliases': {'train': 'D0', 'validation-a': 'D1', 'validation-b': 'D2'}}


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'SCOPE', None)
    monkeypatch.setattr(study, 'SOURCE', None)
    monkeypatch.setattr(study, 'ALIASES', {})
    monkeypatch.setattr(study, 'RUN_CONFIGURATION', None)
    return tmp_path


def test_configuration_only_resolves_paths_and_has_no_asset_io(scoped):
    path = scoped / 'run.json'
    write_new(path, settings())
    config = study.configure_run(path)
    assert config == settings()
    assert study.SCOPE == scoped / 'new_run'
    assert study.SOURCE == scoped / 'inputs/parent'
    assert list(scoped.iterdir()) == [path]


@pytest.mark.parametrize('patch', [
    {'scope': '../outside'}, {'scope': 'inputs'}, {'scope': 'inputs/parent/output'},
    {'runtime': 'new_run/runtime'}, {'source_aliases': {'x': 'D0'}},
    {'undeclared': True}, {'source': ''},
])
def test_bad_configuration_is_rejected_before_state_or_file_changes(scoped, patch):
    path = scoped / 'bad.json'
    write_new(path, settings() | patch)
    with pytest.raises(ValueError):
        study.configure_run(path)
    assert study.SCOPE is None and study.RUN_CONFIGURATION is None
    assert list(scoped.iterdir()) == [path]


def test_cli_help_or_missing_configuration_never_reads_or_runs(scoped, capsys):
    with pytest.raises(SystemExit) as error:
        study.main(['--help'])
    assert error.value.code == 0
    assert '--run-config' in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        study.main(['execute'])
    assert error.value.code == 2
    assert not list(scoped.iterdir())
    with pytest.raises(ValueError, match='--run-config'):
        study.prepare()
    with pytest.raises(ValueError, match='--run-config'):
        study.validate_study()


def test_changed_run_configuration_cannot_resume(scoped):
    path = scoped / 'run.json'
    write_new(path, settings())
    study.configure_run(path)
    study.SCOPE.mkdir()
    write_new(study.SCOPE / 'prospective_protocol.json', {
        'run_configuration': settings() | {'source': 'different'},
        'workflow_binding': study.workflow_binding()})
    with pytest.raises(ValueError, match='Workflow/configuration changed'):
        study.validate_study()


def test_extracted_training_check_keeps_legacy_api():
    from .learning_study import trace_check as legacy
    from .runtime_compat import ConfiguredSandbox as old_sandbox
    from .configured_sandbox import ConfiguredSandbox
    assert legacy is trace_check
    assert old_sandbox is ConfiguredSandbox
    state = {'audit_initial_parameters': [0., 1.], 'audit_final_parameters': [1., 2.],
             'audit_loss_trace': [2., 1.], 'audit_steps': [2]}
    assert trace_check(state, 2)['parameter_delta_l2'] == pytest.approx(2**.5)
    with pytest.raises(ValueError, match='No measured parameter update'):
        trace_check(state | {'audit_final_parameters': [0., 1.]}, 2)


def test_shared_loader_preserves_arrays_and_rejects_changed_input(tmp_path, monkeypatch):
    from . import real_study
    monkeypatch.setattr(real_study, 'ROOT', tmp_path)
    folder = tmp_path / 'prepared'
    folder.mkdir()
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    human = image > 9
    sample = Sample('a', 'train', 'interface', {'image': image, 'valid': np.ones(image.shape, bool)},
                    human, human.astype(float), np.empty((0, 2)))
    path = folder / 'a.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, **sample.fields, human=sample.human,
                            known=sample.known, points_xy=sample.points_xy)
    protocol = Protocol(('train',), ('validation',), ('audit',),
                        engine_version=engine_fingerprint(), data_version=sample_fingerprint([sample]))
    write_new(folder / 'protocol.json', asdict(protocol))
    write_new(folder / 'dataset.json', {'samples': [{'key': 'a', 'group': 'train', 'geometry': 'interface',
              'file': 'a.npz', 'sha256': file_hash(path)}]})
    _, p, samples = load_study(folder, root=tmp_path)
    _, old_p, old_samples = real_study.load_study(folder)
    assert p == old_p == protocol
    assert sample_fingerprint(samples) == sample_fingerprint(old_samples) == protocol.data_version
    assert np.array_equal(samples[0].fields['image'], image)
    # Only the new synthetic fixture is changed, never an existing dataset.
    path.write_bytes(b'changed synthetic fixture')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_study(folder, root=tmp_path)


def test_configured_prepare_and_request_on_synthetic_bound_inputs(scoped):
    """Exercise real loading, metrics, protocol binding and request creation; no model call."""
    write_new(scoped / 'run.json', settings())
    study.configure_run(scoped / 'run.json')
    scope, source = study.SCOPE, study.SOURCE
    scope.mkdir()
    source.mkdir(parents=True)
    fields = scoped / 'inputs/prepared'
    fields.mkdir()
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    human = image >= 10
    samples = [Sample('case-' + g, g, 'interface',
                      {'image': image, 'valid': np.ones(image.shape, bool)},
                      human, human.astype(float), np.empty((0, 2)))
               for g in settings()['source_aliases']]
    protocol = Protocol(('train',), ('validation-a', 'validation-b'), ('audit',), seeds=(41,),
                        engine_version=engine_fingerprint(), data_version=sample_fingerprint(samples))
    write_new(fields / 'protocol.json', asdict(protocol))
    entries, gaps = [], []
    for name in ('original_output', 'rot90_output'):
        (scope / name).mkdir()
    for index, sample in enumerate(samples):
        path = fields / (sample.key + '.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, **sample.fields, human=sample.human,
                                known=sample.known, points_xy=sample.points_xy)
        entries.append({'key': sample.key, 'group': sample.group, 'geometry': sample.geometry,
                        'file': path.name, 'sha256': file_hash(path)})
        prediction = {'score': image + 1, 'support': np.ones(image.shape, bool)}
        score = evaluate(sample.validation_target(), prediction, protocol)
        gaps.append({'group': study.ALIASES[sample.group], 'current_coverage': score['coverage'],
                     'best_existing_branch_coverage': score['coverage'],
                     'candidate_budget_coverage_upper_bound': 1.})
        for mode in ('original', 'rot90'):
            with (scope / (mode + '_output') / f'prediction-{index:04d}.npz').open('xb') as stream:
                np.savez_compressed(stream, **{k: np.rot90(v) if mode == 'rot90' else v
                                              for k, v in prediction.items()})
    write_new(fields / 'dataset.json', {'samples': entries})
    write_new(source / 'protocol.json', {'evaluation': asdict(protocol),
              'fields': str(fields.relative_to(scoped)), 'fields_manifest_sha256': file_hash(fields / 'dataset.json'),
              'development_transforms': ['rot90'], 'held_back_transforms': ['flip_lr']})
    write_new(source / 'development.json', {'context': {'bank_gap_diagnosis': gaps,
              'anonymous_observations': {}, 'observable_feature_definitions': {},
              'observable_field_information': [{'group': g} for g in ('D0', 'D1', 'D2')],
              'all_attempt_feedback': []}})
    write_new(scope / 'parent_replay_verified_v2.json', {'status': 'verified_parent_ready_for_new_study',
              'rows': [{'mode': 'original', 'case': sample.key} for sample in samples]})
    program = scope / 'frozen_parent/program'
    program.mkdir(parents=True)
    (program / 'pipeline.py').write_text('def predict(s,w,c): return {}\n')
    write_new(program.parent / 'manifest.json', {'exported_hashes': {'pipeline.py': file_hash(program / 'pipeline.py')}})
    write_new(scope / 'protected_before.json', {})
    original_hash = file_hash(scope / 'original_output/prediction-0000.npz')
    study.prepare()
    config = study.validate_study()
    assert config['run_configuration'] == settings()
    assert config['workflow_binding'] == study.workflow_binding()
    study.request('full', 23)
    request = study.read(scope / 'full-s23/r01/request.json')
    assert request['evidence']['parent_node'] == 'frozen'
    assert len(request['evidence']['current_cases']) == 3
    assert file_hash(scope / 'original_output/prediction-0000.npz') == original_hash
    with pytest.raises(FileExistsError):
        study.prepare()  # Existing protocol is not overwritten.
