"""One favorable final transform cannot hide another transform's regression."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from .test_bank_report import evidence
from . import stable_study as study, stable_report as reporting
from .contracts import file_hash, write_new
from .system_state import read
from .test_research_state_v2 import answer
from .test_stable_study import experiment, respond
from .stable_report import paired_stability_decision


def inputs():
    full, route, config, required, novelty = evidence()
    for index, row in enumerate(full['rounds']):
        row.update(candidate=f'candidate-{index}', actual_execution=True, novelty=deepcopy(novelty),
                   parent_checkpoint={'system': {'executable_system': f'candidate-{index-1}'}})
    config['held_back_transforms'] = ['flip_lr', 'rot180']
    for role in ('frozen', 'selected'):
        old = full['stress_audit'][role]['receipt']['summary']
        full['stress_audit'][role]['receipt']['summary'] = {
            f'held-{t}-' + g.removeprefix('stress-'): deepcopy(v)
            for t in config['held_back_transforms'] for g, v in old.items()}
    return full, route, config, required, novelty


def test_every_declared_transform_must_retain_its_own_original_and_old_system():
    args = inputs()
    assert paired_stability_decision(*args)['passed']
    args[0]['stress_audit']['selected']['receipt']['summary']['held-rot180-B']['mean'] = .6
    result = paired_stability_decision(*args)
    assert result['held_back_checks']['flip_lr']['passed']
    assert not result['held_back_checks']['rot180']['passed'] and not result['passed']


def test_missing_transform_or_new_capacity_cannot_pass():
    args = inputs()
    del args[0]['stress_audit']['selected']['receipt']['summary']['held-flip_lr-B']
    assert not paired_stability_decision(*args)['passed']
    args = inputs()
    args[-1]['mean_selected_new_gain_over_old_casewise_ceiling'] = dict.fromkeys('ABC', 0.)
    assert not paired_stability_decision(*args)['passed']


def test_gain_only_in_terminal_update_is_not_inherited_new_capacity():
    args = inputs()
    for row in args[0]['rounds'][:-1]:
        row['novelty']['mean_selected_new_gain_over_old_casewise_ceiling'] = dict.fromkeys('ABC', 0.)
    args[0]['rounds'][-1]['status'] = 'accepted'
    result = paired_stability_decision(*args)
    assert not result['admitted_new_capacity_inherited'] and not result['passed']


def test_verifier_recomputes_real_synthetic_rounds_and_binds_the_complete_chain(tmp_path, monkeypatch):
    loaders = {}
    for arm in ('route', 'full'):
        _, loaders[arm] = experiment(tmp_path / f'{arm}-s23', monkeypatch)
    def load(directory, arm, seed):
        return loaders[arm]()
    monkeypatch.setattr(study, 'load', load); monkeypatch.setattr(reporting, 'load', load)
    monkeypatch.setattr(study, 'SEEDS', (23,)); monkeypatch.setattr(reporting, 'SEEDS', (23,))
    for arm in ('route', 'full'):
        for index in range(1, 4):
            study.queue(tmp_path, arm, 23)
            folder = tmp_path / f'{arm}-s23/requests/r{index:02d}'
            request = read(folder / 'request.json')
            if arm == 'full' and index < 3:
                code = ("import numpy as np\ndef predict(s,x,c):\n return {'score':np.where(s['physics']>1.02,s['physics'],0),'support':s['valid']}\n"
                    if index == 1 else "def predict(s,x,c):\n return {'score':s['physics'],'support':s['valid']}\n")
                respond(folder, request, code, f'/synthetic/verified-{arm}-{index}')
            else:
                name = 'b3' if arm == 'full' else 'b0'
                raw = answer(parent=request['context']['current_system'], active_branches=['b0', name] if name != 'b0' else ['b0'],
                    observation=f'synthetic executed round {index}',
                    router_code=f"def route(s,b,x,c):\n return {{'branch':'{name}','reason':'synthetic retained choice'}}\n")
                write_new(folder / 'raw_response.json', raw)
                study.previous.publish(folder, f'/synthetic/verified-{arm}-{index}')
            study.execute(tmp_path, arm, 23)
    _, config, _, archive, _ = loaders['full']()
    protocol = archive.protocol; archive.close()
    # Explicit synthetic metadata for exercising the verifier, not a scientific
    # preflight certificate and never imported into a real study initialization.
    xml = tmp_path / 'synthetic-preflight.xml'
    xml.write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>')
    write_new(tmp_path / 'protocol.json', dict(config, evaluation=asdict(protocol),
        preflight_tests={'path': str(xml.relative_to(study.ROOT)), 'sha256': file_hash(xml)},
        synthetic_fixture_only=True))
    write_new(tmp_path / 'protected_before.json', {})
    study.audit(tmp_path)
    results = read(tmp_path / 'results.json')
    proof = reporting.verify(tmp_path, results)
    assert proof['independent_proposals'] == proof['recomputed_admission_decisions'] == 6
    assert proof['paired_initial_inputs'] == 1
    assert proof['recomputed_system_rows'] > 20 and proof['verified_reverse_analysis_rows'] == 4
    altered = deepcopy(results)
    altered['trajectories'][0]['rounds'][0]['status'] = 'accepted'
    with pytest.raises(ValueError, match='research attempts'):
        reporting.verify(tmp_path, altered)
