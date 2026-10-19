"""Real sandbox integration of the new bank study, using synthetic images."""
from pathlib import Path

import numpy as np
import pytest

from . import bank_study as study
from .archive import Archive
from .contracts import Protocol, file_hash, write_new
from .executor import Sample
from .expansion_state import CompactSystemExecutor
from .real_study import ROOT
from .research_state_v2 import ResearchBank
from .runtime_compat import ConfiguredSandbox
from .system_state import read
from .test_research_state_v2 import answer
from .test_system_state import prepare_initial


def experiment(tmp_path, monkeypatch):
    groups = ('synthetic-training-source', 'synthetic-validation-source', 'synthetic-audit-source')
    protocol = Protocol((groups[0],), (groups[1],), (groups[2],), seeds=(23,))
    samples = []
    for i, group in enumerate(groups):
        image = np.full((8, 8), .2 * (i+1), np.float32)
        base = np.arange(64, dtype=np.float32).reshape(8, 8)
        valid = np.ones((8, 8), bool)
        human = np.zeros((8, 8), bool); human[2:4, 2:4] = True
        fields = {'image': image, 'base': base, 'valid': valid, 'support': valid,
            'physics': base, 'student': image, 'variation': image, 'physics_valid': valid}
        samples.append(Sample(f's{i}', group, 'interface', fields, human, human.astype(float), np.empty((0, 2))))
    sandbox = ConfiguredSandbox(ROOT, ROOT / '.venvs/sam2', timeout=45)
    config = {'rounds_per_arm': 3, 'minimum_free_disk_bytes': 0}
    def load(directory=None, arm=None, seed=None):
        archive = Archive(tmp_path / 'archive', protocol, create=not (tmp_path / 'archive').exists())
        runner = CompactSystemExecutor(ROOT, archive, sandbox, samples, tmp_path / 'pool')
        return tmp_path, config, tmp_path, archive, runner
    _, _, _, archive, runner = load()
    try:
        initial, receipt = prepare_initial(runner)
        bank = ResearchBank(runner); bank.initialize()
        write_new(tmp_path / 'prepared.json', {'baseline': initial, 'summary': receipt['summary'],
            'baseline_library': bank.inherited_manifest()['library'], 'checkpoint': bank.checkpoint()})
    finally:
        archive.close()
    monkeypatch.setattr(study, 'load', load)
    return initial, load


def test_new_baseline_uses_complete_admitted_system_not_stress_or_a_branch():
    def record(seed, values, stress):
        return {'arm': 'full', 'seed': seed, 'checkpoint': {'executable_system': str(seed)},
            'development': {'summary': {g: {'mean': v} for g, v in zip(('A', 'B', 'C'), values)}},
            'stress_audit': {'mean': stress}}
    best, all_options = study.choose_baseline({'trajectories': [
        record(23, (.99, .99, .0), 1.), record(41, (.7, .7, .7), .0)]})
    assert best['seed'] == 41 and len(all_options) == 2


def test_real_round_retains_unadmitted_program_and_exposes_it_to_next_request(tmp_path, monkeypatch):
    initial, load = experiment(tmp_path, monkeypatch)
    study.queue(tmp_path, 'full', 23)
    folder = tmp_path / 'requests/r01'
    request = read(folder / 'request.json')
    assert request['context']['next_branch_name'] == 'b2'
    assert request['context']['observable_feature_definitions']['physical_valid_fraction_of_image'].endswith('physics_valid.size')
    assert request['context']['anonymous_observations'][0]['positive_error_partition']['known_positive_pixels'] == 4
    code = "def predict(sample,state,context):\n return {'score':-sample['base'],'support':sample['valid']}\n"
    branch = {'name': 'b2', 'pipeline_code': code, 'training_steps': 0, 'parent_branch': None, 'replaces': None}
    raw = answer('create_analysis', parent=initial, new_branch=branch, active_branches=['b0', 'b2'],
        router_code="def route(s,b,x,c):\n return {'branch':'b2','reason':'synthetic experiment'}\n")
    write_new(folder / 'raw_response.json', raw)
    original_hash = file_hash(folder / 'raw_response.json')
    study.publish(folder, '/synthetic/independent-01')
    study.execute(tmp_path, 'full', 23)
    result = read(folder / 'result.json')
    assert result['status'] == 'retained_for_research', result
    assert result['research_bank_additions'] == ['b2'] and result['exact_replay']
    assert result['analysis_replay']['sha256']
    assert file_hash(folder / 'raw_response.json') == original_hash
    study.queue(tmp_path, 'full', 23)
    after = read(tmp_path / 'requests/r02/request.json')['context']
    assert after['current_system'] == initial and after['next_branch_name'] == 'b3'
    assert after['capability_library']['b2']['pipeline_code'] == code
    assert not after['capability_library']['b2']['active_in_incumbent']
    assert after['checkpoint'] == result['checkpoint']
    assert after['all_attempt_feedback'][0]['status'] == 'retained_for_research'
    assert after['prior_proposals'][0]['proposal'] == raw
    _, _, _, archive, _ = load()
    try:
        assert archive.incumbent() == initial
        assert len(archive.events(study.ROUND_EVENT)) == 1
        archive.verify()
    finally:
        archive.close()


def test_invalid_raw_reply_consumes_attempt_without_silent_repair(tmp_path, monkeypatch):
    initial, load = experiment(tmp_path, monkeypatch)
    study.queue(tmp_path, 'route', 23)
    folder = tmp_path / 'requests/r01'
    value = answer(parent=initial)
    value['system']['operation'] = 'route_only'  # forbidden duplicate semantic field
    write_new(folder / 'raw_response.json', value)
    raw_hash = file_hash(folder / 'raw_response.json')
    study.publish(folder, '/synthetic/invalid-01')
    study.execute(tmp_path, 'route', 23)
    result = read(folder / 'result.json')
    assert result['status'] == 'invalid_proposal' and not result['actual_execution']
    assert result['candidate'] is None and result['counted_against_proposal_budget']
    assert result['parent_checkpoint'] == result['checkpoint']
    assert file_hash(folder / 'raw_response.json') == raw_hash
    study.queue(tmp_path, 'route', 23)
    assert (tmp_path / 'requests/r02/request.json').is_file()


def test_analysis_replay_reexecutes_code_and_detects_case_order_dependency(tmp_path, monkeypatch):
    _, load = experiment(tmp_path, monkeypatch)
    _, _, _, archive, runner = load()
    try:
        code = '''counter=0
def predict(sample,state,context):
    global counter
    counter+=1
    return {'score':sample['base']+counter,'support':sample['valid']}
'''
        binding = runner.capability({'pipeline_code': code, 'training_steps': 0, 'parent_state': None})
        runner.predict_capability(binding, 'development')
        with pytest.raises(RuntimeError, match='case order'):
            study.analysis_replay(runner, binding)
        assert read(runner.checked_ref(binding))['checkpoint'] is None
    finally:
        archive.close()


def test_bootstrap_collects_measured_rejected_capability_not_unexecuted_files(tmp_path, monkeypatch):
    initial, load = experiment(tmp_path, monkeypatch)
    _, _, _, archive, runner = load()
    try:
        measured = runner.capability({'pipeline_code': "def predict(s,x,c):\n return {'score':1-s['base'],'support':s['valid']}\n",
            'training_steps': 0, 'parent_state': None})
        runner.predict_capability(measured, 'development')
        runner.capability({'pipeline_code': 'def predict(s,x,c):\n raise RuntimeError("not measured")\n',
            'training_steps': 0, 'parent_state': None})
        assert archive.incumbent() == initial
    finally:
        archive.close()
    monkeypatch.setattr(study.previous, 'load', load)
    extras, known = study.collect_known(tmp_path, {'arm': 'full', 'seed': 23, 'candidate': initial})
    assert len(extras) == 1 and len(known) == 3
    assert extras[0]['binding'] == measured


def test_positive_error_partition_distinguishes_support_and_ranking_with_exact_budget():
    from .evaluation import evaluate
    human = np.ones((10, 10), bool)
    valid = human.copy(); valid[9:] = False
    support = human.copy(); support[7:] = False
    score = np.zeros((10, 10), float); score[:5] = np.arange(50, 0, -1).reshape(5, 10)
    fields = {'image': score, 'valid': valid, 'support': support}
    record = study.positive_error_partition(fields, human, score, support, .1)
    assert record['fractions'] == {'outside_valid': .1, 'valid_outside_support': .2,
        'supported_nonpositive_score': .2, 'eligible_not_selected': .4, 'selected': .1}
    assert sum(record['fractions'].values()) == pytest.approx(1.)
    assert record['whole_image_candidate_budget'] == 10
    assert record['support_only_coverage_upper_bound'] == .1
    protocol = Protocol(('train',), ('development',), ('audit',), area_fraction=.1)
    measured = evaluate({**fields, 'human': human, 'geometry': 'interface'},
        {'score': score, 'support': support}, protocol)
    assert measured['coverage'] == record['fractions']['selected']
    empty = study.positive_error_partition(fields, ~human, score, support, .1)
    assert empty['known_positive_pixels'] == 0
    assert all(v is None for v in empty['fractions'].values())
    assert empty['support_only_coverage_upper_bound'] is None
    with pytest.raises(ValueError, match='aligned prediction'):
        study.positive_error_partition(fields, human, score, human, .1)
