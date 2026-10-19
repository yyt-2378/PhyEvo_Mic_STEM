"""Real isolated rounds: gain, stability admission, rejection, and inheritance."""
import numpy as np
import pytest

from . import stable_study as study
from .archive import Archive
from .contracts import Protocol, file_hash, write_new
from .executor import Sample
from .real_study import ROOT
from .research_state_v2 import ResearchBank
from .runtime_compat import ConfiguredSandbox
from .stability_state import StabilitySystemExecutor
from .system_state import read
from .test_research_state_v2 import answer
from .test_system_state import prepare_initial


def experiment(tmp_path, monkeypatch):
    groups = ('synthetic-training-source', 'synthetic-validation-source', 'synthetic-audit-source')
    protocol = Protocol((groups[0],), (groups[1],), (groups[2],), seeds=(23,), maximum_group_regression=.002)
    samples = []
    for index, group in enumerate(groups):
        image = np.full((8, 8), .2 * (index + 1), np.float32)
        base = np.arange(64, dtype=np.float32).reshape(8, 8)
        valid = np.ones((8, 8), bool); human = np.zeros((8, 8), bool); human[2:4, 2:4] = True
        fields = {'image': image, 'base': base, 'valid': valid, 'support': valid,
            'physics': human.astype(np.float32) + .001 * base, 'student': image,
            'variation': image, 'physics_valid': valid}
        samples.append(Sample(f's{index}', group, 'interface', fields, human, human.astype(float), np.empty((0, 2))))
    sandbox = ConfiguredSandbox(ROOT, ROOT / '.venvs/sam2', timeout=45)
    config = {'rounds_per_arm': 3, 'minimum_free_disk_bytes': 0, 'development_transforms': ['rot90']}
    def load(directory=None, arm=None, seed=None):
        archive = Archive(tmp_path / 'archive', protocol, create=not (tmp_path / 'archive').exists())
        runner = StabilitySystemExecutor(ROOT, archive, sandbox, samples, tmp_path / 'pool', ('rot90',))
        return tmp_path, config, tmp_path, archive, runner
    _, _, _, archive, runner = load()
    try:
        initial, receipt = prepare_initial(runner)
        bank = ResearchBank(runner); bank.initialize()
        write_new(tmp_path / 'prepared.json', {'baseline': initial, 'summary': receipt['summary'],
            'trial': next(t['id'] for t in archive.trials(initial) if t['split'] == 'validation'),
            'baseline_library': bank.inherited_manifest()['library'], 'checkpoint': bank.checkpoint(),
            'development_stability': study.probes(runner, initial), 'field_diagnostics': study.field_diagnostics(runner)})
    finally:
        archive.close()
    monkeypatch.setattr(study, 'load', load)
    return initial, load


def respond(folder, request, code, origin):
    context = request['context']; name = context['next_branch_name']
    branch = {'name': name, 'pipeline_code': code, 'training_steps': 0, 'parent_branch': None, 'replaces': None}
    raw = answer('create_analysis', parent=context['current_system'], new_branch=branch,
        active_branches=['b0', name],
        router_code=f"def route(s,b,x,c):\n return {{'branch':'{name}','reason':'synthetic observable field experiment'}}\n")
    write_new(folder / 'raw_response.json', raw)
    study.previous.publish(folder, origin)
    return file_hash(folder / 'raw_response.json')


def test_executed_improvements_are_accepted_and_actually_inherited_twice(tmp_path, monkeypatch):
    initial, load = experiment(tmp_path, monkeypatch)
    hashes, states = [], []
    codes = [
        "import numpy as np\ndef predict(s,x,c):\n return {'score':np.where(s['physics']>1.02,s['physics'],0),'support':s['valid']}\n",
        "def predict(s,x,c):\n return {'score':s['physics'],'support':s['valid']}\n"]
    for index, code in enumerate(codes, 1):
        study.queue(tmp_path, 'full', 23)
        folder = tmp_path / 'requests' / f'r{index:02d}'; request = read(folder / 'request.json')
        if states:
            assert request['context']['checkpoint'] == states[-1]
            assert request['context']['current_system'] == states[-1]['system']['executable_system']
        assert 'synthetic-audit-source' not in str(request)
        assert request['context']['observable_field_information']
        assert request['context']['current_development_stability']
        hashes.append(respond(folder, request, code, f'/synthetic/stable-{index}'))
        study.execute(tmp_path, 'full', 23)
        result = read(folder / 'result.json')
        assert result['status'] == 'accepted', result
        assert result['decision']['quality_decision']['accepted'] and result['decision']['stability_passed']
        assert result['exact_replay'] and result['analysis_replay']['sha256']
        assert all(v['mean'] == index * .5 for v in result['summary'].values())
        assert file_hash(folder / 'raw_response.json') == hashes[-1]
        states.append(result['checkpoint'])
    study.queue(tmp_path, 'full', 23)
    next_request = read(tmp_path / 'requests/r03/request.json')['context']
    assert next_request['checkpoint'] == states[-1]
    assert next_request['current_system'] != initial and len(next_request['experience']) == 2
    assert next_request['next_branch_name'] == 'b4'
    _, _, _, archive, runner = load()
    try:
        assert archive.incumbent() == states[-1]['system']['executable_system']
        assert len(archive.events(study.ROUND_EVENT)) == 2
        assert not archive.events('search_sealed')
        with pytest.raises(ValueError, match='sealed'):
            runner.execute(archive.incumbent(), split='audit')
        archive.verify()
    finally:
        archive.close()


def test_original_gain_with_rotation_failure_is_banked_but_never_deployed(tmp_path, monkeypatch):
    initial, _ = experiment(tmp_path, monkeypatch)
    study.queue(tmp_path, 'full', 23); folder = tmp_path / 'requests/r01'
    request = read(folder / 'request.json')
    # Deliberately invalid scientific strategy, useful only as a synthetic
    # stability-gate adversary: fixed coordinates fit the original label patch.
    code = "import numpy as np\ndef predict(s,x,c):\n v=np.zeros(s['image'].shape);v[2:4,2:4]=1\n return {'score':v,'support':s['valid']}\n"
    respond(folder, request, code, '/synthetic/coordinate-adversary')
    study.execute(tmp_path, 'full', 23)
    result = read(folder / 'result.json')
    assert result['status'] == 'retained_for_research', result
    assert result['decision']['quality_decision']['accepted']
    assert not result['decision']['stability_passed'] and len(result['decision']['stability_regressions']) == 2
    assert result['research_bank_additions'] == ['b2']
    assert result['checkpoint']['system']['executable_system'] == initial
    study.queue(tmp_path, 'full', 23)
    next_context = read(tmp_path / 'requests/r02/request.json')['context']
    assert next_context['current_system'] == initial
    assert next_context['capability_library']['b2']['pipeline_code'] == code
    assert not next_context['capability_library']['b2']['active_in_incumbent']
    expected = study.anonymize(result['decision'], {
        'synthetic-training-source': 'D0', 'synthetic-validation-source': 'D1'})
    assert next_context['all_attempt_feedback'][0]['decision'] == expected


def test_invalid_reply_keeps_raw_bytes_and_consumes_round(tmp_path, monkeypatch):
    initial, _ = experiment(tmp_path, monkeypatch)
    study.queue(tmp_path, 'route', 23); folder = tmp_path / 'requests/r01'
    raw = answer(parent=initial); raw['system']['operation'] = 'route_only'
    write_new(folder / 'raw_response.json', raw); before = file_hash(folder / 'raw_response.json')
    study.previous.publish(folder, '/synthetic/invalid')
    study.execute(tmp_path, 'route', 23)
    result = read(folder / 'result.json')
    assert result['status'] == 'invalid_proposal' and result['counted_against_proposal_budget']
    assert result['parent_checkpoint'] == result['checkpoint']
    assert file_hash(folder / 'raw_response.json') == before


def test_baseline_is_best_complete_system_not_chosen_after_stress():
    def record(seed, scores, stress):
        return {'arm': 'full', 'seed': seed, 'checkpoint': {'system': {'executable_system': str(seed)}},
            'development': {'summary': {g: {'mean': v} for g, v in zip('ABC', scores)}}, 'stress_audit': stress}
    best, candidates = study.choose_baseline({'trajectories': [record(23, (.8, .8, .8), 0.), record(41, (.99, .99, 0.), 1.)]})
    assert best['seed'] == 23 and len(candidates) == 2


def test_held_back_transforms_preserve_all_aligned_fields_without_adding_training(tmp_path, monkeypatch):
    _, load = experiment(tmp_path, monkeypatch)
    _, _, _, archive, runner = load()
    try:
        originals = runner.cohorts('development')
        transformed = study.held_back_samples(originals, ('flip_lr', 'rot180'))
        assert len(transformed) == 4
        assert transformed[0].key == 'held-flip_lr-s0'
        assert transformed[0].group == 'held-flip_lr-synthetic-training-source'
        assert np.array_equal(transformed[0].human, np.fliplr(originals[0].human))
        assert np.array_equal(transformed[-1].fields['physics'], np.rot90(originals[-1].fields['physics'], 2))
        assert [s.key for s in runner.samples] == ['s0', 's1', 's2']
    finally:
        archive.close()
