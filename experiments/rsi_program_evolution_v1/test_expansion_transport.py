"""Invalid research attempts consume budget without changing scientific state."""
import json

import pytest

from .contracts import file_hash, write_new
from .expansion_transport import record_rejection, validation_error
from .system_state import read
from .system_study import checkpoint
from .test_expansion import envelope
from .test_system_state import fixture, prepare_initial


def reserved(tmp_path, invalid=True):
    archive, runner = fixture(tmp_path)
    identity, _ = prepare_initial(runner)
    state = runner.frozen_state(identity)
    context = {'arm': 'route', 'current_system': identity,
        'capability_library': state['library'], 'checkpoint': checkpoint(archive, runner)}
    folder = runner.root / 'requests/r01'
    folder.mkdir(parents=True)
    write_new(folder / 'request.json', {'context': context, 'model': 'synthetic'})
    write_new(folder / 'reservation.json', {'checkpoint': context['checkpoint'],
        'request_sha256': file_hash(folder / 'request.json')})
    value = envelope(parent=identity)
    if invalid:
        value['research_decision']['operation_target'] = 'analysis'
    write_new(folder / 'raw_response.json', value)
    return archive, runner, folder, context


def test_invalid_attempt_preserves_raw_and_incumbent_and_consumes_one_round(tmp_path):
    archive, runner, folder, context = reserved(tmp_path)
    try:
        raw_hash = file_hash(folder / 'raw_response.json')
        trials = len(archive.trials(visible_only=False))
        result = record_rejection(folder, '/independent/model', {'rounds_per_arm': 2}, archive, runner)
        assert result['status'] == 'invalid_proposal'
        assert 'Routing-only control' in result['error']
        assert result['checkpoint'] == context['checkpoint'] == checkpoint(archive, runner)
        assert result['actual_execution'] is False and result['candidate'] is None
        assert len(archive.trials(visible_only=False)) == trials
        assert file_hash(folder / 'raw_response.json') == raw_hash
        assert read(folder / 'response.json')['proposal'] == read(folder / 'raw_response.json')
        assert [r['body'] for r in archive.events('expansion_round_finished')] == [result]
        with pytest.raises(FileExistsError):
            record_rejection(folder, '/independent/model', {'rounds_per_arm': 2}, archive, runner)
        assert len(archive.events('expansion_round_finished')) == 1
        archive.verify()
    finally:
        archive.close()


def test_valid_proposal_cannot_be_skipped_as_transport_failure(tmp_path):
    archive, runner, folder, _ = reserved(tmp_path, invalid=False)
    try:
        with pytest.raises(ValueError, match='Valid proposals'):
            record_rejection(folder, '/independent/model', {'rounds_per_arm': 2}, archive, runner)
        assert not (folder / 'response.json').exists()
        assert not archive.events('expansion_round_finished')
    finally:
        archive.close()


def test_invalid_json_is_retained_as_failure_not_silently_repaired():
    value, error = validation_error('{not JSON', {})
    assert value is None and 'JSONDecodeError' in error
    raw = json.dumps(envelope())
    value, error = validation_error(raw, {'arm': 'route', 'current_system': 'a'*64,
        'capability_library': {'b0': {}, 'b1': {}}})
    assert value == json.loads(raw) and error is None
