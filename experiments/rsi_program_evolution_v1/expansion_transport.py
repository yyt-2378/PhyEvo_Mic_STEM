"""Record invalid proposals as consumed attempts without repairing their answers.

Additive transport recovery: frozen scientific programs, criteria, requests and
raw responses are unchanged. No candidate code is executed by this module.
"""
import argparse
import fcntl
import json
from pathlib import Path

from .contracts import file_hash, inside, write_new
from .expansion_study import load, parse_proposal
from .real_study import ROOT
from .system_state import read
from .system_study import checkpoint


def validation_error(raw, context):
    value = None
    try:
        value = json.loads(raw)
        parse_proposal(value, context['arm'], context['current_system'], context['capability_library'])
    except (ValueError, TypeError, KeyError, AttributeError, SyntaxError) as error:
        return value, f'{type(error).__name__}: {error}'
    return value, None


def record_rejection(folder, origin, config, archive, runner):
    """Append exactly one failed round; valid proposals must use normal execution."""
    folder = inside(runner.workspace, folder)
    if not origin or not isinstance(origin, str):
        raise ValueError('Independent proposer origin required')
    if any((folder / name).exists() for name in ('response.json', 'response.ready.json', 'result.json')):
        raise FileExistsError('Existing transport evidence is preserved')
    rounds = archive.events('expansion_round_finished')
    index = len(rounds) + 1
    if index > config['rounds_per_arm'] or archive.events('search_sealed'):
        raise ValueError('Closed search')
    if folder != runner.root / 'requests' / f'r{index:02d}':
        raise ValueError('Only the next reserved round may be rejected')
    request = read(folder / 'request.json')
    reservation = read(folder / 'reservation.json')
    current = checkpoint(archive, runner)
    context = request['context']
    manifest = runner.frozen_state(archive.incumbent())
    if (reservation['checkpoint'] != current or context['checkpoint'] != current
            or reservation['request_sha256'] != file_hash(folder / 'request.json')
            or context['current_system'] != archive.incumbent()
            or set(context['capability_library']) != set(manifest['library'])):
        raise ValueError('Changed reservation or inherited system')
    raw = (folder / 'raw_response.json').read_text()
    value, error = validation_error(raw, context)
    if error is None:
        raise ValueError('Valid proposals cannot be rejected before scientific execution')
    response = {'raw_text': raw, 'proposal': value, 'origin': origin,
        'request_sha256': file_hash(folder / 'request.json'),
        'requested_model': request['model'], 'returned_model': None, 'usage': None}
    write_new(folder / 'response.json', response)
    write_new(folder / 'response.ready.json', {'response_sha256': file_hash(folder / 'response.json')})
    research = value.get('research_decision', {}) if isinstance(value, dict) else {}
    if not isinstance(research, dict):
        research = {}
    system = value.get('system', {}) if isinstance(value, dict) else {}
    if not isinstance(system, dict):
        system = {}
    result = {'round': index, 'origin': origin,
        'research_decision': {k: research.get(k, 'invalid_or_missing') for k in
            ('diagnosis', 'operation_target', 'evidence', 'alternatives', 'new_capacity_test')},
        'operation': system.get('operation', 'invalid_or_missing'),
        'parent_checkpoint': current, 'checkpoint': current,
        'request_sha256': response['request_sha256'],
        'response_sha256': file_hash(folder / 'response.json'),
        'candidate': None, 'status': 'invalid_proposal', 'error': error,
        'actual_execution': False, 'counted_against_proposal_budget': True,
        'raw_response_unchanged': True, 'transport_source_sha256': file_hash(Path(__file__))}
    write_new(folder / 'result.json', result)
    archive.event('expansion_round_finished', result)
    return result


def reject(directory, arm, seed, origin):
    directory = inside(ROOT, directory)
    lock_path = directory.with_name(directory.name + f'.{arm}-{seed}.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _, config, episode, archive, runner = load(directory, arm, seed)
        try:
            index = len(archive.events('expansion_round_finished')) + 1
            return record_rejection(episode / 'requests' / f'r{index:02d}', origin, config, archive, runner)
        finally:
            archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--arm', required=True, choices=('route', 'full'))
    parser.add_argument('--seed', required=True, type=int, choices=(23, 41, 67))
    parser.add_argument('--origin', required=True)
    args = parser.parse_args()
    print(json.dumps(reject(args.directory, args.arm, args.seed, args.origin)), flush=True)
