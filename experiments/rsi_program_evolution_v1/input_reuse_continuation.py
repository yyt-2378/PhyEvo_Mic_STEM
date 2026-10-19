"""Explicit input-reference transport; unchanged RSI proposals and admission."""
import argparse
import fcntl
import json
from pathlib import Path

from . import routed_continuation as routed
from . import input_reuse_storage as storage
from .contracts import file_hash, write_new
from .field_storage import allocated_bytes, exact_child
from .research_state_v2 import ResearchBank, parse_action

bounded, study, POLICY = routed.bounded, routed.study, routed.POLICY
RUNTIME = 'storage_runtime_input_reuse_v1'


def estimate(runner, proposal, library, policy):
    plan = routed.estimate(runner, proposal, library, policy)
    saving = plan['uncatalogued_existing_router_pairs_reservation']
    planned = plan['planned_prediction_bytes'] - saving
    return plan | {'prior_route_storage_plan': plan['planned_prediction_bytes'],
        'input_pair_payload_reservation_removed': saving, 'planned_prediction_bytes': planned,
        'planned_total_new_bytes': plan['planned_total_new_bytes'] - saving,
        'prediction_pool_fits_expected_replay': allocated_bytes(runner.sandbox.prediction_pool.root) + planned <= policy['max_prediction_pool_bytes'],
        'planning_assumption': 'Exact computed router bytes may reference same-case staged input pairs; unknown branch reserves and actual storage caps unchanged'}


def binding():
    return routed.binding() | {n: file_hash(Path(__file__).with_name(n)) for n in
                              ('input_reuse_storage.py', 'input_reuse_continuation.py')}


def declaration(directory, proof):
    ancestor = directory / 'storage_runtime_routed_v1/binding.json'
    value = json.loads(ancestor.read_text())
    if value != routed.declaration(directory, Path(value['preflight']['path'])):
        raise ValueError('Previous tested runtime changed')
    evidence = json.loads(proof.read_text()); tests = evidence['tests']
    if (evidence['implementations'] != binding() or evidence['policy'] != POLICY
            or tests['tests'] < 20 or tests['failures'] or tests['errors'] or tests['skipped']
            or file_hash(exact_child(study.ROOT, Path(tests['path']))) != tests['sha256']):
        raise ValueError('Passing input-reuse and original runtime tests required')
    return {'implementations': binding(), 'policy': POLICY, 'ancestor_sha256': file_hash(ancestor),
        'preflight': {'path': str(proof), 'sha256': file_hash(proof)},
        'scientific_protocol_sha256': file_hash(directory / 'protocol.json'),
        'shared_existing_pools': 'storage_runtime_members_v1', 'scientific_criteria_unchanged': True,
        'reclamation_receipt_sha256': bounded.CREDIT_SHA, 'old_files_replaced': 0}


def configure(directory, proof):
    expected = declaration(directory, proof)
    runtime = exact_child(directory, directory / RUNTIME); runtime.mkdir(exist_ok=True)
    path = runtime / 'binding.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Input-reuse binding changed')
    else:
        write_new(path, expected)


def worker(directory, action, arm, seed):
    runtime = directory / RUNTIME
    record = json.loads((runtime / 'binding.json').read_text())
    if record != declaration(directory, Path(record['preflight']['path'])):
        raise ValueError('Input-reuse declaration changed')
    lock = directory.with_name(directory.name + f'.{arm}-{seed}.lock')
    with lock.open('a') as stream, storage.installed(directory / 'storage_runtime_members_v1', POLICY):
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == 'queue':
            study.queue(directory, arm, seed)
            return
        _, _, episode, archive, runner = study.load(directory, arm, seed)
        try:
            number = len(archive.events(study.ROUND_EVENT)) + 1
            request = episode / 'requests' / f'r{number:02d}'
            before, raw_hash = archive.verify(), file_hash(request / 'raw_response.json')
            for name in ('storage_runtime_v1', 'storage_runtime_members_v1', 'storage_runtime_bounded_v1', 'storage_runtime_routed_v1'):
                prior = directory / name / f'{arm}-s{seed}-r{number:02d}-volume.json'
                if prior.exists():
                    value = json.loads(prior.read_text())
                    if (value['fits'] or value['scientific_round_started'] or value['archive_tail_unchanged'] != before
                            or value['raw_proposal_sha256'] != raw_hash):
                        raise ValueError('Prior attempt may have started or changed')
            library = ResearchBank(runner).inherited_manifest()['library']
            try:
                proposal, _ = parse_action(json.loads((request / 'raw_response.json').read_text()), arm, archive.incumbent(), library)
            except (ValueError, TypeError, KeyError, AttributeError, SyntaxError):
                plan = {'input_pool_fits': True, 'prediction_pool_fits_expected_replay': True,
                    'planned_total_new_bytes': POLICY['metadata_reserve_bytes'], 'scientific_parser_will_decide': True}
            else:
                plan = estimate(runner, proposal, library, POLICY)
            measured = bounded.budget(directory).check()
            fits = (not measured['reason'] and plan['input_pool_fits'] and plan['prediction_pool_fits_expected_replay']
                and measured['growth_bytes'] + plan['planned_total_new_bytes'] <= POLICY['max_growth_bytes'])
            assert archive.verify() == before
            report = runtime / f'{arm}-s{seed}-r{number:02d}-volume.json'
            write_new(report, {'plan': plan, 'storage': measured, 'fits': fits,
                'archive_tail_unchanged': before, 'raw_proposal_sha256': raw_hash, 'scientific_round_started': False})
            print(json.dumps({'volume_preflight': str(report), 'fits': fits}), flush=True)
            if not fits:
                raise RuntimeError('Input-reference capacity preflight refused; proposal remains pending')
        finally:
            archive.close()
        study.execute(directory, arm, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('guard', 'worker'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--action', choices=('queue', 'execute'), required=True)
    parser.add_argument('--arm', choices=('full', 'route'), required=True)
    parser.add_argument('--seed', type=int, choices=(23, 41, 67), required=True)
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args(); directory = exact_child(study.ROOT, args.directory)
    if args.mode == 'worker':
        worker(directory, args.action, args.arm, args.seed)
    else:
        if args.preflight is None:
            parser.error('--preflight required')
        configure(directory, exact_child(study.ROOT, args.preflight))
        command = bounded.member.old.inner_command(directory, args.action, args.arm, args.seed)
        command[command.index(bounded.member.old.__spec__.name)] = __spec__.name
        result = bounded.budget(directory).run(command, timeout_seconds=3600, poll_seconds=1)
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['status'] == 'ok' else 1)
