"""One explicitly bound attempt under unchanged actual storage caps.

An incomplete upfront prediction reservation is reported as such, never as a
passing strict preflight. The existing per-write, pool, disk and whole-study
guards remain active. Completion is not guaranteed; partial results are kept.
"""
import argparse
import fcntl
import json
from pathlib import Path

from . import input_reuse_continuation as prior
from .contracts import file_hash, write_new
from .field_storage import allocated_bytes, exact_child
from .research_state_v2 import ResearchBank, parse_action

bounded, study, POLICY = prior.bounded, prior.study, prior.POLICY
RUNTIME = 'storage_runtime_capped_attempt_v1'


def assess(plan, measured, pool_bytes, cohort_size, policy, *, allow_uncertain=False):
    remaining = policy['max_prediction_pool_bytes'] - pool_bytes
    worker_metadata = 65536 + cohort_size * 16384
    hard_gates = (measured['reason'] is None and plan['input_pool_fits']
                  and measured['growth_bytes'] + plan['planned_total_new_bytes']
                  <= policy['max_growth_bytes'])
    strict = bool(hard_gates and plan['prediction_pool_fits_expected_replay'])
    limited = bool(allow_uncertain and hard_gates and not strict
                   and plan.get('missing_predictions')
                   and plan.get('training_jobs') == 0
                   and plan.get('checkpoint_bytes_bound') == 0
                   and plan.get('planned_prediction_bytes', 0) > remaining
                   and remaining > worker_metadata)
    return {'strict_preflight_fits': strict, 'bounded_attempt_allowed': strict or limited,
            'uncertain_prediction_reservation': limited,
            'completion_space_guaranteed': False,
            'remaining_pool_bytes': remaining,
            'next_worker_metadata_reserve_bytes': worker_metadata,
            'unchanged_actual_storage_caps': True,
            'capacity_failure_is_not_a_scientific_falsification': True}


def verify_prior_refusals(directory, label, before, raw_hash):
    for name in ('storage_runtime_v1', 'storage_runtime_members_v1',
                 'storage_runtime_bounded_v1', 'storage_runtime_routed_v1',
                 'storage_runtime_input_reuse_v1'):
        path = directory / name / (label + '-volume.json')
        if path.exists():
            value = json.loads(path.read_text())
            if (value['fits'] or value['scientific_round_started']
                    or value['archive_tail_unchanged'] != before
                    or value['raw_proposal_sha256'] != raw_hash):
                raise ValueError('Prior attempt may have started or changed')


def binding():
    return prior.binding() | {'capped_attempt_continuation.py': file_hash(Path(__file__))}


def declaration(directory, proof):
    ancestor = directory / prior.RUNTIME / 'binding.json'
    value = json.loads(ancestor.read_text())
    if value != prior.declaration(directory, Path(value['preflight']['path'])):
        raise ValueError('Existing tested runtime changed')
    evidence = json.loads(proof.read_text()); tests = evidence['tests']
    if (evidence['implementations'] != binding() or evidence['policy'] != POLICY
            or tests['tests'] < 12 or tests['failures'] or tests['errors'] or tests['skipped']
            or file_hash(exact_child(study.ROOT, Path(tests['path']))) != tests['sha256']):
        raise ValueError('Passing limited-attempt and write-cap tests required')
    return {'implementations': binding(), 'policy': POLICY,
            'ancestor_sha256': file_hash(ancestor),
            'preflight': {'path': str(proof), 'sha256': file_hash(proof)},
            'scientific_protocol_sha256': file_hash(directory / 'protocol.json'),
            'shared_existing_pools': 'storage_runtime_members_v1',
            'scientific_criteria_unchanged': True, 'actual_storage_caps_unchanged': True,
            'reclamation_receipt_sha256': bounded.CREDIT_SHA,
            'whole_study_cap_is_monitored_not_an_os_quota': True}


def configure(directory, proof):
    expected = declaration(directory, proof)
    runtime = exact_child(directory, directory / RUNTIME); runtime.mkdir(exist_ok=True)
    path = runtime / 'binding.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Limited-attempt binding changed')
    else:
        write_new(path, expected)


def worker(directory, arm, seed, raw_proposal_sha256):
    runtime = directory / RUNTIME
    value = json.loads((runtime / 'binding.json').read_text())
    if value != declaration(directory, Path(value['preflight']['path'])):
        raise ValueError('Limited-attempt declaration changed')
    lock = directory.with_name(directory.name + f'.{arm}-{seed}.lock')
    with lock.open('a') as stream, prior.storage.installed(directory / 'storage_runtime_members_v1', POLICY):
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, _, episode, archive, runner = study.load(directory, arm, seed)
        try:
            number = len(archive.events(study.ROUND_EVENT)) + 1
            request = episode / 'requests' / f'r{number:02d}'
            before, raw_hash = archive.verify(), file_hash(request / 'raw_response.json')
            if raw_hash != raw_proposal_sha256 or (request / 'result.json').exists():
                raise ValueError('Only the explicitly bound, unfinished proposal may run')
            label = f'{arm}-s{seed}-r{number:02d}'
            verify_prior_refusals(directory, label, before, raw_hash)
            library = ResearchBank(runner).inherited_manifest()['library']
            proposal, _ = parse_action(json.loads((request / 'raw_response.json').read_text()),
                                       arm, archive.incumbent(), library)
            plan = prior.estimate(runner, proposal, library, POLICY)
            measured = bounded.budget(directory).check()
            size = max(len(runner.cohorts(split)) for split in ['development'] +
                       ['stability-' + t for t in runner.development_transforms])
            decision = assess(plan, measured, allocated_bytes(runner.sandbox.prediction_pool.root),
                              size, POLICY, allow_uncertain=True)
            assert archive.verify() == before
            report = runtime / (label + '-volume.json')
            write_new(report, {'plan': plan, 'storage': measured, 'decision': decision,
                'archive_tail_unchanged': before, 'raw_proposal_sha256': raw_hash,
                'scientific_round_started_at_preflight': False})
            print(json.dumps({'limited_attempt_preflight': str(report), **decision}), flush=True)
            if not decision['bounded_attempt_allowed']:
                raise RuntimeError('Insufficient bounded capacity; proposal remains pending')
        finally:
            archive.close()
        # Original execution, replay, admission, failure accounting and state inheritance.
        study.execute(directory, arm, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('guard', 'worker'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--action', choices=('execute',), required=True)
    parser.add_argument('--arm', choices=('full', 'route'), required=True)
    parser.add_argument('--seed', type=int, choices=(23, 41, 67), required=True)
    parser.add_argument('--raw-proposal-sha256', required=True)
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args(); directory = exact_child(study.ROOT, args.directory)
    if args.mode == 'worker':
        worker(directory, args.arm, args.seed, args.raw_proposal_sha256)
    else:
        if args.preflight is None:
            parser.error('--preflight required')
        configure(directory, exact_child(study.ROOT, args.preflight))
        command = bounded.member.old.inner_command(directory, args.action, args.arm, args.seed)
        command[command.index(bounded.member.old.__spec__.name)] = __spec__.name
        command += ['--raw-proposal-sha256', args.raw_proposal_sha256]
        result = bounded.budget(directory).run(command, timeout_seconds=3600, poll_seconds=1)
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['status'] == 'ok' else 1)
