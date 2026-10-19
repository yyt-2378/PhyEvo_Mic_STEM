"""Serial continuation with reclamation-aware accounting and bounded writes.

Frozen science and old storage records are unchanged. Replay is genuinely
executed. Byte-identical replay is planned to share storage, not required for
scientific acceptance: the original numeric-array equality check still decides.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
from pathlib import Path

from . import member_continuation as member
from .contracts import file_hash, write_new
from .field_storage import allocated_bytes, exact_child
from .member_storage import MemberSandbox
from .research_state_v2 import ResearchBank, parse_action
from .storage_budget import StorageBudget

study, POLICY = member.study, member.POLICY
CREDIT = study.ROOT / 'outputs/rsi_v4_storage_migration_20261017_v1/prediction_sharing_completed.json'
CREDIT_SHA = '705b5d0f797c34d3ba675a0773ca53ed9d2f0ad150bd6d04068fe54eeeddb428'


class ReclaimAwareBudget(StorageBudget):
    def __init__(self, workspace, scope, *, reclamation_receipt, receipt_sha256, **kwargs):
        self.credit_path = exact_child(workspace, reclamation_receipt)
        self.credit_sha = receipt_sha256
        if file_hash(self.credit_path) != self.credit_sha:
            raise ValueError('Reclamation receipt changed')
        credit = json.loads(self.credit_path.read_text())
        amount = credit.get('reclaimed_allocated_bytes')
        if (credit.get('status') != 'completed' or type(amount) is not int or amount <= 0
                or credit.get('changed_contents') != 0 or credit.get('missing_paths') != 0
                or credit.get('reclaimed_space_is_not_additional_experiment_budget') is not True):
            raise ValueError('Verified lossless reclamation receipt required')
        self.reclaimed = amount
        super().__init__(workspace, scope, **kwargs)
        if self.reclaimed > self.policy['baseline_allocated_bytes']:
            raise ValueError('Reclamation exceeds the original experiment baseline')

    def check(self):
        measured = super().check()
        if file_hash(self.credit_path) != self.credit_sha:
            raise ValueError('Reclamation receipt changed during execution')
        # Add credit BEFORE clipping: adding it to the old clipped growth would
        # incorrectly charge the whole reclaimed 8.81 GiB as new experiment work.
        growth = max(0, measured['allocated_bytes'] + self.reclaimed - self.policy['baseline_allocated_bytes'])
        excess = max(0, growth - self.policy['max_growth_bytes'])
        reason = 'growth_limit' if excess else (
            'disk_reserve' if measured['available_bytes'] < self.policy['minimum_free_bytes'] else None)
        return measured | {'net_growth_before_reclamation_adjustment': measured['growth_bytes'],
            'verified_reclaimed_bytes_added_back': self.reclaimed,
            'reclamation_receipt_sha256': self.credit_sha,
            'growth_bytes': growth, 'over_budget_bytes': excess, 'reason': reason}


class RemainingCapacitySandbox(MemberSandbox):
    """Used only under the whole-study serial execution lock."""
    def run(self, program, inputs, output):
        inputs = exact_child(self.workspace, inputs)
        output = exact_child(self.workspace, output)
        request = json.loads((inputs / 'request.json').read_text())
        if request['mode'] != 'predict':
            return MemberSandbox.run(self, program, inputs, output)
        remaining = self.prediction_pool.guard.policy['max_pool_bytes'] - allocated_bytes(self.prediction_pool.root)
        # Reserve allocated-block rounding and per-case reference metadata.
        metadata = 65536 + len(request['samples']) * 16384
        limit = min(self.max_written_prediction_bytes, remaining - metadata)
        if limit <= 0:
            raise OSError('Prediction-pool capacity exhausted before worker launch')
        receipt = MemberSandbox.run(replace(self, max_written_prediction_bytes=limit), program, inputs, output)
        record = {'remaining_pool_bytes_before_job': remaining, 'metadata_reserved_bytes': metadata,
                  'new_payload_limit_bytes': limit, 'no_byte_equality_requirement_added': True,
                  'scientific_worker_unchanged': True}
        write_new(output / 'remaining_capacity.json', record)
        return receipt | {'remaining_capacity': record}


def adapt_runner(original, pool_directory, policy):
    runner = member.adapt_runner(original, pool_directory, policy)
    s = runner.sandbox
    runner.sandbox = RemainingCapacitySandbox(s.workspace, s.runtime, timeout=s.timeout, gpu=s.gpu,
        prediction_pool=s.prediction_pool, max_written_prediction_bytes=s.max_written_prediction_bytes)
    return runner


@contextmanager
def installed(pool_directory, policy):
    original = study.load
    def load(*args, **kwargs):
        values = original(*args, **kwargs)
        try:
            return (*values[:-1], adapt_runner(values[-1], pool_directory, policy))
        except BaseException:
            values[-2].close()
            raise
    study.load = load
    try:
        yield
    finally:
        study.load = original


def estimate(runner, proposal, library, policy):
    plan = member.estimate(runner, proposal, library, policy)
    replay = (sum(int(s.fields['image'].size) * 5 + 16384 for s in runner.cohorts('development'))
              if proposal.new_branch else 0)
    planned = plan['prediction_bytes_bound'] - replay
    return plan | {'replay_raw_bound_already_in_original_estimate': replay,
        'planned_prediction_bytes': planned, 'planned_total_new_bytes': plan['total_new_bytes_bound'] - replay,
        'planning_assumption': 'Byte-identical replay shares bytes; novel bytes still face per-job remaining-capacity limits',
        'prediction_pool_fits_expected_replay': allocated_bytes(runner.sandbox.prediction_pool.root) + planned <= policy['max_prediction_pool_bytes'],
        'original_worst_case_bounds_preserved_in_record': True, 'all_replay_inference_still_executed': True}


def binding():
    return member.binding() | {'bounded_continuation.py': file_hash(Path(__file__))}


def budget(directory):
    return ReclaimAwareBudget(study.ROOT, directory, reclamation_receipt=CREDIT, receipt_sha256=CREDIT_SHA,
        max_growth_bytes=POLICY['max_growth_bytes'], minimum_free_bytes=POLICY['minimum_free_bytes'])


def declaration(directory, proof):
    ancestor = directory / 'storage_runtime_members_v1/binding.json'
    value = json.loads(ancestor.read_text())
    if value['implementations'] != member.binding() or value['policy'] != POLICY:
        raise ValueError('Existing runtime implementation or storage caps changed')
    if file_hash(CREDIT) != CREDIT_SHA:
        raise ValueError('Approved reclamation receipt changed')
    return {'implementations': binding(), 'policy': POLICY, 'ancestor_sha256': file_hash(ancestor),
            'scientific_protocol_sha256': file_hash(directory / 'protocol.json'),
            'preflight': {'path': str(proof), 'sha256': file_hash(proof)},
            'reclamation_receipt_sha256': CREDIT_SHA,
            'shared_existing_pools': 'storage_runtime_members_v1', 'scientific_criteria_unchanged': True}


def configure(directory, preflight):
    directory = exact_child(study.ROOT, directory); preflight = exact_child(study.ROOT, preflight)
    proof = json.loads(preflight.read_text()); tests = proof['tests']
    if (proof['implementations'] != binding() or tests['tests'] < 8 or tests['failures']
            or tests['errors'] or tests['skipped'] or file_hash(Path(tests['path'])) != tests['sha256']):
        raise ValueError('Successful bound continuation tests required')
    expected = declaration(directory, preflight)
    runtime = directory / 'storage_runtime_bounded_v1'; runtime.mkdir(exist_ok=True)
    path = runtime / 'binding.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Bound continuation declaration changed')
    else:
        write_new(path, expected)
    return runtime


def worker(directory, action, arm, seed):
    runtime = directory / 'storage_runtime_bounded_v1'
    record = json.loads((runtime / 'binding.json').read_text())
    if record != declaration(directory, Path(record['preflight']['path'])):
        raise ValueError('Bound continuation declaration changed')
    lock = directory.with_name(directory.name + f'.{arm}-{seed}.lock')
    with lock.open('a') as stream, installed(directory / 'storage_runtime_members_v1', POLICY):
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == 'queue':
            study.queue(directory, arm, seed)
            return
        _, _, episode, archive, runner = study.load(directory, arm, seed)
        try:
            number = len(archive.events(study.ROUND_EVENT)) + 1
            request = episode / 'requests' / f'r{number:02d}'; before = archive.verify()
            raw_hash = file_hash(request / 'raw_response.json')
            for name in ('storage_runtime_v1', 'storage_runtime_members_v1'):
                prior = directory / name / f'{arm}-s{seed}-r{number:02d}-volume.json'
                if prior.exists():
                    value = json.loads(prior.read_text())
                    if (value['fits'] or value['scientific_round_started']
                            or value['archive_tail_unchanged'] != before or value['raw_proposal_sha256'] != raw_hash):
                        raise ValueError('Prior attempt may have started or changed; no automatic replay')
            library = ResearchBank(runner).inherited_manifest()['library']
            try:
                proposal, _ = parse_action(json.loads((request / 'raw_response.json').read_text()), arm, archive.incumbent(), library)
            except (ValueError, TypeError, KeyError, AttributeError, SyntaxError):
                plan = {'input_pool_fits': True, 'prediction_pool_fits_expected_replay': True,
                        'planned_total_new_bytes': POLICY['metadata_reserve_bytes'], 'scientific_parser_will_decide': True}
            else:
                plan = estimate(runner, proposal, library, POLICY)
            measured = budget(directory).check()
            fits = (not measured['reason'] and plan['input_pool_fits'] and plan['prediction_pool_fits_expected_replay']
                    and measured['growth_bytes'] + plan['planned_total_new_bytes'] <= POLICY['max_growth_bytes'])
            assert archive.verify() == before
            report = runtime / f'{arm}-s{seed}-r{number:02d}-volume.json'
            write_new(report, {'plan': plan, 'storage': measured, 'fits': fits,
                'archive_tail_unchanged': before, 'raw_proposal_sha256': raw_hash, 'scientific_round_started': False})
            print(json.dumps({'volume_preflight': str(report), 'fits': fits}), flush=True)
            if not fits:
                raise RuntimeError('Bounded capacity preflight refused; original proposal remains pending')
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
        configure(directory, args.preflight)
        command = member.old.inner_command(directory, args.action, args.arm, args.seed)
        command[command.index(member.old.__spec__.name)] = __spec__.name
        result = budget(directory).run(command, timeout_seconds=3600, poll_seconds=1)
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['status'] == 'ok' else 1)
