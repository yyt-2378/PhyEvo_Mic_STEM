"""Explicitly authorized 512 MiB prediction pool; unchanged 1 GiB study cap.

Historical policy files and bindings remain immutable. A checked, per-process
guard exposes the authorized effective policy to the existing writer/finalizer.
The original study, scientific proposal, replay and admission code is reused.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from . import input_reuse_continuation as prior
from .contracts import file_hash, write_new
from .field_storage import exact_child
from .research_state_v2 import ResearchBank, parse_action

study, bounded, BASE = prior.study, prior.bounded, prior.POLICY
POLICY = BASE | {'max_prediction_pool_bytes': 512 * 1024**2}
RUNTIME = 'storage_runtime_authorized_pool_512mib_v1'
FORMAT = 'phyevo-authorized-prediction-pool-amendment-v1'


def validate_authorization(value, directory, old_policy_sha):
    if (value.get('format') != FORMAT or value.get('study_scope') != str(directory)
            or value.get('approved_by') != 'user' or not value.get('approval_text')
            or value.get('preserve_existing_files') is not True
            or value.get('scientific_protocol_unchanged') is not True
            or value.get('previous_policy') != BASE or value.get('effective_policy') != POLICY
            or value.get('original_pool_policy_sha256') != old_policy_sha):
        raise ValueError('Explicit matching 256-to-512 MiB authorization required')
    return dict(POLICY)


class AuthorizedGuard:
    def __init__(self, original, authorization):
        self.original, self.root, self.workspace = original, original.root, original.workspace
        self.authorization = exact_child(self.workspace, authorization)
        self.authorization_sha = file_hash(self.authorization)
        self.original_sha = file_hash(self.root / 'policy.json')
        value = json.loads(self.authorization.read_text())
        validate_authorization(value, self.root.parent.parent, self.original_sha)
        if (original.policy['max_pool_bytes'] != BASE['max_prediction_pool_bytes']
                or original.policy['minimum_free_bytes'] != BASE['minimum_free_bytes']):
            raise ValueError('Original pool does not match the authorized baseline')
        self._policy = original.policy | {'max_pool_bytes': POLICY['max_prediction_pool_bytes']}

    def verify(self):
        if (file_hash(self.authorization) != self.authorization_sha
                or file_hash(self.root / 'policy.json') != self.original_sha):
            raise ValueError('Authorized or historical pool policy changed during execution')

    @property
    def policy(self):
        self.verify()
        return dict(self._policy)

    @contextmanager
    def locked(self):
        with self.original.locked():
            self.verify()
            yield
            self.verify()


def adapt_runner(original, directory, authorization):
    # Initialize through the unchanged historical policy; amend only this
    # explicit runner's guard. Input-pool and per-job limits are not increased.
    runner = prior.storage.adapt_runner(original, directory, BASE)
    pool = runner.sandbox.prediction_pool
    pool.guard = AuthorizedGuard(pool.guard, authorization)
    return runner


@contextmanager
def installed(directory, authorization):
    original = study.load
    def load(*args, **kwargs):
        values = original(*args, **kwargs)
        try:
            return (*values[:-1], adapt_runner(values[-1], directory, authorization))
        except BaseException:
            values[-2].close()
            raise
    study.load = load
    try:
        yield
    finally:
        study.load = original


def binding():
    return prior.binding() | {'pool_amendment_continuation.py': file_hash(Path(__file__))}


def declaration(directory, proof, authorization):
    ancestor = directory / prior.RUNTIME / 'binding.json'
    value = json.loads(ancestor.read_text())
    if value != prior.declaration(directory, Path(value['preflight']['path'])):
        raise ValueError('Original tested runtime changed')
    old_policy = directory / 'storage_runtime_members_v1/predictions/policy.json'
    policy = validate_authorization(json.loads(authorization.read_text()), directory, file_hash(old_policy))
    evidence = json.loads(proof.read_text()); tests = evidence['tests']
    if (evidence['implementations'] != binding() or evidence['effective_policy'] != policy
            or evidence['authorization_sha256'] != file_hash(authorization)
            or tests['tests'] < 10 or tests['failures'] or tests['errors'] or tests['skipped']
            or file_hash(exact_child(study.ROOT, Path(tests['path']))) != tests['sha256']):
        raise ValueError('Bound passing amendment and original write-cap tests required')
    return {'implementations': binding(), 'policy': policy, 'ancestor_sha256': file_hash(ancestor),
        'preflight': {'path': str(proof), 'sha256': file_hash(proof)},
        'authorization': {'path': str(authorization), 'sha256': file_hash(authorization)},
        'original_pool_policy_sha256': file_hash(old_policy),
        'scientific_protocol_sha256': file_hash(directory / 'protocol.json'),
        'shared_existing_pools': 'storage_runtime_members_v1',
        'reclamation_receipt_sha256': bounded.CREDIT_SHA,
        'whole_study_limit_and_baseline_unchanged': True, 'historical_files_replaced': 0}


def configure(directory, proof, authorization):
    expected = declaration(directory, proof, authorization)
    runtime = exact_child(directory, directory / RUNTIME); runtime.mkdir(exist_ok=True)
    path = runtime / 'binding.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Authorized runtime binding changed')
    else:
        write_new(path, expected)


def verify_refusals(directory, label, before, raw_hash):
    for name in ('storage_runtime_v1', 'storage_runtime_members_v1',
                 'storage_runtime_bounded_v1', 'storage_runtime_routed_v1', prior.RUNTIME):
        path = directory / name / (label + '-volume.json')
        if path.exists():
            value = json.loads(path.read_text())
            if (value['fits'] or value['scientific_round_started']
                    or value['archive_tail_unchanged'] != before
                    or value['raw_proposal_sha256'] != raw_hash):
                raise ValueError('Prior scientific execution may have started or changed')
    path = directory / 'storage_runtime_capped_attempt_v1' / (label + '-volume.json')
    if path.exists():
        value = json.loads(path.read_text())
        if (value['decision']['bounded_attempt_allowed'] or value['archive_tail_unchanged'] != before
                or value['raw_proposal_sha256'] != raw_hash):
            raise ValueError('A prior capped scientific attempt may have started')


def worker(directory, action, arm, seed, raw_hash):
    runtime = directory / RUNTIME
    record = json.loads((runtime / 'binding.json').read_text())
    authorization = Path(record['authorization']['path'])
    if record != declaration(directory, Path(record['preflight']['path']), authorization):
        raise ValueError('Authorized runtime declaration changed')
    lock = directory.with_name(directory.name + f'.{arm}-{seed}.lock')
    with lock.open('a') as stream, installed(directory / 'storage_runtime_members_v1', authorization):
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == 'queue':
            study.queue(directory, arm, seed)
            return
        _, _, episode, archive, runner = study.load(directory, arm, seed)
        try:
            number = len(archive.events(study.ROUND_EVENT)) + 1
            request = episode / 'requests' / f'r{number:02d}'
            before = archive.verify()
            if file_hash(request / 'raw_response.json') != raw_hash or (request / 'result.json').exists():
                raise ValueError('Only the explicitly bound unfinished proposal may run')
            label = f'{arm}-s{seed}-r{number:02d}'
            verify_refusals(directory, label, before, raw_hash)
            library = ResearchBank(runner).inherited_manifest()['library']
            try:
                proposal, _ = parse_action(json.loads((request / 'raw_response.json').read_text()),
                                           arm, archive.incumbent(), library)
            except (ValueError, TypeError, KeyError, AttributeError, SyntaxError):
                plan = {'input_pool_fits': True, 'prediction_pool_fits_expected_replay': True,
                    'planned_total_new_bytes': POLICY['metadata_reserve_bytes'], 'scientific_parser_will_decide': True}
            else:
                plan = prior.estimate(runner, proposal, library, POLICY)
            measured = bounded.budget(directory).check()
            fits = (not measured['reason'] and plan['input_pool_fits']
                and plan['prediction_pool_fits_expected_replay']
                and measured['growth_bytes'] + plan['planned_total_new_bytes'] <= POLICY['max_growth_bytes'])
            assert archive.verify() == before
            report = runtime / (label + '-volume.json')
            write_new(report, {'plan': plan, 'storage': measured, 'fits': fits,
                'effective_policy': POLICY, 'authorization_sha256': file_hash(authorization),
                'archive_tail_unchanged': before, 'raw_proposal_sha256': raw_hash,
                'scientific_round_started': False})
            print(json.dumps({'authorized_preflight': str(report), 'fits': fits}), flush=True)
            if not fits:
                raise RuntimeError('Authorized capacity still insufficient; proposal remains pending')
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
    parser.add_argument('--raw-proposal-sha256')
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args(); directory = exact_child(study.ROOT, args.directory)
    if args.action == 'execute' and not args.raw_proposal_sha256:
        parser.error('An exact raw proposal hash is required for execute')
    if args.mode == 'worker':
        worker(directory, args.action, args.arm, args.seed, args.raw_proposal_sha256)
    else:
        if args.preflight is None or args.authorization is None:
            parser.error('Passing preflight and explicit authorization required')
        configure(directory, exact_child(study.ROOT, args.preflight), exact_child(study.ROOT, args.authorization))
        command = bounded.member.old.inner_command(directory, args.action, args.arm, args.seed)
        command[command.index(bounded.member.old.__spec__.name)] = __spec__.name
        if args.raw_proposal_sha256:
            command += ['--raw-proposal-sha256', args.raw_proposal_sha256]
        result = bounded.budget(directory).run(command, timeout_seconds=3600, poll_seconds=1)
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['status'] == 'ok' else 1)
