"""Explicit, additive member-storage continuation; original RSI gates unchanged."""
import argparse
from contextlib import contextmanager
import fcntl
import io
import json
from pathlib import Path

import numpy as np

from . import storage_continuation as old
from .contracts import file_hash, write_new
from .field_storage import allocated_bytes, array_identity, exact_child
from .member_storage import adapt_runner
from .research_state_v2 import ResearchBank, parse_action
from .storage_budget import StorageBudget

study, POLICY = old.study, old.POLICY


def binding():
    return old.binding() | {n: file_hash(Path(__file__).with_name(n))
                           for n in ('member_storage.py', 'member_continuation.py')}


@contextmanager
def installed(directory, policy):
    original = study.load
    def load(*args, **kwargs):
        values = original(*args, **kwargs)
        try:
            return (*values[:-1], adapt_runner(values[-1], directory, policy))
        except BaseException:
            values[-2].close()
            raise
    study.load = load
    try:
        yield
    finally:
        study.load = original


def estimate(runner, proposal, library, policy):
    seen, field_bytes, prediction_bytes, references = set(), 0, 0, 0
    missing = []
    def add(value):
        nonlocal field_bytes
        value, _, identity = array_identity(value)
        if identity in seen:
            return
        seen.add(identity)
        if (runner.array_pool.root / (identity + '.json')).exists():
            return
        buffer = io.BytesIO(); np.savez_compressed(buffer, array=value)
        field_bytes += ((len(buffer.getvalue()) + 4095)//4096 + 2)*4096
    for split in ['development'] + ['stability-' + t for t in runner.development_transforms]:
        cohort = runner.cohorts(split)
        bound = sum(int(s.fields['image'].size)*5 + 16384 for s in cohort)
        prediction_bytes += bound
        for sample in cohort:
            for value in sample.fields.values():
                add(value)
        for name in proposal.active_branches:
            feedback = None
            if name in library:
                cap = runner.checked_ref(library[name])
                path = cap.parent / split / 'predictions.json'
                if path.exists():
                    feedback = json.loads(path.read_text())
            if feedback is None:
                prediction_bytes += bound
                missing.append({'branch': name, 'split': split})
            else:
                for ref in feedback['predictions'].values():
                    path = runner.checked_ref(ref)
                    if path.stat().st_dev != runner.array_pool.root.stat().st_dev:
                        raise ValueError('Existing prediction is on another filesystem; no silent copy')
                    references += 1
            # Existing AND future prediction pairs are linked, never repacked.
    training_jobs = int(proposal.router_training_steps > 0) + int(bool(proposal.new_branch and proposal.new_branch['training_steps']))
    if training_jobs:
        for sample in runner.samples:
            if runner.protocol.split(sample.group) == 'train':
                for value in sample.fields.values():
                    add(value)
                add(sample.human); add(sample.known)
    if proposal.new_branch:
        prediction_bytes += sum(int(s.fields['image'].size)*5 + 16384 for s in runner.cohorts('development'))
    checkpoint_bytes = training_jobs*256*1024**2
    metadata = policy['metadata_reserve_bytes'] + references*2*4096
    return {'new_known_input_objects': len(seen), 'new_input_bytes_bound': field_bytes,
            'prediction_bytes_bound': prediction_bytes, 'checkpoint_bytes_bound': checkpoint_bytes,
            'metadata_reserve_bytes': metadata, 'total_new_bytes_bound': field_bytes+prediction_bytes+checkpoint_bytes+metadata,
            'missing_predictions': missing, 'training_jobs': training_jobs,
            'borrowable_existing_prediction_pairs': references, 'uncomputed_prediction_members_repacked': False,
            'input_pool_fits': allocated_bytes(runner.array_pool.root)+field_bytes <= policy['max_input_pool_bytes'],
            'prediction_pool_fits': allocated_bytes(runner.sandbox.prediction_pool.root)+prediction_bytes <= policy['max_prediction_pool_bytes'],
            'inference_skipped': False, 'labels_used_for_selection': False}


def configure(directory, preflight):
    directory = exact_child(study.ROOT, directory); preflight = exact_child(study.ROOT, preflight)
    proof = json.loads(preflight.read_text())
    if proof['implementations'] != binding():
        raise ValueError('Tested member-storage implementation changed')
    tests = proof['tests']
    if (tests['failures'] or tests['errors'] or tests['skipped'] or tests['tests'] < 8
            or file_hash(Path(tests['path'])) != tests['sha256']):
        raise ValueError('Successful bound member-storage tests required')
    ancestor = directory / 'storage_runtime_v1/binding.json'
    ancestor_record = json.loads(ancestor.read_text())
    if ancestor_record['implementations'] != old.binding() or ancestor_record['policy'] != POLICY:
        raise ValueError('Previous runtime or storage policy changed')
    runtime = exact_child(directory, directory / 'storage_runtime_members_v1'); runtime.mkdir(exist_ok=True)
    expected = {'implementations': binding(), 'policy': POLICY, 'ancestor_sha256': file_hash(ancestor),
                'scientific_protocol_sha256': file_hash(directory / 'protocol.json'),
                'preflight': {'path': str(preflight), 'sha256': file_hash(preflight)},
                'scientific_algorithms_and_admission_unchanged': True, 'existing_files_replaced': 0}
    path = runtime / 'binding.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Member continuation binding changed')
    else:
        write_new(path, expected)
    return runtime


def worker(directory, action, arm, seed):
    runtime = directory / 'storage_runtime_members_v1'
    declaration = json.loads((runtime / 'binding.json').read_text())
    if (declaration['implementations'] != binding() or declaration['policy'] != POLICY
            or declaration['scientific_protocol_sha256'] != file_hash(directory / 'protocol.json')
            or declaration['ancestor_sha256'] != file_hash(directory / 'storage_runtime_v1/binding.json')):
        raise ValueError('Member continuation binding mismatch')
    lock = directory.with_name(directory.name + f'.{arm}-{seed}.lock')
    with lock.open('a') as stream, installed(runtime, POLICY):
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == 'queue':
            study.queue(directory, arm, seed)
            return
        _, _, episode, archive, runner = study.load(directory, arm, seed)
        try:
            number = len(archive.events(study.ROUND_EVENT))+1
            request = episode / 'requests' / f'r{number:02d}'
            before = archive.verify()
            previous = directory / f'storage_runtime_v1/{arm}-s{seed}-r{number:02d}-volume.json'
            if previous.exists():
                previous = json.loads(previous.read_text())
                if previous['fits'] or previous['scientific_round_started'] or previous['archive_tail_unchanged'] != before:
                    raise ValueError('Previous execution may have begun; no automatic restart')
            library = ResearchBank(runner).inherited_manifest()['library']
            try:
                proposal, _ = parse_action(json.loads((request/'raw_response.json').read_text()), arm, archive.incumbent(), library)
            except (ValueError, TypeError, KeyError, AttributeError, SyntaxError):
                plan = {'input_pool_fits': True, 'prediction_pool_fits': True,
                        'total_new_bytes_bound': POLICY['metadata_reserve_bytes'], 'scientific_parser_will_decide': True}
            else:
                plan = estimate(runner, proposal, library, POLICY)
            measured = StorageBudget(study.ROOT, directory, max_growth_bytes=POLICY['max_growth_bytes'],
                                     minimum_free_bytes=POLICY['minimum_free_bytes']).check()
            fits = (not measured['reason'] and plan['input_pool_fits'] and plan['prediction_pool_fits']
                    and measured['growth_bytes']+plan['total_new_bytes_bound'] <= POLICY['max_growth_bytes'])
            assert archive.verify() == before
            record = {'plan': plan, 'storage': measured, 'fits': fits, 'archive_tail_unchanged': before,
                      'raw_proposal_sha256': file_hash(request/'raw_response.json'), 'scientific_round_started': False}
            report = runtime / f'{arm}-s{seed}-r{number:02d}-volume.json'
            write_new(report, record)
            print(json.dumps({'volume_preflight': str(report), 'fits': fits, 'bound_bytes': plan['total_new_bytes_bound']}), flush=True)
            if not fits:
                raise RuntimeError('Storage preflight refused; proposal remains pending')
        finally:
            archive.close()
        study.execute(directory, arm, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('guard', 'worker'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--action', choices=('queue', 'execute'), required=True)
    parser.add_argument('--arm', choices=('full', 'route'), required=True)
    parser.add_argument('--seed', type=int, choices=(23,41,67), required=True)
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args(); directory = exact_child(study.ROOT, args.directory)
    if args.mode == 'worker':
        worker(directory, args.action, args.arm, args.seed)
    else:
        if args.preflight is None:
            parser.error('--preflight required')
        configure(directory, args.preflight)
        command = old.inner_command(directory, args.action, args.arm, args.seed)
        command[command.index(old.__spec__.name)] = __spec__.name
        result = StorageBudget(study.ROOT, directory, max_growth_bytes=POLICY['max_growth_bytes'],
            minimum_free_bytes=POLICY['minimum_free_bytes']).run(command, timeout_seconds=3600, poll_seconds=1)
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['status'] == 'ok' else 1)
