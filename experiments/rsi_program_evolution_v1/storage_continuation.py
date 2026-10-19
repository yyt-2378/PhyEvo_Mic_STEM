"""Explicit storage-only continuation of the frozen stable study.

Scientific code, thresholds, proposals and old records remain unchanged. New
jobs use field objects and content-shared outputs. Estimate before registering
a scientific candidate; an infrastructure refusal leaves the proposal pending.
"""
import argparse
from contextlib import contextmanager
import fcntl
import io
import json
import os
from pathlib import Path

import numpy as np

from . import stable_study as study
from .contracts import file_hash, write_new
from .field_storage import array_identity, exact_child, allocated_bytes
from .field_storage_runtime import FieldStorageExecutor
from .prediction_storage import PredictionPool
from .prediction_storage_runtime import SharedPredictionSandbox, prediction_binding
from .research_state_v2 import ResearchBank, parse_action
from .storage_budget import StorageBudget

MIB = 1024**2
POLICY = {'max_growth_bytes': 1024*MIB, 'max_input_pool_bytes': 512*MIB,
          'max_prediction_pool_bytes': 256*MIB, 'minimum_free_bytes': 8*1024**3,
          'max_written_prediction_bytes': 128*MIB, 'metadata_reserve_bytes': 16*MIB}


def binding():
    folder = Path(__file__).parent
    return prediction_binding() | {n: file_hash(folder / n) for n in
                                    ('storage_budget.py', 'storage_continuation.py')}


def adapt_runner(old, directory, policy):
    directory = exact_child(old.workspace, directory)
    pool = PredictionPool(old.workspace, directory / 'predictions',
                          max_pool_bytes=policy['max_prediction_pool_bytes'],
                          minimum_free_bytes=policy['minimum_free_bytes'])
    sandbox = SharedPredictionSandbox(old.workspace, old.sandbox.runtime,
        timeout=old.sandbox.timeout, gpu=old.sandbox.gpu, prediction_pool=pool,
        max_written_prediction_bytes=policy['max_written_prediction_bytes'])
    return FieldStorageExecutor(old.workspace, old.archive, sandbox, old.samples,
        directory / 'fields', old.development_transforms,
        max_pool_bytes=policy['max_input_pool_bytes'], minimum_free_bytes=policy['minimum_free_bytes'])


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
    """Read-only volume estimate; no new program, array or archive event.

    Known field payloads are compressed in memory exactly as the field codec.
    Uncomputed scores/support use raw-byte upper bounds plus ZIP overhead.
    Training reserves the existing worker's per-file limit, not a smaller model.
    This is conservative planning, not a proof against arbitrary custom output;
    whole-scope monitoring remains necessary.
    """
    seen = set()
    field_bytes = 0
    known_objects = 0
    unknown_field_bytes = 0
    prediction_bytes = 0
    missing_predictions = []

    def add(value):
        nonlocal field_bytes, known_objects
        value, _, identity = array_identity(value)
        if identity in seen:
            return
        seen.add(identity)
        marker = runner.array_pool.root / (identity + '.json')
        if marker.exists():
            return
        buffer = io.BytesIO(); np.savez_compressed(buffer, array=value)
        field_bytes += ((len(buffer.getvalue()) + 4095)//4096 + 2)*4096
        known_objects += 1

    splits = ['development'] + ['stability-' + t for t in runner.development_transforms]
    if proposal.new_branch and proposal.new_branch['name'] not in proposal.active_branches:
        raise ValueError('New branch must be actively evaluated before volume planning')
    for split in splits:
        cohort = runner.cohorts(split)
        shape_bound = sum(int(s.fields['image'].size)*5 + 16384 for s in cohort)
        # Router outputs always recompute; allow a fresh copy even if sharing wins.
        prediction_bytes += shape_bound
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
                # A branch not measured on this split needs both prediction
                # bytes and the subsequently staged score/support field objects.
                prediction_bytes += shape_bound
                unknown_field_bytes += shape_bound
                missing_predictions.append({'branch': name, 'split': split})
            else:
                for ref in feedback['predictions'].values():
                    with np.load(runner.checked_ref(ref), allow_pickle=False) as values:
                        add(values['score']); add(values['support'])
    training_jobs = int(proposal.router_training_steps > 0) + int(bool(proposal.new_branch and proposal.new_branch['training_steps']))
    if training_jobs:
        for sample in runner.samples:
            if runner.protocol.split(sample.group) == 'train':
                add(sample.human); add(sample.known)
    # Reverse-order analysis/system replay is actually rerun; its same bytes
    # share storage. Allow one additional unknown branch output if replay differs
    # (which the unchanged scientific replay gate would reject).
    if proposal.new_branch:
        prediction_bytes += sum(int(s.fields['image'].size)*5+16384 for s in runner.cohorts('development'))
    checkpoint_bytes = training_jobs * 256*MIB
    input_bytes = field_bytes + unknown_field_bytes
    total = input_bytes + prediction_bytes + checkpoint_bytes + policy['metadata_reserve_bytes']
    current_input = allocated_bytes(runner.array_pool.root)
    current_prediction = allocated_bytes(runner.sandbox.prediction_pool.root)
    return {'new_known_input_objects': known_objects, 'known_input_allocated_estimate': field_bytes,
            'uncomputed_input_raw_bound': unknown_field_bytes, 'new_input_bytes_bound': input_bytes,
            'prediction_bytes_bound': prediction_bytes, 'checkpoint_bytes_bound': checkpoint_bytes,
            'metadata_reserve_bytes': policy['metadata_reserve_bytes'], 'total_new_bytes_bound': total,
            'missing_predictions': missing_predictions, 'training_jobs': training_jobs,
            'input_pool_fits': current_input + input_bytes <= policy['max_input_pool_bytes'],
            'prediction_pool_fits': current_prediction + prediction_bytes <= policy['max_prediction_pool_bytes'],
            'inference_skipped': False, 'labels_used_for_selection': False}


def configure(directory, preflight):
    directory = exact_child(study.ROOT, directory)
    preflight = exact_child(study.ROOT, preflight)
    proof = json.loads(preflight.read_text())
    if (proof['tests']['failures'] or proof['tests']['errors'] or proof['tests']['skipped']
            or proof['tests']['tests'] < 45 or file_hash(Path(proof['tests']['path'])) != proof['tests']['sha256']):
        raise ValueError('Bound successful storage preflight required')
    for name, expected in proof['implementations'].items():
        if file_hash(Path(__file__).with_name(name)) != expected:
            raise ValueError('Previously tested storage implementation changed: ' + name)
    runtime = exact_child(directory, directory / 'storage_runtime_v1')
    runtime.mkdir(exist_ok=True)
    expected = {'implementations': binding(), 'policy': POLICY,
                'scientific_protocol_sha256': file_hash(directory / 'protocol.json'),
                'preflight': {'path': str(preflight), 'sha256': file_hash(preflight)},
                'scientific_algorithms_and_admission_unchanged': True,
                'only_future_output_storage_changes': True}
    path = runtime / 'binding.json'
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Continuation runtime binding changed')
    else:
        write_new(path, expected)
    return runtime


def inner_command(directory, action, arm, seed):
    root = study.ROOT
    lock = directory.with_name(directory.name + f'.{arm}-{seed}.lock')
    command = ['/usr/bin/bwrap', '--unshare-all', '--die-with-parent', '--new-session',
        '--cap-drop', 'ALL', '--clearenv', '--ro-bind', '/usr', '/usr',
        '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
        '--proc', '/proc', '--dev', '/dev', '--dir', '/tmp', '--ro-bind', str(root), str(root),
        '--bind', str(directory), str(directory), '--bind', str(lock), str(lock), '--chdir', str(root)]
    for path in ('/dev/nvidia3', '/dev/nvidia4', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools'):
        if Path(path).exists():
            command += ['--dev-bind', path, path]
    for path in ('/sys', '/etc/ld.so.cache', '/nvidia-smi-real'):
        if Path(path).exists():
            command += ['--ro-bind', path, path]
    # Public GPU UUID map from the validated migration namespace, never secrets.
    settings = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONPATH': str(root)+':'+str(root/'src'),
                'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
                'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                'CUDA_CACHE_DISABLE': '1', 'PHYEVO_GPU_UUID_MAP': os.environ['PHYEVO_GPU_UUID_MAP']}
    for key, value in settings.items():
        command += ['--setenv', key, value]
    return command + [str(root/'.venvs/sam2/bin/python'), '-B', '-m', __spec__.name, 'worker',
                      str(directory), '--action', action, '--arm', arm, '--seed', str(seed)]


def worker(directory, action, arm, seed):
    runtime = directory / 'storage_runtime_v1'
    declaration = json.loads((runtime / 'binding.json').read_text())
    if (declaration['implementations'] != binding()
            or declaration['scientific_protocol_sha256'] != file_hash(directory / 'protocol.json')):
        raise ValueError('Frozen continuation binding mismatch')
    policy = declaration['policy']
    lock = directory.with_name(directory.name + f'.{arm}-{seed}.lock')
    with lock.open('a') as stream, installed(runtime, policy):
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == 'queue':
            study.queue(directory, arm, seed)
            return
        _, _, episode, archive, runner = study.load(directory, arm, seed)
        try:
            round_number = len(archive.events(study.ROUND_EVENT)) + 1
            request = episode / 'requests' / f'r{round_number:02d}'
            before = archive.verify()
            library = ResearchBank(runner).inherited_manifest()['library']
            try:
                proposal, _ = parse_action(json.loads((request/'raw_response.json').read_text()), arm,
                                           archive.incumbent(), library)
            except (ValueError, TypeError, KeyError, AttributeError, SyntaxError):
                # Preserve the original scientific handling/counting of invalid
                # proposals. Such a reply cannot start array staging or fitting.
                plan = {'input_pool_fits': True, 'prediction_pool_fits': True,
                        'total_new_bytes_bound': policy['metadata_reserve_bytes'],
                        'scientific_parser_will_decide': True}
            else:
                plan = estimate(runner, proposal, library, policy)
            budget = StorageBudget(study.ROOT, directory, max_growth_bytes=policy['max_growth_bytes'],
                                   minimum_free_bytes=policy['minimum_free_bytes'])
            measured = budget.check()
            fits = (not measured['reason'] and plan['input_pool_fits'] and plan['prediction_pool_fits']
                    and measured['growth_bytes'] + plan['total_new_bytes_bound'] <= policy['max_growth_bytes'])
            assert archive.verify() == before
            # Unique receipt per proposal; never overwrite a prior preflight.
            report = runtime / f'{arm}-s{seed}-r{round_number:02d}-volume.json'
            record = {'plan': plan, 'storage': measured, 'fits': fits, 'archive_tail_unchanged': before,
                      'raw_proposal_sha256': file_hash(request/'raw_response.json'), 'scientific_round_started': False}
            if report.exists():
                raise FileExistsError('This proposal already has a storage preflight; inspect before retry')
            write_new(report, record)
            print(json.dumps({'volume_preflight': str(report), 'fits': fits, 'bound_bytes': plan['total_new_bytes_bound']}), flush=True)
            if not fits:
                raise RuntimeError('Storage preflight refused before scientific candidate registration; proposal remains pending')
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
    args = parser.parse_args()
    directory = exact_child(study.ROOT, args.directory)
    if args.mode == 'worker':
        worker(directory, args.action, args.arm, args.seed)
    else:
        if args.preflight is None:
            parser.error('--preflight required for the guarded controller')
        configure(directory, args.preflight)
        budget = StorageBudget(study.ROOT, directory, max_growth_bytes=POLICY['max_growth_bytes'],
                               minimum_free_bytes=POLICY['minimum_free_bytes'])
        result = budget.run(inner_command(directory, args.action, args.arm, args.seed),
                            timeout_seconds=3600, poll_seconds=1)
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['status'] == 'ok' else 1)
