"""Share a computed router output with its already staged, read-only input.

No payload is imported into the global prediction pool merely to reference it.
Only complete same-case score/support pairs already visible in this job qualify.
Every prediction is still computed and serialized by the unchanged worker.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import re

from . import bounded_continuation as bounded
from .field_storage import allocated_bytes, exact_child, new_json, sha256
from .prediction_storage import DIGEST, PREDICTION_NAME, PredictionPool


class InputReusePool(PredictionPool):
    def __init__(self, original, inputs):
        self.guard, self.root = original.guard, original.root
        self.inputs = exact_child(self.guard.workspace, inputs)
        self.borrowed = {}
        request = json.loads((self.inputs / 'request.json').read_text())
        if request['mode'] != 'predict':
            return
        for sample in request['samples']:
            name = sample['file']
            if not re.fullmatch(r'sample-\d{4}\.fields\.json', name):
                raise ValueError('Explicit member input manifest required')
            manifest = json.loads(exact_child(self.inputs, self.inputs / name).read_text())
            fields = manifest['fields']
            key = next(r['identity'] for r in fields if r['name'] == 'image')
            if not DIGEST.fullmatch(key):
                raise ValueError('Invalid input identity')
            pairs = {}
            for ref in fields:
                if 'member' not in ref:
                    continue
                match = re.fullmatch(r'branch_(b\d+)_(score|support)', ref['name'])
                if (not match or match[2] != ref['member'] or not DIGEST.fullmatch(ref['sha256'])
                        or ref['file'] != ref['sha256'] + '.npz'):
                    raise ValueError('Only explicit branch prediction members qualify')
                pairs.setdefault(match[1], {})[match[2]] = ref
            for pair in pairs.values():
                if set(pair) != {'score', 'support'} or pair['score']['file'] != pair['support']['file']:
                    raise ValueError('Complete co-located prediction pair required')
                ref = pair['score']
                path = exact_child(self.inputs, self.inputs / 'arrays' / ref['file'])
                if path.stat().st_dev != self.root.stat().st_dev or sha256(path) != ref['sha256']:
                    raise ValueError('Borrowed input changed or requires copying')
                self.borrowed.setdefault(key, {})[ref['sha256']] = path

    def snapshot(self, sample_keys):
        known = super().snapshot(sample_keys)
        for key in sample_keys:
            for digest, path in self.borrowed.get(key, {}).items():
                known[digest] = path.stat().st_size
        return known

    def finalize(self, output, sample_keys, known):
        """Original append-only finalizer plus direct, validated input references."""
        output = exact_child(self.guard.workspace, output)
        if (output.stat().st_dev != self.root.stat().st_dev
                or any(not isinstance(k, str) or not DIGEST.fullmatch(k) for k in sample_keys)):
            raise ValueError('Same-device outputs and explicit image identities required')
        result = json.loads(exact_child(output, output / 'result.json').read_text())
        expected = [{'index': i, 'file': f'prediction-{i:04d}.npz'} for i in range(len(sample_keys))]
        trace = json.loads(exact_child(output, output / 'prediction_storage_worker.json').read_text())
        records = trace['computed_predictions']
        if (result.get('predictions') != expected
                or [r['name'] for r in records] != [r['file'] for r in expected]
                or trace.get('inference_skipped') is not False):
            raise ValueError('Missing actual prediction serialization records')
        linked, new_objects, independent, borrowed = 0, 0, 0, 0
        with self.guard.locked():
            for record, key in zip(records, sample_keys):
                digest, size = record['sha256'], record['size']
                if not DIGEST.fullmatch(digest) or type(size) is not int or size < 1:
                    raise ValueError('Invalid serialized reference')
                path = exact_child(output, output / record['name'])
                canonical = exact_child(self.root, self.root / 'objects' / (digest + '.npz'))
                source_input = self.borrowed.get(key, {}).get(digest)
                kind = record['kind']
                if kind in ('cached', 'local'):
                    reference = exact_child(output, path.with_suffix('.npz.ref.json'))
                    if json.loads(reference.read_text()) != record or path.exists():
                        raise ValueError('Ambiguous or changed prediction reference')
                    if kind == 'cached':
                        if known.get(digest) != size:
                            raise ValueError('Reference was not visible in this job snapshot')
                        source = source_input if source_input is not None else canonical
                    else:
                        if not PREDICTION_NAME.fullmatch(record['source']) or record['source'] >= path.name:
                            raise ValueError('Local reference must name an earlier prediction')
                        source = exact_child(output, output / record['source'])
                    if source.stat().st_size != size or sha256(source) != digest:
                        raise ValueError('Referenced prediction content changed')
                    space = os.statvfs(self.root)
                    if space.f_bavail * space.f_frsize < self.guard.policy['minimum_free_bytes'] + 4096:
                        raise OSError('Prediction disk reserve violated')
                    os.link(source, path)
                    linked += 1
                elif kind != 'new':
                    raise ValueError('Unknown prediction storage kind')
                if path.stat().st_size != size or sha256(path) != digest:
                    raise ValueError('Actual computed prediction hash mismatch')
                if kind == 'cached' and source_input is not None:
                    borrowed += 1
                    continue  # Existing immutable input, not a new pool payload.
                membership = exact_child(self.root, self.root / 'cases' / key / (digest + '.json'))
                metadata = (4096 if not membership.exists() else 0) + (4096 if not membership.parent.exists() else 0)
                growth = (0 if canonical.exists() else ((size + 4095)//4096)*4096) + metadata
                if allocated_bytes(self.root) + growth > self.guard.policy['max_pool_bytes']:
                    raise OSError('Prediction-pool storage budget exceeded')
                space = os.statvfs(self.root)
                if space.f_bavail * space.f_frsize < self.guard.policy['minimum_free_bytes'] + metadata:
                    raise OSError('Prediction-pool disk reserve violated')
                if canonical.exists():
                    if canonical.stat().st_size != size or sha256(canonical) != digest:
                        raise ValueError('Immutable prediction object changed')
                    independent += int(not os.path.samefile(path, canonical))
                else:
                    os.link(path, canonical)
                    new_objects += 1
                membership.parent.mkdir(exist_ok=True)
                value = {'sha256': digest, 'size': size}
                if membership.exists():
                    if json.loads(membership.read_text()) != value:
                        raise ValueError('Prediction membership changed')
                else:
                    new_json(membership, value)
        return {'computed_predictions': len(records), 'new_canonical_objects': new_objects,
                'hardlinked_new_paths': linked, 'borrowed_input_predictions': borrowed,
                'concurrent_or_unexposed_duplicates_retained': independent,
                'new_prediction_payload_bytes': trace['new_prediction_payload_bytes'],
                'existing_files_replaced': 0, 'inference_skipped': False}


class InputReuseSandbox(bounded.RemainingCapacitySandbox):
    def run(self, program, inputs, output):
        pool = InputReusePool(self.prediction_pool, inputs)
        return bounded.RemainingCapacitySandbox.run(replace(self, prediction_pool=pool), program, inputs, output)


def adapt_runner(original, directory, policy):
    runner = bounded.adapt_runner(original, directory, policy)
    s = runner.sandbox
    runner.sandbox = InputReuseSandbox(s.workspace, s.runtime, timeout=s.timeout, gpu=s.gpu,
        prediction_pool=s.prediction_pool, max_written_prediction_bytes=s.max_written_prediction_bytes)
    return runner


@contextmanager
def installed(directory, policy):
    original = bounded.study.load
    def load(*args, **kwargs):
        values = original(*args, **kwargs)
        try:
            return (*values[:-1], adapt_runner(values[-1], directory, policy))
        except BaseException:
            values[-2].close()
            raise
    bounded.study.load = load
    try:
        yield
    finally:
        bounded.study.load = original
