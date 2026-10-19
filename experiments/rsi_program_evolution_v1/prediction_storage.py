"""Deduplicate newly computed prediction bytes without removing any files.

The worker computes every prediction, serializes it in bounded memory, and
either writes fresh bytes or an exact hash reference. The host resolves only
predeclared references into new hardlinks. No global arrays enter the sandbox.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import re

PREDICTION_NAME = re.compile(r'prediction-\d{4}\.npz')
DIGEST = re.compile(r'[a-f0-9]{64}')


class PredictionWriter:
    def __init__(self, config, work, original_open):
        if (type(config.get('enabled')) is not bool
                or any(type(config.get(k)) is not int or config[k] <= 0 for k in
                       ('max_prediction_bytes', 'max_written_prediction_bytes'))
                or not isinstance(config.get('known_hashes'), dict)
                or any(not DIGEST.fullmatch(k) or type(v) is not int or v < 1
                       for k, v in config['known_hashes'].items())):
            raise ValueError('Explicit bounded prediction storage configuration required')
        self.config, self.work, self.original_open = config, Path(work), original_open
        self.emitted, self.records, self.written = {}, [], 0

    def open(self, path, mode='r', *args, **kwargs):
        path = Path(path)
        if (self.config['enabled'] and path.parent == self.work and mode == 'xb'
                and PREDICTION_NAME.fullmatch(path.name) and not args and not kwargs):
            if path.exists() or path.with_suffix('.npz.ref.json').exists():
                raise FileExistsError('Prediction or exact reference already exists')
            return _PredictionBuffer(self, path)
        return self.original_open(path, mode, *args, **kwargs)

    def commit(self, path, payload):
        digest = hashlib.sha256(payload).hexdigest()
        record = {'name': path.name, 'sha256': digest, 'size': len(payload)}
        if digest in self.config['known_hashes']:
            if self.config['known_hashes'][digest] != len(payload):
                raise ValueError('Known prediction size differs')
            record['kind'] = 'cached'
        elif digest in self.emitted:
            record.update(kind='local', source=self.emitted[digest])
        else:
            if self.written + len(payload) > self.config['max_written_prediction_bytes']:
                raise OSError('New prediction-byte budget exceeded before write')
            with self.original_open(path, 'xb') as stream:
                stream.write(payload)
            self.written += len(payload)
            self.emitted[digest] = path.name
            record['kind'] = 'new'
        if record['kind'] != 'new':
            with self.original_open(path.with_suffix('.npz.ref.json'), 'x') as stream:
                json.dump(record, stream, sort_keys=True)
        self.records.append(record)

    def finish(self):
        with self.original_open(self.work / 'prediction_storage_worker.json', 'x') as stream:
            json.dump({'computed_predictions': self.records,
                       'new_prediction_payload_bytes': self.written,
                       'inference_skipped': False}, stream, sort_keys=True)


class _PredictionBuffer(io.BytesIO):
    def __init__(self, owner, path):
        super().__init__()
        self.owner, self.path = owner, path

    def write(self, value):
        if self.tell() + len(value) > self.owner.config['max_prediction_bytes']:
            raise OSError('Prediction memory serialization limit exceeded')
        return super().write(value)

    def seek(self, offset, whence=0):
        position = super().seek(offset, whence)
        if position > self.owner.config['max_prediction_bytes']:
            raise OSError('Prediction memory serialization limit exceeded')
        return position

    def truncate(self, size=None):
        if (self.tell() if size is None else size) > self.owner.config['max_prediction_bytes']:
            raise OSError('Prediction memory serialization limit exceeded')
        return super().truncate(size)

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.owner.commit(self.path, self.getvalue())
        finally:
            self.close()
        return False


class PredictionPool:
    """Host-side append-only catalog; never mounted into generated programs."""
    def __init__(self, workspace, root, *, max_pool_bytes, minimum_free_bytes):
        from .field_storage import ArrayPool, exact_child
        # Reuse its canonical-root, immutable-policy, flock and reserve contract.
        self.guard = ArrayPool(workspace, root, max_pool_bytes=max_pool_bytes,
                               minimum_free_bytes=minimum_free_bytes)
        self.root = self.guard.root
        exact_child(self.root, self.root / 'objects').mkdir(exist_ok=True)
        exact_child(self.root, self.root / 'cases').mkdir(exist_ok=True)

    def snapshot(self, sample_keys):
        from .field_storage import exact_child
        known = {}
        with self.guard.locked():
            for key in set(sample_keys):
                if not DIGEST.fullmatch(key):
                    raise ValueError('Invalid input-image identity')
                directory = exact_child(self.root, self.root / 'cases' / key)
                if not directory.exists():
                    continue
                for path in directory.glob('*.json'):
                    path = exact_child(directory, path)
                    value = json.loads(path.read_text())
                    digest = path.stem
                    if (not DIGEST.fullmatch(digest) or type(value.get('size')) is not int
                            or value['size'] < 1 or value != {'sha256': digest, 'size': value['size']}):
                        raise ValueError('Invalid immutable prediction membership')
                    known[digest] = value['size']
        return known

    def finalize(self, output, sample_keys, known):
        from .field_storage import exact_child, sha256, new_json, allocated_bytes
        output = exact_child(self.guard.workspace, output)
        if any(not isinstance(k, str) or not DIGEST.fullmatch(k) for k in sample_keys):
            raise ValueError('Invalid input-image identity')
        if output.stat().st_dev != self.root.stat().st_dev:
            raise ValueError('Prediction pool and new outputs must share one filesystem')
        result = json.loads(exact_child(output, output / 'result.json').read_text())
        expected = [{'index': i, 'file': f'prediction-{i:04d}.npz'} for i in range(len(sample_keys))]
        if result.get('predictions') != expected:
            raise ValueError('Prediction output cohort differs from the executed request')
        trace = json.loads(exact_child(output, output / 'prediction_storage_worker.json').read_text())
        records = trace['computed_predictions']
        if ([r['name'] for r in records] != [r['file'] for r in expected]
                or trace.get('inference_skipped') is not False):
            raise ValueError('Missing actual serialization records')
        linked, new_objects, existing_independent = 0, 0, 0
        with self.guard.locked():
            for record, key in zip(records, sample_keys):
                digest, size = record['sha256'], record['size']
                if not DIGEST.fullmatch(digest) or type(size) is not int or size < 1:
                    raise ValueError('Invalid serialized prediction reference')
                path = exact_child(output, output / record['name'])
                canonical = exact_child(self.root, self.root / 'objects' / (digest + '.npz'))
                kind = record['kind']
                if kind in ('cached', 'local'):
                    reference = exact_child(output, path.with_suffix('.npz.ref.json'))
                    if json.loads(reference.read_text()) != record or path.exists():
                        raise ValueError('Ambiguous or changed prediction reference')
                    if kind == 'cached':
                        if known.get(digest) != size:
                            raise ValueError('Reference was not visible in this job snapshot')
                        source = canonical
                    else:
                        if not PREDICTION_NAME.fullmatch(record['source']) or record['source'] >= path.name:
                            raise ValueError('Local reference must name an earlier prediction')
                        source = exact_child(output, output / record['source'])
                    if source.stat().st_size != size or sha256(source) != digest:
                        raise ValueError('Referenced prediction content changed')
                    os.link(source, path)
                    linked += 1
                elif kind != 'new':
                    raise ValueError('Unknown prediction storage kind')
                if path.stat().st_size != size or sha256(path) != digest:
                    raise ValueError('Actual computed prediction hash mismatch')
                membership = exact_child(self.root, self.root / 'cases' / key / (digest + '.json'))
                metadata_growth = (4096 if not membership.exists() else 0) + (4096 if not membership.parent.exists() else 0)
                growth = (0 if canonical.exists() else ((size + 4095)//4096)*4096) + metadata_growth
                if allocated_bytes(self.root) + growth > self.guard.policy['max_pool_bytes']:
                    raise OSError('Prediction-pool storage budget exceeded')
                space = os.statvfs(self.root)
                if space.f_bavail * space.f_frsize < self.guard.policy['minimum_free_bytes'] + metadata_growth:
                    raise OSError('Prediction-pool disk reserve violated')
                if canonical.exists():
                    if canonical.stat().st_size != size or sha256(canonical) != digest:
                        raise ValueError('Immutable prediction object changed')
                    if not os.path.samefile(path, canonical):
                        existing_independent += 1  # Concurrent first writes remain; no deletion.
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
                'hardlinked_new_paths': linked, 'concurrent_or_unexposed_duplicates_retained': existing_independent,
                'new_prediction_payload_bytes': trace['new_prediction_payload_bytes'],
                'existing_files_replaced': 0, 'inference_skipped': False}
