"""Append-only, lossless per-array storage for future experiment input staging.

No existing artifacts are removed or replaced. Job inputs contain only hard
links to their permitted arrays, never a mount of the entire shared store.
"""
from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

import numpy as np

FORMAT = 'phyevo-field-references-v1'
MAX_ARRAY_BYTES = 256 * 1024**2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def exact_child(root, path):
    root, path = Path(root).absolute(), Path(path).absolute()
    if root.resolve() != root or path.resolve() != path or path == root:
        raise ValueError('Canonical real child paths required')
    path.relative_to(root)
    return path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def new_json(path, value):
    with path.open('x') as stream:
        stream.write(canonical(value) + '\n')


def array_identity(value):
    value = np.asarray(value)
    order = 'F' if value.flags.f_contiguous and not value.flags.c_contiguous else 'C'
    value = np.array(value, order=order, copy=True)
    if (value.dtype.kind not in 'biufc' or value.dtype.fields or value.nbytes > MAX_ARRAY_BYTES
            or not np.isfinite(value).all()):
        raise ValueError('Finite bounded numeric arrays required')
    spec = {'dtype': value.dtype.str, 'shape': list(value.shape), 'nbytes': value.nbytes, 'order': order}
    h = hashlib.sha256(canonical(spec).encode() + b'\0' + value.tobytes(order=order)).hexdigest()
    return value, spec, h


def allocated_bytes(root):
    seen, total = set(), 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(directory) / name
            s = path.lstat()
            if not (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)) or path.resolve() != path:
                raise ValueError('Unexpected link or special storage entry')
            key = s.st_dev, s.st_ino
            if key not in seen:
                total += s.st_blocks * 512
                seen.add(key)
    return total


class ArrayPool:
    def __init__(self, workspace, root, *, max_pool_bytes, minimum_free_bytes):
        self.workspace = Path(workspace).absolute()
        self.root = exact_child(workspace, root)
        if (type(max_pool_bytes) is not int or max_pool_bytes < 65536
                or type(minimum_free_bytes) is not int or minimum_free_bytes < 0):
            raise ValueError('An explicit finite input-pool budget and disk reserve are required')
        if self.root.exists() and not (self.root / 'policy.json').exists():
            if any(p.name != 'pool.lock' for p in self.root.iterdir()):
                raise ValueError('Existing nonempty directory is not an initialized field pool')
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = {'format': FORMAT, 'max_pool_bytes': max_pool_bytes,
                       'minimum_free_bytes': minimum_free_bytes}
        with self.locked():
            policy = self.root / 'policy.json'
            if policy.exists():
                if json.loads(policy.read_text()) != self.policy:
                    raise ValueError('Input-pool policy cannot silently change')
            else:
                new_json(policy, self.policy)

    @contextmanager
    def locked(self):
        path = exact_child(self.root, self.root / 'pool.lock')
        with path.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def put(self, value):
        value, spec, identity = array_identity(value)
        path = exact_child(self.root, self.root / (identity + '.npz'))
        marker = exact_child(self.root, self.root / (identity + '.json'))
        with self.locked():
            if marker.exists():
                reference = json.loads(marker.read_text())
                if reference != {'identity': identity, 'file': path.name, 'array': spec,
                                 'sha256': sha256(path)}:
                    raise ValueError('Immutable array object changed')
                return reference
            if path.exists():
                raise FileExistsError('Incomplete object preserved, not overwritten')
            buffer = io.BytesIO()
            np.savez_compressed(buffer, array=value)
            payload = buffer.getvalue()
            reference = {'identity': identity, 'file': path.name, 'array': spec,
                         'sha256': hashlib.sha256(payload).hexdigest()}
            # Reserve filesystem blocks for the payload and its metadata before
            # writing. This bounds this pool, not arbitrary candidate outputs.
            growth = ((len(payload) + 4095) // 4096 + 2) * 4096
            available = os.statvfs(self.root)
            if allocated_bytes(self.root) + growth > self.policy['max_pool_bytes']:
                raise OSError('Input-pool storage budget exceeded before payload write')
            if available.f_bavail * available.f_frsize - growth < self.policy['minimum_free_bytes']:
                raise OSError('Disk reserve would be violated before payload write')
            with path.open('xb') as stream:
                stream.write(payload)
                stream.flush(); os.fsync(stream.fileno())
            new_json(marker, reference)
            return reference

    def stage(self, folder, name, fields):
        folder = exact_child(self.workspace, folder)
        if not re.fullmatch(r'sample-\d{4}\.fields\.json', name):
            raise ValueError('Canonical input directory and explicit sample name required')
        destination = exact_child(folder, folder / name)
        if destination.exists():
            raise FileExistsError('Sample manifest already exists')
        entries = []
        arrays = exact_child(folder, folder / 'arrays')
        arrays.mkdir(exist_ok=True)
        for key, value in fields.items():
            if not isinstance(key, str) or not key:
                raise ValueError('Named fields required')
            ref = self.put(value)
            source = exact_child(self.root, self.root / ref['file'])
            target = exact_child(arrays, arrays / ref['file'])
            if target.exists():
                if not os.path.samefile(source, target):
                    raise ValueError('Unexpected pre-existing staged array')
            else:
                os.link(source, target)
            entries.append({'name': key, **ref})
        new_json(destination, {'format': FORMAT, 'fields': entries})
        return {'file': name}


class ArrayBundle(Mapping):
    """The NpzFile surface used by the unchanged trusted legacy worker."""
    def __init__(self, path, loader):
        self.path, self.loader = Path(path), loader
        manifest = json.loads(self.path.read_text())
        if manifest.get('format') != FORMAT:
            raise ValueError('Unknown field-reference schema')
        self.fields = manifest['fields']
        self.files = [item['name'] for item in self.fields]
        if len(set(self.files)) != len(self.files):
            raise ValueError('Duplicate field names')

    def __getitem__(self, key):
        ref = next((r for r in self.fields if r['name'] == key), None)
        if ref is None:
            raise KeyError(key)
        if (not re.fullmatch(r'[a-f0-9]{64}', ref['identity'])
                or ref['file'] != ref['identity'] + '.npz'):
            raise ValueError('Invalid array object reference')
        path = exact_child(self.path.parent, self.path.parent / 'arrays' / ref['file'])
        if sha256(path) != ref['sha256']:
            raise ValueError('Staged array hash mismatch')
        with self.loader(path, allow_pickle=False) as values:
            if values.files != ['array']:
                raise ValueError('Unexpected array object fields')
            value, spec, identity = array_identity(values['array'])
        if spec != ref['array'] or identity != ref['identity']:
            raise ValueError('Staged array metadata mismatch')
        return value

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def close(self):
        pass


def input_loader(legacy_load, input_root):
    input_root = Path(input_root)

    def load(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and str(file).endswith('.fields.json'):
            path = exact_child(input_root, Path(file))
            if (path.parent != input_root or args or kwargs != {'allow_pickle': False}
                    or not re.fullmatch(r'sample-\d{4}\.fields\.json', path.name)):
                raise ValueError('Only the explicit trusted-worker manifest load is supported')
            return ArrayBundle(path, legacy_load)
        return legacy_load(file, *args, **kwargs)

    return load
