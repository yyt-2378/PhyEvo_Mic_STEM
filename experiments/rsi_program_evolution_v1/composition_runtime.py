"""Sandbox-only callable capability graph, copied as /program/capabilities.py.

This changes execution composition, not the evaluator or scientific methods.
Every inherited node resolves its own parent; rejected nodes are never parents.
"""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

_ROOT = Path('/program')
_CONFIG = None
_MODULES = {}
_STATES = {}
_STACK = []
_TRACE = []


def config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = json.loads((_ROOT / 'composition.json').read_text())
    return _CONFIG


def _module(name, filename):
    if Path(filename).name != filename:
        raise ValueError('Explicit program basename required')
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, _ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def _state(node):
    if node not in _STATES:
        filename = config()['nodes'][node]['checkpoint']
        if filename is None:
            _STATES[node] = {}
        else:
            if Path(filename).name != filename:
                raise ValueError('Explicit checkpoint basename required')
            with np.load(_ROOT / filename, allow_pickle=False) as arrays:
                _STATES[node] = {k: arrays[k] for k in arrays.files}
            for value in _STATES[node].values():
                value.setflags(write=False)
    return _STATES[node]


@contextmanager
def _enter(node):
    if node in _STACK or len(_STACK) >= 32:
        raise ValueError('Cyclic or excessive capability ancestry')
    _STACK.append(node)
    try:
        yield
    finally:
        assert _STACK.pop() == node


def _observable(sample):
    if any(k in sample for k in ('human', 'known', 'points_xy', 'label')):
        raise ValueError('Strip training supervision before capability inference')
    return sample


def _result(result, sample):
    score = np.asarray(result['score'])
    support = np.asarray(result['support'])
    if (score.shape != sample['image'].shape or support.shape != score.shape
            or not np.isfinite(score).all() or support.dtype != np.bool_
            or np.any(support & ~np.asarray(sample['valid'], bool))):
        raise ValueError('Invalid composed prediction')
    return {'score': score.astype(np.float32), 'support': support}


def branch(name, sample, context):
    """Unmodified retained branch; no labels, source IDs or evaluator access."""
    _observable(sample)
    legacy = _module('retained', 'retained.py')
    if name not in legacy.CONFIG['active_branches']:
        raise KeyError(name)
    _TRACE.append({'call': 'branch', 'branch': name})
    return _result(legacy.Branches(sample, context)[name], sample)


def frozen_best(sample, context):
    """The actually replay-verified F41-r06 whole system, not an oracle."""
    _observable(sample)
    legacy = _module('retained', 'retained.py')
    branches = legacy.Branches(sample, context)
    choice = legacy.module('router', 'router.py').route(
        sample, branches, legacy.weights('router', legacy.CONFIG['router_checkpoint']), context)
    if choice['branch'] not in legacy.CONFIG['active_branches']:
        raise ValueError('Unknown retained branch')
    _TRACE.append({'call': 'frozen_best', 'branch': choice['branch']})
    return _result(branches[choice['branch']], sample)


def parent(sample, context):
    """Parent of the executing node, not the root request's global parent."""
    if not _STACK:
        raise ValueError('parent() requires an executing node')
    node = config()['nodes'][_STACK[-1]]['parent']
    _TRACE.append({'call': 'parent', 'node': node})
    return frozen_best(sample, context) if node == 'frozen' else execute(node, sample, context)


def execute(node, sample, context, state=None):
    _observable(sample)
    definition = config()['nodes'][node]
    with _enter(node):
        module = _module(node, definition['program'])
        state = _state(node) if state is None else state
        if definition['arm'] == 'route':
            choice = module.route(sample, state, context)
            if not isinstance(choice, dict) or not isinstance(choice.get('reason'), str):
                raise ValueError('Route control must return branch and reason')
            if choice.get('branch') == 'parent':
                result = parent(sample, context)
            elif choice.get('branch') == 'frozen':
                result = frozen_best(sample, context)
            else:
                result = branch(choice.get('branch'), sample, context)
        else:
            result = module.predict(sample, state, context)
    return _result(result, sample)


def fit(samples, context):
    node = config()['current']
    with _enter(node):
        return _module(node, config()['nodes'][node]['program']).fit(samples, context)


def predict(sample, state, context):
    _TRACE.clear()
    result = execute(config()['current'], sample, context, state)
    # The adapter, not candidate code, records actual inherited calls.
    index = predict.counter
    with Path('/work', f'composition-trace-{index:04d}.json').open('x') as stream:
        json.dump({'node': config()['current'], 'calls': _TRACE}, stream)
    predict.counter += 1
    return result


predict.counter = 0
