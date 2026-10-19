"""Whole-system candidates: immutable capabilities, checkpoints and a router.

Generated programs run only in the existing OS sandbox. The label-side scorer
checks the selected output against the immutable capability prediction, so a
router cannot quietly rewrite an algorithm or enlarge the output budget.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from .contracts import Proposal, canonical, digest, file_hash, inside, write_new
from .controller import BASELINE_POLICY
from .evaluation import compare, evaluate, summarize
from .learning_study import trace_check


RAW_PIPELINE = "def predict(sample,state,context):\n    return {'score':sample['base'],'support':sample['support']}\n"
RAW_ROUTER = "def route(sample,branches,state,context):\n    return {'branch':'b0','reason':'Unchanged original framework'}\n"


def read(path):
    return json.loads(Path(path).read_text())


def code_contract(code, entry, steps):
    if not isinstance(code, str) or len(code.encode()) > 80000:
        raise ValueError('Invalid or oversized program')
    names = {n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
    if entry not in names or (steps and 'fit' not in names):
        raise ValueError('Missing executable entry point or fit')
    if type(steps) is not int or not 0 <= steps <= 4096:
        raise ValueError('Invalid optimizer step count')


@dataclass(frozen=True)
class SystemProposal:
    parent: str
    operation: str
    observation: str
    hypothesis: str
    intervention: str
    falsifier: str
    active_branches: list[str]
    router_code: str
    router_training_steps: int
    inherit_router_state: bool
    new_branch: dict | None

    def validate(self, incumbent, library):
        if self.parent != incumbent:
            raise ValueError('System proposal must inherit the current whole-system checkpoint')
        if self.operation not in {'route_only', 'add_branch', 'replace_branch'}:
            raise ValueError('Unknown integration operation')
        if not all(isinstance(v, str) and v.strip() for v in
                   (self.observation, self.hypothesis, self.intervention, self.falsifier)):
            raise ValueError('Concrete observation, mechanism and falsifier required')
        if type(self.inherit_router_state) is not bool:
            raise ValueError('Explicit router-state inheritance required')
        code_contract(self.router_code, 'route', self.router_training_steps)
        available = set(library)
        if self.operation == 'route_only':
            if self.new_branch is not None:
                raise ValueError('route_only cannot conceal a new algorithm')
        else:
            b = self.new_branch
            if not isinstance(b, dict) or set(b) != {'name', 'pipeline_code', 'training_steps', 'parent_branch', 'replaces'}:
                raise ValueError('Explicit new capability definition required')
            if not re.fullmatch(r'b[0-9]+', b['name']) or b['name'] in available:
                raise ValueError('New capability needs a fresh anonymous name')
            if b['parent_branch'] is not None and b['parent_branch'] not in available:
                raise ValueError('Unknown weight-state parent')
            code_contract(b['pipeline_code'], 'predict', b['training_steps'])
            if self.operation == 'replace_branch':
                if b['replaces'] not in available - {'b0'} or b['replaces'] in self.active_branches:
                    raise ValueError('Replace an optional active branch, not the protected base')
            elif b['replaces'] is not None:
                raise ValueError('An addition cannot silently replace a branch')
            available.add(b['name'])
            if b['name'] not in self.active_branches:
                raise ValueError('New capability must be available to the proposed router')
        if (not isinstance(self.active_branches, list) or len(self.active_branches) != len(set(self.active_branches))
                or 'b0' not in self.active_branches or not set(self.active_branches) <= available):
            raise ValueError('Unique existing branches and the protected base are required')
        return self


SYSTEM_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': list(SystemProposal.__dataclass_fields__),
    'properties': {
        **{k: {'type': 'string'} for k in ('parent', 'observation', 'hypothesis', 'intervention', 'falsifier', 'router_code')},
        'operation': {'enum': ['route_only', 'add_branch', 'replace_branch']},
        'active_branches': {'type': 'array', 'items': {'type': 'string'}},
        'router_training_steps': {'type': 'integer', 'minimum': 0, 'maximum': 4096},
        'inherit_router_state': {'type': 'boolean'},
        'new_branch': {'anyOf': [{'type': 'null'}, {'type': 'object', 'additionalProperties': False,
            'required': ['name', 'pipeline_code', 'training_steps', 'parent_branch', 'replaces'],
            'properties': {'name': {'type': 'string'}, 'pipeline_code': {'type': 'string'},
                'training_steps': {'type': 'integer', 'minimum': 0, 'maximum': 4096},
                'parent_branch': {'type': ['string', 'null']}, 'replaces': {'type': ['string', 'null']}}}]},
    },
}


def wrapper(active):
    """Trusted adapter. Candidate routing code cannot change the selected arrays."""
    return '''import importlib.util
import json
import numpy as np
spec=importlib.util.spec_from_file_location('router','/program/router.py')
router=importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)
ACTIVE=ACTIVE_VALUE
counter=0
def unpack(sample):
    observation={k:v for k,v in sample.items() if not k.startswith('branch_')}
    branches={name:{'score':sample['branch_'+name+'_score'],
                    'support':sample['branch_'+name+'_support']} for name in ACTIVE}
    return observation,branches
def fit(samples,context):
    prepared=[]
    for sample in samples:
        observation,branches=unpack(sample)
        prepared.append(dict(observation,branches=branches))
    return router.fit(prepared,context)
def predict(sample,state,context):
    global counter
    observation,branches=unpack(sample)
    result=router.route(observation,branches,state,context)
    if not isinstance(result,dict) or result.get('branch') not in ACTIVE:
        raise ValueError('Router must select an available capability')
    if not isinstance(result.get('reason'),str):
        raise ValueError('Router must explain the observable adoption condition')
    with open('/work/route-%04d.json'%counter,'x') as f:
        json.dump({'branch':result['branch'],'reason':result['reason']},f)
    counter+=1
    return branches[result['branch']]
'''.replace('ACTIVE_VALUE', repr(list(active)))


def joint_admission(new, old, base, protocol, best_by_group):
    """Mean gain cannot purchase regression on an already retained capability."""
    decision = compare(new, old, base, protocol)
    summary = summarize(new)
    if len(summary) < 2:
        raise ValueError('Whole-system admission requires at least two development sources')
    memory_regressions = [g for g in summary if summary[g]['mean'] < best_by_group[g] - protocol.maximum_group_regression]
    decision['retained_capability_regressions'] = memory_regressions
    decision['accepted'] = bool(decision['accepted'] and not memory_regressions)
    decision['reason'] = 'whole_system_gain_and_retention' if decision['accepted'] else 'insufficient_system_gain_or_retention'
    return decision


class SystemExecutor:
    def __init__(self, root, archive, sandbox, samples):
        self.workspace, self.archive, self.sandbox = Path(root), archive, sandbox
        self.samples = tuple(samples)
        self.protocol = archive.protocol
        self.root = archive.root.parent
        (self.root / 'capabilities').mkdir(exist_ok=True)
        (self.root / 'system_jobs').mkdir(exist_ok=True)

    def checked_ref(self, ref):
        path = inside(self.root, ref['path'])
        if file_hash(path) != ref['sha256']:
            raise ValueError('Bound program, state or prediction changed')
        return path

    def ref(self, path):
        path = inside(self.root, path)
        return {'path': str(path.relative_to(self.root)), 'sha256': file_hash(path)}

    def cohorts(self, split):
        if split == 'audit':
            if not self.archive.events('search_sealed'):
                raise ValueError('All search must be sealed before transfer inference')
            return [s for s in self.samples if self.protocol.split(s.group) == 'audit']
        if split != 'development':
            raise ValueError('Unknown system evaluation cohort')
        return [s for s in self.samples if self.protocol.split(s.group) != 'audit']

    def stage(self, folder, cohort, *, mode='predict', steps=0, state=None, branches=None):
        folder.mkdir(parents=True, exist_ok=False)
        records = []
        for index, sample in enumerate(cohort):
            fields = dict(sample.fields)
            if branches:
                for name, prediction in branches.items():
                    with np.load(self.checked_ref(prediction[sample.key]), allow_pickle=False) as z:
                        fields['branch_' + name + '_score'] = z['score']
                        fields['branch_' + name + '_support'] = z['support']
            if mode == 'fit':
                fields.update(human=sample.human, known=sample.known)
            name = f'sample-{index:04d}.npz'
            with (folder / name).open('xb') as stream:
                np.savez_compressed(stream, **fields)
            records.append({'file': name})
        if state:
            source = self.checked_ref(state)
            with (folder / 'state.npz').open('xb') as stream:
                stream.write(source.read_bytes())
        write_new(folder / 'request.json', {'mode': mode, 'seed': self.protocol.seeds[0],
            'training_steps': steps, 'samples': records, 'state_file': bool(state),
            'device': 'cuda' if self.sandbox.gpu is not None else 'cpu'})

    def fit(self, program, folder, steps, parent_state=None, branches=None):
        if not steps:
            return parent_state, None
        cohort = [s for s in self.samples if self.protocol.split(s.group) == 'train']
        self.stage(folder / 'input', cohort, mode='fit', steps=steps, state=parent_state, branches=branches)
        receipt = self.sandbox.run(program, folder / 'input', folder / 'output')
        if receipt['status'] != 'ok':
            raise RuntimeError((folder / 'output/stderr.log').read_text()[-4000:])
        checkpoint = folder / 'output/state.npz'
        with np.load(checkpoint, allow_pickle=False) as arrays:
            trace = trace_check(dict(arrays), steps)
        return self.ref(checkpoint), {'trace': trace, 'execution': receipt,
            'fit_groups': sorted({s.group for s in cohort})}

    def capability(self, definition):
        code_contract(definition['pipeline_code'], 'predict', definition['training_steps'])
        identity = digest({'definition': definition, 'data_version': self.protocol.data_version,
                           'train_groups': self.protocol.train_groups, 'seed': self.protocol.seeds[0]})
        folder = self.root / 'capabilities' / identity
        if (folder / 'capability.json').exists():
            value = read(folder / 'capability.json')
            if file_hash(folder / 'definition.json') != value['definition_sha256']:
                raise ValueError('Capability definition changed')
            self.checked_ref(value['program'])
            if value['checkpoint']:
                self.checked_ref(value['checkpoint'])
            return self.ref(folder / 'capability.json')
        folder.mkdir(exist_ok=False)
        (folder / 'program').mkdir()
        with (folder / 'program/pipeline.py').open('x') as stream:
            stream.write(definition['pipeline_code'])
        write_new(folder / 'definition.json', definition)
        state, training = self.fit(folder / 'program', folder / 'fit', definition['training_steps'], definition.get('parent_state'))
        write_new(folder / 'capability.json', {'identity': identity, 'definition_sha256': file_hash(folder / 'definition.json'),
            'program': self.ref(folder / 'program/pipeline.py'), 'checkpoint': state, 'training': training,
            'weight_training_groups': sorted({s.group for s in self.samples if self.protocol.split(s.group) == 'train'}) if state else [],
            'data_version': self.protocol.data_version})
        return self.ref(folder / 'capability.json')

    def predict_capability(self, binding, split):
        cap_path = self.checked_ref(binding); cap = read(cap_path)
        program = self.checked_ref(cap['program']).parent
        if cap['checkpoint']:
            self.checked_ref(cap['checkpoint'])
        cohort = self.cohorts(split)
        folder = cap_path.parent / split
        if (folder / 'predictions.json').exists():
            event = next(e['body'] for e in self.archive.events('capability_predictions_recorded')
                         if e['body']['capability'] == cap['identity'] and e['body']['split'] == split)
            if file_hash(folder / 'predictions.json') != event['sha256']:
                raise ValueError('Capability feedback cache changed')
            receipt = read(folder / 'predictions.json')
            for ref in receipt['predictions'].values():
                self.checked_ref(ref)
            return receipt
        folder.mkdir(exist_ok=False)
        self.stage(folder / 'input', cohort, state=cap['checkpoint'])
        execution = self.sandbox.run(program, folder / 'input', folder / 'output')
        if execution['status'] != 'ok':
            raise RuntimeError((folder / 'output/stderr.log').read_text()[-4000:])
        predictions, rows = {}, []
        for index, sample in enumerate(cohort):
            path = folder / 'output' / f'prediction-{index:04d}.npz'
            with np.load(path, allow_pickle=False) as arrays:
                result = evaluate(sample.validation_target(), dict(arrays), self.protocol)
            predictions[sample.key] = self.ref(path)
            rows.append({'case': sample.key, 'group': sample.group, 'seed': self.protocol.seeds[0],
                         'role': self.protocol.split(sample.group), **result})
        receipt = {'predictions': predictions, 'rows': rows, 'summary': summarize(rows),
                   'execution': execution, 'checkpoint': cap['checkpoint'], 'capability': binding}
        write_new(folder / 'predictions.json', receipt)
        self.archive.event('capability_predictions_recorded', {'capability': cap['identity'], 'split': split,
            'sha256': file_hash(folder / 'predictions.json')})
        return receipt

    def register_system(self, proposal, parent_manifest=None):
        if parent_manifest is None:
            raise ValueError('Initial systems use register_initial')
        proposal.validate(self.archive.incumbent(), parent_manifest['library'])
        library = dict(parent_manifest['library'])
        if proposal.new_branch:
            b = proposal.new_branch
            parent_state = read(self.checked_ref(library[b['parent_branch']]))['checkpoint'] if b['parent_branch'] else None
            library[b['name']] = self.capability({'pipeline_code': b['pipeline_code'],
                'training_steps': b['training_steps'], 'parent_state': parent_state})
        parent_state = parent_manifest['router_checkpoint'] if proposal.inherit_router_state else None
        manifest = {'library': library, 'active_branches': proposal.active_branches,
            'parent_system': proposal.parent, 'router_parent_state': parent_state,
            'router_training_steps': proposal.router_training_steps, 'operation': proposal.operation}
        return self._register(proposal.router_code, manifest, proposal)

    def register_initial(self, library):
        manifest = {'library': library, 'active_branches': list(library), 'parent_system': None,
            'router_parent_state': None, 'router_training_steps': 0, 'operation': 'initial_old_framework'}
        return self._register(RAW_ROUTER, manifest, None)

    def _register(self, router_code, manifest, raw):
        pipeline = wrapper(manifest['active_branches'])
        manifest = dict(manifest, router_sha256=digest(router_code))
        p = Proposal('tune' if raw else 'generate', (raw.parent,) if raw else (),
            (raw.observation if raw else 'Preserve the original guarded framework') + '\nSystem specification: ' + canonical(manifest),
            raw.hypothesis if raw else 'Existing behavior is reproduced exactly',
            raw.intervention if raw else 'Register capabilities but choose only the original branch',
            'Whole-system gain without losing retained capabilities',
            raw.falsifier if raw else 'Any changed original prediction', pipeline, BASELINE_POLICY,
            target='routing', training_steps=manifest['router_training_steps'])
        identity = self.archive.add(p)
        folder = Path(self.archive.candidate(identity)['directory'])
        with (folder / 'router.py').open('x') as stream:
            stream.write(router_code)
        write_new(folder / 'system_spec.json', manifest)
        self.archive.event('system_registered', {'candidate': identity, 'specification_sha256': file_hash(folder / 'system_spec.json'),
            'router_sha256': file_hash(folder / 'router.py')})
        return identity

    def system(self, identity):
        folder = Path(self.archive.candidate(identity)['directory'])
        registered = next(e['body'] for e in self.archive.events('system_registered') if e['body']['candidate'] == identity)
        if file_hash(folder / 'system_spec.json') != registered['specification_sha256'] or file_hash(folder / 'router.py') != registered['router_sha256']:
            raise ValueError('Whole-system specification or router changed')
        spec = read(folder / 'system_spec.json')
        for binding in spec['library'].values():
            cap = read(self.checked_ref(binding)); self.checked_ref(cap['program'])
            if cap['checkpoint']:
                self.checked_ref(cap['checkpoint'])
        return folder, spec

    def frozen_state(self, identity):
        folder, spec = self.system(identity)
        state_path = folder / 'system_state.json'
        value = read(state_path)
        event = next(e['body'] for e in self.archive.events('system_state_frozen') if e['body']['candidate'] == identity)
        if file_hash(state_path) != event['sha256'] or value['specification_sha256'] != file_hash(folder / 'system_spec.json'):
            raise ValueError('Executable system state changed')
        if value['router_checkpoint']:
            self.checked_ref(value['router_checkpoint'])
        return dict(spec, **value)

    def execute(self, identity, split='development', *, replay=False):
        folder, spec = self.system(identity)
        cohort = self.cohorts(split)
        if replay:
            cohort = list(reversed(cohort))
        run_name = identity + '-' + split + ('-replay' if replay else '')
        run = self.root / 'system_jobs' / run_name
        run.mkdir(exist_ok=False)
        try:
            branches = {name: self.predict_capability(spec['library'][name], split)['predictions']
                        for name in spec['active_branches']}
            if split == 'development' and not replay:
                state, training = self.fit(folder, run / 'fit', spec['router_training_steps'],
                                           spec['router_parent_state'], branches)
            else:
                state = self.frozen_state(identity)['router_checkpoint']; training = None
            self.stage(run / 'input', cohort, state=state, branches=branches)
            execution = self.sandbox.run(folder, run / 'input', run / 'output')
            if execution['status'] != 'ok':
                raise RuntimeError((run / 'output/stderr.log').read_text()[-4000:])
            rows, predictions, choices = [], {}, Counter()
            for index, sample in enumerate(cohort):
                path = run / 'output' / f'prediction-{index:04d}.npz'
                route = read(run / 'output' / f'route-{index:04d}.json')
                if route['branch'] not in branches:
                    raise ValueError('Unknown selected branch')
                with np.load(path, allow_pickle=False) as result:
                    with np.load(self.checked_ref(branches[route['branch']][sample.key]), allow_pickle=False) as expected:
                        if any(not np.array_equal(result[k], expected[k]) for k in ('score', 'support')):
                            raise ValueError('Router changed a capability output instead of selecting it')
                    measured = evaluate(sample.validation_target(), dict(result), self.protocol)
                choices[route['branch']] += 1
                rows.append({'case': sample.key, 'group': sample.group, 'seed': self.protocol.seeds[0],
                    'role': self.protocol.split(sample.group), 'branch': route['branch'], 'reason': route['reason'], **measured})
                predictions[sample.key] = self.ref(path)
            receipt = {'rows': rows, 'summary': summarize(rows), 'predictions': predictions, 'choices': dict(choices),
                'execution': execution, 'router_training': training, 'router_checkpoint': state,
                'system_specification_sha256': file_hash(folder / 'system_spec.json'), 'actual_execution': True,
                'cohort_semantics': 'training-source retention replay plus source-held-out development' if split == 'development' else 'sealed transfer audit',
                'no_test_oracle_selection': True}
            if split == 'development' and not replay:
                write_new(folder / 'system_state.json', {'candidate': identity, 'router_checkpoint': state,
                    'specification_sha256': file_hash(folder / 'system_spec.json')})
                self.archive.event('system_state_frozen', {'candidate': identity, 'sha256': file_hash(folder / 'system_state.json')})
        except (RuntimeError, ValueError, KeyError, OSError) as error:
            receipt = {'error': str(error), 'actual_execution': True}
        write_new(run / 'result.json', receipt)
        archive_split = 'audit' if split == 'audit' else 'smoke' if replay else 'validation'
        trial = self.archive.record_trial(identity, archive_split, 'ok' if 'rows' in receipt else 'failed', receipt)
        return trial, receipt
