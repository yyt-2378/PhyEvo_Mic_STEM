"""Additive research-state contracts for a NEW study, not the frozen v2 batch.

Diagnosis is not an executable action. A single action determines the old
executor operation without repairing a model reply. Successfully measured but
unadmitted capabilities enter a separate research bank, not the live system.
"""
from copy import deepcopy
import re

import numpy as np

from .contracts import digest, file_hash
from .system_state import SYSTEM_SCHEMA, SystemProposal, read
from .system_study import checkpoint


ACTIONS = {
    'route_existing': 'route_only',
    'create_analysis': 'add_branch',
    'train_student': 'add_branch',
    'replace_analysis': 'replace_branch',
    'replace_student': 'replace_branch',
}
EVENT = 'research_bank_capability_retained_v2'
DIAGNOSES = ('routing_gap', 'capability_gap', 'mixed', 'inconclusive')
MASK_FEATURE_CONTRACT = {
    'physical_valid_fraction_of_image': 'count_nonzero(physics_valid) / physics_valid.size',
    'physical_valid_fraction_within_valid': 'count_nonzero(physics_valid & valid) / count_nonzero(valid)',
    'physical_valid_fraction_within_support': 'count_nonzero(physics_valid & support) / count_nonzero(support)',
    'valid_fraction_of_image': 'count_nonzero(valid) / valid.size',
    'support_fraction_of_image': 'count_nonzero(support) / support.size',
}


def mask_observables(fields):
    """Explicit denominators; an unavailable conditional fraction is not zero.

    Only observable masks are consumed. No image, coordinates or labels are
    exported. A threshold learned from the global fraction must not silently be
    used on a within-support fraction (often identically one).
    """
    masks = {k: np.asarray(fields[k]) for k in ('valid', 'support', 'physics_valid')}
    v, s, p = (masks[k] for k in ('valid', 'support', 'physics_valid'))
    if (v.ndim != 2 or not v.size or any(x.dtype != np.bool_ or x.shape != v.shape for x in masks.values())):
        raise ValueError('Aligned, nonempty 2D boolean observable masks required')
    conditional = lambda a, b: float(np.count_nonzero(a & b) / np.count_nonzero(b)) if np.any(b) else None
    return {'physical_valid_fraction_of_image': float(p.mean()),
        'physical_valid_fraction_within_valid': conditional(p, v),
        'physical_valid_fraction_within_support': conditional(p, s),
        'valid_fraction_of_image': float(v.mean()), 'support_fraction_of_image': float(s.mean())}


def allowed_actions(arm):
    if arm not in ('route', 'full'):
        raise ValueError('Unknown research arm')
    return ['route_existing'] if arm == 'route' else list(ACTIONS)


def response_schema(arm):
    system = deepcopy(SYSTEM_SCHEMA)
    system['required'].remove('operation')
    del system['properties']['operation']
    diagnosis = {'type': 'object', 'additionalProperties': False,
        'required': ['kind', 'evidence', 'alternatives', 'new_capacity_test'],
        'properties': {'kind': {'enum': list(DIAGNOSES)},
            **{k: {'type': 'string'} for k in ('evidence', 'alternatives', 'new_capacity_test')}}}
    return {'type': 'object', 'additionalProperties': False,
        'required': ['diagnosis', 'action', 'system'], 'properties': {
            'diagnosis': diagnosis, 'action': {'enum': allowed_actions(arm)}, 'system': system}}


def next_branch_name(library):
    if not library or any(re.fullmatch(r'b[0-9]+', name) is None for name in library):
        raise ValueError('Anonymous nonempty capability library required')
    return 'b' + str(max(int(name[1:]) for name in library) + 1)


def parse_action(value, arm, incumbent, library):
    """Diagnosis may identify a capability gap even in a routing-only control."""
    if not isinstance(value, dict) or set(value) != {'diagnosis', 'action', 'system'}:
        raise ValueError('Expected exactly diagnosis, action and system')
    diagnosis = value['diagnosis']
    fields = {'kind', 'evidence', 'alternatives', 'new_capacity_test'}
    if (not isinstance(diagnosis, dict) or set(diagnosis) != fields
            or diagnosis['kind'] not in DIAGNOSES
            or any(not isinstance(diagnosis[k], str) or not diagnosis[k].strip()
                   for k in fields - {'kind'})):
        raise ValueError('Explicit diagnosis, evidence, alternatives and falsifier required')
    action = value['action']
    if action not in allowed_actions(arm):
        raise ValueError('Action is unavailable in this research arm')
    raw = value['system']
    required = set(SystemProposal.__dataclass_fields__) - {'operation'}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError('System fields must match the schema; action alone specifies operation')
    proposal = SystemProposal(**raw, operation=ACTIONS[action]).validate(incumbent, library)
    branch = proposal.new_branch
    if branch:
        if branch['name'] != next_branch_name(library):
            raise ValueError('Use the next reserved bank alias for a new capability')
        learning = action in ('train_student', 'replace_student')
        if learning != (branch['training_steps'] > 0):
            raise ValueError('Student action must train; analysis action must be deterministic')
    if proposal.router_training_steps + (branch['training_steps'] if branch else 0) > 4096:
        raise ValueError('Combined optimizer budget exceeded')
    return proposal, diagnosis


class ResearchBank:
    """Measured programs and states remain addressable after whole-system rejection.

    Entries are immutable, content-bound archive events. Keeping an entry grants
    no deployment authority: the incumbent and its active branches do not change.
    A subsequent model proposal must explicitly select/modify an entry, then pass
    the unchanged whole-system gate before it becomes the incumbent.
    """
    def __init__(self, runner):
        self.runner = runner
        self.archive = runner.archive

    def entries(self):
        result = {}
        for event in self.archive.events(EVENT):
            entry = event['body']
            name = entry['name']
            if name in result:
                raise ValueError('Research bank alias was rebound')
            cap = read(self.runner.checked_ref(entry['binding']))
            program = self.runner.checked_ref(cap['program'])
            state_hash = None
            if cap['checkpoint']:
                state_hash = file_hash(self.runner.checked_ref(cap['checkpoint']))
            if (file_hash(program) != entry['program_sha256']
                    or state_hash != entry['checkpoint_sha256']):
                raise ValueError('Research capability state changed')
            if set(cap['weight_training_groups']) & set(
                    self.runner.protocol.validation_groups + self.runner.protocol.audit_groups):
                raise ValueError('Research bank checkpoint trained on excluded sources')
            feedback_path = self.runner.checked_ref(entry['feedback'])
            feedback = read(feedback_path)
            if feedback['capability'] != entry['binding']:
                raise ValueError('Research feedback belongs to another capability')
            recorded = [e['body'] for e in self.archive.events('capability_predictions_recorded')
                        if e['body']['capability'] == cap['identity'] and e['body']['split'] == 'development']
            if len(recorded) != 1 or recorded[0]['sha256'] != file_hash(feedback_path):
                raise ValueError('Research feedback has no unique executed provenance')
            # Do not accept a hand-written summary detached from actual arrays.
            for reference in feedback['predictions'].values():
                self.runner.checked_ref(reference)
            result[name] = entry
        return result

    def initialize(self):
        if self.archive.events(EVENT):
            raise FileExistsError('Research bank is already initialized')
        identity = self.archive.incumbent()
        _, spec = self.runner.system(identity)
        for binding in spec['library'].values():
            self.runner.predict_capability(binding, 'development')
        return self.retain_measured(identity)

    def retain_measured(self, candidate):
        """Record only completed analysis executions; never fit or select here."""
        before = self.archive.incumbent()
        _, spec = self.runner.system(candidate)
        existing = self.entries()
        additions = []
        for name, binding in spec['library'].items():
            if name in existing:
                if existing[name]['binding'] != binding:
                    raise ValueError('Cannot overwrite an existing research capability')
                continue
            cap_path = self.runner.checked_ref(binding)
            feedback_path = cap_path.parent / 'development/predictions.json'
            if not feedback_path.is_file():
                # Fit or inference failures retain their logs, not a usable capability.
                continue
            cap = read(cap_path)
            feedback = read(feedback_path)
            recorded = [e['body'] for e in self.archive.events('capability_predictions_recorded')
                        if e['body']['capability'] == cap['identity'] and e['body']['split'] == 'development']
            if (len(recorded) != 1 or recorded[0]['sha256'] != file_hash(feedback_path)
                    or feedback['capability'] != binding):
                raise ValueError('Unverified capability feedback')
            entry = {'name': name, 'binding': binding, 'origin_system': candidate,
                'feedback': self.runner.ref(feedback_path),
                'program_sha256': file_hash(self.runner.checked_ref(cap['program'])),
                'checkpoint_sha256': cap['checkpoint']['sha256'] if cap['checkpoint'] else None,
                'bank_entry_is_system_admission': False}
            self.archive.event(EVENT, entry)
            additions.append(name)
        self.entries()
        if self.archive.incumbent() != before:
            raise RuntimeError('Research retention must never change the deployed system')
        return additions

    def inherited_manifest(self):
        current = self.runner.frozen_state(self.archive.incumbent())
        library = {name: entry['binding'] for name, entry in self.entries().items()}
        if any(library.get(name) != binding for name, binding in current['library'].items()):
            raise ValueError('Research bank lost an incumbent capability')
        return dict(current, library=library)

    def checkpoint(self):
        state = {'system': checkpoint(self.archive, self.runner),
                 'research_bank_sha256': digest(self.entries())}
        return dict(state, research_checkpoint=digest(state))

    def catalog(self):
        """Caller anonymizes development source names; no arrays or labels exposed."""
        active = set(self.runner.frozen_state(self.archive.incumbent())['active_branches'])
        catalog = {}
        for name, entry in self.entries().items():
            cap = read(self.runner.checked_ref(entry['binding']))
            feedback = read(self.runner.checked_ref(entry['feedback']))
            catalog[name] = {'pipeline_code': self.runner.checked_ref(cap['program']).read_text(),
                'checkpoint_present': cap['checkpoint'] is not None,
                'checkpoint_sha256': entry['checkpoint_sha256'],
                'training_trace': cap['training']['trace'] if cap['training'] else None,
                'development_summary': feedback['summary'], 'active_in_incumbent': name in active,
                'available_for_adoption_or_state_inheritance': True}
        return catalog
