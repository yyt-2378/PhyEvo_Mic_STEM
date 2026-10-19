"""Bounded error-feedback replay and a research-only retention controller.

Outer acquisitions never enter either component. Internal control frames/tiles
are not independent acquisitions. The original physics and expert labels are
immutable; unmarked pixels remain unknown, not expert-negative examples.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .rsi_evaluation import EvaluationProtocol, evaluate_case, matched_expert_indices, student_mask, student_points
from .rsi_feedback import FEEDBACK_BRANCH, feedback_predictions
from .rsi_refinement import POLICIES, RefinedSampler


def make_partitions(entries, outer_group, *, guard_pixels=128, gap_frames=2):
    """Order-defined internal control; never inspect image values or outcomes."""
    eligible = [e for e in entries if e['geometry'] in ('point', 'interface')
                and e['action'] == 'correct_geometry' and e['human_positive_pixels'] > 0]
    result = {'outer_group': outer_group, 'train': [], 'control': [], 'outer': [],
              'unused_gap_cases': [], 'control_is_independent_acquisition': False}
    for group in sorted({e['group'] for e in eligible}):
        cases = sorted([e for e in eligible if e['group'] == group], key=lambda e: e['case_number'])
        if group == outer_group:
            result['outer'] = [{'case_number': e['case_number'], 'rows': [0, e['shape'][0]]} for e in cases]
        elif len(cases) >= 6:
            ncontrol = max(1, math.ceil(len(cases)*.2))
            boundary = len(cases)-ncontrol
            for role, subset in [('train', cases[:boundary-gap_frames]), ('control', cases[boundary:])]:
                result[role].extend({'case_number': e['case_number'], 'rows': [0, e['shape'][0]]} for e in subset)
            result['unused_gap_cases'].extend(e['case_number'] for e in cases[boundary-gap_frames:boundary])
        elif len(cases) == 1:
            e = cases[0]
            mid = e['shape'][0]//2
            if mid-guard_pixels//2 < 128:
                raise ValueError('Single-image source too small for separated control tiles')
            result['train'].append({'case_number': e['case_number'], 'rows': [0, mid-guard_pixels//2]})
            result['control'].append({'case_number': e['case_number'], 'rows': [mid+guard_pixels//2, e['shape'][0]]})
        else:
            raise ValueError('No registered control partition rule for this source')
    if not result['outer'] or not result['train'] or not result['control']:
        raise ValueError('Incomplete outer/train/control partition')
    validate_partitions(entries, result)
    return result


def validate_partitions(entries, partition):
    by_id = {e['case_number']: e for e in entries}
    for role in ('train', 'control', 'outer'):
        seen = set()
        for item in partition[role]:
            n, (a, b) = item['case_number'], item['rows']
            if n in seen or not 0 <= a < b <= by_id[n]['shape'][0]:
                raise ValueError('Duplicate or invalid partition')
            seen.add(n)
            if (by_id[n]['group'] == partition['outer_group']) != (role == 'outer'):
                raise ValueError('Outer acquisition leakage')
    for a in partition['train']:
        for b in partition['control']:
            if a['case_number'] == b['case_number'] and max(a['rows'][0], b['rows'][0]) < min(a['rows'][1], b['rows'][1]):
                raise ValueError('Train/control pixels overlap')


def materialize(entries, arrays, records, items):
    """Copy a role's pixels; adjust point coordinates and physics centroids."""
    by_id = {e['case_number']: e for e in entries}
    selected, planes, candidates = [], {}, {}
    for item in items:
        n, (a, b) = item['case_number'], item['rows']
        original, plane = by_id[n], arrays[n]
        e = deepcopy(original)
        cropped = {k: v[a:b].copy() for k, v in plane.items()}
        e['shape'] = list(cropped['image'].shape)
        e['human_known_pixels'] = int((cropped['human_weight'] > 0).sum())
        e['human_positive_pixels'] = int((cropped['human'] > 0).sum())
        e['teacher_positive_pixels'] = int((cropped['teacher'] > 0).sum())
        if e['geometry'] == 'point':
            e['points_xy'] = [[x, y-a] for x, y in original['points_xy'] if a <= y < b]
        e['original_rows'] = [a, b]
        if not e['human_positive_pixels'] or (e['geometry'] == 'point' and not e['points_xy']):
            raise ValueError('Registered split has no positive supervision; do not adapt the split to outcomes')
        candidate_list = []
        for c in records[n]['prediction']['candidates']:
            x, y = c['centroid_xy']
            if a <= y < b:
                c = deepcopy(c)
                c['centroid_xy'] = [x, y-a]
                candidate_list.append(c)
        selected.append(e)
        planes[n], candidates[n] = cropped, candidate_list
    return selected, planes, candidates


def predictions(image, teacher, candidates, score, geometry):
    # All branches consume image/physics/model scores only, never human labels.
    p = feedback_predictions(image, teacher, candidates, score, geometry, protocol=EvaluationProtocol())
    return {k: p[k] for k in ('physics_budgeted', 'student_expert', FEEDBACK_BRANCH)}


def measure(entry, plane, candidates, score):
    return evaluate_case(entry, plane, predictions(plane['image'], plane['teacher'], candidates, score, entry['geometry']))


def error_state(entry, plane, score, previous_best=0.0):
    """Hard positives under the fixed review budget; no new negative labels."""
    protocol = EvaluationProtocol()
    human = plane['human'] > 0
    if entry['geometry'] == 'point':
        points = np.asarray(entry['points_xy'])
        predicted = student_points(score, protocol.point_budget(score.shape), protocol.point_min_distance_px)
        hits = matched_expert_indices(predicted, points, protocol.point_tolerance_px)
        coverage = len(hits)/len(points)
        misses = [p for i, p in enumerate(points) if i not in hits]
        hard = np.zeros(human.shape, bool)
        for x, y in misses:
            iy, ix = int(round(y)), int(round(x))
            hard[max(0, iy-4):iy+5, max(0, ix-4):ix+5] = True
        hard &= human
    else:
        mask = student_mask(score, protocol.area_budget(score.shape))
        coverage = float((human & mask).sum()/human.sum())
        hard = human & ~mask
    forgetting = max(0., previous_best-coverage)
    # Bounded replay: no case disappears, even if its error becomes zero.
    weight = min(5, 1+math.ceil(2*(1-coverage)+2*forgetting))
    return {'coverage': coverage, 'best_coverage': max(previous_best, coverage),
            'forgetting': forgetting, 'weight': weight, 'hard_pixels': int(hard.sum())}, hard


class FeedbackSampler(RefinedSampler):
    """Reuse the frozen crop/loss machinery; modify only next-round replay.

    All cases keep at least one copy; hard cases have at most five copies.
    Source probability is a 50/50 base and hard-weighted mixture. Guided
    positives retain all old positives plus two copies of missed positives.
    """

    def __init__(self, entries, arrays, *, excluded_groups=(), state=None, hard_masks=None, enabled=True):
        super().__init__(entries, arrays, POLICIES['photometric'], excluded_groups=excluded_groups)
        allowed = {str(e['case_number']) for e in entries}
        state, hard_masks = state or {}, hard_masks or {}
        if set(state)-allowed or set(map(str, hard_masks))-allowed:
            raise ValueError('Non-training feedback in sampler')
        if state and set(state) != allowed:
            raise ValueError('Feedback must cover the complete training inventory')
        if enabled and state:
            for geometry, (groups, base) in self.sources.items():
                hardness = []
                for source in groups:
                    cases = self.cases[(geometry, source)]
                    weights = [int(state[str(e['case_number'])]['weight']) for e in cases]
                    if any(w < 1 or w > 5 for w in weights):
                        raise ValueError('Unbounded replay weight')
                    hardness.append(float(np.mean(weights)))
                    self.cases[(geometry, source)] = [e for e, w in zip(cases, weights) for _ in range(w)]
                weighted = base*np.asarray(hardness)
                self.sources[geometry] = groups, .5*base+.5*weighted/weighted.sum()
            for e in entries:
                n = e['case_number']
                hard = np.asarray(hard_masks[n], bool)
                if hard.shape != arrays[n]['image'].shape or np.any(hard & ~(arrays[n]['human'] > 0)):
                    raise ValueError('Hard replay must be a subset of training human positives')
                self.positive_yx[n] = np.concatenate([self.positive_yx[n], np.argwhere(hard), np.argwhere(hard)])


def aggregate_control(rows, geometry, branch):
    groups = sorted({r['group'] for r in rows if r['geometry'] == geometry})
    result = {}
    for g in groups:
        values = [r['methods'][branch]['positive_coverage'] for r in rows
                  if r['geometry'] == geometry and r['group'] == g and r['main_comparison_eligible']]
        if values:
            result[g] = float(np.mean(values))
    return result


def retention_gate(rows, incumbent, round_number, *, minimum_gain=.005, maximum_loss=.01):
    """Internal control selects a research route, NOT a production promotion.

    This is explicitly weaker evidence than cross-acquisition inner validation.
    Independent outer acquisition results must be reported, including losses.
    """
    result = deepcopy(incumbent)
    decisions = []
    for geometry in sorted({r['geometry'] for r in rows}):
        base = incumbent.get(geometry)
        if base is None:
            base = {'round': 0, 'branch': 'physics_budgeted',
                    'control_by_source': aggregate_control(rows, geometry, 'physics_budgeted')}
        trials = []
        for branch in ('student_expert', FEEDBACK_BRANCH):
            values = aggregate_control(rows, geometry, branch)
            if values.keys() != base['control_by_source'].keys() or not values:
                raise ValueError('Control source inventory changed')
            gains = {g: values[g]-base['control_by_source'][g] for g in values}
            gain = float(np.mean(list(gains.values())))
            accepted = gain >= minimum_gain and min(gains.values()) >= -maximum_loss
            trials.append({'round': round_number, 'branch': branch, 'control_by_source': values,
                           'gains': gains, 'mean_gain': gain, 'eligible': accepted})
        candidates = [t for t in trials if t['eligible']]
        best = sorted(candidates, key=lambda t: (-t['mean_gain'], t['branch']))[0] if candidates else None
        result[geometry] = ({k: best[k] for k in ('round', 'branch', 'control_by_source')} if best else base)
        decisions.append({'geometry': geometry, 'previous': base, 'selected': result[geometry],
                          'accepted': bool(best), 'trials': trials})
    return result, decisions
