"""Diagnostics must compare real, bound, aligned prediction artifacts."""
from copy import deepcopy

import numpy as np
import pytest

from .contracts import file_hash
from .stability_report import bound, compare_saved_receipts


def receipts(tmp_path):
    score = np.arange(80).reshape(8, 10).astype(float)
    support = np.ones(score.shape, bool)
    np.savez(tmp_path / 'a.npz', score=score, support=support)
    np.savez(tmp_path / 'b.npz', score=np.rot90(score), support=np.rot90(support))
    ref = lambda name: {'path': name, 'sha256': file_hash(tmp_path / name)}
    original = {'rows': [{'case': 'case', 'group': 'anonymous-source', 'branch': 'b1', 'coverage': .5}],
                'predictions': {'case': ref('a.npz')}}
    stress = {'rows': [{'case': 'stress-case', 'group': 'stress-anonymous-source', 'branch': 'b1', 'coverage': .5}],
              'predictions': {'stress-case': ref('b.npz')}}
    return original, stress


def test_aligned_bound_executions_are_compared_without_exporting_coordinates(tmp_path):
    original, stress = receipts(tmp_path)
    result = compare_saved_receipts(tmp_path, original, stress, .1, {'anonymous-source': 'D0'})
    assert result['groups']['D0']['mean_coverage_change'] == 0.
    assert result['groups']['D0']['changed_branch_count'] == 0
    assert result['observations'][0]['observable_stability']['selection_jaccard'] == 1.
    assert result['observations'][0]['sample'] == 'observation-000'
    assert 'anonymous-source' not in str(result)


def test_stale_prediction_or_unmatched_case_is_not_silently_accepted(tmp_path):
    original, stress = receipts(tmp_path)
    stale = dict(original['predictions']['case'], sha256='0'*64)
    with pytest.raises(ValueError, match='changed'):
        bound(tmp_path, stale)
    bad = deepcopy(stress)
    bad['rows'][0]['case'] = 'wrong-case'
    with pytest.raises(ValueError, match='cohorts'):
        compare_saved_receipts(tmp_path, original, bad, .1, {'anonymous-source': 'D0'})
    bad = deepcopy(stress)
    bad['rows'][0]['group'] = 'wrong-source'
    with pytest.raises(ValueError, match='source'):
        compare_saved_receipts(tmp_path, original, bad, .1, {'anonymous-source': 'D0'})
