"""Current workflows must not acquire dependencies on historical study drivers."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import phyevo


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'experiments/rsi_program_evolution_v1'


def dependencies(name):
    result = set()
    for node in ast.walk(ast.parse((PACKAGE / (name + '.py')).read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            candidates = [node.module.split('.')[0]] if node.module else [a.name for a in node.names]
            result.update(n for n in candidates if (PACKAGE / (n + '.py')).is_file())
    return result


def test_current_rsi_does_not_import_historical_drivers():
    pending = ['controller', 'composition_study', 'composition_runtime',
               'composition_continuation', 'composition_retention_audit',
               'composition_final_audit', 'retained_portfolio', 'replay_retained_portfolio', 'broker',
               'worker', 'field_storage_worker', 'prediction_storage_worker']
    visited = set()
    forbidden = {'real_study', 'learning_study', 'meta_study', 'active_study',
                 'frontier_study', 'stable_study', 'expansion_study', 'system_study',
                 'bank_study', 'runtime_compat'}
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        assert name not in forbidden, f'Current RSI depends on historical driver: {name}'
        visited.add(name)
        pending.extend(dependencies(name) - visited)
    assert {'study_io', 'training_trace', 'configured_sandbox'} <= visited


def test_importing_current_workflows_does_not_load_historical_drivers():
    code = """
import sys
from experiments.rsi_program_evolution_v1 import composition_study, composition_final_audit
for name in ('real_study', 'learning_study', 'meta_study', 'active_study',
             'frontier_study', 'stable_study', 'expansion_study', 'system_study', 'runtime_compat'):
    assert 'experiments.rsi_program_evolution_v1.' + name not in sys.modules, name
assert composition_study.SCOPE is None and composition_study.SOURCE is None
"""
    completed = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize('workflow,args,expected', [
    ('detect', ['--input', 'a.png'], ['-m', 'tfree_detection', 'detect', '--input', 'a.png']),
    ('rsi', ['report', '--run-config', 'config.json'], ['-m', phyevo.RSI, 'report', '--run-config', 'config.json']),
    ('train', [], ['scripts/run_expert_learning_cycle.py', '--help']),
    ('review', [], ['-m', 'streamlit', 'run', 'apps/review_dataset_viewer.py', '--help']),
    ('platform', [], ['scripts/serve_platform.py', '--help']),
    ('benchmark', ['locked', '--help'], ['scripts/run_locked_external_test.py', '--help']),
])
def test_dispatch_preserves_arguments(workflow, args, expected):
    assert phyevo.command_for(workflow, args)[2:] == expected


def test_dry_run_never_launches_a_child(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run must not execute anything')
    monkeypatch.setattr(phyevo.subprocess, 'run', forbidden)
    assert phyevo.main(['--dry-run', 'review', '--server.address', '127.0.0.1']) == 0
    assert json.loads(capsys.readouterr().out)['executed'] is False


def test_platform_help_does_not_create_private_storage(tmp_path):
    storage = tmp_path / 'uninitialized-platform'
    completed = subprocess.run(
        [sys.executable, '-B', 'scripts/phyevo.py', 'platform', '--help', '--root', str(storage)],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert 'supervise' in completed.stdout
    assert not storage.exists()


@pytest.mark.parametrize('kind', ['fair', 'author'])
def test_baseline_never_defaults_to_old_run(kind):
    with pytest.raises(ValueError, match='--output explicitly'):
        phyevo.command_for('benchmark', [kind, 'launch'])
    assert '--help' in phyevo.command_for('benchmark', [kind])
    assert '--output=outputs/new' in phyevo.command_for('benchmark', [kind, 'launch', '--output=outputs/new'])


def test_dispatch_forwards_status_and_uses_no_shell(monkeypatch):
    observed = {}
    def run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return type('Result', (), {'returncode': 7})()
    monkeypatch.setattr(phyevo.subprocess, 'run', run)
    assert phyevo.main(['check', '--profile', 'syntax']) == 7
    assert isinstance(observed['command'], list)
    assert observed['cwd'] == ROOT
    assert not observed.get('shell', False)
