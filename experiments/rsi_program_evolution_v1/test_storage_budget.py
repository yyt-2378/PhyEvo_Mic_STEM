"""KB-sized storage growth checks; no real scientific trial or cleanup."""
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from .storage_budget import StorageBudget, usage


def command(scope, code):
    return ['/usr/bin/bwrap', '--unshare-all', '--die-with-parent', '--new-session',
            '--cap-drop', 'ALL', '--clearenv', '--ro-bind', '/usr', '/usr',
            '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/lib', '/lib',
            '--symlink', 'usr/lib64', '/lib64', '--proc', '/proc', '--dev', '/dev',
            '--bind', str(scope), '/work', '--chdir', '/work',
            '/usr/bin/python3', '-I', '-B', '-c', code]


def guard(tmp_path, budget=1024**2, reserve=0):
    scope = tmp_path / 'run'; scope.mkdir(exist_ok=True)
    return StorageBudget(tmp_path, scope, max_growth_bytes=budget, minimum_free_bytes=reserve)


def test_usage_counts_hardlinks_once(tmp_path):
    path = tmp_path / 'one'
    with path.open('xb') as stream: stream.write(b'x'*8192)
    before = usage(tmp_path)
    os.link(path, tmp_path / 'two')
    after = usage(tmp_path)
    assert before['allocated_bytes'] == after['allocated_bytes']
    assert after['unique_files'] == 1


def test_policy_cannot_reset_growth_and_refuses_alias_scope(tmp_path):
    g = guard(tmp_path); baseline = g.policy['baseline_allocated_bytes']
    with (g.scope / 'new').open('xb') as stream: stream.write(b'x'*16384)
    assert guard(tmp_path).policy['baseline_allocated_bytes'] == baseline
    assert guard(tmp_path).check()['growth_bytes'] >= 16384
    with pytest.raises(ValueError, match='cannot silently change'):
        guard(tmp_path, budget=2*1024**2)
    with pytest.raises(ValueError):
        StorageBudget(tmp_path, tmp_path, max_growth_bytes=1024**2, minimum_free_bytes=0)


def test_guard_checks_successful_run_and_records_owned_pid(tmp_path):
    g = guard(tmp_path)
    result = g.run(command(g.scope, "from pathlib import Path\nwith Path('/work/result').open('xb') as f: f.write(b'ok')"),
                   timeout_seconds=5, poll_seconds=.02)
    assert result['status'] == 'ok' and result['returncode'] == 0
    assert result['owned_pid'] and result['files_deleted'] == 0
    assert (g.scope / 'result').read_bytes() == b'ok'
    assert json.loads(Path(result['receipt']).read_text()) == result


def test_exceeded_growth_stops_only_its_owned_namespace_and_keeps_partial_files(tmp_path):
    g = guard(tmp_path, budget=96*1024)
    other = subprocess.Popen(['/usr/bin/sleep', '10'])
    try:
        code = "import time\nfrom pathlib import Path\nfor i in range(50):\n    with Path('/work/block'+str(i)).open('xb') as f: f.write(b'x'*16384)\n    time.sleep(.03)\n"
        result = g.run(command(g.scope, code), timeout_seconds=5, poll_seconds=.02)
        assert result['status'] == 'storage_limit' and result['owned_process_stopped']
        assert result['observed_overshoot_bytes'] > 0
        assert other.poll() is None
        files = sorted(g.scope.glob('block*'))
        assert 0 < len(files) < 50
        time.sleep(.1)
        assert sorted(g.scope.glob('block*')) == files
        assert result['scientific_rejection'] is False and result['files_deleted'] == 0
    finally:
        other.terminate(); other.wait(timeout=2)


def test_no_launch_when_scope_is_already_over_budget_or_disk_reserve(tmp_path):
    for name, reserve in [('growth', 0), ('reserve', 2**63)]:
        directory = tmp_path / name; directory.mkdir()
        g = guard(directory, budget=65536, reserve=reserve)
        if not reserve:
            with (g.scope / 'already_full').open('xb') as stream: stream.write(b'x'*65536)
        result = g.run(command(g.scope, "raise AssertionError('must not run')"), timeout_seconds=5)
        assert result['status'] == 'storage_limit' and result['owned_pid'] is None


def test_timeout_stops_owned_process_and_invalid_command_is_refused(tmp_path):
    g = guard(tmp_path)
    with pytest.raises(ValueError, match='bubblewrap'):
        g.run(['/usr/bin/python3', '-c', 'pass'], timeout_seconds=1)
    result = g.run(command(g.scope, 'import time; time.sleep(5)'), timeout_seconds=.2, poll_seconds=.02)
    assert result['status'] == 'timeout' and result['owned_process_stopped']


def test_monitor_failure_stops_owned_writer_and_leaves_a_receipt(tmp_path, monkeypatch):
    g = guard(tmp_path)
    original = g.check
    count = 0
    def interrupted_scan():
        nonlocal count
        count += 1
        if count > 2:
            raise ValueError('synthetic monitor failure')
        return original()
    monkeypatch.setattr(g, 'check', interrupted_scan)
    code = 'import time; time.sleep(5)'
    result = g.run(command(g.scope, code), timeout_seconds=3, poll_seconds=.02)
    assert result['status'] == 'monitor_error' and result['owned_process_stopped']
    assert result['after']['final_scan_unavailable'] is True
    assert Path(result['receipt']).is_file()


def test_link_metadata_is_counted_without_following_targets(tmp_path):
    g = guard(tmp_path)
    (g.scope / 'alias').symlink_to('does-not-exist')
    result = g.check()
    assert result['reason'] is None
    assert result['symlink_entries_not_followed'] == 1
