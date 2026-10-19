"""Persistent whole-study growth budget for owned, isolated controller runs.

This is a monitored limit, not a filesystem quota. A writer can overshoot
between scans; receipts explicitly record that overshoot. Nothing is deleted.
Callers must wrap the entire controller, including staging, fitting and logs,
and restrict its writable mounts to this scope before calling run().
"""
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import time
import uuid

from .field_storage import exact_child, new_json, sha256


def usage(scope):
    """Count allocated blocks once per inode, including directory metadata."""
    seen, total, files, links = set(), 0, 0, 0
    device = scope.stat().st_dev
    for directory, dirs, names in os.walk(scope, followlinks=False):
        for name in ['.'] + names + dirs:
            path = Path(directory) if name == '.' else Path(directory) / name
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue  # A scan may overlap creation/rename, not a permission grant to delete.
            # Count the link entry itself, never its target. E.g. pytest creates
            # current-directory aliases; ignoring their targets avoids both
            # double counting and any traversal beyond the permitted scope.
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)) or info.st_dev != device:
                raise ValueError('Unexpected special file or nested filesystem in budget scope')
            identity = info.st_dev, info.st_ino
            if identity not in seen:
                total += info.st_blocks * 512
                files += stat.S_ISREG(info.st_mode)
                links += stat.S_ISLNK(info.st_mode)
                seen.add(identity)
    space = os.statvfs(scope)
    return {'allocated_bytes': total, 'unique_files': files, 'symlink_entries_not_followed': links,
            'available_bytes': space.f_bavail * space.f_frsize}


class StorageBudget:
    def __init__(self, workspace, scope, *, max_growth_bytes, minimum_free_bytes):
        self.workspace = Path(workspace).absolute()
        self.scope = exact_child(self.workspace, scope)
        if (not self.scope.is_dir() or type(max_growth_bytes) is not int or max_growth_bytes < 65536
                or type(minimum_free_bytes) is not int or minimum_free_bytes < 0):
            raise ValueError('Existing exact scope, finite growth budget and disk reserve required')
        self.directory = exact_child(self.scope, self.scope / 'storage_guard')
        expected = {'scope': str(self.scope), 'max_growth_bytes': max_growth_bytes,
                    'minimum_free_bytes': minimum_free_bytes, 'limit_kind': 'monitored_allocated_growth_v1'}
        if not self.directory.exists():
            before = usage(self.scope)
            self.directory.mkdir()
            new_json(self.directory / 'policy.json', expected | {'baseline_allocated_bytes': before['allocated_bytes']})
        policy = exact_child(self.directory, self.directory / 'policy.json')
        self.policy = json.loads(policy.read_text())
        if ({k: self.policy.get(k) for k in expected} != expected
                or type(self.policy.get('baseline_allocated_bytes')) is not int):
            raise ValueError('Storage policy cannot silently change or reset its baseline')
        self.policy_hash = sha256(policy)

    def check(self):
        if sha256(self.directory / 'policy.json') != self.policy_hash:
            raise ValueError('Storage policy changed during execution')
        measured = usage(self.scope)
        measured['growth_bytes'] = max(0, measured['allocated_bytes'] - self.policy['baseline_allocated_bytes'])
        measured['over_budget_bytes'] = max(0, measured['growth_bytes'] - self.policy['max_growth_bytes'])
        measured['reason'] = ('growth_limit' if measured['over_budget_bytes'] else
                              'disk_reserve' if measured['available_bytes'] < self.policy['minimum_free_bytes'] else None)
        return measured

    @staticmethod
    def stop_owned(process):
        # Popen identity is owned by this run. No PID lookup, pkill or other-job cleanup.
        if process.poll() is not None:
            return False
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return True

    def run(self, command, *, timeout_seconds, poll_seconds=.25):
        if (not command or command[0] != '/usr/bin/bwrap'
                or not all(flag in command for flag in ('--unshare-all', '--die-with-parent'))):
            raise ValueError('Only owned bubblewrap PID-namespace controllers are supported')
        if (not math.isfinite(timeout_seconds) or timeout_seconds <= 0
                or not math.isfinite(poll_seconds) or not .01 <= poll_seconds <= 1):
            raise ValueError('Positive timeout and 10 ms–1 s scan interval required')
        lock = exact_child(self.directory, self.directory / 'run.lock')
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._run(command, timeout_seconds, poll_seconds)

    def _run(self, command, timeout_seconds, poll_seconds):
        record = exact_child(self.directory, self.directory / ('run-' + uuid.uuid4().hex))
        record.mkdir()
        started = time.monotonic()
        initial = self.check()
        new_json(record / 'request.json', {'command': command, 'policy_sha256': self.policy_hash,
                 'before': initial, 'timeout_seconds': timeout_seconds, 'poll_seconds': poll_seconds})
        process, stopped, peak, scans = None, False, initial['growth_bytes'], 1
        status = 'storage_limit' if initial['reason'] else 'ok'
        latest, error = initial, None
        if status == 'ok':
            try:
                with (record / 'stdout.log').open('xb') as stdout, (record / 'stderr.log').open('xb') as stderr:
                    process = subprocess.Popen(command, cwd=self.workspace, env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'},
                                               stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                               start_new_session=True)
                    new_json(record / 'owned_process.json', {'pid': process.pid, 'started_by_this_run': True})
                    while True:
                        latest = self.check(); scans += 1
                        peak = max(peak, latest['growth_bytes'])
                        if latest['reason']:
                            status = 'storage_limit'; stopped = self.stop_owned(process); break
                        if process.poll() is not None:
                            status = 'ok' if process.returncode == 0 else 'failed'; break
                        if time.monotonic() - started >= timeout_seconds:
                            status = 'timeout'; stopped = self.stop_owned(process); break
                        time.sleep(poll_seconds)
            except BaseException as exc:
                if process is not None:
                    stopped = self.stop_owned(process) or stopped
                error = f'{type(exc).__name__}: {exc}'
                status = 'monitor_error'
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    new_json(record / 'interrupted.json', {'error': error, 'owned_process_stopped': stopped})
                    raise
        if process is not None:
            process.wait()
        try:
            final = self.check()
            if final['reason'] and status in ('ok', 'failed'):
                status = 'storage_limit'
        except (OSError, ValueError) as exc:
            final = latest | {'final_scan_unavailable': True}
            error = error or f'{type(exc).__name__}: {exc}'
            status = 'monitor_error'
        peak = max(peak, final.get('growth_bytes', 0))
        result = {'status': status, 'returncode': None if process is None else process.returncode,
                  'owned_pid': None if process is None else process.pid, 'owned_process_stopped': stopped,
                  'before': initial, 'after': final, 'peak_observed_growth_bytes': peak, 'scans': scans,
                  'observed_overshoot_bytes': max(0, peak - self.policy['max_growth_bytes']),
                  'seconds': time.monotonic() - started, 'poll_seconds': poll_seconds,
                  'not_an_os_hard_quota': True, 'files_deleted': 0, 'error': error,
                  'scientific_rejection': False, 'receipt': str(record / 'result.json')}
        new_json(record / 'result.json', result)
        return result
