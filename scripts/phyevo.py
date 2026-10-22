#!/usr/bin/env python3
"""Current PhyEvo-Mic workflows from a source checkout; no historical auto-run.

Examples: phyevo.py detect --help; phyevo.py rsi --help.
Use --dry-run before the workflow to print the command without running it.
The selected interpreter is reused; no environment is installed or changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RSI = 'experiments.rsi_program_evolution_v1.composition_study'
WORKFLOWS = {
    'detect': ['-m', 'tfree_detection', 'detect'],
    'interface': ['scripts/run_interface_detection.py'],
    'review': ['-m', 'streamlit', 'run', 'apps/review_dataset_viewer.py'],
    'platform': ['scripts/serve_platform.py'],
    'train': ['scripts/run_expert_learning_cycle.py'],
    'rsi': ['-m', RSI],
    'check': ['scripts/check_source_release.py'],
}
BENCHMARKS = {
    'locked': 'scripts/run_locked_external_test.py',
    'fair': 'scripts/run_fair_baselines.py',
    'author': 'scripts/run_author_pretrained.py',
}


def command_for(workflow, arguments):
    """Pure construction: no image, database or experiment asset is opened."""
    arguments = list(arguments)
    if arguments[:1] == ['--']:
        arguments.pop(0)
    if workflow == 'benchmark':
        if not arguments or arguments[0] not in BENCHMARKS:
            raise ValueError('benchmark requires one of: locked, fair, author')
        kind, arguments = arguments[0], arguments[1:]
        base = [BENCHMARKS[kind]]
        if (kind in ('fair', 'author') and arguments
                and not any(a in ('-h', '--help') for a in arguments)
                and not any(a == '--output' or a.startswith('--output=') for a in arguments)):
            raise ValueError('Specify --output explicitly; never default to a historical baseline run')
    else:
        base = WORKFLOWS[workflow]
    return [sys.executable, '-B', *base, *(arguments or ['--help'])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'benchmark: locked / fair / author. Each current command has one implementation; '
        'legacy research scripts are not alternative default workflows.'))
    parser.add_argument('--dry-run', action='store_true', help='Print command only; no child process')
    parser.add_argument('workflow', choices=(*WORKFLOWS, 'benchmark'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.workflow == 'benchmark' and args.arguments in ([], ['--help'], ['-h']):
        print('benchmark {locked|fair|author} [--help or explicit experiment arguments]')
        return 0
    try:
        command = command_for(args.workflow, args.arguments)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({'cwd': str(ROOT), 'argv': command, 'executed': False}, ensure_ascii=False))
        return 0
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT / 'src'), str(ROOT)))
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
