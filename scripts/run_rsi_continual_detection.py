#!/usr/bin/env python3
"""Bind a frozen continual route or run it explicitly as a research side branch."""
import argparse
import json
import os
from pathlib import Path

from tfree_detection.config import load_config
from tfree_detection.framework import TrainingFreeDetectionFramework
from tfree_detection.rsi_continual_runtime import create_binding, detect_with_continual
from tfree_detection.rsi_runtime import inside, save_rsi_detection

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    bind = sub.add_parser('bind')
    bind.add_argument('--training', required=True)
    bind.add_argument('--run', required=True)
    bind.add_argument('--output', required=True)
    detect = sub.add_parser('detect')
    for key in ('input', 'config', 'output'):
        detect.add_argument('--'+key, required=True)
    detect.add_argument('--binding')
    detect.add_argument('--experimental-research', action='store_true')
    detect.add_argument('--student-device', default='cpu')
    detect.add_argument('--dataset-path')
    detect.add_argument('--frame-index', type=int)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    output = inside(ROOT, args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'bind':
        value = create_binding(ROOT, args.training, args.run, output)
        print(json.dumps({'binding': str(output), 'routes': value['routes']}))
    else:
        config = load_config(inside(ROOT, args.config))
        if config.stage2.sam3.enabled:
            raise ValueError('Use a physics-only config; no external service provisioning')
        result = detect_with_continual(TrainingFreeDetectionFramework(config), args.input, project_root=ROOT,
            binding=args.binding, experimental_research=args.experimental_research, student_device=args.student_device,
            dataset_path=args.dataset_path, frame_index=args.frame_index)
        save_rsi_detection(result, config, output, project_root=ROOT)
        print(json.dumps({'output': str(output), **result.metadata}, ensure_ascii=False))


if __name__ == '__main__':
    main()
