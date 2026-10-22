"""Resolve preserved FAU inputs after consolidating the October 21–22 reports.

No files are read at import, no symlinks are created, and historical manifests
stay byte-for-byte unchanged. Only the explicitly consolidated directories map.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'real_data_test/4D数据计算晶格变化_0901'
SELECTED = BASE / 'selected_site_comparison_20261022_v1'
DEPENDENCIES = SELECTED / '_dependencies'
CONSOLIDATED = frozenset({
    'FAU_average_strain_report_20261021',
    'expert_negative_comparison_20261021_v1',
    'expert_spatial_displacement_20261021_v1',
    'expert_morphology_comparison_20261021_v1',
    'v20_expert_negative_and_recall_revision_20261021',
    'v20_exemplar_only_correction_20261021',
    'same_plane_skeleton_three_class_20261021_v1',
    'review_feedback_three_color_20261022_v1',
    'review_feedback_three_color_20261022_v2',
    'review_feedback_three_color_20261022_v3',
    'atomic_center_strain_20261022_v1',
    'neighbor_site_structure_20261022_v1',
})


def resolve_input(path):
    """Find the same input bytes at their new location; never change hashes."""
    path = Path(path).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('Project-local path required')
    if path.is_relative_to(BASE):
        rel = path.relative_to(BASE)
        if rel.parts and rel.parts[0] in CONSOLIDATED:
            relocated = (DEPENDENCIES / rel).resolve()
            if not relocated.is_relative_to(DEPENDENCIES.resolve()) or not relocated.is_relative_to(ROOT):
                raise ValueError('Invalid consolidated dependency path')
            if relocated.exists():
                return relocated
    return path


def result_dir(name):
    if name not in CONSOLIDATED:
        raise ValueError('Not an explicitly consolidated FAU release')
    return resolve_input(BASE / name)
