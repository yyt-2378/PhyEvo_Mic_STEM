import json
import numpy as np
from PIL import Image
import pytest
import torch
from scripts.emseek_author_weights import extract_state, strict_load, preprocess, official_preprocess
from scripts.fetch_emseek_author_weights import parse_folder


def test_official_preprocessing_identical_non_square_input():
    raw = np.random.default_rng(7).integers(0, 256, (37, 91), dtype=np.uint8)
    image = Image.fromarray(raw)
    author = official_preprocess()
    for size in [(1024, 1024), (64, 64)]:
        assert torch.equal(preprocess(image, size), author(None, image, input_size=size))


def test_missing_tensors_cannot_silently_keep_random_weights():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match='missing'):
        strict_load(model, {'weight': model.weight.detach().clone()})


def test_wrong_shapes_and_extra_tensors_rejected():
    model = torch.nn.Linear(2, 1)
    bad = {**model.state_dict(), 'weight': torch.zeros(1, 3)}
    with pytest.raises(ValueError, match='shape'):
        strict_load(model, bad)
    with pytest.raises(ValueError, match='unexpected'):
        strict_load(model, {**model.state_dict(), 'extra': torch.zeros(1)})


def test_full_state_and_module_prefix_restored_exactly():
    model = torch.nn.Linear(2, 1)
    state = extract_state({'state_dict': {'module.weight': torch.ones(1, 2), 'module.bias': torch.zeros(1)}})
    receipt = strict_load(model, state)
    assert receipt['all_tensors_restored_exactly'] and not receipt['random_weight_fallback']
    assert torch.equal(model.weight, torch.ones(1, 2))


def test_duplicate_or_non_tensor_state_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        extract_state({'module.weight': torch.ones(1), 'weight': torch.ones(1)})
    with pytest.raises(ValueError, match='Non-tensor'):
        extract_state({'weight': 'not a tensor'})


def test_parse_public_drive_inventory_without_javascript_execution():
    row = ['public-id', [], 'task2.pth', 'application/x-zip'] + [None] * 9 + [123]
    payload = json.dumps([[row], 1]).replace('"', '\\x22')
    value = parse_folder("window['_DRIVE_ivd'] = '" + payload + "';")
    assert value == [{'id': 'public-id', 'name': 'task2.pth', 'mime': 'application/x-zip', 'size': 123}]
    with pytest.raises(ValueError, match='No public'):
        parse_folder('Please sign in')


def test_task_checkpoint_is_selected_without_scores_or_labels():
    from scripts.run_emseek_author_pretrained import variants_for_task
    point = variants_for_task({'kind': 'expert', 'geometry': 'point', 'group': 'held-out'})
    interface = variants_for_task({'kind': 'expert', 'geometry': 'interface', 'group': 'held-out'})
    assert ('task2_no_prompt', 'task2.pth', None) in point
    assert all(row[1] == 'unet_ref.pth' for row in interface)
    assert variants_for_task({'kind': 'fecral'})[-1] == ('task5_no_prompt', 'task5.pth', None)
    assert all(row[1] != 'task1.pth' for row in point)  # all atoms != atomic defects


def test_reference_bank_excludes_entire_query_acquisition():
    from scripts.run_emseek_author_pretrained import training_references
    entries = [{'case_number': i, 'action': 'correct_geometry', 'geometry': 'point',
                'geometry_index': 1, 'group': group} for i, group in enumerate(['query', 'train'])]
    arrays = {i: {'image': np.zeros((64, 64), np.float32), 'human': np.zeros((64, 64), np.float32)} for i in range(2)}
    arrays[0]['human'][0, 0] = 1
    arrays[1]['human'][30, 30] = 1
    _, audit = training_references(entries, arrays, ['query'])
    assert len(audit) == 1 and audit[0]['group'] == 'train' and audit[0]['case_number'] == 1


def test_empty_native_mask_does_not_invent_component_points():
    from scripts.run_emseek_author_pretrained import point_metrics
    from tfree_detection.rsi_evaluation import EvaluationProtocol
    result = point_metrics(np.zeros((32, 32), np.float32), np.zeros((32, 32), bool),
                           np.asarray([[16., 16.]]), EvaluationProtocol())
    assert result['native_component_points'] == {'matched': 0, 'predicted': 0, 'truth': 1}
