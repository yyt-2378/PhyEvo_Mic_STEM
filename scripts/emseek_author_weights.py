"""Strict official EMSeek segmentation weights, with unchanged author preprocessing."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor/fair_baselines_20261014/emseek'
WEIGHTS = ROOT / 'vendor/author_pretrained_emseek_20261014_v2/SegMentor'


def make_model():
    path = VENDOR / 'emseek/models/unetpp_ref.py'
    spec = importlib.util.spec_from_file_location('emseek_author_pinned', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Complete checkpoint replaces BOTH encoders; never download ImageNet implicitly.
    return module.UnetPlusPlus(encoder_name='resnet18', pretrained=False, prompt=True, classes=2)


def extract_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected tensor state dictionary')
    for key in ('state_dict', 'model_state_dict'):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError('Missing state dictionary')
    result = {}
    for name, value in checkpoint.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError('Non-tensor checkpoint state')
        key = name.removeprefix('module.')
        if key in result:
            raise ValueError('Duplicate checkpoint tensor after prefix removal')
        result[key] = value
    return result


def strict_load(model, state):
    expected = model.state_dict()
    if set(expected) != set(state):
        raise ValueError('Author checkpoint has missing or unexpected tensors')
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f'Author checkpoint shape/dtype mismatch: {key}')
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite checkpoint')
    model.load_state_dict(state, strict=True)
    if not all(torch.equal(v, state[k]) for k, v in model.state_dict().items()):
        raise ValueError('Checkpoint not fully restored')
    model.eval()
    return {'state_tensors': len(state), 'parameters': sum(p.numel() for p in model.parameters()),
            'all_tensors_restored_exactly': True, 'random_weight_fallback': False}


def load_model(name, device='cpu'):
    if name not in ['unet_ref.pth'] + [f'task{i}.pth' for i in range(1, 6)]:
        raise ValueError('Unregistered author checkpoint')
    path = WEIGHTS / name
    receipt = json.loads(path.with_suffix('.receipt.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
        raise ValueError('Author weight hash mismatch')
    model = make_model()
    state = extract_state(torch.load(path, map_location='cpu', weights_only=True))
    mapping = strict_load(model, state)
    return model.to(device), {**mapping, 'checkpoint': name, 'sha256': receipt['sha256']}


def preprocess(image, size=(1024, 1024)):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return transforms.Compose([
        transforms.Resize(size), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(image.convert('RGB')).unsqueeze(0)


def predict(model, image, *, patch=None, device='cpu'):
    image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    tensor = preprocess(image).to(device)
    prompt = None if patch is None else preprocess(patch, (64, 64)).to(device)
    with torch.inference_mode():
        logits = model(tensor, prompt)
        score = F.interpolate(torch.softmax(logits, 1)[:, 1:2], size=(image.height, image.width),
                              mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        native = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    mask = np.asarray(Image.fromarray(native).resize(image.size, Image.Resampling.NEAREST))
    return score, mask


def official_preprocess():
    """Extract only the inspected pure preprocessing method, not the LLM platform."""
    source = VENDOR / 'emseek/agents/segmentor_agent.py'
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'SegMentorAgent')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_image')
    scope = {'Image': Image, 'np': np, 'torch': torch, 'transforms': transforms, 'Tuple': tuple}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    return scope['preprocess_image']
