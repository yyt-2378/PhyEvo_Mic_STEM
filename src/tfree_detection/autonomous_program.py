"""Typed spatial-program synthesis, not a menu of source-specific solutions.

The grammar is human-designed; expressions and fitted coefficients are generated
by the controller. Programs cannot read files, annotations, source IDs or globals.
No eval/exec, arbitrary Python, imports, network or host writes are available.
Generated Python is an audit representation of this exact interpreter.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

CHANNELS = ("base", "physics", "student", "variation", "image", "valid")
UNARY = ("abs", "square", "negate", "smooth", "contrast", "spread", "dilate", "erode", "gradient")
BINARY = ("add", "multiply", "minimum", "maximum")
SCALES = (3, 7, 15, 31, 63)
MAX_DEPTH = 5
MAX_NODES = 31


def validate(node, *, depth=0):
    if not isinstance(node, dict) or depth > MAX_DEPTH:
        raise ValueError("Invalid or too-deep spatial program")
    op = node.get("op")
    if op == "field":
        if set(node) != {"op", "name"} or node["name"] not in CHANNELS:
            raise ValueError("Forbidden field or payload")
        return 1
    if op in UNARY:
        spatial = op in {"smooth", "contrast", "spread", "dilate", "erode", "gradient"}
        expected = {"op", "arg", "scale"} if spatial else {"op", "arg"}
        if set(node) != expected or (spatial and node["scale"] not in SCALES):
            raise ValueError("Invalid unary program")
        count = 1 + validate(node["arg"], depth=depth + 1)
    elif op in BINARY:
        if set(node) != {"op", "left", "right"}:
            raise ValueError("Invalid binary program")
        count = 1 + validate(node["left"], depth=depth + 1) + validate(node["right"], depth=depth + 1)
    else:
        raise ValueError("Forbidden operator")
    if count > MAX_NODES:
        raise ValueError("Program node budget exceeded")
    return count


def identity(node):
    validate(node)
    return hashlib.sha256(json.dumps(node, sort_keys=True).encode()).hexdigest()


def normalize(x):
    """Finite per-image scale, not a probability calibration."""
    x = x.float()
    mean = x.mean((-2, -1), keepdim=True)
    scale = x.std((-2, -1), correction=0, keepdim=True).clamp(min=1e-5)
    return ((x - mean) / scale).clamp(-8, 8)


def spatial(x, scale, fn="mean"):
    # Reflection is undefined for kernels larger than tiny synthetic images.
    k = min(scale, 2 * ((min(x.shape[-2:]) - 1) // 2) + 1)
    pad = k // 2
    z = F.pad(x[None, None], (pad, pad, pad, pad), mode="replicate")
    pool = F.avg_pool2d if fn == "mean" else F.max_pool2d
    # Separable box/max filters keep the grammar computationally bounded even
    # at its largest scale. No O(kernel_area) full-image sliding windows.
    return pool(pool(z, (1, k), stride=1), (k, 1), stride=1)[0, 0]


def evaluate(node, fields, *, checked=False):
    if not checked:
        validate(node)
        if set(fields) != set(CHANNELS):
            raise ValueError("Exact field-only inference contract required")
        shapes = {tuple(v.shape) for v in fields.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError("Expected equal 2-D fields")
    op = node["op"]
    if op == "field":
        return fields[node["name"]]
    if op in BINARY:
        a = evaluate(node["left"], fields, checked=True)
        b = evaluate(node["right"], fields, checked=True)
        if op == "add":
            result = (a + b) / 2
        elif op == "multiply":
            result = a * b
        elif op == "minimum":
            result = torch.minimum(a, b)
        else:
            result = torch.maximum(a, b)
    else:
        a = evaluate(node["arg"], fields, checked=True)
        if op == "abs":
            result = a.abs()
        elif op == "square":
            result = a.square()
        elif op == "negate":
            result = -a
        elif op == "smooth":
            result = spatial(a, node["scale"])
        elif op == "contrast":
            result = a - spatial(a, node["scale"])
        elif op == "spread":
            result = (spatial(a.square(), node["scale"]) - spatial(a, node["scale"]).square()).clamp(min=0).sqrt()
        elif op == "dilate":
            result = spatial(a, node["scale"], "max")
        elif op == "erode":
            result = -spatial(-a, node["scale"], "max")
        elif op == "gradient":
            z = spatial(a, node["scale"])
            dy = F.pad(z[1:] - z[:-1], (0, 0, 0, 1))
            dx = F.pad(z[:, 1:] - z[:, :-1], (0, 1, 0, 0))
            result = (dy.square() + dx.square() + 1e-12).sqrt()
    return normalize(result)


def random_program(rng, *, depth=0):
    if depth >= 3 or (depth > 0 and rng.random() < .45):
        return {"op": "field", "name": str(rng.choice(CHANNELS))}
    if rng.random() < .7:
        op = str(rng.choice(UNARY))
        node = {"op": op, "arg": random_program(rng, depth=depth + 1)}
        if op in {"smooth", "contrast", "spread", "dilate", "erode", "gradient"}:
            node["scale"] = int(rng.choice(SCALES))
    else:
        node = {"op": str(rng.choice(BINARY)), "left": random_program(rng, depth=depth + 1),
                "right": random_program(rng, depth=depth + 1)}
    return node


def mutate(parent, rng):
    """Grow or replace a subtree; not just alter scalar hyperparameters."""
    node = json.loads(json.dumps(parent))
    choices = [(None, None, node)]

    def walk(value):
        for key in ("arg", "left", "right"):
            if key in value:
                choices.append((value, key, value[key]))
                walk(value[key])
    walk(node)
    owner, key, old = choices[int(rng.integers(len(choices)))]
    if rng.random() < .5:
        replacement = {"op": str(rng.choice(BINARY)), "left": old, "right": random_program(rng, depth=2)}
    else:
        replacement = random_program(rng, depth=1)
    if owner is None:
        node = replacement
    else:
        owner[key] = replacement
    try:
        validate(node)
    except ValueError:
        node = random_program(rng)
    return node


def render(node):
    """Readable source of the synthesized formula; never executed with eval."""
    validate(node)

    def expression(n):
        op = n["op"]
        if op == "field":
            return f'fields[{n["name"]!r}]'
        if op in BINARY:
            return f'{op}({expression(n["left"])}, {expression(n["right"])})'
        scale = f', scale={n["scale"]}' if "scale" in n else ""
        return f'{op}({expression(n["arg"])}{scale})'
    return ("# Generated spatial DSL source; operators are normalized tensor primitives.\n"
            "# This is an audit rendering, not unrestricted executable Python.\n"
            "def generated_feature(fields):\n    return " + expression(node) + "\n")


def make_pool(rng, parent_programs, *, size=32, seen=()):
    pool = []; used = set(seen); attempts = 0
    while len(pool) < size and attempts < size * 100:
        attempts += 1
        node = mutate(parent_programs[int(rng.integers(len(parent_programs)))], rng) if parent_programs and rng.random() < .65 else random_program(rng)
        key = identity(node)
        if key not in used:
            used.add(key); pool.append(node)
    if len(pool) != size:
        raise RuntimeError("Could not create requested distinct program pool")
    return pool


def top_mask(score, allowed, fraction=.1):
    if score.shape != allowed.shape or not torch.isfinite(score).all():
        raise ValueError("Invalid candidate score")
    budget = int(np.ceil(fraction * score.numel()))
    indices = torch.nonzero(allowed.flatten(), as_tuple=True)[0]
    order = torch.argsort(score.flatten()[indices], descending=True, stable=True)
    mask = torch.zeros_like(allowed, dtype=torch.bool)
    mask.flatten()[indices[order[:budget]]] = True
    return mask
