from copy import deepcopy

import numpy as np
import pytest
import torch

from tfree_detection.autonomous_program import (BINARY, CHANNELS, UNARY, evaluate,
    identity, make_pool, normalize, render, spatial, top_mask, validate)


def fields(size=23):
    rng = np.random.default_rng(23)
    return {key: normalize(torch.as_tensor(rng.normal(size=(size, size)), dtype=torch.float32)) for key in CHANNELS}


@pytest.mark.parametrize("node", [
    {"op": "field", "name": "human"},
    {"op": "field", "name": "source_id"},
    {"op": "field", "name": "base", "path": "/somewhere"},
    {"op": "exec", "arg": "print(1)"},
    {"op": "smooth", "arg": {"op": "field", "name": "image"}, "scale": 999999},
])
def test_grammar_forbids_labels_identity_io_and_unbounded_work(node):
    with pytest.raises(ValueError):
        validate(node)


@pytest.mark.parametrize("op", UNARY + BINARY)
def test_all_primitives_finite_and_same_shape(op):
    leaf = {"op": "field", "name": "image"}
    node = {"op": op, "left": leaf, "right": {"op": "field", "name": "student"}} if op in BINARY else {"op": op, "arg": leaf}
    if op in {"smooth", "contrast", "spread", "dilate", "erode", "gradient"}:
        node["scale"] = 63
    out = evaluate(node, fields())
    assert out.shape == (23, 23) and torch.isfinite(out).all()
    assert "def generated_feature" in render(node)


def test_no_extra_fields_even_when_program_does_not_reference_them():
    data = fields(); data["human"] = torch.ones((23, 23))
    with pytest.raises(ValueError):
        evaluate({"op": "field", "name": "base"}, data)


def test_pool_generates_compositions_and_does_not_repeat_rejected_programs():
    pool = make_pool(np.random.default_rng(2), [], size=32)
    ids = [identity(p) for p in pool]
    assert len(set(ids)) == 32 and any(validate(p) > 3 for p in pool)
    more = make_pool(np.random.default_rng(2), pool[:2], size=32, seen=ids)
    assert not set(ids) & {identity(p) for p in more}


def test_exact_budget_validity_and_stable_ties():
    score = torch.ones((10, 10)); allowed = torch.ones_like(score, dtype=torch.bool)
    allowed[0] = False
    mask = top_mask(score, allowed)
    assert int(mask.sum()) == 10 and mask[1].all() and not mask[0].any()
    empty = top_mask(score, torch.zeros_like(allowed))
    assert not empty.any()


def test_refuses_infinite_scores():
    score = torch.zeros((10, 10)); score[0, 0] = float("inf")
    with pytest.raises(ValueError):
        top_mask(score, torch.ones_like(score, dtype=torch.bool))


def test_constant_and_tiny_image_primitives_do_not_create_nonfinite_values():
    x = torch.ones((4, 4))
    assert torch.equal(spatial(x, 63), x)
    assert not normalize(x).any()


def test_depth_budget_is_enforced():
    node = {"op": "field", "name": "image"}
    for _ in range(10):
        node = {"op": "abs", "arg": deepcopy(node)}
    with pytest.raises(ValueError):
        validate(node)
