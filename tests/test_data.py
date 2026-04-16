"""CLRS data correctness — verifies generators produce the right outputs and
the accuracy metric pipeline is internally consistent.

Mirrors `Section 2` and `Section 4` of the original `scripts/test_all.sh`
harness. Runs on CPU; no GPU required.
"""

from __future__ import annotations

from collections import deque

import torch

from nanogpt.data import (
    CLRSDataset,
    PAD,
    SEPARATOR,
    TaskConfig,
)


def _output_after_sep(tokens):
    sep_idx = tokens.index(SEPARATOR)
    return tokens[:sep_idx], [t for t in tokens[sep_idx + 1 :] if t != PAD]


def test_sorting_output_is_sorted():
    cfg = TaskConfig(task_name="sorting", num_samples=100, seq_len=64, max_arr_len=12)
    ds = CLRSDataset(cfg, seed=42)
    for i in range(100):
        inp, out = _output_after_sep(ds.samples[i])
        assert out == sorted(inp), f"sample {i}: {inp} -> {out} != {sorted(inp)}"


def test_max_output_correct():
    cfg = TaskConfig(task_name="max", num_samples=100, seq_len=64, max_arr_len=12)
    ds = CLRSDataset(cfg, seed=42)
    for i in range(100):
        inp, out = _output_after_sep(ds.samples[i])
        assert len(out) == 2, f"sample {i}: expected 2 output tokens, got {len(out)}"
        assert out[0] == max(inp), f"sample {i}: max wrong"
        assert out[1] == inp.index(max(inp)), f"sample {i}: argmax wrong"


def test_binary_search_output_correct():
    cfg = TaskConfig(task_name="binary_search", num_samples=100, seq_len=64, max_arr_len=12)
    ds = CLRSDataset(cfg, seed=42)
    for i in range(100):
        tokens = ds.samples[i]
        seps = [j for j, t in enumerate(tokens) if t == SEPARATOR]
        assert len(seps) >= 2
        arr = tokens[: seps[0]]
        target = tokens[seps[0] + 1 : seps[1]]
        assert len(target) == 1
        out = [t for t in tokens[seps[1] + 1 :] if t != PAD]
        assert len(out) == 1
        assert arr[out[0]] == target[0], f"sample {i}: arr[{out[0]}] != {target[0]}"


def test_needle_task_correctness():
    cfg = TaskConfig(task_name="needle", num_samples=100, seq_len=4096, max_arr_len=2000)
    ds = CLRSDataset(cfg, seed=42)
    for i in range(100):
        inp, out = _output_after_sep(ds.samples[i])
        assert len(out) == 1, f"sample {i}: expected 1 output, got {len(out)}"
        needle_val = out[0]
        assert 128 <= needle_val <= 254, f"sample {i}: needle value out of range"
        assert needle_val in inp, f"sample {i}: needle not in input"
        n_needles = sum(1 for t in inp if t >= 128)
        assert n_needles == 1, f"sample {i}: {n_needles} needles (expected 1)"


def test_bfs_no_vocab_collision_at_max_scale():
    cfg = TaskConfig(task_name="bfs", num_samples=50, seq_len=4096, max_arr_len=255, max_val=256)
    ds = CLRSDataset(cfg, seed=99)
    for i in range(50):
        for j, t in enumerate(ds.samples[i]):
            if t == SEPARATOR:
                break
            assert t < SEPARATOR, f"sample {i} pos {j}: token {t} collides with SEPARATOR"


def test_bfs_output_correct():
    cfg = TaskConfig(task_name="bfs", num_samples=50, seq_len=128, max_arr_len=12)
    ds = CLRSDataset(cfg, seed=42)
    for i in range(50):
        tokens = ds.samples[i]
        sep_idx = tokens.index(SEPARATOR)
        inp = tokens[:sep_idx]
        n_nodes = inp[0]
        pos = 1
        adj = [[] for _ in range(n_nodes)]
        for node in range(n_nodes):
            deg = inp[pos]
            pos += 1
            for _ in range(deg):
                adj[node].append(inp[pos])
                pos += 1
        # Reference BFS from node 0 using sorted adjacency.
        visited = []
        seen = {0}
        q = deque([0])
        while q:
            node = q.popleft()
            visited.append(node)
            for nb in sorted(adj[node]):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        out = [t for t in tokens[sep_idx + 1 :] if t != PAD]
        assert out == visited, f"sample {i}: BFS mismatch"


def test_output_only_accuracy_metric():
    """`compute_accuracy` in train.py counts only positions after the last
    SEPARATOR. Verify that the (x, y) target stream actually has output
    tokens at those positions."""
    cfg = TaskConfig(task_name="sorting", num_samples=10, seq_len=32, max_arr_len=6)
    ds = CLRSDataset(cfg, seed=42)
    x, y = ds[0]
    seps = (x == SEPARATOR).nonzero(as_tuple=True)[0]
    assert len(seps) > 0
    out_start = seps[-1].item()
    out_mask = torch.zeros_like(y, dtype=torch.bool)
    out_mask[out_start:] = True
    out_mask &= y != PAD
    assert out_mask.sum().item() > 0
    first_out = y[out_start].item()
    assert first_out != PAD and first_out != SEPARATOR


def test_binary_search_target_idx():
    cfg = TaskConfig(task_name="binary_search", num_samples=10, seq_len=32, max_arr_len=6)
    ds = CLRSDataset(cfg, seed=42)
    x, y = ds[0]
    seps = (x == SEPARATOR).nonzero(as_tuple=True)[0]
    assert len(seps) >= 2
    target_idx = y[seps[-1].item()].item()
    assert target_idx not in (PAD, SEPARATOR)


def test_pad_constant_matches_model_ignore_index():
    """model.py hardcodes ignore_index=257 in cross-entropy. data.py defines
    PAD=257. Drift between them silently changes the loss."""
    from nanogpt.data import PAD
    import inspect
    from nanogpt.model import GPT

    src = inspect.getsource(GPT.forward)
    assert "ignore_index=257" in src
    assert PAD == 257
