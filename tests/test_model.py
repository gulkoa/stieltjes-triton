"""Model forward/backward smoke tests on CPU.

Mirrors `Section 3` of the original `scripts/test_all.sh` harness. Runs on
CPU using softmax attention by default; the Stieltjes reference path also
runs on CPU so we test it end-to-end here as well.
"""

from __future__ import annotations

import torch

from nanogpt.model import GPT, GPTConfig


def _tiny_cfg(attn_type="softmax", **kw):
    return GPTConfig(
        vocab_size=258,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        attn_type=attn_type,
        **kw,
    )


def test_softmax_forward_backward():
    torch.manual_seed(0)
    model = GPT(_tiny_cfg(attn_type="softmax"))
    x = torch.randint(0, 258, (2, 32))
    targets = torch.randint(0, 258, (2, 32))
    logits, loss = model(x, targets=targets)
    assert logits.shape == (2, 32, 258)
    assert torch.isfinite(loss)
    loss.backward()


def test_stieltjes_ref_forward_backward():
    """Stieltjes via the PyTorch reference path — same code path used by all
    training experiments in the paper."""
    torch.manual_seed(0)
    model = GPT(
        _tiny_cfg(
            attn_type="stieltjes",
            stieltjes_q=2.0,
            stieltjes_num_iter=5,
            stieltjes_use_triton=False,
        )
    )
    x = torch.randint(0, 258, (2, 32))
    targets = torch.randint(0, 258, (2, 32))
    logits, loss = model(x, targets=targets)
    assert logits.shape == (2, 32, 258)
    assert torch.isfinite(loss)
    loss.backward()
    # gradients exist and are finite for every parameter that requires_grad
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad has NaN/inf"


def test_nope_position_encoding_runs():
    """NoPE config should accept sequences longer than block_size."""
    torch.manual_seed(0)
    model = GPT(_tiny_cfg(attn_type="softmax", pos_enc="none"))
    x = torch.randint(0, 258, (2, 200))  # > block_size=64
    logits = model(x)
    assert logits.shape == (2, 200, 258)


def test_param_count_in_expected_range():
    """6-layer 384-embd default config (paper's main backbone)."""
    cfg = GPTConfig(
        vocab_size=258, block_size=512, n_layer=6, n_head=6, n_embd=384, dropout=0.1,
    )
    model = GPT(cfg)
    n = model.num_params()
    # tied embedding + 6 blocks; the paper's main backbone is ~11M parameters.
    assert 8_000_000 < n < 15_000_000, f"unexpected param count: {n:,}"
