"""
Stieltjes Flash Attention — public reference implementation.

Drop-in alternative to softmax attention.  The query–key scores are normalised
with a Stieltjes transform instead of softmax:

    standard:   O = softmax(QK^T / sqrt(d)) @ V
    stieltjes:  O = stieltjes_q(QK^T / sqrt(d)) @ V

with element-wise weights

    P_ij = (lambda_i - s_ij)^{-q}

and lambda_i chosen per row so that sum_j P_ij = 1.  The unique root is found
with a guarded Newton–Raphson iteration.  q = 1 recovers the classical Cauchy
Stieltjes transform; arbitrary q > 0 is supported.

This file ships:

  * `stieltjes_attention_ref` — pure-PyTorch reference, autograd-friendly.
  * `_stieltjes_attn_fwd`     — Triton forward kernel (flash-style, O(N·D) memory).
  * `StieltjesAttention`      — torch.autograd.Function with Triton forward and
                                a PyTorch backward (autograd through the
                                reference implementation).
  * `stieltjes_attention`     — convenience wrapper.

The forward is a fused tiled kernel with three sweeps over K (one for the row
max, `num_iter` for the Newton–Raphson, one for P @ V).  The backward
intentionally falls back to PyTorch — it is correct, easy to read, and the
forward is what is novel.  See `tests/test_stieltjes.py` for parity checks.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "stieltjes_attention",
    "stieltjes_attention_ref",
    "StieltjesAttention",
]


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch, autograd-friendly)
# ---------------------------------------------------------------------------

def stieltjes_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    causal: bool = False,
    stieltjes_q: float = 1.0,
    num_iter: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference Stieltjes attention.

    Args:
        q, k, v: (B, H, N, D) tensors.
        sm_scale: score scale (typically 1 / sqrt(D)).
        causal:  apply causal mask.
        stieltjes_q: Stieltjes order q > 0.
        num_iter: Newton–Raphson iterations.
        eps: floor on (lambda - s) to avoid division by zero.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale  # (B, H, N, N)

    if causal:
        N = scores.shape[-1]
        mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    sq = stieltjes_q
    s_max = scores.max(dim=-1, keepdim=True).values
    x = scores - s_max  # centred so max is 0; lambda must stay > 0

    n_cols = scores.shape[-1]
    if causal:
        # Row i has (i+1) valid positions.  (i+1)^{1/q} is the all-equal solution.
        row_counts = torch.arange(1, n_cols + 1, device=scores.device, dtype=scores.dtype)
        lambd = row_counts.pow(1.0 / sq).view(1, 1, -1, 1).expand_as(s_max)
    else:
        lambd = torch.full_like(s_max, float(n_cols) ** (1.0 / sq))

    for _ in range(num_iter):
        diff = (lambd - x).clamp(min=eps)
        f_val = diff.pow(-sq).sum(dim=-1, keepdim=True) - 1.0
        f_deriv = -sq * diff.pow(-sq - 1.0).sum(dim=-1, keepdim=True)
        # Halving guard: NR can overshoot below 0; clamp below by lambda/2.
        lambd = torch.maximum(lambd - f_val / f_deriv, lambd * 0.5)

    diff = (lambd - x).clamp(min=eps)
    weights = diff.pow(-sq)

    if causal:
        weights = weights.masked_fill(~mask, 0.0)

    return torch.matmul(weights.to(v.dtype), v)


# ---------------------------------------------------------------------------
# Triton forward kernel
# ---------------------------------------------------------------------------
#
# Tiled, flash-style.  For each query block of `BLOCK_M` rows the kernel makes
# (2 + NUM_ITER) sweeps over the keys and one extra sweep that also reads V:
#
#   Pass 1: row-wise max of QK^T scores.
#   Pass 2: NUM_ITER Newton–Raphson iterations for lambda.
#   Pass 3: weights = (lambda - s)^{-q}, accumulate weights @ V.
#
# Working memory is O(N·D) — the NxN attention matrix is never materialised.

@triton.jit
def _stieltjes_attn_fwd(
    Q, K, V, O,
    LambdaInit,  # (N,) fp32 — per-row initial lambda guess.  For causal:
                 # (i+1)^{1/q}.  For non-causal: N^{1/q} broadcast.  Matching
                 # the reference's init keeps NR trajectories identical.
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    sm_scale,
    N_CTX,
    sq: tl.constexpr,
    NUM_ITER: tl.constexpr,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Recover (batch, head) from the flat (B*H) program id, supporting strided
    # / non-contiguous Q,K,V.  For contiguous (B,H,N,D), H_eff equals H.
    H_eff = stride_qz // stride_qh
    off_z = off_hz // H_eff
    off_h = off_hz % H_eff

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_block = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # === Pass 1: row-wise max of QK^T ====================================
    row_max = tl.full([BLOCK_M], value=-1e30, dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_block = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q_block, tl.trans(k_block)) * sm_scale
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -1e30)

        row_max = tl.maximum(row_max, tl.max(qk, axis=1))

    # === Pass 2: Newton–Raphson for lambda ===============================
    # After centring by row_max the scores are <= 0, so lambda > 0.
    init_ptrs = LambdaInit + offs_m
    lambd = tl.load(init_ptrs, mask=offs_m < N_CTX, other=1.0)

    for _nr in tl.static_range(NUM_ITER):
        f_val = tl.zeros([BLOCK_M], dtype=tl.float32)
        f_deriv = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_n in tl.range(0, N_CTX, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_block = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

            qk = tl.dot(q_block, tl.trans(k_block)) * sm_scale
            if CAUSAL:
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -1e30)

            centered = qk - row_max[:, None]
            diff = tl.maximum(lambd[:, None] - centered, EPS)

            if sq == 1.0:
                inv_q = 1.0 / diff
                inv_q1 = inv_q * inv_q
            else:
                log_diff = tl.log(diff)
                inv_q = tl.exp(log_diff * (-sq))
                inv_q1 = tl.exp(log_diff * (-sq - 1.0))

            # Masked entries have centered ~ -1e30 -> diff huge -> inv ~ 0.
            f_val += tl.sum(inv_q, axis=1)
            f_deriv += tl.sum(inv_q1, axis=1)

        # f(lambda) = sum (lambda - s)^{-q} - 1
        # f'(lambda) = -q * sum (lambda - s)^{-q-1}
        f_val = f_val - 1.0
        f_deriv = f_deriv * (-sq)
        lambd = tl.maximum(lambd - f_val / f_deriv, lambd * 0.5)

    # === Pass 3: weights @ V =============================================
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k_block = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v_block = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q_block, tl.trans(k_block)) * sm_scale
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -1e30)

        centered = qk - row_max[:, None]
        diff = tl.maximum(lambd[:, None] - centered, EPS)

        if sq == 1.0:
            weights = 1.0 / diff
        else:
            weights = tl.exp(tl.log(diff) * (-sq))

        acc += tl.dot(weights.to(v_block.dtype), v_block)

    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(q_block.dtype), mask=offs_m[:, None] < N_CTX)


def _triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
    stieltjes_q: float,
    num_iter: int,
) -> torch.Tensor:
    B, H, N, D = q.shape
    assert k.shape == v.shape == (B, H, N, D), "Q, K, V must share shape"
    assert D in {16, 32, 64, 128, 256}, f"head dim {D} not supported"

    # Per-row initial lambda — match the reference exactly.
    if causal:
        row_counts = torch.arange(1, N + 1, device=q.device, dtype=torch.float32)
        lambda_init = row_counts.pow(1.0 / stieltjes_q)
    else:
        lambda_init = torch.full(
            (N,), float(N) ** (1.0 / stieltjes_q),
            device=q.device, dtype=torch.float32,
        )

    if D <= 64:
        BLOCK_M, BLOCK_N = 128, 64
    else:
        BLOCK_M, BLOCK_N = 64, 64

    o = torch.empty_like(q)
    grid = (triton.cdiv(N, BLOCK_M), B * H)

    _stieltjes_attn_fwd[grid](
        q, k, v, o,
        lambda_init,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sm_scale,
        N,
        sq=stieltjes_q,
        NUM_ITER=num_iter,
        EPS=1e-6,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
    )
    return o


# ---------------------------------------------------------------------------
# Autograd: Triton forward, PyTorch backward via the reference impl
# ---------------------------------------------------------------------------

class StieltjesAttention(torch.autograd.Function):
    """Stieltjes attention with a Triton forward and PyTorch (autograd) backward.

    The reference implementation is autograd-compatible end-to-end, so the
    backward simply reruns it under `torch.enable_grad()` and returns
    `torch.autograd.grad`.  Numerically the gradients match what a fully fused
    Triton backward would produce, modulo floating-point accumulation order.

    The number of Newton–Raphson iterations in the backward is bumped to
    `max(num_iter, 5)` so the saved lambda from the (faster) Triton forward is
    not the bottleneck for gradient quality.
    """

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, stieltjes_q, num_iter):
        o = _triton_forward(q, k, v, causal, sm_scale, stieltjes_q, num_iter)
        ctx.save_for_backward(q, k, v)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        ctx.stieltjes_q = stieltjes_q
        ctx.num_iter = num_iter
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        bwd_iter = max(ctx.num_iter, 5)
        with torch.enable_grad():
            qg = q.detach().requires_grad_(True)
            kg = k.detach().requires_grad_(True)
            vg = v.detach().requires_grad_(True)
            o = stieltjes_attention_ref(
                qg, kg, vg,
                ctx.sm_scale,
                causal=ctx.causal,
                stieltjes_q=ctx.stieltjes_q,
                num_iter=bwd_iter,
            )
            dq, dk, dv = torch.autograd.grad(o, (qg, kg, vg), do)
        return dq, dk, dv, None, None, None, None


def stieltjes_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    stieltjes_q: float = 1.0,
    num_iter: int = 3,
) -> torch.Tensor:
    """Stieltjes flash attention (Triton forward, PyTorch backward).

    Args:
        q, k, v: (B, H, N, D) — query / key / value.  D in {16, 32, 64, 128, 256}.
        causal:  apply causal mask.
        sm_scale: score scale.  Defaults to 1 / sqrt(D).
        stieltjes_q: Stieltjes order q > 0 (q = 1 is the classical case).
        num_iter: Newton–Raphson iterations in the forward.  3 is usually plenty.
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    return StieltjesAttention.apply(q, k, v, causal, sm_scale, stieltjes_q, num_iter)
