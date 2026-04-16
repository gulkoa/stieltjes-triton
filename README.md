# Stieltjes Flash Attention (Triton)

Reference Triton implementation of **Stieltjes attention** — a drop-in
replacement for softmax in Transformer attention that normalises scores with
an algebraic Stieltjes transform instead of the exponential map.

This repository accompanies the paper
*Efficient GPU Implementation of Stieltjes Attention: A Flash-Style Triton
Kernel for Algebraic Normalization* (Gulko & Kumar, 2026).

## What is Stieltjes attention?

For row-wise scores `s_ij = (Q K^T / sqrt(d))_ij`, softmax computes
`p_ij = exp(s_ij) / sum_k exp(s_ik)`. Stieltjes attention replaces this with

```
p_ij = (lambda_i - s_ij)^(-q) / Z_i,    Z_i = sum_k (lambda_i - s_ik)^(-q)
```

where `q > 0` is an order parameter and `lambda_i > max_j s_ij` is the
row-specific *pole* chosen so that the row sums to 1. Because `Z_i` depends
on `lambda_i`, we find the pole with a guarded Newton–Raphson iteration. The
resulting weights have polynomial (rather than exponential) tails — see the
paper for the motivation.

The forward pass is implemented as a flash-style fused Triton kernel that
runs `2 + T` tiled sweeps over the keys (one for the row max, `T` for the
Newton solve, one for the `P @ V` accumulation) and never materialises the
`N x N` attention matrix. Working memory is `O(N · d)`.

## What is in this repository

This is the **reference release** that pairs with the paper. It contains:

| File | Purpose |
| --- | --- |
| `stieltjes_attention.py` | Single-file implementation: pure-PyTorch reference, Triton forward kernel, and `torch.autograd.Function` wrapper. |
| `tests/test_stieltjes.py` | Forward and backward correctness tests against the PyTorch reference. |
| `pyproject.toml` | Minimal install metadata. |

The autograd wrapper uses the **Triton kernel for the forward pass** (fast,
low memory) and a **PyTorch backward** that reruns the autograd-friendly
reference and returns the gradients. This is the same code path used for all
training experiments in the paper — see *Backward pass* below.

## Why is the backward in PyTorch?

A naive Triton backward that treats `lambda_i` as a constant saved from the
forward pass passes random-input correctness tests but diverges training
within a single bf16 step. The reason is documented in the paper: the
forward defines `lambda_i = lambda_i(S)` *implicitly* via the Newton root
condition, and the missing implicit-function term

```
∂lambda_i / ∂s_ij = -(∂F/∂s_ij) / (∂F/∂lambda_i)
```

makes gradients six to seven orders of magnitude too large under trained
attention patterns. Tracing autograd through the NR iteration graph in
PyTorch is correct (the implicit dependence is captured automatically) and
fast enough for our nanoGPT-scale experiments. A fused Triton backward with
the rank-one correction is left for future work.

## Installation

```bash
pip install -e .
# or, with the test extras:
pip install -e ".[test]"
```

Requires Python 3.10+, a CUDA GPU, PyTorch 2.4+, and Triton 3.0+. The kernel
has been exercised on A100 (CUDA 13) and H100 (CUDA 13).

## Usage

```python
import torch
from stieltjes_attention import stieltjes_attention

B, H, N, D = 4, 8, 2048, 64
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)

o = stieltjes_attention(
    q, k, v,
    causal=True,
    sm_scale=1.0 / (D ** 0.5),  # default
    stieltjes_q=4.0,            # paper's preferred q for CLRS tasks
    num_iter=3,                 # NR iterations; see "iteration count" below
)
o.sum().backward()  # backward goes through the PyTorch reference
```

Supported configurations:

* Shapes: `(B, H, N, D)` with head dimension `D ∈ {16, 32, 64, 128, 256}`.
* Dtypes: fp16 / bf16 / fp32 inputs (the kernel accumulates in fp32).
* `causal=True` for autoregressive masking.
* `stieltjes_q` is any positive float; `q = 1` triggers an internal fast
  path that skips the `exp(-q * log(·))` pair.

### Reference-only (no Triton)

For debugging or for environments without a GPU, the autograd-friendly
reference is exposed directly:

```python
from stieltjes_attention import stieltjes_attention_ref
o = stieltjes_attention_ref(q, k, v, sm_scale=1.0/D**0.5,
                            causal=True, stieltjes_q=4.0, num_iter=10)
```

## Iteration count `T` is a training hyperparameter

The Newton–Raphson initialisation `lambda_i^(0) = (i+1)^(1/q)` is the
closed-form root only for uniform scores. With trained attention patterns
the residual after a finite `T` is non-trivial and the model learns to
compensate. **The training and inference iteration counts must match** —
training at `T = 3` and evaluating at `T = 10` (or vice versa) degrades
accuracy by tens of points. The paper fixes `T = 3` for both phases; we
recommend the same default.

## Tests

```bash
# pytest
pytest tests/test_stieltjes.py -v

# or stand-alone
python tests/test_stieltjes.py
```

The suite exercises a grid of shapes (including non-power-of-two `N` that
crosses multiple Triton tiles), `causal ∈ {False, True}`, and
`q ∈ {1, 2}`. Forward parity is checked against an fp32 reference within
fp16 tolerance; backward parity is checked against autograd through the
reference.

## Reported performance (paper, Section 3.2)

| Hardware | Forward speedup vs PyTorch ref (geomean) | Memory reduction at (B=4, H=8, N=2048, D=64) |
| --- | --- | --- |
| A100-PCIE-40GB | 3.37× | 4.9× |
| H100-SXM5-80GB | 3.78× | 4.9× |

These come from the throughput-and-memory grid (B ∈ {1,4}, H = 8,
N ∈ {128, 512, 1024, 2048}, D = 64, q = 4, causal). The headline memory
result is operational: the dense PyTorch reference OOMs on a single
A100-40GB at `N = 32k`, batch 1; the Triton kernel runs cleanly at the same
configuration.

## Citation

```bibtex
@misc{gulko2026stieltjes,
  title  = {Efficient GPU Implementation of Stieltjes Attention:
            A Flash-Style Triton Kernel for Algebraic Normalization},
  author = {Gulko, Alex and Kumar, Sachin},
  year   = {2026},
}
```

The Stieltjes-transform simplex-map framework underlying this kernel is
joint work with Jack Taylor and Tarun Kathuria; see the paper's
acknowledgement footnote for the full attribution.
