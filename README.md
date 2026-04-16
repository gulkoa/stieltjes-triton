# Stieltjes Flash Attention (Triton)

Reference Triton implementation of **Stieltjes attention** — a drop-in
replacement for softmax in Transformer attention that normalises scores with
an algebraic Stieltjes transform instead of the exponential map.

This repository accompanies the paper
*Efficient GPU Implementation of Stieltjes Attention: A Flash-Style Triton
Kernel for Algebraic Normalization* (Gulko & Kumar, 2026).

---

## Table of contents

1. [What is Stieltjes attention?](#what-is-stieltjes-attention)
2. [What is in this repository](#what-is-in-this-repository)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Usage in detail](#usage-in-detail)
7. [API reference](#api-reference)
8. [Numerical considerations](#numerical-considerations)
9. [Backward pass: why PyTorch and not Triton](#backward-pass-why-pytorch-and-not-triton)
10. [Test harness](#test-harness)
11. [Reported performance](#reported-performance)
12. [Troubleshooting](#troubleshooting)
13. [Limitations](#limitations)
14. [Citation](#citation)

---

## What is Stieltjes attention?

For row-wise scores `s_ij = (Q K^T / sqrt(d))_ij`, softmax computes
`p_ij = exp(s_ij) / Σ_k exp(s_ik)`. Stieltjes attention replaces this with

```
p_ij = (lambda_i - s_ij)^(-q) / Z_i,    Z_i = Σ_k (lambda_i - s_ik)^(-q)
```

where `q > 0` is an order parameter and `lambda_i > max_j s_ij` is the
row-specific *pole* chosen so that the row sums to 1. Because `Z_i` depends
on `lambda_i`, we find the pole with a guarded Newton–Raphson iteration. The
resulting weights have polynomial (rather than exponential) tails — see the
paper for motivation and properties.

The forward pass is implemented as a flash-style fused Triton kernel that
runs `2 + T` tiled sweeps over the keys (one for the row max, `T` for the
Newton solve, one for the `P @ V` accumulation) and never materialises the
`N × N` attention matrix. Working memory is `O(N · d)`.

---

## What is in this repository

| Path | Purpose |
| --- | --- |
| `stieltjes_attention.py` | Single-file implementation: PyTorch reference, Triton forward kernel, and `torch.autograd.Function` wrapper. |
| `tests/test_stieltjes.py` | Forward and backward correctness tests against the PyTorch reference. Pytest-friendly and stand-alone runnable. |
| `pyproject.toml` | Install metadata. Module is published as `stieltjes_attention`. |
| `README.md` | This file. |

The autograd wrapper uses the **Triton kernel for the forward pass** (fast,
low memory) and a **PyTorch backward** that reruns the autograd-friendly
reference and returns the gradients. This is exactly the code path used for
all training experiments in the paper — see
[Backward pass](#backward-pass-why-pytorch-and-not-triton) below.

---

## Prerequisites

### Hardware

* NVIDIA GPU with compute capability ≥ 7.0. Validated on:
  * A100-PCIE-40GB (CUDA 13)
  * H100-SXM5-80GB (CUDA 13)
* CUDA toolkit and driver matching the PyTorch wheel you install.
  PyTorch ships its own CUDA runtime, so usually you only need a recent
  driver (`nvidia-smi` shows ≥ the version printed by
  `torch.version.cuda`).

### Software

* Python 3.10 or newer.
* PyTorch ≥ 2.4 with CUDA support.
* Triton ≥ 3.0 (Triton ships with PyTorch on Linux x86-64; no separate
  install is normally needed).
* (Optional) `pytest` ≥ 8.0 to run the test harness via pytest.

The kernel is Linux/x86-64 only (Triton's platform support).

---

## Installation

The project is a single Python module with a `pyproject.toml`. Pick the
flow that matches your environment.

### 1. Plain `pip` (simplest)

```bash
git clone https://github.com/gulkoa/stieltjes-triton.git
cd stieltjes-triton

# editable install — picks up edits without reinstalling
pip install -e .

# with the test extras (adds pytest)
pip install -e ".[test]"
```

### 2. Fresh virtual environment

```bash
git clone https://github.com/gulkoa/stieltjes-triton.git
cd stieltjes-triton

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[test]"
```

### 3. With [`uv`](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/gulkoa/stieltjes-triton.git
cd stieltjes-triton

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[test]"
```

### 4. No install, just on `PYTHONPATH`

If you only need the module and don't want to touch your environment:

```bash
git clone https://github.com/gulkoa/stieltjes-triton.git
export PYTHONPATH="$PWD/stieltjes-triton:$PYTHONPATH"
```

### Verifying the install

```bash
python -c "import torch; import triton; import stieltjes_attention; \
print('torch', torch.__version__, 'triton', triton.__version__, \
      'cuda', torch.cuda.is_available())"
```

You should see `cuda True`. If you see `cuda False`, the kernel will not
run — install a CUDA-enabled PyTorch wheel from
<https://pytorch.org/get-started/locally/>.

---

## Quick start

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
    num_iter=3,                 # NR iterations; see "Numerical considerations"
)
o.sum().backward()              # backward goes through the PyTorch reference
```

---

## Usage in detail

### Bidirectional (encoder-style) attention

```python
o = stieltjes_attention(q, k, v, causal=False, stieltjes_q=2.0, num_iter=3)
```

### Causal (decoder-style, e.g. GPT)

```python
o = stieltjes_attention(q, k, v, causal=True, stieltjes_q=4.0, num_iter=3)
```

### Drop-in replacement inside an `nn.Module`

A typical Transformer self-attention block exposes Q, K, V tensors of
shape `(B, H, N, D)`. Stieltjes attention plugs in wherever softmax /
scaled-dot-product attention is invoked:

```python
import torch.nn as nn
from stieltjes_attention import stieltjes_attention

class StieltjesSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True, q=4.0, num_iter=3):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.causal = causal
        self.q = q
        self.num_iter = num_iter

    def forward(self, x):                              # x: (B, N, d_model)
        B, N, _ = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                        # each (B, H, N, D)
        o = stieltjes_attention(
            q, k, v,
            causal=self.causal,
            stieltjes_q=self.q,
            num_iter=self.num_iter,
        )                                              # (B, H, N, D)
        return self.proj(o.transpose(1, 2).reshape(B, N, -1))
```

### Reference-only (CPU or no Triton)

For debugging, gradient checking, or environments without a GPU, the
autograd-friendly reference is exposed directly:

```python
from stieltjes_attention import stieltjes_attention_ref

o = stieltjes_attention_ref(
    q, k, v,
    sm_scale=1.0 / (D ** 0.5),
    causal=True,
    stieltjes_q=4.0,
    num_iter=10,
)
```

The reference works on CPU and any dtype PyTorch supports. It is the
ground truth used by the test harness, but it materialises the full
`N × N` weight matrix and is roughly 3× slower than the Triton path for
the shapes in the paper.

### Inference at long context

For inference-only workloads (no `.backward()` call) the saved-tensor
graph is short-circuited and only Triton runs — this is the regime where
the paper's memory-and-throughput numbers matter most. Wrap the call in
`torch.inference_mode()` to skip autograd bookkeeping entirely:

```python
with torch.inference_mode():
    o = stieltjes_attention(q, k, v, causal=True,
                            stieltjes_q=4.0, num_iter=3)
```

---

## API reference

### `stieltjes_attention(q, k, v, causal=False, sm_scale=None, stieltjes_q=1.0, num_iter=3)`

The main entry point. Triton forward, PyTorch backward.

| Argument | Type | Description |
| --- | --- | --- |
| `q, k, v` | `Tensor (B, H, N, D)` | Queries / keys / values. `D ∈ {16, 32, 64, 128, 256}`. fp16, bf16, or fp32. |
| `causal` | `bool` | Apply lower-triangular causal mask. |
| `sm_scale` | `float \| None` | Score scale. Defaults to `1 / sqrt(D)`. |
| `stieltjes_q` | `float` | Stieltjes order `q > 0`. `q = 1` triggers an internal fast path. |
| `num_iter` | `int` | Newton–Raphson iterations in the Triton forward. |

Returns: `Tensor (B, H, N, D)` of the same dtype/device as `q`.

### `stieltjes_attention_ref(q, k, v, sm_scale, causal=False, stieltjes_q=1.0, num_iter=5, eps=1e-6)`

Pure-PyTorch reference implementation (autograd compatible end-to-end).
Materialises the `N × N` attention matrix; use for testing and as a
fallback on non-CUDA hardware.

### `StieltjesAttention(torch.autograd.Function)`

Low-level autograd Function. Most users should call
`stieltjes_attention` instead. The forward signature is
`forward(ctx, q, k, v, causal, sm_scale, stieltjes_q, num_iter)` and the
backward returns `(dq, dk, dv, None, None, None, None)`.

---

## Numerical considerations

### Iteration count `T` is a training hyperparameter

The Newton–Raphson initialisation `lambda_i^(0) = (i+1)^(1/q)` is the
closed-form root *only* for uniform scores. With trained attention
patterns the residual after a finite `T` is non-trivial and the model
learns to compensate.

> **Training and inference `num_iter` must match.** Training at `T = 3`
> and evaluating at `T = 10` (or vice versa) degrades accuracy by tens of
> points on the paper's CLRS tasks.

The paper fixes `T = 3` for both phases; we recommend the same default.

### Choice of `q`

* `q = 1` is the classical Cauchy Stieltjes transform and the cheapest
  to compute (one reciprocal, one squaring — no `exp/log`).
* `q ∈ {2, 4}` are the values that match softmax accuracy on the
  algorithmic tasks in the paper. Larger `q` gives sharper attention
  (closer to argmax).
* Very large `q` (e.g. `q ≥ 16`) is brittle to train; see the paper's
  *high-q brittleness* finding.

### Scale `sm_scale`

Defaults to `1 / sqrt(D)` to match standard scaled-dot-product attention.
Custom scales are supported (e.g. ALiBi-style position-dependent
biasing on the score tensor before this call).

### Dtype

The kernel accepts fp16, bf16, and fp32 inputs and accumulates in fp32
internally. fp16 is the recommended I/O dtype on A100/H100; bf16 is fine
on H100 and is what the paper uses for long-context training.

---

## Backward pass: why PyTorch and not Triton

A naive Triton backward that treats `lambda_i` as a constant saved from
the forward pass passes random-input correctness tests but diverges
training within a single bf16 step. The reason is documented in
Section 2.3 of the paper: the forward defines `lambda_i = lambda_i(S)`
*implicitly* via the Newton root condition, and the missing
implicit-function term

```
∂lambda_i / ∂s_ij = -(∂F/∂s_ij) / (∂F/∂lambda_i)
```

makes gradients six to seven orders of magnitude too large under trained
attention patterns. Tracing PyTorch autograd through the NR iteration
graph captures this implicit dependence automatically and is fast enough
for nanoGPT-scale experiments. A fused Triton backward with the rank-one
correction is left for future work.

---

## Test harness

The test file lives at `tests/test_stieltjes.py`. It can be invoked
either through pytest (recommended for CI) or as a stand-alone script
(useful when pytest is not installed).

All tests automatically skip on machines without CUDA.

### Run everything

```bash
pytest tests/test_stieltjes.py
```

Verbose output (one line per parametrised configuration):

```bash
pytest tests/test_stieltjes.py -v
```

Compact failure tracebacks:

```bash
pytest tests/test_stieltjes.py -v --tb=short
```

### List the test grid without running

```bash
pytest tests/test_stieltjes.py --collect-only -q
```

You will see entries such as

```
tests/test_stieltjes.py::test_forward_matches_reference[1-1-64-64-False-1.0]
tests/test_stieltjes.py::test_forward_matches_reference[1-1-64-64-True-1.0]
...
tests/test_stieltjes.py::test_backward_matches_reference[1-1-1024-64-False-1.0]
```

The bracketed suffix is `[B-H-N-D-causal-q]`.

### Run only the forward tests

```bash
pytest tests/test_stieltjes.py -v -k forward
```

### Run only the backward tests

```bash
pytest tests/test_stieltjes.py -v -k backward
```

### Run a single configuration

```bash
# any matching id substring works
pytest tests/test_stieltjes.py -v -k "backward and 1024"
pytest tests/test_stieltjes.py -v -k "1-2-512-64-True-1.0"
```

### Stop on the first failure

```bash
pytest tests/test_stieltjes.py -x
```

### Stand-alone (no pytest required)

```bash
python tests/test_stieltjes.py
```

This walks the same forward and backward grid and prints `PASS`/`FAIL`
per configuration. Exit code is `0` on success, `1` if any case failed.
Useful when running on a freshly-spawned compute node where you would
rather not install `pytest`.

### What the suite checks

* **Forward parity.** fp16 Triton output vs an fp32 reference run with
  10 NR iterations. Tolerance: `max_err < 0.05`. The grid covers
  `B ∈ {1, 2}`, `H ∈ {1, 2, 4}`, `N ∈ {64, 128, 256, 512, 1024}`,
  `D ∈ {64, 128}`, `causal ∈ {False, True}`, `q ∈ {1.0, 2.0}`.
* **Backward parity.** Gradients from `stieltjes_attention.backward`
  vs `torch.autograd.grad` through the reference at fp32. Tolerance:
  `max(dQ_err, dK_err, dV_err) < 0.15`. (Backward is currently routed
  through the same reference path under PyTorch autograd, so this test
  is also a regression check on the reference's own gradient.)

---

## Reported performance

From the paper, Section 3.2 (forward kernel only, vs the dense PyTorch
reference, causal mask, `D = 64`, `q = 4`):

| Hardware | Forward speedup (geomean) | Memory at (B=4, H=8, N=2048, D=64) |
| --- | --- | --- |
| A100-PCIE-40GB | **3.37×** | **329 MB** vs 1609 MB (4.9× reduction) |
| H100-SXM5-80GB | **3.78×** | same 4.9× reduction |

The headline operational result: the dense PyTorch reference OOMs on a
single A100-40GB at `N = 32k`, batch 1, while the Triton kernel runs
cleanly at the same configuration.

---

## Troubleshooting

**`ImportError: triton`** — On Linux x86-64, `pip install torch` already
brings Triton in. On unsupported platforms (macOS, Windows, ARM) Triton
will not install and only `stieltjes_attention_ref` is usable.

**`torch.cuda.is_available() == False`** — The Triton path requires CUDA.
Install a CUDA-enabled PyTorch wheel from
<https://pytorch.org/get-started/locally/>.

**`AssertionError: head dim … not supported`** — Only `D ∈ {16, 32, 64,
128, 256}` are compiled. Pad/truncate to one of these for the kernel
call.

**Out-of-memory on the reference path** — switch to
`stieltjes_attention` (Triton). The reference materialises the `N × N`
weight matrix; the Triton kernel does not.

**Gradients explode / loss diverges in the first step** — confirm you
are calling `stieltjes_attention` (Triton fwd, PyTorch bwd) and **not** a
hand-rolled Triton backward; see
[Backward pass](#backward-pass-why-pytorch-and-not-triton).

**Train/eval accuracy mismatch** — check that `num_iter` is identical at
training and inference time; see *Numerical considerations*.

---

## Limitations

* No fused Triton backward (intentional; see above). Backward runs the
  reference under PyTorch autograd.
* Head dimension restricted to `{16, 32, 64, 128, 256}`.
* Single-GPU only. No tensor-parallel / sequence-parallel partitioning is
  performed inside the kernel; combine with PyTorch's distributed
  primitives at the module level.
* No support yet for sliding-window or arbitrary additive attention
  masks; only `causal ∈ {False, True}` is exposed.

---

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
