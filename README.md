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
10. [Training pipeline (nanoGPT)](#training-pipeline-nanogpt)
11. [Benchmarks](#benchmarks)
12. [Figures](#figures)
13. [Test harness](#test-harness)
14. [Continuous integration](#continuous-integration)
15. [Reported performance](#reported-performance)
16. [Troubleshooting](#troubleshooting)
17. [Limitations](#limitations)
18. [Citation](#citation)

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

This is the **complete training and evaluation pipeline** that produced the
paper. It contains:

| Path | Purpose |
| --- | --- |
| `stieltjes_attention.py` | Triton forward kernel + PyTorch reference + autograd Function. |
| `nanogpt/` | Training pipeline: CLRS data, GPT-2-style backbone with swappable softmax/Stieltjes attention, training loop, q-curriculum trainer, checkpoint evaluation, attention analysis. |
| `benchmarks/` | Throughput benchmarks (`bench_triton_vs_ref.py` produces the paper's headline speedup CSV; `bench_stieltjes.py` benchmarks the standalone NR vs binary-search kernels). |
| `figures/` | Plotting scripts that produced every paper figure from CSV/JSON outputs. |
| `tests/` | Pytest harness — converted from the original `scripts/test_all.sh` and split by responsibility (`test_imports`, `test_data`, `test_model`, `test_kernel`). Same checks the paper's authors ran before every commit. |
| `.github/workflows/tests.yml` | CPU CI on every push/PR. |
| `pyproject.toml` | Install metadata + extras (`nanogpt`, `figures`, `test`, `dev`). |

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

# kernel only
pip install -e .

# kernel + test harness (pytest)
pip install -e ".[test]"

# everything: kernel + nanogpt training + figures + tests
pip install -e ".[dev]"
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

## Training pipeline (nanoGPT)

The `nanogpt/` package is the training pipeline used for every accuracy
experiment in the paper. It is a stripped-down GPT-2-style backbone with
swappable softmax / Stieltjes attention, byte-level CLRS-task data
generation, an output-only accuracy metric, and a curriculum trainer
that anneals the Stieltjes order `q` over epochs.

Install the pipeline extras alongside the kernel:

```bash
pip install -e ".[nanogpt]"   # adds numpy
# or, for everything (training + figures + tests):
pip install -e ".[dev]"
```

### Layout

| Module | Responsibility |
| --- | --- |
| `nanogpt/data.py` | CLRS task generators (`sorting`, `binary_search`, `bfs`, `max`, `needle`) and the `CLRSDataset` PyTorch wrapper. Vocabulary is `0..255` byte values + `SEPARATOR=256` + `PAD=257`. |
| `nanogpt/model.py` | `GPTConfig` and `GPT`. Causal self-attention block dispatches to either softmax or Stieltjes (Triton kernel or PyTorch reference). `pos_enc="learned"` (default) or `"none"` (NoPE). |
| `nanogpt/train.py` | Single-config training loop. Logs per-epoch CSV; designed for a compute node (no plotting). |
| `nanogpt/train_curriculum.py` | Same loop but mutates `stieltjes_q` per a `q@epoch` schedule string. |
| `nanogpt/eval_accuracy.py` | Re-evaluates a saved checkpoint under three accuracy metrics (`output_only`, `all_positions`, `input_echo`) and emits JSON. |
| `nanogpt/analyze.py` | Extracts per-head attention patterns, entropy and concentration statistics from a checkpoint; saves CSV and `.pt` tensors for downstream figure scripts. |

### Train softmax baseline

```bash
python nanogpt/train.py \
    --task binary_search \
    --attn softmax \
    --seq-len 128 \
    --epochs 30 \
    --num-samples 20000 \
    --out results/binary_search_softmax/
```

### Train Stieltjes (PyTorch reference path — what the paper uses)

```bash
python nanogpt/train.py \
    --task binary_search \
    --attn stieltjes \
    --q 4.0 \
    --stieltjes-num-iter 3 \
    --seq-len 128 \
    --epochs 30 \
    --num-samples 20000 \
    --out results/binary_search_stieltjes_q4/
```

Add `--stieltjes-use-triton` to route training through the Triton forward
(plus PyTorch backward via the autograd Function). Numerically equivalent
on the paper's tasks but slightly faster at long context.

### q-curriculum

Anneal q over epochs. Schedule is a comma-separated list of `q@start_epoch`
(1-indexed); each value is held until the next entry kicks in.

```bash
python nanogpt/train_curriculum.py \
    --task binary_search --attn stieltjes \
    --q-schedule "1@1,2@11,4@21,8@31,16@41" \
    --epochs 50 \
    --out results/bsearch_curriculum_q1to16/
```

### Evaluate a checkpoint

```bash
python nanogpt/eval_accuracy.py \
    --checkpoint results/binary_search_stieltjes_q4/model.pt \
    --task binary_search --attn stieltjes --q 4.0 \
    --seq-len 128 --val-samples 5000 --seed 42
```

Writes `accuracy_fixed.json` next to the checkpoint with the three
accuracy variants and a sanity-check histogram of output-start positions.

### Extract attention patterns

```bash
python nanogpt/analyze.py \
    --checkpoint results/binary_search_stieltjes_q4/model.pt \
    --task binary_search --attn stieltjes --q 4.0 \
    --out results/binary_search_stieltjes_q4/analysis/
```

Each script accepts `--help` for the full argument list.

---

## Benchmarks

Both benchmarks require a CUDA GPU.

### Triton vs PyTorch reference (paper headline)

```bash
python benchmarks/bench_triton_vs_ref.py
```

Sweeps the shape grid from Section 3.2 of the paper and writes
`bench_triton_vs_ref_<host>_<gpu>.csv`. Forward speedup, backward
speedup, and per-shape times are recorded. The `figures/`
throughput plots consume this CSV.

### Standalone Stieltjes kernel (NR vs binary search)

```bash
python benchmarks/bench_stieltjes.py
```

Benchmarks four implementations of the row-wise Stieltjes
normalisation (Triton NR, PyTorch NR, Triton binary search, PyTorch
binary search) plus a softmax baseline across context lengths from 256
to 131k. Prints a table and saves a PNG.

---

## Figures

Every figure in the paper is reproduced by a script in `figures/`:

| Script | Paper figure |
| --- | --- |
| `fig_throughput_a100_vs_h100.py` | Per-shape forward speedup, A100 vs H100 (headline). |
| `fig_throughput_and_memory.py` | Throughput + memory grid on a single hardware. |
| `fig_training_curves.py` | Selected `metrics.csv` curves for the main needle/binary-search runs. |
| `fig_all_training_curves.py` | One plot per task — debugging view of every run under `results/`. |
| `fig_q_curve_bsearch.py` | Accuracy vs Stieltjes order `q` on `binary_search`. |
| `fig_entropy_curriculum_boundary.py` | Attention-entropy phase transition under the q-curriculum. |
| `fig_velickovic_attention_maps.py` | Velickovic-style attention heatmaps across context lengths. |
| `fig_velickovic_entropy_vs_seq.py` | Entropy vs sequence length per attention type. |

Install the figure extras:

```bash
pip install -e ".[figures]"
```

Each script reads from `results/` (relative to the repo root) and writes
PDFs to `figures/out/`. Run them after producing the corresponding
`metrics.csv` / benchmark CSV / analysis output via the pipeline above:

```bash
python figures/fig_throughput_a100_vs_h100.py
python figures/fig_q_curve_bsearch.py
# ...
```

---

## Test harness

The test harness mirrors the pre-commit suite the paper's authors ran
before every code change. It is split by responsibility across four
files in `tests/` so failures localise quickly:

| File | Coverage | GPU? |
| --- | --- | --- |
| `tests/test_imports.py` | Every public module imports without crashing. | CPU |
| `tests/test_data.py` | All five CLRS task generators produce algorithmically correct outputs; the accuracy-metric pipeline is internally consistent; `model.ignore_index` matches `data.PAD`. | CPU |
| `tests/test_model.py` | Forward + backward of the GPT backbone with both attention types; NoPE accepts long sequences; param count of the default config. | CPU |
| `tests/test_kernel.py` | Triton forward parity vs the fp32 reference and Triton backward parity vs `torch.autograd.grad` through the reference. Auto-skips on CPU runners. | GPU |

Together: 20 CPU tests + 21 GPU tests.

### Run everything

```bash
pytest tests/
```

Verbose output (one line per parametrised configuration):

```bash
pytest tests/ -v
```

Compact failure tracebacks:

```bash
pytest tests/ -v --tb=short
```

### List the test grid without running

```bash
pytest tests/ --collect-only -q
```

You will see entries such as

```
tests/::test_forward_matches_reference[1-1-64-64-False-1.0]
tests/::test_forward_matches_reference[1-1-64-64-True-1.0]
...
tests/::test_backward_matches_reference[1-1-1024-64-False-1.0]
```

The bracketed suffix is `[B-H-N-D-causal-q]`.

### Run only the forward tests

```bash
pytest tests/ -v -k forward
```

### Run only the backward tests

```bash
pytest tests/ -v -k backward
```

### Run a single configuration

```bash
# any matching id substring works
pytest tests/ -v -k "backward and 1024"
pytest tests/ -v -k "1-2-512-64-True-1.0"
```

### Stop on the first failure

```bash
pytest tests/ -x
```

### Stand-alone (no pytest required)

The kernel test file also runs without pytest:

```bash
python tests/test_kernel.py
```

This walks the same forward and backward grid and prints `PASS`/`FAIL`
per configuration. Exit code is `0` on success, `1` if any case failed.
Useful when running on a freshly-spawned compute node where pytest
isn't installed.

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

## Continuous integration

`.github/workflows/tests.yml` runs the CPU subset (imports, data, model)
on every push and pull request against `main`, across Python 3.10 / 3.11 /
3.12 on `ubuntu-latest`. The CUDA-only `test_kernel.py` cases auto-skip
on the GitHub-hosted runners via the module-level `pytestmark`.

The workflow is one job; it installs the CPU PyTorch wheel (no CUDA bits)
and the package with `--no-deps` so dependency resolution is deterministic
and fast.

To run the GPU portion in CI you need a self-hosted runner with CUDA. The
existing CPU job is the canonical CI signal; the GPU portion is exercised
locally on the paper's H100/A100 hosts before each release.

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
