"""
Triton kernel correctness tests.

Forward: Triton kernel vs PyTorch reference (fp16 vs fp32).
Backward: torch.autograd.grad through `stieltjes_attention` vs the same
through the reference impl.

Requires a CUDA device with a working Triton install. All tests in this
file auto-skip on CPU-only runners via the module-level `pytestmark`
below.

Run:
    pytest tests/test_kernel.py -v
or, stand-alone:
    python tests/test_kernel.py
"""

from __future__ import annotations

import pytest
import torch

from stieltjes_attention import stieltjes_attention, stieltjes_attention_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Stieltjes Triton kernel requires CUDA",
)


FWD_CONFIGS = [
    # (B, H, N, D, causal, q)
    (1, 1,  64,  64, False, 1.0),
    (1, 1,  64,  64, True,  1.0),
    (2, 4, 128,  64, False, 1.0),
    (2, 4, 128,  64, True,  1.0),
    (1, 2, 256,  64, False, 2.0),
    (1, 2, 256,  64, True,  2.0),
    (1, 1, 128, 128, False, 1.0),
    (1, 1, 128, 128, True,  1.0),
    (1, 2, 512,  64, False, 1.0),
    (1, 2, 512,  64, True,  1.0),
    (1, 1, 1024, 64, False, 1.0),
    (1, 1, 1024, 64, True,  2.0),
]

BWD_CONFIGS = [
    (1, 1,  64,  64, False, 1.0),
    (1, 1,  64,  64, True,  1.0),
    (2, 2, 128,  64, False, 1.0),
    (2, 2, 128,  64, True,  1.0),
    (1, 1, 128, 128, False, 1.0),
    (1, 2, 128,  64, False, 2.0),
    (1, 2, 512,  64, False, 1.0),
    (1, 2, 512,  64, True,  1.0),
    (1, 1, 1024, 64, False, 1.0),
]


@pytest.mark.parametrize("B,H,N,D,causal,sq", FWD_CONFIGS)
def test_forward_matches_reference(B, H, N, D, causal, sq):
    """fp16 Triton output should match fp32 reference within fp16 tolerance."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    sm_scale = 1.0 / (D ** 0.5)

    ref = stieltjes_attention_ref(
        q.float(), k.float(), v.float(),
        sm_scale, causal=causal, stieltjes_q=sq, num_iter=10,
    ).half()

    tri = stieltjes_attention(
        q, k, v, causal=causal, sm_scale=sm_scale,
        stieltjes_q=sq, num_iter=5,
    )

    max_err = (tri - ref).abs().max().item()
    assert max_err < 0.05, f"max_err={max_err} too large for B={B} H={H} N={N} D={D} causal={causal} q={sq}"


@pytest.mark.parametrize("B,H,N,D,causal,sq", BWD_CONFIGS)
def test_backward_matches_reference(B, H, N, D, causal, sq):
    """Gradients from `stieltjes_attention` should match the reference's autograd."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    sm_scale = 1.0 / (D ** 0.5)

    q_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32, requires_grad=True)
    k_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32, requires_grad=True)
    v_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32, requires_grad=True)
    do = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    o_ref = stieltjes_attention_ref(
        q_ref, k_ref, v_ref, sm_scale,
        causal=causal, stieltjes_q=sq, num_iter=10,
    )
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(o_ref, (q_ref, k_ref, v_ref), do)

    q_tri = q_ref.detach().clone().to(torch.float16).requires_grad_(True)
    k_tri = k_ref.detach().clone().to(torch.float16).requires_grad_(True)
    v_tri = v_ref.detach().clone().to(torch.float16).requires_grad_(True)

    o_tri = stieltjes_attention(
        q_tri, k_tri, v_tri,
        causal=causal, sm_scale=sm_scale,
        stieltjes_q=sq, num_iter=5,
    )
    o_tri.backward(do.to(torch.float16))

    dq_err = (q_tri.grad.float() - dq_ref).abs().max().item()
    dk_err = (k_tri.grad.float() - dk_ref).abs().max().item()
    dv_err = (v_tri.grad.float() - dv_ref).abs().max().item()

    assert max(dq_err, dk_err, dv_err) < 0.15, (
        f"grad err too large: dQ={dq_err:.4f} dK={dk_err:.4f} dV={dv_err:.4f} "
        f"for B={B} H={H} N={N} D={D} causal={causal} q={sq}"
    )


def _run_cli():
    """Stand-alone runner so `python tests/test_stieltjes.py` works without pytest."""
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping.")
        return 0

    n_fail = 0

    print("Forward correctness")
    print("-" * 70)
    for cfg in FWD_CONFIGS:
        try:
            test_forward_matches_reference(*cfg)
            print(f"  PASS  {cfg}")
        except AssertionError as e:
            n_fail += 1
            print(f"  FAIL  {cfg}: {e}")

    print("\nBackward correctness")
    print("-" * 70)
    for cfg in BWD_CONFIGS:
        try:
            test_backward_matches_reference(*cfg)
            print(f"  PASS  {cfg}")
        except AssertionError as e:
            n_fail += 1
            print(f"  FAIL  {cfg}: {e}")

    print("-" * 70)
    if n_fail == 0:
        print("All tests passed.")
        return 0
    print(f"{n_fail} test(s) FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(_run_cli())
