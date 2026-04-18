import math
import random
import statistics

import pytest
import torch
import torch.nn.functional as F

from flash_atten import flash_attention_2


# Fixed seed to keep randomized sequence lengths reproducible.
random.seed(20260328)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton flash_attention_2 tests")


def _sample_log_uniform_lengths(n_samples: int, low: int = 1, high: int = 1000):
    log_low = math.log10(low)
    log_high = math.log10(high)
    vals = []
    for _ in range(n_samples):
        u = random.uniform(log_low, log_high)
        vals.append(max(low, min(high, int(round(10 ** u)))))
    return vals


def _build_length_suite():
    # Explicit anchor points requested by the user plus random log-uniform points.
    anchors = [1, 3, 9, 23, 43, 98, 120, 430, 783]
    sampled = _sample_log_uniform_lengths(n_samples=12, low=1, high=1000)
    lengths = sorted(set(anchors + sampled))
    return lengths


SEQ_LENGTH_SUITE = _build_length_suite()

# Small shapes for quick validation.
SMALL_CONFIGS = [
    (1, 4, 32),
    (2, 8, 64),
]

# LLM-like shapes commonly seen in recent 8B-20B class models.
LARGE_CONFIGS = [
    (1, 32, 128),
    (1, 40, 128),
]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,D", SMALL_CONFIGS)
@pytest.mark.parametrize("S", SEQ_LENGTH_SUITE)
def test_flash_attention_2_matches_torch_small(B, H, D, S, causal):
    _require_cuda()

    device = "cuda"
    dtype = torch.float16

    # Disable TF32 to make the comparison stricter and more stable.
    torch.backends.cuda.matmul.allow_tf32 = False

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    out_flash = flash_attention_2(q, k, v, causal=causal)
    out_torch = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # In fp16, exact bitwise identity is usually too strict; use a tight tolerance.
    torch.testing.assert_close(out_flash, out_torch, rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,D", LARGE_CONFIGS)
@pytest.mark.parametrize("S", [120, 430, 783])
def test_flash_attention_2_matches_torch_large(B, H, D, S, causal):
    _require_cuda()

    device = "cuda"
    dtype = torch.float16

    torch.backends.cuda.matmul.allow_tf32 = False

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    out_flash = flash_attention_2(q, k, v, causal=causal)
    out_torch = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    torch.testing.assert_close(out_flash, out_torch, rtol=1e-2, atol=2e-2)


def _time_cuda_ms(fn, warmup: int = 10, iters: int = 30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


@pytest.mark.parametrize(
    "B,H,D,S,causal",
    [
        (1, 4, 32, 98, False),
        (1, 8, 64, 120, True),
        (1, 32, 128, 430, False),
        (1, 40, 128, 783, True),
    ],
)
def test_profile_flash_attention_2_vs_torch(B, H, D, S, causal):
    _require_cuda()

    device = "cuda"
    dtype = torch.float16

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # Compile/warm run before profiling so timings reflect steady-state execution.
    flash_attention_2(q, k, v, causal=causal)
    F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.cuda.synchronize()

    flash_stats = _time_cuda_ms(lambda: flash_attention_2(q, k, v, causal=causal))
    torch_stats = _time_cuda_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))

    ratio = flash_stats["mean_ms"] / torch_stats["mean_ms"] if torch_stats["mean_ms"] > 0 else float("inf")

    print(
        "\n"
        f"[profile] B={B} H={H} S={S} D={D} causal={causal} | "
        f"flash mean={flash_stats['mean_ms']:.3f}ms p50={flash_stats['p50_ms']:.3f}ms | "
        f"torch mean={torch_stats['mean_ms']:.3f}ms p50={torch_stats['p50_ms']:.3f}ms | "
        f"flash/torch={ratio:.3f}x"
    )

    # Keep profiling test informational; correctness is covered in dedicated tests.
    assert flash_stats["mean_ms"] > 0
    assert torch_stats["mean_ms"] > 0