import json
from pathlib import Path

import pytest
import torch
import triton
from PIL import Image, ImageDraw, ImageFont
from triton.runtime.errors import OutOfResources

from flash_atten import FlashAttentionProfiler, _profile_event_count, flash_attention_2


DEVICE = "cuda"
DTYPE = torch.float16
PROJECT_LOG_DIR = Path(__file__).resolve().parent / "test_log" / "profile"
TIMELINE_SUFFIX = "_timeline.png"
QWEN25_7B_CASE = (1, 28, 120, 128)
PROFILE_RUNS_PER_CASE = 100


def _block_n_sweep_values(seq_len: int) -> list[int]:
    start = seq_len / 8
    values = []
    for idx in range(8):
        raw = start + idx * (seq_len - start) / 7
        values.append(max(1, min(seq_len, int(round(raw)))))
    return list(dict.fromkeys(values))


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton flash_attention_2 profiling tests")


def _benchmark_flash_attention_ns(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_n: int,
) -> dict[str, int | list[int]]:
    def _run_once():
        flash_attention_2(
            q,
            k,
            v,
            causal=False,
            return_lse=True,
            enable_profiling=False,
            block_n=block_n,
        )

    bench_times_ms = triton.testing.do_bench(_run_once, return_mode="all")
    bench_times_ns = [int(round(time_ms * 1_000_000.0)) for time_ms in bench_times_ms]
    return {
        "times_ns": bench_times_ns,
        "min_ns": min(bench_times_ns),
        "max_ns": max(bench_times_ns),
        "mean_ns": sum(bench_times_ns) / len(bench_times_ns),
    }


@pytest.mark.parametrize("block_n", _block_n_sweep_values(QWEN25_7B_CASE[2]))
def test_flash_attention_host_runtime_only_qwen25_7b_case(block_n: int):
    _require_cuda()
    run_count = PROFILE_RUNS_PER_CASE
    warmup_count = 20

    B, H, S, D = QWEN25_7B_CASE
    out = None
    lse = None
    mean_runtime_ns = 0
    run_times_ns: list[int] = []

    PROJECT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        di = triton.runtime.driver.active.get_device_interface()
        cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
        for _ in range(warmup_count):
            out, lse = flash_attention_2(
                q,
                k,
                v,
                causal=False,
                return_lse=True,
                block_n=block_n,
            )
        di.synchronize()
        start_events = [di.Event(enable_timing=True) for _ in range(run_count)]
        end_events = [di.Event(enable_timing=True) for _ in range(run_count)]
        for run_idx in range(run_count):
            cache.zero_()
            start_events[run_idx].record()
            out, lse = flash_attention_2(
                q,
                k,
                v,
                causal=False,
                return_lse=True,
                block_n=block_n,
            )
            end_events[run_idx].record()
        di.synchronize()
        run_times_ns = [
            int(round(start_events[run_idx].elapsed_time(end_events[run_idx]) * 1_000_000.0))
            for run_idx in range(run_count)
        ]
        mean_runtime_ns = int(round(sum(run_times_ns) / len(run_times_ns)))
    except OutOfResources as exc:
        pytest.fail(f"BLOCK_N={block_n} is not supported on this GPU: {exc}")

    assert out is not None
    assert lse is not None
    assert out.shape == (B, H, S, D)
    assert lse.shape == (B, H, S)
    assert len(run_times_ns) == run_count
    assert mean_runtime_ns > 0

    q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    bench_stats = _benchmark_flash_attention_ns(q, k, v, block_n=block_n)
    assert len(bench_stats["times_ns"]) > 0
    assert bench_stats["min_ns"] <= bench_stats["mean_ns"] <= bench_stats["max_ns"]
    assert bench_stats["mean_ns"] > 0

    print(
        f"BLOCK_N={block_n} runtime-only mean per run: {int(round(mean_runtime_ns / 1_000.0))} us; "
        f"do_bench mean: {int(round(bench_stats['mean_ns'] / 1_000.0))} us"
    )