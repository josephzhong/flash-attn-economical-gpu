import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
import triton

from exceptions import OutOfResourcesWithDetail, _resource_failure_text, _resource_usage_summary
from flash_atten import flash_attention

from tuner import (
    FlashAttenTuneArguments,
    FlashAttentionTuner,
    FlashAttentionOptimizeSharedMemTuner,
)


DEVICE = "cuda"
DTYPE = torch.float16
PROJECT_LOG_DIR = Path(__file__).resolve().parent / "test_log" / "profile"
QWEN35_9B_CASE = (1, 16, 783, 256)
PROFILE_RUNS_PER_CASE = 100
FLASH_ATTENTION_COMPARISON_CASE = (32, 28, 240, 128)
FLASH_ATTENTION_COMPARISON_RUNS = 300
FLASH_ATTENTION_COMPARISON_WARMUP_RUNS = 100
FLASH_ATTENTION_COMPARISON_NUM_STAGES = 1
FLASH_ATTENTION_COMPARISON_WARPS = [1, 2, 4, 8, 16, 32]
logger = logging.getLogger(__name__)


def _block_n_sweep_values(seq_len: int) -> list[int]:
    start = 32
    values = [32]
    for idx in range(8):
        raw = start + idx * (seq_len - start) / 7
        if raw > 128:
            break
        values.append(max(1, min(seq_len, int(round(raw)))))
    return list(dict.fromkeys(values))

def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton flash_attention_2 profiling tests")

@pytest.mark.parametrize("block_n", _block_n_sweep_values(QWEN35_9B_CASE[2]))
def test_flash_attention_profiler_qwen35_9b_case(
    block_n: int,
    caplog: pytest.LogCaptureFixture,
):
    _require_cuda()
    caplog.set_level("INFO", logger="flash_atten")
    run_count = PROFILE_RUNS_PER_CASE
    
    B, H, S, D = QWEN35_9B_CASE

    PROJECT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    tuner = FlashAttentionTuner(
        (B, H, S, D),
        (B, H, S, D),
        (B, H, S, D),
        DTYPE.itemsize,
        causal=False,
        return_lse=True,
        tune_args=FlashAttenTuneArguments(BLOCK_N=block_n),
    )
    try:
        for run_idx in range(run_count):
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            launch_spec = tuner.build_flash_attention_launch_spec(
                q,
                k,
                v
            )
            if run_idx == run_count - 1:
                logger.info(
                    _resource_usage_summary(
                        "_flash_attn_fwd_kernel_basic",
                        launch_spec.launch_context,
                        launch_spec.tune_args,
                        launch_spec.input_shapes,
                    )
                )
            out, lse = flash_attention(launch_spec)
    except OutOfResourcesWithDetail as exc:
        pytest.fail(_resource_failure_text(f"BLOCK_N={block_n} is not supported on this GPU", exc))

    assert out.shape == (B, H, S, D)
    assert lse.shape == (B, H, S)

def _run_flash_attention_profile_sweep(
    *,
    function_name: str,
    output_dir: Path,
):
    B, H, S, D = FLASH_ATTENTION_COMPARISON_CASE
    tuner_cls_by_function = {
        "_flash_attn_fwd_kernel_basic": FlashAttentionTuner,
        "_flash_attn_fwd_kernel_optimize_shared_mem": FlashAttentionOptimizeSharedMemTuner,
    }
    tuner_cls = tuner_cls_by_function[function_name]
    kernel_fn = flash_attention
    unsupported_warps = []
    total_warps = len(FLASH_ATTENTION_COMPARISON_WARPS)

    output_dir.mkdir(parents=True, exist_ok=True)
    for warp_idx, num_warps in enumerate(FLASH_ATTENTION_COMPARISON_WARPS, start=1):
        print(f"[{function_name}] testing num_warps={num_warps} ({warp_idx}/{total_warps})")
        resource_usage_summary = None
        tuner = tuner_cls(
            (B, H, S, D),
            (B, H, S, D),
            (B, H, S, D),
            DTYPE.itemsize,
            causal=False,
            return_lse=True,
            tune_args=FlashAttenTuneArguments(
                NUM_WARPS=num_warps,
                NUM_STAGES=FLASH_ATTENTION_COMPARISON_NUM_STAGES,
            ),
        )
        try:
            print(
                f"[{function_name}] warming up num_warps={num_warps} "
                f"for {FLASH_ATTENTION_COMPARISON_WARMUP_RUNS} runs"
            )
            warmup_q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            warmup_k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            warmup_v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            for _ in range(FLASH_ATTENTION_COMPARISON_WARMUP_RUNS):
                warmup_launch_spec = tuner.build_flash_attention_launch_spec(
                    warmup_q,
                    warmup_k,
                    warmup_v,
                )
                warmup_out, warmup_lse = kernel_fn(warmup_launch_spec)
            torch.cuda.synchronize()

            for run_idx in range(FLASH_ATTENTION_COMPARISON_RUNS):
                q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
                k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
                v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
                launch_spec = tuner.build_flash_attention_launch_spec(
                    q,
                    k,
                    v,
                )
                out, lse = kernel_fn(launch_spec)
                if run_idx == 0:
                    resource_usage_summary = _resource_usage_summary(
                        function_name,
                        launch_spec.launch_context,
                        launch_spec.tune_args,
                        launch_spec.input_shapes,
                    )
                    ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                    torch.testing.assert_close(out.detach().cpu().float(), ref_out.detach().cpu().float(), rtol=1e-2, atol=2e-2)
            torch.cuda.synchronize()
            assert out.shape == (B, H, S, D)
            assert lse.shape == (B, H, S)
            assert resource_usage_summary is not None
            
        except OutOfResourcesWithDetail:
            print(f"[{function_name}] skipped num_warps={num_warps} due to resource limits")
            unsupported_warps.append(num_warps)

    return {
        "function_name": function_name,
        "unsupported_warps": unsupported_warps,
    }


def test_flash_attention_basic_vs_optimized_shared_mem():

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_output_dir = PROJECT_LOG_DIR / f"fa_basic_vs_optimized_shared_mem_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    flash_attention_basic_results = _run_flash_attention_profile_sweep(
        function_name="_flash_attn_fwd_kernel_basic",
        output_dir=run_output_dir / "kernel_basic",
    )
    flash_attention_double_pipelines_results = _run_flash_attention_profile_sweep(
        function_name="_flash_attn_fwd_kernel_optimize_shared_mem",
        output_dir=run_output_dir / "double_pipelines",
    )
    
    assert flash_attention_basic_results["results"]
    assert flash_attention_double_pipelines_results["results"]