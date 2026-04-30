import json
import logging
from pathlib import Path

import pytest
import torch
import triton

from exceptions import OutOfResourcesWithDetail, _resource_failure_text, _resource_usage_summary
from flash_atten import flash_attention_2

from tuner import (
    FlashAttenTuneArguments,
    FlashAttentionTuner,
)


DEVICE = "cuda"
DTYPE = torch.float16
PROJECT_LOG_DIR = Path(__file__).resolve().parent / "test_log" / "profile"
QWEN35_9B_CASE = (1, 16, 783, 256)
PROFILE_RUNS_PER_CASE = 100
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
                v,
            )
            if run_idx == run_count - 1:
                logger.info(
                    _resource_usage_summary(
                        "flash_attention_2",
                        launch_spec.launch_context,
                        launch_spec.tune_args,
                        launch_spec.input_shapes,
                    )
                )
            out, lse = flash_attention_2(launch_spec)
    except OutOfResourcesWithDetail as exc:
        pytest.fail(_resource_failure_text(f"BLOCK_N={block_n} is not supported on this GPU", exc))

    assert out.shape == (B, H, S, D)
    assert lse.shape == (B, H, S)