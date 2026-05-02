import math
import random
import statistics
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import triton
from triton.compiler import make_backend
from triton.runtime import driver

from flash_atten import (
    _flash_attn_fwd_kernel_basic,
    flash_attention,
)
from tuner import FlashAttentionTuner

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None


random.seed(20260328)

DEVICE = "cuda"
DTYPE = torch.float16
COMPARE_RTOL = 1e-2
COMPARE_ATOL = 2e-2
TEST_LOG_DIR = Path(__file__).resolve().parent / "test_log"
TEST_RUN_START_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
TEST_RUN_LOG_PATH = TEST_LOG_DIR / f"{TEST_RUN_START_TIMESTAMP}_test_attention_kernel_pipeline.log"


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton flash_attention tests")


def _require_sdpa_kernel_api():
    if sdpa_kernel is None or SDPBackend is None:
        pytest.skip("This PyTorch build does not expose torch.nn.attention.sdpa_kernel")


def _sample_log_uniform_lengths(n_samples: int, low: int = 1, high: int = 1000):
    log_low = math.log10(low)
    log_high = math.log10(high)
    vals = []
    for _ in range(n_samples):
        u = random.uniform(log_low, log_high)
        vals.append(max(low, min(high, int(round(10**u)))))
    return vals


def _build_length_suite():
    anchors = [1, 3, 9, 23, 43, 98, 120, 430, 783]
    sampled = _sample_log_uniform_lengths(n_samples=12, low=1, high=1000)
    return sorted(set(anchors + sampled))


SEQ_LENGTH_SUITE = _build_length_suite()
SMALL_CONFIGS = [
    (1, 4, 32),
    (2, 8, 64),
]
LARGE_CONFIGS = [
    (1, 28, 128),  # Qwen2.5-7B style shape: hidden_size=3584 with 28 attention heads.
    (1, 32, 128),  # Common 4096-hidden / 32-head shape used by many 7B-8B LLMs, e.g. Llama 2 7B, Mistral 7B, Llama 3 8B, and Qwen3-8B.
    (1, 16, 256),  # Qwen3.5 text full-attention shape, e.g. Qwen3.5-4B and Qwen3.5-9B.
    (1, 40, 128),  # Common 5120-hidden / 40-head shape seen in Falcon-class style models, e.g. Falcon-7B and Falcon-7B-Instruct.
]
PROFILE_CASES = [
    (1, 4, 32, 98, False),
    (1, 8, 64, 120, True),
    (1, 28, 128, 120, False),  # Qwen2.5-7B style shape: hidden_size=3584 with 28 attention heads.
    (1, 32, 128, 430, False),  # Common 4096-hidden / 32-head shape used by many 7B-8B LLMs, e.g. Llama 2 7B, Mistral 7B, Llama 3 8B, and Qwen3-8B.
    (1, 16, 256, 120, False),  # Qwen3.5 text full-attention shape, e.g. Qwen3.5-4B and Qwen3.5-9B.
    (1, 40, 128, 783, True),  # Common 5120-hidden / 40-head shape seen in Falcon-class style models, e.g. Falcon-7B and Falcon-7B-Instruct.
]

SMALL_CASES = [(B, H, D, S, causal) for B, H, D in SMALL_CONFIGS for S in SEQ_LENGTH_SUITE for causal in (False, True)]
LARGE_CASES = [(B, H, D, S, causal) for B, H, D in LARGE_CONFIGS for S in (120, 430, 783) for causal in (False, True)]
ALL_COMPARE_CASES = SMALL_CASES + LARGE_CASES
PIPELINE_CASES = list(dict.fromkeys(ALL_COMPARE_CASES + PROFILE_CASES))

TORCH_SDPA_BACKENDS = {
    "math": SDPBackend.MATH if SDPBackend is not None else None,
    "flash": SDPBackend.FLASH_ATTENTION if SDPBackend is not None else None,
}

def _case_id(case):
    B, H, D, S, causal = case
    return f"B{B}_H{H}_D{D}_S{S}_{'causal' if causal else 'noncausal'}"


def _format_case_summary(case, case_status: str, stats, ratios, statuses):
    B, H, D, S, causal = case
    summary = [
        f"{name} mean={kernel_stats['mean_ms']:.3f}ms p50={kernel_stats['p50_ms']:.3f}ms"
        if kernel_stats is not None
        else f"{name} mean=n/a p50=n/a"
        for name, kernel_stats in stats.items()
    ]
    return (
        f"[case] status={case_status} B={B} H={H} S={S} D={D} causal={causal} | "
        + " | ".join(summary + ratios + statuses)
    )


def _get_triton_flash_kernel_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    return_lse: bool,
):
    launch_spec = FlashAttentionTuner(
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        q.element_size(),
        causal=causal,
        return_lse=return_lse,
    ).build_flash_attention_launch_spec(q, k, v)
    kwargs = dict(launch_spec.kernel_kwargs)
    kwargs["debug"] = False
    o = launch_spec.kernel_args[3]
    lse = launch_spec.kernel_args[4] if isinstance(launch_spec.kernel_args[4], torch.Tensor) else None
    return launch_spec.kernel_args, kwargs, o, lse

def _get_exact_triton_cache_key(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    return_lse: bool,
):
    target = driver.active.get_current_target()
    backend = make_backend(target)
    if _flash_attn_fwd_kernel_basic.binder is None:
        _flash_attn_fwd_kernel_basic.create_binder(backend)

    args, kwargs, o, lse = _get_triton_flash_kernel_call(q, k, v, causal=causal, return_lse=return_lse)
    try:
        _, sig_and_spec, constexpr_vals, _, excess_kwargs = _flash_attn_fwd_kernel_basic.binder(*args, **kwargs)
        return "".join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
    finally:
        del o
        if lse is not None:
            del lse


def _compile_triton_flash_with_real_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    return_lse: bool,
):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    device = driver.active.get_current_device()
    cache = _flash_attn_fwd_kernel_basic.cache[device]
    cache_key = _get_exact_triton_cache_key(q, k, v, causal=causal, return_lse=return_lse)
    if cache_key not in cache:
        try:
            launch_spec = FlashAttentionTuner(
                tuple(q.shape),
                tuple(k.shape),
                tuple(v.shape),
                q.element_size(),
                causal=causal,
                return_lse=return_lse,
            ).build_flash_attention_launch_spec(q, k, v)
            out = flash_attention(launch_spec)
            torch.cuda.synchronize()
            del out
        except Exception as exc:
            if cache_key not in cache:
                raise RuntimeError(
                    "kernel execution failed before Triton cache entry became available"
                ) from exc
    if cache_key not in cache:
        raise KeyError("unable to find the Triton compiled kernel in the exact runtime cache")
    return cache[cache_key]


def _get_triton_flash_compile_report(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, return_lse: bool = False):
    compiled = _compile_triton_flash_with_real_inputs(q, k, v, causal=causal, return_lse=return_lse)
    ptx = compiled.asm["ptx"]
    tensor_core_tokens = ("mma.sync", "wmma.mma", "wgmma.mma", "tcgen05.mma")
    found_tensor_core_tokens = [token for token in tensor_core_tokens if token in ptx]
    auxiliary_tokens = [token for token in ("ldmatrix",) if token in ptx]
    target_line = next((line.strip() for line in ptx.splitlines() if line.strip().startswith(".target ")), "target=n/a")

    return {
        "target": target_line,
        "tensor_core_used": bool(found_tensor_core_tokens),
        "tensor_core_tokens": found_tensor_core_tokens,
        "auxiliary_tokens": auxiliary_tokens,
    }


def _format_triton_flash_compile_status(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
    try:
        report = _get_triton_flash_compile_report(q, k, v, causal=causal, return_lse=False)
    except Exception as exc:
        return f"triton_flash_ptx_tensor_core=unknown (PTX inspection failed: {exc})"

    evidence = report["tensor_core_tokens"] or report["auxiliary_tokens"] or ["no tensor-core PTX opcodes found"]
    tensor_core_state = "yes" if report["tensor_core_used"] else "no"
    return (
        f"triton_flash_ptx_tensor_core={tensor_core_state} "
        f"(target={report['target']}; evidence={','.join(evidence)})"
    )


def _write_case_summary_log(summary_text: str):
    TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with TEST_RUN_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(summary_text + "\n")
    return TEST_RUN_LOG_PATH


def _torch_sdpa_context(backend_name: str):
    if backend_name == "default":
        return nullcontext()

    _require_sdpa_kernel_api()
    return sdpa_kernel(backends=[TORCH_SDPA_BACKENDS[backend_name]])


def _run_torch_sdpa(q, k, v, causal: bool, backend_name: str):
    with _torch_sdpa_context(backend_name):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _maybe_run_torch_sdpa(q, k, v, causal: bool, backend_name: str):
    try:
        out = _run_torch_sdpa(q, k, v, causal=causal, backend_name=backend_name)
        torch.cuda.synchronize()
        return out, None
    except Exception as exc:
        return None, f"torch SDPA backend '{backend_name}' is unavailable for this shape/device: {exc}"


def _maybe_run_triton_flash(q, k, v, causal: bool):
    try:
        launch_spec = FlashAttentionTuner(
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            q.element_size(),
            causal=causal,
            return_lse=False,
        ).build_flash_attention_launch_spec(q, k, v)
        out = flash_attention(launch_spec)
        torch.cuda.synchronize()
        return out, None
    except Exception as exc:
        return None, f"Triton flash_attention is unavailable for this shape/device: {exc}"


def _make_inputs(B: int, H: int, D: int, S: int):
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    return q, k, v


def _make_cpu_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # Use float32 on CPU for a stable baseline reference.
    return q.detach().cpu().float(), k.detach().cpu().float(), v.detach().cpu().float()


def _clear_cuda_memory():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.fixture
def attention_case(request):
    _require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    return request.param


@pytest.fixture
def attention_inputs(attention_case):
    B, H, D, S, causal = attention_case
    q, k, v = _make_inputs(B, H, D, S)
    return {
        "case": attention_case,
        "q": q,
        "k": k,
        "v": v,
        "causal": causal,
    }


def _assert_attention_close(actual: torch.Tensor, expected: torch.Tensor):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=COMPARE_RTOL,
        atol=COMPARE_ATOL,
    )


def _run_cpu_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _time_cuda_ms(fn, warmup: int = 10, iters: int = 30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    output = None
    for _ in range(iters):
        start.record()
        output = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }, output


def _time_cpu_ms(fn, warmup: int = 3, iters: int = 10):
    for _ in range(warmup):
        fn()

    times = []
    output = None
    for _ in range(iters):
        start = time.perf_counter()
        output = fn()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }, output


def _probe_kernel_availability(name: str, run_kernel):
    try:
        out = run_kernel()
        del out
        _clear_cuda_memory()
        return None
    except Exception as exc:
        _clear_cuda_memory()
        return f"{name} unavailable: {exc}"


def _format_speedup_ratios(stats):
    ratios = []
    cpu_stats = stats.get("cpu_sdpa")
    cpu_mean = cpu_stats["mean_ms"] if cpu_stats is not None else 0.0
    for name, kernel_stats in stats.items():
        if name == "cpu_sdpa" or kernel_stats is None:
            continue
        kernel_mean = kernel_stats["mean_ms"]
        ratio = cpu_mean / kernel_mean if kernel_mean > 0 else float("inf")
        ratios.append(f"{name}/cpu_sdpa={ratio:.3f}x")
    return ratios


def _run_kernel_pipeline(attention_inputs):
    q = attention_inputs["q"]
    k = attention_inputs["k"]
    v = attention_inputs["v"]
    case = attention_inputs["case"]
    B, H, D, S, causal = case
    q_cpu, k_cpu, v_cpu = _make_cpu_inputs(q, k, v)
    triton_compile_status = _format_triton_flash_compile_status(q, k, v, causal=causal)

    kernels = [
        ("cpu_sdpa", lambda: _run_cpu_sdpa(q_cpu, k_cpu, v_cpu, causal=causal), "cpu"),
        ("gpu_math_sdpa", lambda: _run_torch_sdpa(q, k, v, causal=causal, backend_name="math"), "cuda"),
        ("gpu_flash_sdpa", lambda: _run_torch_sdpa(q, k, v, causal=causal, backend_name="flash"), "cuda"),
        (
            "triton_flash",
            lambda: flash_attention(
                FlashAttentionTuner(
                    tuple(q.shape),
                    tuple(k.shape),
                    tuple(v.shape),
                    q.element_size(),
                    causal=causal,
                    return_lse=False,
                ).build_flash_attention_launch_spec(q, k, v)
            ),
            "cuda",
        ),
    ]

    availability = {}
    runnable_kernels = []
    for name, run_kernel, device_kind in kernels:
        error = _probe_kernel_availability(name, run_kernel)
        availability[name] = error
        if error is None:
            runnable_kernels.append((name, run_kernel, device_kind))

    baseline_output = None
    stats = {}
    statuses = []

    for name, error in availability.items():
        if error is not None:
            stats[name] = None
            statuses.append(f"{name}=unavailable ({error})")

    statuses.append(triton_compile_status)

    if availability["cpu_sdpa"] is not None:
        summary_text = _format_case_summary(case, "failed", stats, [], statuses)
        print("\n" + summary_text)
        _write_case_summary_log(summary_text)
        raise AssertionError(availability["cpu_sdpa"])
    if availability["triton_flash"] is not None:
        summary_text = _format_case_summary(case, "failed", stats, [], statuses)
        print("\n" + summary_text)
        _write_case_summary_log(summary_text)
        raise AssertionError(availability["triton_flash"])
    if availability["gpu_math_sdpa"] is not None:
        summary_text = _format_case_summary(case, "skipped", stats, [], statuses)
        print("\n" + summary_text)
        _write_case_summary_log(summary_text)
        pytest.skip(availability["gpu_math_sdpa"])

    for name, run_kernel, device_kind in runnable_kernels:
        output = None
        try:
            if device_kind == "cpu":
                stats[name], output = _time_cpu_ms(run_kernel)
            else:
                stats[name], output = _time_cuda_ms(run_kernel)

            if name == "cpu_sdpa":
                baseline_output = output.detach().cpu()
            else:
                _assert_attention_close(output.detach().cpu().float(), baseline_output.float())
                statuses.append(f"{name}=passed")
        except AssertionError:
            stats.setdefault(name, None)
            statuses.append(f"{name}=failed")
            ratios = _format_speedup_ratios(stats)
            summary_text = _format_case_summary(case, "failed", stats, ratios, statuses)
            print("\n" + summary_text)
            _write_case_summary_log(summary_text)
            raise
        finally:
            if output is not None:
                del output
            if device_kind == "cuda":
                _clear_cuda_memory()

    ratios = _format_speedup_ratios(stats)
    summary_text = _format_case_summary(case, "passed", stats, ratios, statuses)
    print("\n" + summary_text)
    _write_case_summary_log(summary_text)

    for kernel_stats in stats.values():
        if kernel_stats is not None:
            assert kernel_stats["mean_ms"] > 0


@pytest.mark.parametrize("attention_case", PIPELINE_CASES, ids=_case_id, indirect=True)
def test_attention_kernel_pipeline(attention_inputs):
    _run_kernel_pipeline(attention_inputs)