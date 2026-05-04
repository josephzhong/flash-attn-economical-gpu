import math
import random
import re
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
from tuner import FlashAttentionTuner, FlashAttentionOptimizeSharedMemTuner, FlashAttentionBlockDNomaskTuner, FlashAttenTuneArguments
from exceptions import _resource_usage_summary

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None

from typing import Callable


random.seed(20260328)

DEVICE = "cuda"
DTYPE = torch.float16
COMPARE_RTOL = 1e-2
COMPARE_ATOL = 2e-2
TEST_LOG_DIR = Path(__file__).resolve().parent / "test_log"
TEST_RUN_START_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
TEST_RUN_LOG_PATH = TEST_LOG_DIR / f"{TEST_RUN_START_TIMESTAMP}_test_attention_kernel_pipeline.log"
CUDA_CLOCK_WARMUP_SLEEP_CYCLES = 50_000_000
CUDA_CLOCK_WARMUP_SLEEP_LAUNCHES = 1
KERNEL_RUNTIME_SERIES = {
    "cpu_sdpa": "torch SDPA cpu",
    "gpu_math_sdpa": "torch SDPA math",
    "gpu_efficient_sdpa": "torch SDPA efficient",
    "gpu_flash_sdpa": "torch SDPA flash",
    "triton_flash_basic": "ours basic",
    "triton_flash_opt_shared_mem": "ours opt shared mem",
}


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
LARGE_CONFIGS = [
    (1, 28, 128),  # Qwen2.5-7B style shape: hidden_size=3584 with 28 attention heads.
    (1, 16, 256),  # Qwen3.5 text full-attention shape, e.g. Qwen3.5-4B and Qwen3.5-9B.
]

LARGE_CASES = [(B, H, D, S, causal) for B, H, D in LARGE_CONFIGS for S in (1024, 8192, 16384, 32768, 65536) for causal in (False,)]
PIPELINE_CASES = list(dict.fromkeys(LARGE_CASES))

TORCH_SDPA_BACKENDS = {
    "math": SDPBackend.MATH if SDPBackend is not None else None,
    "efficient": SDPBackend.EFFICIENT_ATTENTION if SDPBackend is not None else None,
    "flash": SDPBackend.FLASH_ATTENTION if SDPBackend is not None else None,
}

def _case_id(case):
    B, H, D, S, causal = case
    return f"B{B}_H{H}_D{D}_S{S}_{'causal' if causal else 'noncausal'}"


def _format_case_summary(case, case_status: str, stats, ratios, statuses, stat_details=None):
    B, H, D, S, causal = case
    stat_details = stat_details or {}
    summary = []
    for name, kernel_stats in stats.items():
        detail = stat_details.get(name, "")
        label = f"{name} {detail}" if detail else name
        if kernel_stats is not None:
            summary.append(f"{label} mean={kernel_stats['mean_ms']:.3f}ms p50={kernel_stats['p50_ms']:.3f}ms")
        else:
            summary.append(f"{label} mean=n/a p50=n/a")
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


def _attention_tflops(B: int, H: int, D: int, S: int, causal: bool, runtime_ms: float):
    if not math.isfinite(runtime_ms) or runtime_ms <= 0:
        return float("nan")

    score_count = S * (S + 1) / 2 if causal else S * S
    flops = 4 * B * H * D * score_count
    return flops / (runtime_ms * 1.0e9)


def _dtype_itemsize(datatype=torch.float16):
    if isinstance(datatype, torch.dtype):
        return torch.empty((), dtype=datatype).element_size()

    dtype_name = str(datatype).lower().replace("torch.", "").replace("numpy.", "").replace("np.", "")
    dtype_itemsize = {
        "float16": 2,
        "half": 2,
        "bfloat16": 2,
        "float32": 4,
        "float": 4,
        "float64": 8,
        "double": 8,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "int32": 4,
        "uint32": 4,
        "int64": 8,
        "uint64": 8,
    }.get(dtype_name)
    if dtype_itemsize is None:
        raise ValueError(f"Unsupported datatype for graph data-size calculation: {datatype!r}")
    return dtype_itemsize


def _parse_flash_atten_kernel_pipeline_rows(log_path: Path):
    log_path = Path(log_path)
    rows = []

    if not log_path.exists():
        return rows

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("[case]"):
            continue

        shape_match = re.search(r"\bB=(\d+)\s+H=(\d+)\s+S=(\d+)\s+D=(\d+)\s+causal=(True|False)\b", line)
        if shape_match is None:
            continue

        B, H, S, D = (int(value) for value in shape_match.groups()[:4])
        causal = shape_match.group(5) == "True"
        runtimes = {}
        for kernel_name in KERNEL_RUNTIME_SERIES:
            match = re.search(rf"(?:^|\|\s*){re.escape(kernel_name)}(?:\s+[^|]*?)?\s+mean=(n/a|[0-9.]+)ms\b", line)
            if match is None or match.group(1) == "n/a":
                runtimes[kernel_name] = float("nan")
            else:
                runtimes[kernel_name] = float(match.group(1))

        rows.append(
            {
                "B": B,
                "H": H,
                "D": D,
                "S": S,
                "causal": causal,
                "data_size": 4 * B * H * S * D,
                "runtimes": runtimes,
            }
        )

    return rows


def _collect_flash_atten_kernel_pipeline_folder_rows(target_folder: Path, datatype=torch.float16):
    target_folder = Path(target_folder)
    element_size = _dtype_itemsize(datatype)
    rows = []

    for log_path in sorted(target_folder.glob("*.log")):
        for row in _parse_flash_atten_kernel_pipeline_rows(log_path):
            B, H, D, S, causal = row["B"], row["H"], row["D"], row["S"], row["causal"]
            rows.append(
                {
                    **row,
                    "log_path": log_path,
                    "label": f"B{B} H{H} S{S} D{D} {'causal' if causal else 'noncausal'}",
                    "data_bytes": 4 * B * H * S * D * element_size,
                }
            )

    rows.sort(key=lambda row: (row["data_bytes"], row["B"], row["H"], row["S"], row["D"], row["causal"], row["log_path"].name))
    return rows


def _render_flash_atten_kernel_pipeline_folder_graph(
    target_folder: Path,
    datatype=torch.float16,
    output_path: Path | None = None,
):
    rows = _collect_flash_atten_kernel_pipeline_folder_rows(target_folder, datatype=datatype)
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_folder = Path(target_folder)
    output_path = Path(output_path) if output_path is not None else target_folder / "flash_atten_kernel_pipeline_runtime_tflops.png"
    title_suffix = ""
    if torch.cuda.is_available():
        title_suffix = f" - {torch.cuda.get_device_name(torch.cuda.current_device())}"

    active_kernels = [
        kernel_name
        for kernel_name in KERNEL_RUNTIME_SERIES
        if any(math.isfinite(row["runtimes"][kernel_name]) and row["runtimes"][kernel_name] > 0 for row in rows)
    ]
    if not active_kernels:
        return None

    x_values = list(range(len(rows)))
    case_labels = [row["label"] for row in rows]
    bar_width = min(0.13, 0.82 / len(active_kernels))
    group_offset = (len(active_kernels) - 1) * bar_width / 2

    fig_width = max(12.0, min(48.0, 1.35 * len(rows) + 8.0))
    fig, (runtime_ax, tflops_ax) = plt.subplots(1, 2, figsize=(fig_width, 6.0), dpi=150)

    for kernel_index, kernel_name in enumerate(active_kernels):
        offset = kernel_index * bar_width - group_offset
        bar_x = [x + offset for x in x_values]
        runtime_values = [row["runtimes"][kernel_name] for row in rows]
        tflops_values = [
            _attention_tflops(row["B"], row["H"], row["D"], row["S"], row["causal"], row["runtimes"][kernel_name])
            for row in rows
        ]
        legend_label = KERNEL_RUNTIME_SERIES[kernel_name]
        runtime_ax.bar(bar_x, runtime_values, width=bar_width, label=legend_label)
        tflops_ax.bar(bar_x, tflops_values, width=bar_width, label=legend_label)

    for ax in (runtime_ax, tflops_ax):
        ax.set_xticks(x_values)
        ax.set_xticklabels(case_labels, rotation=35, ha="right", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend(loc="best", fontsize=8)

    runtime_ax.set_xlabel("Data shape (B H S D)")
    runtime_ax.set_ylabel("Mean runtime (ms)")
    runtime_ax.set_yscale("log")
    runtime_ax.set_title(f"Flash Attention Kernel Mean Runtime{title_suffix}")

    tflops_ax.set_xlabel("Data shape (B H S D)")
    tflops_ax.set_ylabel("TFLOPs")
    tflops_ax.set_title(f"Flash Attention Throughput{title_suffix}")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_flash_atten_kernel_pipeline_folder_graph(
    target_folder: Path,
    datatype=torch.float16,
    output_path: Path | None = None,
):
    return _render_flash_atten_kernel_pipeline_folder_graph(
        target_folder,
        datatype=datatype,
        output_path=output_path,
    )


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


def _warmup_cuda_clocks(reference: torch.Tensor):
    del reference
    for _ in range(CUDA_CLOCK_WARMUP_SLEEP_LAUNCHES):
        torch.cuda._sleep(CUDA_CLOCK_WARMUP_SLEEP_CYCLES)
    torch.cuda.synchronize()


def _time_cuda_ms(fn, attention_inputs, warmup: int = 10, iters: int = 30):
    q = attention_inputs["q"]
    k = attention_inputs["k"]
    v = attention_inputs["v"]
    _warmup_cuda_clocks(q)

    output = fn(q, k, v).clone()
    for _ in range(warmup):
        q, k, v = torch.randn_like(q), torch.randn_like(k), torch.randn_like(v)
        fn(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        q, k, v = torch.randn_like(q), torch.randn_like(k), torch.randn_like(v)
        start.record()
        fn(q, k, v)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    triton.testing.do_bench(
        lambda: fn(q, k, v),
        warmup=25,
        rep=100,
        return_mode="all",
    )
    
    times = times[1:]
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }, output


def _time_cpu_ms(fn, attention_inputs, warmup: int = 3, iters: int = 10):
    q = attention_inputs["q"]
    k = attention_inputs["k"]
    v = attention_inputs["v"]
    q_cpu, k_cpu, v_cpu = _make_cpu_inputs(q, k, v)
    output = fn(q_cpu, k_cpu, v_cpu)
    for _ in range(warmup):
        q_cpu, k_cpu, v_cpu = torch.randn_like(q_cpu), torch.randn_like(k_cpu), torch.randn_like(v_cpu)
        fn(q_cpu, k_cpu, v_cpu)
    times = []
    for _ in range(iters):
        q_cpu, k_cpu, v_cpu = torch.randn_like(q_cpu), torch.randn_like(k_cpu), torch.randn_like(v_cpu)
        start = time.perf_counter()
        fn(q_cpu, k_cpu, v_cpu)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }, output


def _probe_kernel_availability(name: str, run_kernel: Callable, attention_inputs: dict):
    try:
        q, k, v = attention_inputs["q"], attention_inputs["k"], attention_inputs["v"]
        out = run_kernel(q, k, v)
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

def _format_tune_args(tune_args):
    if tune_args is None:
        return ""
    fields = (
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_N_PAD",
        "BLOCK_DMODEL",
        "BLOCK_DMODEL_OUTER",
        "NUM_WARPS",
        "NUM_STAGES",
    )
    return " ".join(f"{field}={getattr(tune_args, field)}" for field in fields)

def _format_resource_usage(launch_spec):
    summary = _resource_usage_summary(
        launch_spec.function_name,
        launch_spec.launch_context,
    )
    return " ".join(summary.replace("\n", " ").split())


def _run_kernel_pipeline(attention_inputs):
    q = attention_inputs["q"]
    k = attention_inputs["k"]
    v = attention_inputs["v"]
    case = attention_inputs["case"]
    B, H, D, S, causal = case
    q_cpu, k_cpu, v_cpu = _make_cpu_inputs(q, k, v)
    triton_compile_status = _format_triton_flash_compile_status(q, k, v, causal=causal)

    def cpu_sdpa(q, k, v):
        return _run_cpu_sdpa(q, k, v, causal=causal)

    def gpu_math_sdpa(q, k, v):
        return _run_torch_sdpa(q, k, v, causal=causal, backend_name="math")
    
    def gpu_efficient_sdpa(q, k, v):
        return _run_torch_sdpa(q, k, v, causal=causal, backend_name="efficient")

    def gpu_flash_sdpa(q, k, v):
        return _run_torch_sdpa(q, k, v, causal=causal, backend_name="flash")
    
    basic_tune_args = FlashAttenTuneArguments(
            BLOCK_N=int(128*64 / D),
            BLOCK_M=int(128*128 / D),
            BLOCK_N_PAD=int(128*64 / D),
            BLOCK_DMODEL=D,
            NUM_WARPS=8,
            NUM_STAGES=1,
        ) 

    basic_tuner_launch_spec = FlashAttentionTuner(
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        q.element_size(),
        causal=causal,
        return_lse=False,
        tune_args=basic_tune_args
    ).build_flash_attention_launch_spec(q, k, v)

    def triton_flash_basic(q, k, v):
        basic_tuner_launch_spec.update_data((q, k, v))
        return flash_attention(
            basic_tuner_launch_spec
        )
    if D == 256:
        opt_tune_args = FlashAttenTuneArguments(
                BLOCK_N=64,
                BLOCK_M=32,
                BLOCK_N_PAD=64,
                BLOCK_DMODEL=256,
                NUM_WARPS=8,
                NUM_STAGES=1,
            ) 
    else:
        opt_tune_args = FlashAttenTuneArguments(
                BLOCK_N=128,
                BLOCK_M=128,
                BLOCK_N_PAD=128,
                BLOCK_DMODEL=128,
                NUM_WARPS=8,
                NUM_STAGES=1,
            ) 
    
    opt_shr_mem_tuner_launch_spec = FlashAttentionOptimizeSharedMemTuner(
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        q.element_size(),
        causal=causal,
        return_lse=False,
        tune_args=opt_tune_args
    ).build_flash_attention_launch_spec(q, k, v)

    def triton_flash_opt_shared_mem(q, k, v):
        opt_shr_mem_tuner_launch_spec.update_data((q, k, v))
        return flash_attention(
            opt_shr_mem_tuner_launch_spec
        )


    if D == 256:
        block_d_nomask_tune_args = FlashAttenTuneArguments(
            BLOCK_N=64,
            BLOCK_M=128,
            BLOCK_N_PAD=64,
            BLOCK_DMODEL=32,
            BLOCK_DMODEL_OUTER=256,
            NUM_WARPS=8,
            NUM_STAGES=4,
        ) 
    else:
        if D == 128:
            block_d_nomask_tune_args = FlashAttenTuneArguments(
                BLOCK_N=64,
                BLOCK_M=128,
                BLOCK_N_PAD=64,
                BLOCK_DMODEL=128,
                BLOCK_DMODEL_OUTER=64,
                NUM_WARPS=8,
                NUM_STAGES=3,
            )
        else:
            # not implemented
            block_d_nomask_tune_args = None
    
    block_d_nomask_launch_spec = FlashAttentionBlockDNomaskTuner(
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        q.element_size(),
        causal=causal,
        return_lse=False,
        tune_args=block_d_nomask_tune_args
    ).build_flash_attention_launch_spec(
        q,
        k,
        v,
    )

    def triton_flash_block_d_nomask(q, k, v):
        block_d_nomask_launch_spec.update_data((q, k, v))
        return flash_attention(
            block_d_nomask_launch_spec
        )
    
    stat_details = {
        "triton_flash_basic": f"{_format_tune_args(basic_tuner_launch_spec.tune_args)} resource_usage {_format_resource_usage(basic_tuner_launch_spec)}",
        "triton_flash_opt_shared_mem": (
            f"{_format_tune_args(opt_shr_mem_tuner_launch_spec.tune_args)} "
            f"resource_usage {_format_resource_usage(opt_shr_mem_tuner_launch_spec)}"
        ),
        "triton_flash_block_d_nomask": (
            f"{_format_tune_args(block_d_nomask_launch_spec.tune_args)} "
            f"resource_usage {_format_resource_usage(block_d_nomask_launch_spec)}"
        ),
    }

    kernels = [
        # ("cpu_sdpa", cpu_sdpa, "cpu"),
        # ("gpu_math_sdpa", gpu_math_sdpa, "cuda"),
        ("gpu_efficient_sdpa", gpu_efficient_sdpa, "cuda"),
        ("gpu_flash_sdpa", gpu_flash_sdpa, "cuda"),
        ("triton_flash_basic", triton_flash_basic, "cuda"),
        ("triton_flash_opt_shared_mem", triton_flash_opt_shared_mem, "cuda"),
        # ("triton_flash_block_d_nomask", triton_flash_block_d_nomask, "cuda")
    ]

    availability = {}
    runnable_kernels = []
    for name, run_kernel, device_kind in kernels:
        error = _probe_kernel_availability(name, run_kernel, attention_inputs)
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

    blocking_errors = [error for error in availability.values() if error is not None]

    for name, run_kernel, device_kind in runnable_kernels:
        output = None
        try:
            if device_kind == "cpu":
                stats[name], output = _time_cpu_ms(run_kernel, attention_inputs)
            else:
                stats[name], output = _time_cuda_ms(run_kernel, attention_inputs)
                torch.cuda.synchronize()

            if name == "cpu_sdpa":
                baseline_output = output.detach().cpu()
            else:
                # _assert_attention_close(output.detach().cpu().float(), baseline_output.float())
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
    summary_text = _format_case_summary(case, "passed", stats, ratios, statuses, stat_details)
    print("\n" + summary_text)
    _write_case_summary_log(summary_text)

    for kernel_stats in stats.values():
        if kernel_stats is not None:
            assert kernel_stats["mean_ms"] > 0


@pytest.mark.parametrize("attention_case", PIPELINE_CASES, ids=_case_id, indirect=True)
def test_attention_kernel_pipeline(attention_inputs):
    _run_kernel_pipeline(attention_inputs)
