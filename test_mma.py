import math
import random
import re
import statistics
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

from mma import _matmul_contiguous_kernel, _select_2d_contiguous_config, _supports_contiguous_cache_eviction_hints


random.seed(20260328)

DEVICE = "cuda"
DTYPE = torch.float16
COMPARE_RTOL = 2e-1
COMPARE_ATOL = 2e-1

TEST_LOG_DIR = Path(__file__).resolve().parent / "test_log"
TEST_RUN_START_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
TEST_RUN_LOG_PATH = TEST_LOG_DIR / f"{TEST_RUN_START_TIMESTAMP}_test_mma_kernel_pipeline.log"
_TRITON_2D_CONTIGUOUS_OUTPUT_CACHE = {}
CUDA_CLOCK_WARMUP_SLEEP_CYCLES = 50_000_000
CUDA_CLOCK_WARMUP_SLEEP_LAUNCHES = 1

KERNEL_RUNTIME_SERIES = {
    "cpu_torch_matmul": "cpu",
    "gpu_torch_matmul": "torch_cuda",
    "triton_2d_contiguous": "triton_2d_contiguous",
}


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton tl.dot tests")

LARGE_CONFIGS = [
    (1024, 256, 1024),
    (8192, 256, 8192),
]
PIPELINE_CASES = list(dict.fromkeys(LARGE_CONFIGS))


def _case_id(case):
    M, K, N = case
    return f"M{M}_K{K}_N{N}"


def _format_case_summary(case, case_status: str, stats, ratios, statuses, stat_details=None):
    M, K, N = case
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
        f"[case] status={case_status} M={M} K={K} N={N} | "
        + " | ".join(summary + ratios + statuses)
    )


def _write_case_summary_log(summary_text: str):
    TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with TEST_RUN_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(summary_text + "\n")
    return TEST_RUN_LOG_PATH


def _matmul_tflops(M: int, K: int, N: int, runtime_ms: float):
    if not math.isfinite(runtime_ms) or runtime_ms <= 0:
        return float("nan")

    flops = 2 * M * K * N
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


def _parse_mma_kernel_pipeline_rows(log_path: Path):
    log_path = Path(log_path)
    rows = []

    if not log_path.exists():
        return rows

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("[case]"):
            continue

        shape_match = re.search(r"\bM=(\d+)\s+K=(\d+)\s+N=(\d+)\b", line)
        if shape_match is None:
            continue

        M, K, N = (int(value) for value in shape_match.groups())
        runtimes = {}
        for kernel_name in KERNEL_RUNTIME_SERIES:
            match = re.search(rf"(?:^|\|\s*){re.escape(kernel_name)}(?:\s+[^|]*?)?\s+mean=(n/a|[0-9.]+)ms\b", line)
            if match is None or match.group(1) == "n/a":
                runtimes[kernel_name] = float("nan")
            else:
                runtimes[kernel_name] = float(match.group(1))

        rows.append(
            {
                "M": M,
                "K": K,
                "N": N,
                "data_size": M * K + K * N + M * N,
                "runtimes": runtimes,
            }
        )

    return rows


def _collect_mma_kernel_pipeline_folder_rows(target_folder: Path, datatype=torch.float16):
    target_folder = Path(target_folder)
    element_size = _dtype_itemsize(datatype)
    rows = []

    for log_path in sorted(target_folder.glob("*.log")):
        for row in _parse_mma_kernel_pipeline_rows(log_path):
            M, K, N = row["M"], row["K"], row["N"]
            rows.append(
                {
                    **row,
                    "log_path": log_path,
                    "label": f"{M}x{N}x{K}",
                    "data_bytes": (M * K + K * N + M * N) * element_size,
                }
            )

    rows.sort(key=lambda row: (row["data_bytes"], row["M"], row["K"], row["N"], row["log_path"].name))
    return rows


def _render_mma_kernel_pipeline_folder_graph(
    target_folder: Path,
    datatype=torch.float16,
    output_path: Path | None = None,
):
    rows = _collect_mma_kernel_pipeline_folder_rows(target_folder, datatype=datatype)
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_folder = Path(target_folder)
    output_path = Path(output_path) if output_path is not None else target_folder / "mma_kernel_pipeline_runtime_tflops.png"

    active_kernels = [
        kernel_name
        for kernel_name in MMA_FOLDER_GRAPH_KERNELS
        if any(math.isfinite(row["runtimes"][kernel_name]) and row["runtimes"][kernel_name] > 0 for row in rows)
    ]
    if not active_kernels:
        return None

    x_values = list(range(len(rows)))
    case_labels = [row["label"] for row in rows]
    bar_width = min(0.16, 0.82 / len(active_kernels))
    group_offset = (len(active_kernels) - 1) * bar_width / 2

    fig_width = max(12.0, min(42.0, 1.2 * len(rows) + 7.0))
    fig, (runtime_ax, tflops_ax) = plt.subplots(1, 2, figsize=(fig_width, 6.0), dpi=150)

    for kernel_index, kernel_name in enumerate(active_kernels):
        offset = kernel_index * bar_width - group_offset
        bar_x = [x + offset for x in x_values]
        runtime_values = [row["runtimes"][kernel_name] for row in rows]
        tflops_values = [
            _matmul_tflops(row["M"], row["K"], row["N"], row["runtimes"][kernel_name])
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

    runtime_ax.set_xlabel("Data shape (M x N x K)")
    runtime_ax.set_ylabel("Mean runtime (ms)")
    runtime_ax.set_yscale("log")
    runtime_ax.set_title("MMA Kernel Mean Runtime")

    tflops_ax.set_xlabel("Data shape (M x N x K)")
    tflops_ax.set_ylabel("TFLOPs")
    tflops_ax.set_title("MMA Kernel Throughput")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_mma_kernel_pipeline_folder_graph(
    target_folder: Path,
    datatype=torch.float16,
    output_path: Path | None = None,
):
    return _render_mma_kernel_pipeline_folder_graph(
        target_folder,
        datatype=datatype,
        output_path=output_path,
    )


def _make_inputs(M: int, K: int, N: int):
    a = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    b = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
    return a, b


def _make_cpu_inputs(a: torch.Tensor, b: torch.Tensor):
    return a.detach().cpu().to(DTYPE), b.detach().cpu().to(DTYPE)


def _clear_cuda_memory():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.fixture
def mma_case(request):
    _require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    return request.param


@pytest.fixture
def mma_inputs(mma_case):
    M, K, N = mma_case
    a, b = _make_inputs(M, K, N)
    return {
        "case": mma_case,
        "a": a,
        "b": b,
    }


def _assert_matmul_close(actual: torch.Tensor, expected: torch.Tensor):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=COMPARE_RTOL,
        atol=COMPARE_ATOL,
    )


def _run_cpu_torch_matmul(a: torch.Tensor, b: torch.Tensor):
    return torch.matmul(a, b)


def _run_gpu_torch_matmul(a: torch.Tensor, b: torch.Tensor):
    return torch.matmul(a, b)


def _run_triton_2d_contiguous(a: torch.Tensor, b: torch.Tensor):
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    block_m, block_n, block_k, group_m, num_warps, num_stages = _select_2d_contiguous_config(M, N, K)
    cache_key = (a.device, a.dtype, M, N, block_m, block_n)
    cached = _TRITON_2D_CONTIGUOUS_OUTPUT_CACHE.get(cache_key)
    if cached is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
        cached = (c, grid)
        _TRITON_2D_CONTIGUOUS_OUTPUT_CACHE[cache_key] = cached
    c, grid = cached
    _matmul_contiguous_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        USE_CACHE_EVICTION_HINTS=_supports_contiguous_cache_eviction_hints(a.device),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


def _warmup_cuda_clocks(reference: torch.Tensor):
    del reference
    for _ in range(CUDA_CLOCK_WARMUP_SLEEP_LAUNCHES):
        torch.cuda._sleep(CUDA_CLOCK_WARMUP_SLEEP_CYCLES)
    torch.cuda.synchronize()


def _time_cuda_ms(fn, mma_inputs, warmup: int = 10, iters: int = 30):
    a = mma_inputs["a"]
    b = mma_inputs["b"]
    _warmup_cuda_clocks(a)

    output = fn(a, b).clone()
    for _ in range(warmup):
        a, b = torch.randn_like(a), torch.randn_like(b)
        fn(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        a, b = torch.randn_like(a), torch.randn_like(b)
        start.record()
        fn(a, b)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    triton.testing.do_bench(
        lambda: fn(a, b),
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


def _time_cpu_ms(fn, mma_inputs, warmup: int = 3, iters: int = 10):
    a = mma_inputs["a"]
    b = mma_inputs["b"]
    a_cpu, b_cpu = _make_cpu_inputs(a, b)
    output = fn(a_cpu, b_cpu)
    for _ in range(warmup):
        a_cpu, b_cpu = torch.randn_like(a_cpu), torch.randn_like(b_cpu)
        fn(a_cpu, b_cpu)
    times = []
    for _ in range(iters):
        a_cpu, b_cpu = torch.randn_like(a_cpu), torch.randn_like(b_cpu)
        start = time.perf_counter()
        fn(a_cpu, b_cpu)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }, output


def _probe_kernel_availability(name: str, run_kernel: Callable, mma_inputs: dict):
    try:
        a, b = mma_inputs["a"], mma_inputs["b"]
        out = run_kernel(a, b)
        del out
        _clear_cuda_memory()
        return None
    except Exception as exc:
        _clear_cuda_memory()
        return f"{name} unavailable: {exc}"


def _format_speedup_ratios(stats):
    ratios = []
    cpu_stats = stats.get("cpu_torch_matmul")
    cpu_mean = cpu_stats["mean_ms"] if cpu_stats is not None else 0.0
    for name, kernel_stats in stats.items():
        if name == "cpu_torch_matmul" or kernel_stats is None:
            continue
        kernel_mean = kernel_stats["mean_ms"]
        ratio = cpu_mean / kernel_mean if kernel_mean > 0 else float("inf")
        ratios.append(f"{name}/cpu_torch_matmul={ratio:.3f}x")
    return ratios


def _run_kernel_pipeline(mma_inputs):
    a = mma_inputs["a"]
    b = mma_inputs["b"]
    case = mma_inputs["case"]
    triton_2d_detail = "unavailable"
    try:
        M, K = a.shape
        _, N = b.shape
        block_m, block_n, block_k, group_m, num_warps, num_stages = _select_2d_contiguous_config(M, N, K)
        triton_2d_detail = (
            f"BLOCK_M={block_m} BLOCK_N={block_n} BLOCK_K={block_k} "
            f"GROUP_M={group_m} num_warps={num_warps} num_stages={num_stages}"
        )
    except ValueError:
        pass
    stat_details = {
        "triton_2d_contiguous": triton_2d_detail,
    }
    kernels = [
        ("cpu_torch_matmul", _run_cpu_torch_matmul, "cpu"),
        ("gpu_torch_matmul", _run_gpu_torch_matmul, "cuda"),
        ("triton_2d_contiguous", _run_triton_2d_contiguous, "cuda")
    ]

    availability = {}
    runnable_kernels = []
    for name, run_kernel, device_kind in kernels:
        error = _probe_kernel_availability(name, run_kernel, mma_inputs)
        availability[name] = error
        if error is None:
            runnable_kernels.append((name, run_kernel, device_kind))

    baseline_output = None
    stats = {name: None for name, _, _ in kernels}
    statuses = []

    for name, error in availability.items():
        if error is not None:
            stats[name] = None
            statuses.append(f"{name}=unavailable ({error})")
    blocking_errors = [error for error in availability.values() if error is not None]
    if blocking_errors:
        summary_text = _format_case_summary(case, "error", stats, [], statuses, stat_details)
        print("\n" + summary_text)
        _write_case_summary_log(summary_text)
        raise AssertionError("; ".join(blocking_errors))

    try:
        with torch.no_grad():
            outputs = {}
            for name, run_kernel, device_kind in runnable_kernels:
                output = None
                if device_kind == "cpu":
                    stats[name], output = _time_cpu_ms(run_kernel, mma_inputs)
                else:
                    stats[name], output = _time_cuda_ms(run_kernel, mma_inputs)
                    torch.cuda.synchronize()
                outputs[name] = output

            baseline_output = outputs["cpu_torch_matmul"].detach().cpu()
            for name, _, device_kind in runnable_kernels:
                if name == "cpu_torch_matmul":
                    continue
                output = outputs[name]
                _assert_matmul_close(output.detach().cpu().to(DTYPE), baseline_output.to(DTYPE))
                statuses.append(f"{name}=passed")
    except Exception as exc:
        statuses.append(f"{name}=error ({type(exc).__name__}: {exc})")
        ratios = _format_speedup_ratios(stats)
        summary_text = _format_case_summary(case, "error", stats, ratios, statuses, stat_details)
        print("\n" + summary_text)
        _write_case_summary_log(summary_text)
        raise

    ratios = _format_speedup_ratios(stats)
    summary_text = _format_case_summary(case, "passed", stats, ratios, statuses, stat_details)
    print("\n" + summary_text)
    _write_case_summary_log(summary_text)

    for kernel_stats in stats.values():
        if kernel_stats is not None:
            assert kernel_stats["mean_ms"] > 0


@pytest.mark.parametrize("mma_case", PIPELINE_CASES, ids=_case_id, indirect=True)
def test_mma_kernel_pipeline(mma_inputs):
    _run_kernel_pipeline(mma_inputs)
