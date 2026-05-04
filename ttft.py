from __future__ import annotations

import argparse
import contextlib
import gc
import math
import statistics
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from flash_atten import flash_attention
from tuner import FlashAttentionBlockDNomaskTuner, FlashAttenTuneArguments

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - depends on the local torch build.
    SDPBackend = None
    sdpa_kernel = None


DEFAULT_MODELS = (
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen3.5-0.8B",
)
DEFAULT_PROMPT_LENGTHS = (1024, 8192, 16384, 32768, 65536)
DTYPE = torch.float16
DEVICE = "cuda"
PLOT_DIR = Path(__file__).resolve().parent / "test_log"
PLOT_BACKEND_LABELS = {
    "efficient_gqa_expanded": "torch SDPA efficient",
    "flash": "torch SDPA flash_atten",
    "triton_block_d_nomask": "ours",
    "triton_gqa": "ours",
}


@dataclass(frozen=True)
class ModelShape:
    model_id: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int
    full_attention_layers: int
    vocab_size: int


@dataclass(frozen=True)
class BenchResult:
    model_id: str
    prompt_len: int
    backend: str
    mean_ms: float | None
    p50_ms: float | None
    samples_ms: tuple[float, ...]
    status: str


@dataclass
class BackendUseStats:
    backend: str
    calls: int = 0


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TTFT benchmarking.")


def _require_transformers():
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "ttft.py requires transformers. Install it in the project env, for example: "
            "~/miniconda3/envs/py312/bin/python -m pip install transformers accelerate"
        ) from exc
    return AutoConfig, AutoModelForCausalLM


def _text_config(config):
    return getattr(config, "text_config", None) or config


def _get_layer_types(text_config) -> list[str] | None:
    layer_types = getattr(text_config, "layer_types", None)
    return list(layer_types) if layer_types is not None else None


def inspect_model_shape(model_id: str) -> ModelShape:
    AutoConfig, _ = _require_transformers()
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    text_config = _text_config(config)
    hidden_size = int(getattr(text_config, "hidden_size"))
    num_attention_heads = int(getattr(text_config, "num_attention_heads"))
    head_dim = int(getattr(text_config, "head_dim", hidden_size // num_attention_heads))
    num_key_value_heads = int(getattr(text_config, "num_key_value_heads", num_attention_heads))
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers"))
    layer_types = _get_layer_types(text_config)
    if layer_types is None:
        full_attention_layers = num_hidden_layers
    else:
        full_attention_layers = sum(1 for layer_type in layer_types if layer_type == "full_attention")
    vocab_size = int(getattr(text_config, "vocab_size", getattr(config, "vocab_size", 0)))
    return ModelShape(
        model_id=model_id,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        head_dim=head_dim,
        full_attention_layers=full_attention_layers,
        vocab_size=vocab_size,
    )


def estimate_fp16_vram_gib(shape: ModelShape, prompt_len: int, parameter_count: int | None) -> float:
    weight_bytes = 0 if parameter_count is None else parameter_count * 2
    kv_bytes = (
        2
        * shape.full_attention_layers
        * shape.num_key_value_heads
        * prompt_len
        * shape.head_dim
        * 2
    )
    # Leave room for logits, temporary activations, allocator fragmentation, and non-attention state.
    return (weight_bytes + kv_bytes) / (1024**3) * 1.35


def _make_random_input_ids(prompt_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    high = max(1024, vocab_size if vocab_size > 0 else 32000)
    return torch.randint(0, high, (1, prompt_len), device=device, dtype=torch.long)


def _block_d_tune_args(head_dim: int) -> FlashAttenTuneArguments:
    if head_dim == 256:
        return FlashAttenTuneArguments(
            BLOCK_N=64,
            BLOCK_M=128,
            BLOCK_N_PAD=64,
            BLOCK_DMODEL=32,
            BLOCK_DMODEL_OUTER=256,
            NUM_WARPS=8,
            NUM_STAGES=4,
        )
    if head_dim == 128:
        return FlashAttenTuneArguments(
            BLOCK_N=64,
            BLOCK_M=128,
            BLOCK_N_PAD=64,
            BLOCK_DMODEL=128,
            BLOCK_DMODEL_OUTER=64,
            NUM_WARPS=8,
            NUM_STAGES=3,
        )
    raise ValueError(f"block-D nomask benchmark only has known launch settings for head_dim 128 or 256, got {head_dim}")


def _run_block_d_nomask_vanilla(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("custom SDPA expects q, k, v with shape [batch, heads, seq, head_dim]")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise ValueError("custom SDPA only supports float16 tensors")
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        raise ValueError("custom SDPA only supports CUDA tensors")
    if q.shape[0] != 1:
        raise ValueError("custom SDPA benchmark expects batch size 1")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("vanilla block-D attention expects q, k, and v to have the same number of heads")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same head_dim")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    tune_args = _block_d_tune_args(q.shape[-1])
    launch_spec = FlashAttentionBlockDNomaskTuner(
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        q.element_size(),
        causal=False,
        return_lse=False,
        tune_args=tune_args,
    ).build_flash_attention_launch_spec(q, k, v, sm_scale=scale)
    return flash_attention(launch_spec)


def _raise_unsupported_triton(reason: str):
    raise RuntimeError(f"triton_block_d_nomask unsupported SDPA call: {reason}")


def _repeat_kv_for_gqa(k_or_v: torch.Tensor, query_heads: int) -> torch.Tensor:
    kv_heads = k_or_v.shape[1]
    if kv_heads == query_heads:
        return k_or_v
    if query_heads % kv_heads != 0:
        raise ValueError(f"query_heads={query_heads} is not divisible by kv_heads={kv_heads}")
    return k_or_v.repeat_interleave(query_heads // kv_heads, dim=1)


def _run_block_d_nomask_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None) -> torch.Tensor:
    k = _repeat_kv_for_gqa(k, q.shape[1])
    v = _repeat_kv_for_gqa(v, q.shape[1])
    return _run_block_d_nomask_vanilla(q, k, v, scale)


def _run_triton_gqa_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None) -> torch.Tensor:
    return _run_block_d_nomask_sdpa(q, k, v, scale)


def _run_efficient_gqa_expanded_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sdpa_fn,
    *,
    attn_mask,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    if attn_mask is not None:
        raise RuntimeError("efficient_gqa_expanded unsupported SDPA call: attn_mask is not None")
    if dropout_p != 0.0:
        raise RuntimeError(f"efficient_gqa_expanded unsupported SDPA call: dropout_p={dropout_p}")
    q = q.contiguous()
    k = _repeat_kv_for_gqa(k, q.shape[1]).contiguous()
    v = _repeat_kv_for_gqa(v, q.shape[1]).contiguous()
    with torch_sdpa_backend("efficient"):
        return sdpa_fn(
            q,
            k,
            v,
            is_causal=is_causal,
            dropout_p=0.0,
            scale=scale,
        )


@contextlib.contextmanager
def force_noncausal_attention(model) -> Iterator[None]:
    patched_modules = []
    for module in model.modules():
        if hasattr(module, "is_causal"):
            patched_modules.append((module, module.is_causal))
            module.is_causal = False
    try:
        yield
    finally:
        for module, old_value in patched_modules:
            module.is_causal = old_value


@contextlib.contextmanager
def patched_sdpa_with_efficient_gqa_expanded(*, stats: BackendUseStats | None = None) -> Iterator[None]:
    original_sdpa = F.scaled_dot_product_attention

    def replacement(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if stats is not None:
            stats.calls += 1
        return _run_efficient_gqa_expanded_sdpa(
            query,
            key,
            value,
            original_sdpa,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    F.scaled_dot_product_attention = replacement
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa


@contextlib.contextmanager
def patched_sdpa_with_triton_gqa(*, allow_causal: bool, stats: BackendUseStats | None = None) -> Iterator[None]:
    original_sdpa = F.scaled_dot_product_attention

    def replacement(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if dropout_p != 0.0:
            _raise_unsupported_triton(f"dropout_p={dropout_p}")
        if attn_mask is not None:
            _raise_unsupported_triton("attn_mask is not None")
        if is_causal and not allow_causal:
            _raise_unsupported_triton("is_causal=True")
        if stats is not None:
            stats.calls += 1
        return _run_triton_gqa_sdpa(query, key, value, scale)

    F.scaled_dot_product_attention = replacement
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa


@contextlib.contextmanager
def torch_sdpa_backend(name: str) -> Iterator[None]:
    if sdpa_kernel is None or SDPBackend is None:
        raise RuntimeError("This PyTorch build does not expose torch.nn.attention.sdpa_kernel.")
    backend = {
        "math": SDPBackend.MATH,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
    }[name]
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*kernel not used because.*")
        warnings.filterwarnings("error", message=".*No available kernel.*")
        warnings.filterwarnings("error", message=".*both fused kernels require query, key and value to have the same num_heads.*")
        warnings.filterwarnings("error", message=".*has been runtime disabled.*")
        with sdpa_kernel(backends=[backend]):
            yield


@contextlib.contextmanager
def patched_sdpa_enable_gqa(*, stats: BackendUseStats | None = None) -> Iterator[None]:
    original_sdpa = F.scaled_dot_product_attention

    def replacement(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if (
            not enable_gqa
            and query.ndim >= 3
            and key.ndim >= 3
            and value.ndim >= 3
            and query.shape[-3] != key.shape[-3]
            and key.shape[-3] == value.shape[-3]
            and query.shape[-3] % key.shape[-3] == 0
        ):
            enable_gqa = True
        if stats is not None:
            stats.calls += 1
        return original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)

    F.scaled_dot_product_attention = replacement
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa


@contextlib.contextmanager
def backend_context(name: str, *, allow_causal_custom: bool, stats: BackendUseStats | None = None) -> Iterator[None]:
    if name in {"math", "efficient", "flash"}:
        with torch_sdpa_backend(name):
            with patched_sdpa_enable_gqa(stats=stats):
                yield
        return
    if name in {"triton_block_d_nomask", "triton_gqa"}:
        with patched_sdpa_with_triton_gqa(allow_causal=allow_causal_custom, stats=stats):
            yield
        return
    if name == "efficient_gqa_expanded":
        with patched_sdpa_with_efficient_gqa_expanded(stats=stats):
            yield
        return
    raise ValueError(f"unknown backend {name}")


def _forward_final_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=None,
        use_cache=True,
        logits_to_keep=1,
    )
    return outputs.logits[:, -1, :]


def _forward_first_token(model, input_ids: torch.Tensor):
    return _forward_final_logits(model, input_ids).argmax(dim=-1)


def verify_against_flash(
    model,
    input_ids: torch.Tensor,
    *,
    backend: str,
    allow_causal_custom: bool,
    force_noncausal: bool,
) -> str:
    if backend == "flash":
        return "skipped"
    causal_context = force_noncausal_attention(model) if force_noncausal else contextlib.nullcontext()
    with torch.inference_mode():
        with causal_context:
            custom_stats = BackendUseStats(backend)
            flash_stats = BackendUseStats("flash")
            with backend_context(backend, allow_causal_custom=allow_causal_custom, stats=custom_stats):
                custom_logits = _forward_final_logits(model, input_ids).detach().float()
            with backend_context("flash", allow_causal_custom=allow_causal_custom, stats=flash_stats):
                flash_logits = _forward_final_logits(model, input_ids).detach().float()
    if custom_stats.calls <= 0:
        raise RuntimeError(f"{backend} verification did not intercept any SDPA calls")
    if flash_stats.calls <= 0:
        raise RuntimeError("flash verification did not intercept any SDPA calls")
    max_abs = (custom_logits - flash_logits).abs().max().item()
    denom = flash_logits.abs().clamp_min(1.0e-6)
    max_rel = ((custom_logits - flash_logits).abs() / denom).max().item()
    token_match = bool(custom_logits.argmax(dim=-1).eq(flash_logits.argmax(dim=-1)).all().item())
    close = torch.allclose(custom_logits, flash_logits, rtol=1.0e-1, atol=1.0e-1)
    summary = f"close={close} next_token_match={token_match} max_abs={max_abs:.6f} max_rel={max_rel:.6f}"
    if not close:
        raise AssertionError(f"{backend} output differs from Torch flash reference: {summary}")
    return f"{summary} backend_sdpa_calls={custom_stats.calls} flash_sdpa_calls={flash_stats.calls}"


def time_ttft_ms(
    model,
    input_ids: torch.Tensor,
    *,
    backend: str,
    repeats: int,
    warmups: int,
    allow_causal_custom: bool,
    force_noncausal: bool,
) -> tuple[float, ...]:
    samples = []
    total_sdpa_calls = 0
    with torch.inference_mode():
        for idx in range(warmups + repeats):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start = time.perf_counter()
            causal_context = force_noncausal_attention(model) if force_noncausal else contextlib.nullcontext()
            stats = BackendUseStats(backend)
            with causal_context:
                with backend_context(backend, allow_causal_custom=allow_causal_custom, stats=stats):
                    output = _forward_first_token(model, input_ids)
            torch.cuda.synchronize()
            if stats.calls <= 0:
                raise RuntimeError(f"{backend} did not intercept any SDPA calls")
            total_sdpa_calls += stats.calls
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            del output
            if idx >= warmups:
                samples.append(elapsed_ms)
    time_ttft_ms.last_sdpa_calls = total_sdpa_calls
    return tuple(samples)


def load_model(model_id: str):
    _, AutoModelForCausalLM = _require_transformers()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=DTYPE,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    return model.to(device=DEVICE, dtype=DTYPE).eval()


def _safe_plot_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(" ", "_")


def plot_ttft_results(results: list[BenchResult]) -> list[Path]:
    successful = [result for result in results if result.mean_ms is not None]
    if not successful:
        print("[plot] skipped no successful benchmark results")
        return []

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    model_ids = sorted({result.model_id for result in successful})
    fig, axes = plt.subplots(
        1,
        len(model_ids),
        figsize=(8.0 * len(model_ids), 5.0),
        squeeze=False,
        sharey=False,
    )
    for axis, model_id in zip(axes[0], model_ids):
        model_results = [result for result in successful if result.model_id == model_id]
        backends = sorted({result.backend for result in model_results})
        for backend in backends:
            backend_results = sorted(
                [result for result in model_results if result.backend == backend],
                key=lambda result: result.prompt_len,
            )
            x_vals = [result.prompt_len for result in backend_results]
            y_vals = [result.mean_ms for result in backend_results]
            axis.plot(x_vals, y_vals, marker="o", linewidth=2, label=PLOT_BACKEND_LABELS.get(backend, backend))
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Prompt length (tokens)")
        axis.set_ylabel("TTFT (ms)")
        axis.set_title(model_id)
        axis.grid(True, which="both", linestyle="--", alpha=0.35)
        axis.legend()
    fig.suptitle("LLM TTFT Test", fontsize=14)
    fig.tight_layout()
    output_path = PLOT_DIR / f"{timestamp}_llm_ttft.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[plot] path={output_path}")
    return [output_path]


def _print_shape(shape: ModelShape, expected_head_dim: int | None) -> None:
    expected = "n/a" if expected_head_dim is None else str(expected_head_dim)
    ok = expected_head_dim is None or shape.head_dim == expected_head_dim
    status = "ok" if ok else "mismatch"
    print(
        f"[shape] {shape.model_id} hidden={shape.hidden_size} q_heads={shape.num_attention_heads} "
        f"kv_heads={shape.num_key_value_heads} layers={shape.num_hidden_layers} "
        f"full_attn_layers={shape.full_attention_layers} head_dim={shape.head_dim} "
        f"expected={expected} status={status}"
    )


def run(args: argparse.Namespace) -> list[BenchResult]:
    _require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = True
    device_name = torch.cuda.get_device_name()
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[device] {device_name} memory={total_gib:.1f}GiB dtype=float16")
    print("[note] triton_block_d_nomask patches F.scaled_dot_product_attention in-process.")
    print("[note] The benchmark passes no attention_mask and defaults HF attention modules to non-causal mode to match the local nomask/noncausal kernel.")

    expected_dims = {
        "Qwen/Qwen2.5-0.5B": 128,
        "Qwen/Qwen2.5-1.5B": 128,
        "Qwen/Qwen2.5-1.5B-Instruct": 128,
        "Qwen/Qwen3.5-0.8B": 256,
        "Qwen/Qwen3.5-0.8B-Base": 256,
    }
    results: list[BenchResult] = []
    verified_cases: set[tuple[str, int, str]] = set()
    for model_id in args.models:
        shape = inspect_model_shape(model_id)
        _print_shape(shape, expected_dims.get(model_id))
        if expected_dims.get(model_id) is not None and shape.head_dim != expected_dims[model_id]:
            print(
                f"[warn] {model_id} does not match requested head_dim={expected_dims[model_id]}; "
                "custom block-D runs may be skipped or fail for this model."
            )

        model = load_model(model_id)
        param_count = model.num_parameters() if hasattr(model, "num_parameters") else None
        for prompt_len in args.lengths:
            estimated_gib = estimate_fp16_vram_gib(shape, prompt_len, param_count)
            if estimated_gib > total_gib * args.memory_fraction:
                print(
                    f"[skip] {model_id} prompt_len={prompt_len} estimated_vram={estimated_gib:.1f}GiB "
                    f"exceeds {args.memory_fraction:.0%} of available memory"
                )
                continue
            input_ids = _make_random_input_ids(prompt_len, shape.vocab_size, torch.device(DEVICE))
            for backend in args.backends:
                try:
                    verify_key = (model_id, prompt_len, backend)
                    if args.verify_flash and backend != "flash" and verify_key not in verified_cases:
                        verification = verify_against_flash(
                            model,
                            input_ids,
                            backend=backend,
                            allow_causal_custom=args.allow_causal_custom,
                            force_noncausal=args.force_noncausal,
                        )
                        verified_cases.add(verify_key)
                        print(
                            f"[verify] model={model_id} prompt_len={prompt_len} "
                            f"backend={backend} reference=flash {verification}"
                        )
                    samples = time_ttft_ms(
                        model,
                        input_ids,
                        backend=backend,
                        repeats=args.repeats,
                        warmups=args.warmups,
                        allow_causal_custom=args.allow_causal_custom,
                        force_noncausal=args.force_noncausal,
                    )
                    mean_ms = statistics.mean(samples)
                    p50_ms = statistics.median(samples)
                    print(
                        f"[result] model={model_id} prompt_len={prompt_len} backend={backend} "
                        f"mean={mean_ms:.3f}ms p50={p50_ms:.3f}ms "
                        f"sdpa_calls={getattr(time_ttft_ms, 'last_sdpa_calls', 'n/a')} "
                        f"samples={','.join(f'{x:.3f}' for x in samples)}"
                    )
                    results.append(BenchResult(model_id, prompt_len, backend, mean_ms, p50_ms, samples, "ok"))
                except Exception as exc:
                    print(f"[result] model={model_id} prompt_len={prompt_len} backend={backend} status=failed error={exc}")
                    results.append(BenchResult(model_id, prompt_len, backend, None, None, (), f"failed: {exc}"))
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
            del input_ids
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if args.plot:
        plot_ttft_results(results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark HF transformer TTFT across SDPA backends.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_PROMPT_LENGTHS))
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["triton_block_d_nomask", "efficient_gqa_expanded", "flash"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--memory-fraction", type=float, default=0.88)
    parser.add_argument(
        "--force-noncausal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Temporarily set HF attention modules with an is_causal flag to False during timed runs.",
    )
    parser.add_argument(
        "--allow-causal-custom",
        action="store_true",
        help="Force causal self-attention calls through the non-causal local kernel for latency experiments only.",
    )
    parser.add_argument(
        "--verify-flash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare Triton final-token logits with Torch flash SDPA after each successful Triton timing case.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one TTFT-vs-token-length PNG per model after the benchmark.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
