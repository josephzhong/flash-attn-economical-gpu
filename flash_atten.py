from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import cuda as tl_cuda
from triton.runtime.errors import OutOfResources

from exceptions import (
    OutOfResourcesWithDetail,
    estimate_flash_fwd_shared_bytes,
    get_sm_resource_limits,
)
from tuner import LaunchSpec, Tuner


@triton.jit
def _flash_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    B,
    H,
    M,
    N,
    D,
    sm_scale,
    causal: tl.constexpr,
    HAS_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        cur_n = start_n + offs_n

        k_ptrs = (
            k_ptr
            + b * stride_kb
            + h * stride_kh
            + cur_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            v_ptr
            + b * stride_vb
            + h * stride_vh
            + cur_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )

        kv_mask = (cur_n[:, None] < N) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale

        in_bounds = (offs_m[:, None] < M) & (cur_n[None, :] < N)
        qk = tl.where(in_bounds, qk, float("-inf"))

        if causal:
            causal_mask = offs_m[:, None] >= cur_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    out = acc / l_i[:, None]

    o_ptrs = (
        o_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    o_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    tl.store(o_ptrs, out, mask=o_mask)

    if HAS_LSE:
        lse_ptrs = lse_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lm
        lse = m_i + tl.log(l_i)
        tl.store(lse_ptrs, lse, mask=offs_m < M)


def _select_block_dmodel(d: int) -> int:
    return max(16, _next_power_of_two(d))


def _default_block_m(block_n_pad: int) -> int:
    if block_n_pad >= 128:
        return 16
    if block_n_pad >= 64:
        return 32
    return 64


def _get_max_shared_mem_bytes() -> int:
    device = triton.runtime.driver.active.get_current_device()
    return triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]


@dataclass(frozen=True)
class FlashAttentionLaunchSpec(LaunchSpec):
    grid: None
    kernel_args: None
    kernel_kwargs: None
    launch_context: None


@dataclass(frozen=True)
class FlashAttenTuneArguments:
    BLOCK_N: int | None = None
    BLOCK_M: int | None = None
    BLOCK_N_PAD: int | None = None
    BLOCK_DMODEL: int | None = None

    def __post_init__(self) -> None:
        if self.BLOCK_N is not None and self.BLOCK_N <= 0:
            raise ValueError("BLOCK_N must be positive when provided")
        if self.BLOCK_M is not None:
            if self.BLOCK_M <= 0:
                raise ValueError("BLOCK_M must be positive when provided")
            if _next_power_of_two(self.BLOCK_M) != self.BLOCK_M:
                raise ValueError("BLOCK_M must be a power of 2 when provided")
        if self.BLOCK_N_PAD is not None:
            if self.BLOCK_N_PAD <= 0:
                raise ValueError("BLOCK_N_PAD must be positive when provided")
            if _next_power_of_two(self.BLOCK_N_PAD) != self.BLOCK_N_PAD:
                raise ValueError("BLOCK_N_PAD must be a power of 2 when provided")
        if self.BLOCK_DMODEL is not None:
            if self.BLOCK_DMODEL <= 0:
                raise ValueError("BLOCK_DMODEL must be positive when provided")
            if _next_power_of_two(self.BLOCK_DMODEL) != self.BLOCK_DMODEL:
                raise ValueError("BLOCK_DMODEL must be a power of 2 when provided")

class FlashAttentionTuner(Tuner):
    _config_cache: dict[tuple[object, ...], tuple[dict[str, int | bool], dict[str, object], int]] = {}

    def __init__(
        self,
        q_shape: tuple[int, int, int, int],
        k_shape: tuple[int, int, int, int],
        v_shape: tuple[int, int, int, int],
        *,
        causal: bool,
        return_lse: bool,
        tune_args: FlashAttenTuneArguments | None = None,
    ) -> None:
        if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
            raise ValueError("q_shape, k_shape, and v_shape must all be 4D")
        if q_shape[:2] != k_shape[:2] or q_shape[:2] != v_shape[:2]:
            raise ValueError("Batch and head dimensions must match across q_shape, k_shape, and v_shape")
        if q_shape[3] != k_shape[3] or q_shape[3] != v_shape[3]:
            raise ValueError("Last dimension must match across q_shape, k_shape, and v_shape")
        if k_shape[2] != v_shape[2]:
            raise ValueError("k_shape and v_shape sequence lengths must match")
        self.q_shape = tuple(int(v) for v in q_shape)
        self.k_shape = tuple(int(v) for v in k_shape)
        self.v_shape = tuple(int(v) for v in v_shape)
        self.causal = causal
        self.return_lse = return_lse
        self.tune_args = tune_args or FlashAttenTuneArguments()
        self.B, self.H, self.M, self.D = self.q_shape
        _, _, self.N, _ = self.k_shape
        if self.tune_args.BLOCK_DMODEL is not None and self.tune_args.BLOCK_DMODEL < self.D:
            raise ValueError("BLOCK_DMODEL must be >= D when provided")
        if self.tune_args.BLOCK_N_PAD is not None and self.tune_args.BLOCK_N is not None:
            if self.tune_args.BLOCK_N > self.tune_args.BLOCK_N_PAD:
                raise ValueError("BLOCK_N cannot exceed BLOCK_N_PAD")
        self.sm_limits = get_sm_resource_limits()

    def _cache_key(self, q: torch.Tensor) -> tuple[object, ...]:
        return (
            str(q.device),
            str(q.dtype),
            self.B,
            self.H,
            self.M,
            self.N,
            self.D,
            self.causal,
            self.return_lse,
            self.tune_args.BLOCK_N,
            self.tune_args.BLOCK_M,
            self.tune_args.BLOCK_N_PAD,
            self.tune_args.BLOCK_DMODEL,
        )

    def _power_of_two_candidates(self, upper_bound: int, minimum: int = 1) -> list[int]:
        values: list[int] = []
        value = 1 << (max(minimum, upper_bound) - 1).bit_length()
        while value >= minimum:
            values.append(min(value, upper_bound))
            value //= 2
        deduped = []
        seen = set()
        for value in values:
            if value not in seen and value >= minimum:
                deduped.append(value)
                seen.add(value)
        return deduped

    def _block_n_pad_candidates(self) -> list[int]:
        if self.tune_args.BLOCK_N_PAD is not None:
            return [self.tune_args.BLOCK_N_PAD]
        min_pad = _next_power_of_two(self.tune_args.BLOCK_N) if self.tune_args.BLOCK_N is not None else 16
        max_pad = min(128, _next_power_of_two(max(16, min(self.N, 128))))
        if min_pad > max_pad:
            max_pad = min_pad
        return [pad for pad in self._power_of_two_candidates(max_pad, minimum=16) if pad >= min_pad]

    def _block_m_candidates(self, block_n_pad: int) -> list[int]:
        if self.tune_args.BLOCK_M is not None:
            return [self.tune_args.BLOCK_M]
        return self._power_of_two_candidates(max(1, _default_block_m(block_n_pad)), minimum=1)

    def _block_n_candidates(self, block_n_pad: int) -> list[int]:
        if self.tune_args.BLOCK_N is not None:
            return [self.tune_args.BLOCK_N] if self.tune_args.BLOCK_N <= block_n_pad else []
        upper = min(block_n_pad, max(16, min(self.N, 128)))
        exact = min(self.N, block_n_pad)
        values = self._power_of_two_candidates(max(16, upper), minimum=16)
        if exact > 0 and exact <= block_n_pad and exact not in values:
            values.insert(0, exact)
        deduped: list[int] = []
        seen = set()
        for value in values:
            if value <= block_n_pad and value not in seen:
                deduped.append(value)
                seen.add(value)
        return deduped

    def _block_dmodel_candidates(self) -> list[int]:
        if self.tune_args.BLOCK_DMODEL is not None:
            return [self.tune_args.BLOCK_DMODEL]
        return [_select_block_dmodel(self.D)]

    def _num_warps(self, block_n_pad: int) -> int:
        if self.D <= 64 or block_n_pad >= 128:
            return 4
        return 8

    def _num_stages(self, block_n_pad: int) -> int:
        return 1 if block_n_pad >= 128 else 2

    def _build_candidate_launch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sm_scale: float,
        o: torch.Tensor,
        lse: torch.Tensor | None,
        config: dict[str, int | bool],
        launch_context: dict[str, object] | None = None,
    ) -> FlashAttentionLaunchSpec:
        grid = (triton.cdiv(self.M, int(config["BLOCK_M"])), self.B * self.H)
        kernel_args = (
            q,
            k,
            v,
            o,
            lse if lse is not None else 0,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            (lse.stride(0) if lse is not None else 0),
            (lse.stride(1) if lse is not None else 0),
            (lse.stride(2) if lse is not None else 0),
            self.B,
            self.H,
            self.M,
            self.N,
            self.D,
            sm_scale,
        )
        kernel_kwargs = {
            "causal": self.causal,
            "HAS_LSE": self.return_lse,
            "BLOCK_M": int(config["BLOCK_M"]),
            "BLOCK_N": int(config["BLOCK_N"]),
            "BLOCK_N_PAD": int(config["BLOCK_N_PAD"]),
            "BLOCK_DMODEL": int(config["BLOCK_DMODEL"]),
            "num_warps": int(config["num_warps"]),
            "num_stages": int(config["num_stages"]),
        }
        if launch_context is None:
            launch_context = self.build_compiling_context(
                grid=grid,
                kernel_args=kernel_args,
                kernel_kwargs=kernel_kwargs,
                attempted_shared_memory_bytes_per_block=estimate_flash_fwd_shared_bytes(
                    block_m=int(config["BLOCK_M"]),
                    block_n_pad=int(config["BLOCK_N_PAD"]),
                    block_dmodel=int(config["BLOCK_DMODEL"]),
                    element_size=q.element_size(),
                ),
            )
        return FlashAttentionLaunchSpec(
            grid=grid,
            kernel_args=kernel_args,
            kernel_kwargs=kernel_kwargs,
            launch_context=launch_context,
        )

    def build_compiling_context(
        self,
        *,
        grid: tuple[int, ...],
        kernel_args: tuple[object, ...],
        kernel_kwargs: dict[str, object],
        attempted_shared_memory_bytes_per_block: int,
    ) -> dict[str, object]:
        context: dict[str, object] = dict(self.sm_limits)
        num_warps = kernel_kwargs.get("num_warps")
        if num_warps is not None:
            context["threads_per_block"] = int(num_warps) * int(context["warp_size"])
        context["attempted_shared_memory_bytes_per_block"] = attempted_shared_memory_bytes_per_block
        try:
            compiled = _flash_attn_fwd_kernel.warmup(*kernel_args, grid=grid, **kernel_kwargs)
            compiled._init_handles()
            metadata = getattr(compiled, "metadata", None)
            metadata_num_warps = getattr(metadata, "num_warps", None) if metadata is not None else None
            if "threads_per_block" not in context and metadata_num_warps is not None:
                context["threads_per_block"] = int(metadata_num_warps) * int(context["warp_size"])
            compiled_shared = getattr(metadata, "shared", None) if metadata is not None else None
            if compiled_shared is not None:
                context["compiled_shared_memory_bytes_per_block"] = compiled_shared
                context["attempted_shared_memory_bytes_per_block"] = compiled_shared

            registers_per_thread = getattr(compiled, "n_regs", None)
            if registers_per_thread is not None:
                context["attempted_registers_per_thread"] = registers_per_thread
                threads_per_block = context.get("threads_per_block")
                if threads_per_block is not None:
                    attempted_registers = int(registers_per_thread) * int(threads_per_block)
                    context["attempted_registers_per_block"] = attempted_registers
                    context["attempted_register_file_size_bytes_per_block"] = attempted_registers * 4
        except Exception as exc:
            context["resource_introspection_error"] = f"{type(exc).__name__}: {exc}"
        return context

    def evaluate_candidate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sm_scale: float,
        o: torch.Tensor,
        lse: torch.Tensor | None,
        config: dict[str, int | bool],
    ) -> tuple[bool, FlashAttentionLaunchSpec]:
        launch_spec = self._build_candidate_launch(
            q,
            k,
            v,
            sm_scale=sm_scale,
            o=o,
            lse=lse,
            config=config,
        )

        shared_limit = self.sm_limits["available_shared_memory_bytes_per_sm"]
        register_limit = self.sm_limits["available_register_file_size_bytes_per_sm"]
        context = launch_spec.launch_context
        shared_ok = int(context.get("compiled_shared_memory_bytes_per_block", context["attempted_shared_memory_bytes_per_block"])) <= shared_limit
        register_ok = int(context.get("attempted_register_file_size_bytes_per_block", 0)) <= register_limit
        feasible = shared_ok and (register_ok or "attempted_register_file_size_bytes_per_block" not in context)
        return feasible, launch_spec

    def _candidate_configs(self) -> list[dict[str, int | bool]]:
        candidates: list[dict[str, int | bool]] = []
        for block_n_pad in self._block_n_pad_candidates():
            for block_n in self._block_n_candidates(block_n_pad):
                for block_m in self._block_m_candidates(block_n_pad):
                    for block_dmodel in self._block_dmodel_candidates():
                        if block_dmodel < self.D:
                            continue
                        candidates.append(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_N_PAD": block_n_pad,
                                "BLOCK_DMODEL": block_dmodel,
                                "num_warps": self._num_warps(block_n_pad),
                                "num_stages": self._num_stages(block_n_pad),
                            }
                        )
        return sorted(
            candidates,
            key=lambda cfg: (
                estimate_flash_fwd_shared_bytes(
                    block_m=int(cfg["BLOCK_M"]),
                    block_n_pad=int(cfg["BLOCK_N_PAD"]),
                    block_dmodel=int(cfg["BLOCK_DMODEL"]),
                    element_size=2,
                ),
                int(cfg["BLOCK_N"]),
                int(cfg["BLOCK_N_PAD"]),
                int(cfg["BLOCK_M"]),
            ),
            reverse=True,
        )

    def _validate_runtime_tensors(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        if tuple(q.shape) != self.q_shape or tuple(k.shape) != self.k_shape or tuple(v.shape) != self.v_shape:
            raise ValueError("Runtime tensor shapes do not match the tuner shapes")

    def build_launch_spec(
        self,
        *data: torch.Tensor,
        sm_scale: float,
        o: torch.Tensor,
        lse: torch.Tensor | None,
    ) -> FlashAttentionLaunchSpec:
        if len(data) != 3:
            raise ValueError("FlashAttentionTuner.build_launch_spec expects (q, k, v)")
        q, k, v = data
        self._validate_runtime_tensors(q, k, v)

        cached = self._config_cache.get(self._cache_key(q))
        if cached is not None:
            config, launch_context = cached
            launch_spec = self._build_candidate_launch(
                q,
                k,
                v,
                sm_scale=sm_scale,
                o=o,
                lse=lse,
                config=dict(config),
                launch_context=dict(launch_context),
            )
            return FlashAttentionLaunchSpec(
                grid=launch_spec.grid,
                kernel_args=launch_spec.kernel_args,
                kernel_kwargs=launch_spec.kernel_kwargs,
                launch_context=launch_spec.launch_context
            )

        best_spec: FlashAttentionLaunchSpec | None = None
        best_rank: tuple[int, int, int, int] | None = None
        fallback_context: dict[str, object] | None = None

        for config in self._candidate_configs():
            feasible, launch_spec = self.evaluate_candidate(
                q,
                k,
                v,
                sm_scale=sm_scale,
                o=o,
                lse=lse,
                config=config,
            )
            fallback_context = launch_spec.launch_context
            if not feasible:
                continue
            rank = (
                int(launch_spec.launch_context.get("compiled_shared_memory_bytes_per_block", launch_spec.launch_context["attempted_shared_memory_bytes_per_block"])),
                int(launch_spec.launch_context.get("attempted_register_file_size_bytes_per_block", 0)),
                int(config["BLOCK_N_PAD"]),
                int(config["BLOCK_M"]),
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_spec = launch_spec

        if best_spec is None:
            context = fallback_context or dict(self.sm_limits)
            context.setdefault(
                "attempted_shared_memory_bytes_per_block",
                estimate_flash_fwd_shared_bytes(
                    block_m=max(1, self.tune_args.BLOCK_M or 1),
                    block_n_pad=self.tune_args.BLOCK_N_PAD
                    or _next_power_of_two(self.tune_args.BLOCK_N or min(max(16, self.N), 128)),
                    block_dmodel=self.tune_args.BLOCK_DMODEL or _select_block_dmodel(self.D),
                    element_size=q.element_size(),
                ),
            )
            register_bytes = int(context.get("attempted_register_file_size_bytes_per_block", 0))
            if register_bytes > self.sm_limits["available_register_file_size_bytes_per_sm"]:
                raise OutOfResourcesWithDetail(
                    int(context.get("attempted_registers_per_block", 0)),
                    self.sm_limits["available_registers_per_sm"],
                    "registers",
                    kernel_name="_flash_attn_fwd_kernel",
                    launch_context=context,
                    extra_detail="No candidate flash-attention block fit the one-SM register-file limit after lowering shared-memory usage",
                )
            raise OutOfResourcesWithDetail(
                int(context.get("attempted_shared_memory_bytes_per_block", 0)),
                self.sm_limits["available_shared_memory_bytes_per_sm"],
                "shared memory",
                kernel_name="_flash_attn_fwd_kernel",
                launch_context=context,
                extra_detail="No candidate flash-attention block fit the one-SM shared-memory limit",
            )

        self._config_cache[self._cache_key(q)] = (
            dict(best_spec.kernel_kwargs),
            dict(best_spec.launch_context)
        )
        return best_spec


def select_block_m_for_shared_memory(
    *,
    M: int,
    block_n: int=64,
    D: int=128,
    dtype: torch.dtype=torch.float16
) -> int:
    block_n_pad = _next_power_of_two(block_n)
    while block_n_pad > 0:
        block_dmodel = _select_block_dmodel(D)
        element_size = torch.empty((), dtype=dtype).element_size()
        max_shared_mem = _get_max_shared_mem_bytes()
        if M <= 0:
            raise ValueError("M must be positive")
        preferred = _default_block_m(block_n_pad)
        candidate = preferred
        while candidate >= 1:
            required = estimate_flash_fwd_shared_bytes(
                block_m=candidate,
                block_n_pad=block_n_pad,
                block_dmodel=block_dmodel,
                element_size=element_size,
            )
            if required <= max_shared_mem:
                return candidate, block_n_pad
            candidate //= 2
        block_n_pad //= 2

    min_required = estimate_flash_fwd_shared_bytes(
        block_m=1,
        block_n_pad=block_n_pad,
        block_dmodel=block_dmodel,
        element_size=element_size,
    )
    raise OutOfResourcesWithDetail(
        min_required,
        max_shared_mem,
        "shared memory",
        kernel_name="_flash_attn_fwd_kernel",
        launch_context={
            **get_sm_resource_limits(),
            "attempted_shared_memory_bytes_per_block": min_required,
        },
        extra_detail=(
            "No valid block configuration fit within the shared-memory budget for one SM-resident block "
            f"(BLOCK_M=1, BLOCK_N_PAD={block_n_pad}, BLOCK_DMODEL={block_dmodel}, dtype={dtype})"
        ),
    )


def _next_power_of_two(x: int) -> int:
    if x <= 0:
        raise ValueError("x must be positive")
    return 1 << (x - 1).bit_length()


def _select_block_dmodel(d: int) -> int:
    for candidate in (16, 32, 64, 128, 256):
        if d <= candidate:
            return candidate
    raise ValueError(f"Head dimension {d} is too large; expected <= 256")


def flash_attention_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    return_lse: bool = False,
    block_n: int | None = None,
    block_m: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    FlashAttention-2 style forward kernel implemented with OpenAI Triton.

    Expected tensor layout:
    - q: [B, H, M, D]
    - k: [B, H, N, D]
    - v: [B, H, N, D]
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, v must be CUDA tensors")

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must all be 4D tensors shaped [B, H, seq_len, D]")

    B, H, M, D = q.shape
    Bk, Hk, N, Dk = k.shape
    Bv, Hv, Nv, Dv = v.shape

    if (B, H) != (Bk, Hk) or (B, H) != (Bv, Hv):
        raise ValueError("Batch and head dimensions must match across q, k, v")
    if D != Dk or D != Dv:
        raise ValueError("Last dimension (head size) must match across q, k, v")
    if N != Nv:
        raise ValueError("k and v sequence length must match")

    if sm_scale is None:
        sm_scale = D ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty_like(q)
    lse = torch.empty((B, H, M), device=q.device, dtype=torch.float32) if return_lse else None

    if block_m is not None:
        if int(block_m) <= 0:
            raise ValueError("block_m must be a positive integer")
        if _next_power_of_two(int(block_m)) != int(block_m):
            raise ValueError("block_m must be a power of 2")

    if (64 if block_n is None else int(block_n)) <= 0:
        raise ValueError("block_n must be a positive integer")

    tuner = FlashAttentionTuner(
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        causal=causal,
        return_lse=lse is not None,
        tune_args=FlashAttenTuneArguments(
            BLOCK_N=(int(block_n) if block_n is not None else None),
            BLOCK_M=(int(block_m) if block_m is not None else None),
        ),
    )
    launch_spec = tuner.build_launch_spec(
        q,
        k,
        v,
        sm_scale=sm_scale,
        o=o,
        lse=lse,
    )
    try:
        _flash_attn_fwd_kernel[launch_spec.grid](*launch_spec.kernel_args, **launch_spec.kernel_kwargs)
    except OutOfResources as exc:
        raise OutOfResourcesWithDetail.from_out_of_resources(
            exc,
            kernel_name="_flash_attn_fwd_kernel",
            launch_context=launch_spec.launch_context,
            extra_detail="Triton rejected this launch for a single block scheduled on one SM",
        ) from exc

    return (o, lse) if return_lse else o


__all__ = [
    "FlashAttenTuneArguments",
    "_flash_attn_fwd_kernel",
    "FlashAttentionLaunchSpec",
    "FlashAttentionTuner",
    "_get_max_shared_mem_bytes",
    "OutOfResourcesWithDetail",
    "select_block_m_for_shared_memory",
    "_select_block_dmodel",
    "flash_attention_2",
]