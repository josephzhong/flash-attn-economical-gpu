from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch
import triton

from exceptions import (
    OutOfResourcesWithDetail,
    get_sm_resource_limits,
)


@dataclass(frozen=True)
class LaunchSpec:
    grid: tuple[int, ...]
    kernel_args: tuple[object, ...]
    kernel_kwargs: dict[str, object]
    launch_context: dict[str, object]


class Tuner(ABC):
    @abstractmethod
    def build_launch_spec(self, *data: torch.Tensor, **kwargs: object) -> LaunchSpec:
        """Build a fully materialized launch spec from runtime tensors."""

    @abstractmethod
    def build_compiling_context(self, **kwargs: object) -> dict[str, object]:
        """Build resource/compilation context for a candidate launch."""

    @abstractmethod
    def evaluate_candidate(self, *data: torch.Tensor, **kwargs: object) -> tuple[bool, LaunchSpec]:
        """Evaluate a candidate launch and return feasibility plus its launch spec."""


def _next_power_of_two(x: int) -> int:
    if x <= 0:
        raise ValueError("x must be positive")
    return 1 << (x - 1).bit_length()


def _get_max_shared_mem_bytes() -> int:
    device = triton.runtime.driver.active.get_current_device()
    return triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]


@dataclass(frozen=True)
class FlashAttentionLaunchSpec(LaunchSpec):
    function_name: str
    tune_args: FlashAttenTuneArguments
    input_shapes: dict[str, tuple[int, ...] | None]


@dataclass(frozen=True)
class FlashAttenTuneArguments:
    BLOCK_N: int | None = None
    BLOCK_M: int | None = None
    BLOCK_N_PAD: int | None = None
    BLOCK_DMODEL: int | None = None
    NUM_WARPS: int | None = None
    NUM_STAGES: int | None = None

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
        if self.NUM_WARPS is not None and self.NUM_WARPS <= 0:
            raise ValueError("NUM_WARPS must be positive when provided")
        if self.NUM_STAGES is not None and self.NUM_STAGES <= 0:
            raise ValueError("NUM_STAGES must be positive when provided")


class FlashAttentionTuner(Tuner):
    KERNEL_NAME = "_flash_attn_fwd_kernel"
    FUNCTION_NAME = "flash_attention_2"
    _config_cache: dict[tuple[object, ...], tuple[dict[str, int | bool], dict[str, object]]] = {}

    def __init__(
        self,
        q_shape: tuple[int, int, int, int],
        k_shape: tuple[int, int, int, int],
        v_shape: tuple[int, int, int, int],
        dtype_size: int,
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
        self.dtype_size = dtype_size
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
        max_warps = max(1, int(self.sm_limits.get("max_warps_per_block", 1)))
        if self.tune_args.NUM_WARPS is not None and self.tune_args.NUM_WARPS > max_warps:
            raise ValueError(f"NUM_WARPS must be <= {max_warps} for the current device")

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
            self.tune_args.NUM_WARPS,
            self.tune_args.NUM_STAGES,
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

    def _block_n_pad_candidates(self, block_d_model: int) -> list[int]:
        if self.tune_args.BLOCK_N_PAD is not None:
            return [self.tune_args.BLOCK_N_PAD]
        min_pad = _next_power_of_two(self.tune_args.BLOCK_N) if self.tune_args.BLOCK_N is not None else 16
        max_shared_memory_element_cnt = _get_max_shared_mem_bytes() / self.dtype_size
        max_pad = _next_power_of_two(int(max_shared_memory_element_cnt / block_d_model / 4))
        if min_pad > max_pad:
            max_pad = min_pad
        return [pad for pad in self._power_of_two_candidates(max_pad, minimum=16) if pad >= min_pad]

    def _block_m_candidates(self, block_n_pad: int) -> list[int]:
        if self.tune_args.BLOCK_M is not None:
            return [self.tune_args.BLOCK_M]
        return [block_n_pad * 2]

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
        return [max(16, _next_power_of_two(self.D))]

    def _supported_num_warps(self) -> list[int]:
        max_warps = max(1, int(self.sm_limits.get("max_warps_per_block", 1)))
        return [warps for warps in (1, 2, 4, 8, 16, 32) if warps <= max_warps]

    def _num_warps(self, block_m: int, block_n: int, block_n_pad: int, block_dmodel: int) -> int:
        if self.tune_args.NUM_WARPS is not None:
            return int(self.tune_args.NUM_WARPS)
        supported = self._supported_num_warps()
        if not supported:
            return 1
        tile_work = (
            int(block_m) * int(block_n_pad)
            + int(block_n_pad) * int(block_dmodel)
            + int(block_m) * int(block_dmodel)
        )
        logical_span = max(int(block_n), int(block_n_pad))
        estimated = max(1, math.ceil(max(tile_work, int(block_m) * logical_span) / 4096))
        for warps in supported:
            if warps >= estimated:
                return warps
        return supported[-1]

    def _num_stages(self, block_n_pad: int) -> int:
        if self.tune_args.NUM_STAGES is not None:
            return int(self.tune_args.NUM_STAGES)
        return 1

    def estimate_flash_fwd_shared_bytes(
        self,
        *,
        block_m: int,
        block_n_pad: int,
        block_dmodel: int,
        element_size: int,
    ) -> int:
        return element_size * block_dmodel * (block_m + 2 * block_n_pad)

    def _kernel(self):
        import flash_atten

        return getattr(flash_atten, self.KERNEL_NAME)

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
                attempted_shared_memory_bytes_per_block=self.estimate_flash_fwd_shared_bytes(
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
            function_name=self.FUNCTION_NAME,
            tune_args=FlashAttenTuneArguments(
                BLOCK_N=int(config["BLOCK_N"]),
                BLOCK_M=int(config["BLOCK_M"]),
                BLOCK_N_PAD=int(config["BLOCK_N_PAD"]),
                BLOCK_DMODEL=int(config["BLOCK_DMODEL"]),
                NUM_WARPS=int(config["num_warps"]),
                NUM_STAGES=int(config["num_stages"]),
            ),
            input_shapes={
                "q": tuple(int(dim) for dim in q.shape),
                "k": tuple(int(dim) for dim in k.shape),
                "v": tuple(int(dim) for dim in v.shape),
                "o": tuple(int(dim) for dim in o.shape),
                "lse": (tuple(int(dim) for dim in lse.shape) if lse is not None else None),
            },
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
            compiled = self._kernel().warmup(*kernel_args, grid=grid, **kernel_kwargs)
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
        shared_ok = (
            int(
                context.get(
                    "compiled_shared_memory_bytes_per_block",
                    context["attempted_shared_memory_bytes_per_block"],
                )
            )
            <= shared_limit
        )
        register_ok = int(context.get("attempted_register_file_size_bytes_per_block", 0)) <= register_limit
        feasible = shared_ok and (register_ok or "attempted_register_file_size_bytes_per_block" not in context)
        return feasible, launch_spec

    def _candidate_configs(self) -> list[dict[str, int | bool]]:
        candidates: list[dict[str, int | bool]] = []
        for block_dmodel in self._block_dmodel_candidates():
            if block_dmodel < self.D:
                continue
            for block_n_pad in self._block_n_pad_candidates(block_dmodel):
                for block_m in self._block_m_candidates(block_n_pad):
                    for block_n in self._block_n_candidates(block_n_pad):
                        candidates.append(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_N_PAD": block_n_pad,
                                "BLOCK_DMODEL": block_dmodel,
                                "num_warps": self._num_warps(block_m, block_n, block_n_pad, block_dmodel),
                                "num_stages": self._num_stages(block_n_pad),
                            }
                        )
        return sorted(
            candidates,
            key=lambda cfg: (
                self.estimate_flash_fwd_shared_bytes(
                    block_m=int(cfg["BLOCK_M"]),
                    block_n_pad=int(cfg["BLOCK_N_PAD"]),
                    block_dmodel=int(cfg["BLOCK_DMODEL"]),
                    element_size=self.dtype_size,
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
            selected_config, selected_context = cached
            selected_config = dict(selected_config)
            selected_context = dict(selected_context)
            launch_spec = self._build_candidate_launch(
                q,
                k,
                v,
                sm_scale=sm_scale,
                o=o,
                lse=lse,
                config=selected_config,
                launch_context=selected_context,
            )
        else:
            best_spec: FlashAttentionLaunchSpec | None = None
            best_config: dict[str, int | bool] | None = None
            best_rank: tuple[int, int, int, int] | None = None
            fallback_launch_spec: FlashAttentionLaunchSpec | None = None
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
                fallback_launch_spec = launch_spec
                fallback_context = launch_spec.launch_context
                if not feasible:
                    continue
                rank = (
                    int(
                        launch_spec.launch_context.get(
                            "compiled_shared_memory_bytes_per_block",
                            launch_spec.launch_context["attempted_shared_memory_bytes_per_block"],
                        )
                    ),
                    int(launch_spec.launch_context.get("attempted_register_file_size_bytes_per_block", 0)),
                    int(config["BLOCK_N_PAD"]),
                    int(config["BLOCK_M"]),
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_spec = launch_spec
                    best_config = dict(config)

            if best_spec is None or best_config is None:
                context = fallback_context or dict(self.sm_limits)
                fallback_tune_args = fallback_launch_spec.tune_args if fallback_launch_spec is not None else None
                context.setdefault(
                    "attempted_shared_memory_bytes_per_block",
                    self.estimate_flash_fwd_shared_bytes(
                        block_m=max(1, int(getattr(fallback_tune_args, "BLOCK_M", None) or self.tune_args.BLOCK_M)),
                        block_n_pad=int(
                            getattr(fallback_tune_args, "BLOCK_N_PAD", None)
                            or self.tune_args.BLOCK_N_PAD
                        ),
                        block_dmodel=int(
                            getattr(fallback_tune_args, "BLOCK_DMODEL", None)
                            or self.tune_args.BLOCK_DMODEL
                        ),
                        element_size=q.element_size(),
                    ),
                )
                register_bytes = int(context.get("attempted_register_file_size_bytes_per_block", 0))
                if register_bytes > self.sm_limits["available_register_file_size_bytes_per_sm"]:
                    raise OutOfResourcesWithDetail(
                        int(context.get("attempted_registers_per_block", 0)),
                        self.sm_limits["available_registers_per_sm"],
                        "registers",
                        kernel_name=self.KERNEL_NAME,
                        launch_context=context,
                        extra_detail="No candidate flash-attention block fit the one-SM register-file limit after lowering shared-memory usage",
                    )
                raise OutOfResourcesWithDetail(
                    int(context.get("attempted_shared_memory_bytes_per_block", 0)),
                    self.sm_limits["available_shared_memory_bytes_per_sm"],
                    "shared memory",
                    kernel_name=self.KERNEL_NAME,
                    launch_context=context,
                    extra_detail="No candidate flash-attention block fit the one-SM shared-memory limit",
                )

            launch_spec = best_spec
            self._config_cache[self._cache_key(q)] = (
                dict(best_config),
                dict(best_spec.launch_context),
            )
        return launch_spec

    def build_flash_attention_launch_spec(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sm_scale: float | None = None,
    ) -> FlashAttentionLaunchSpec:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if sm_scale is None:
            sm_scale = q.shape[-1] ** -0.5
        o = torch.empty_like(q)
        lse = (
            torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
            if self.return_lse
            else None
        )
        launch_spec = self.build_launch_spec(
            q,
            k,
            v,
            sm_scale=sm_scale,
            o=o,
            lse=lse
        )
        return launch_spec


__all__ = [
    "FlashAttenTuneArguments",
    "FlashAttentionLaunchSpec",
    "FlashAttentionTuner",
    "LaunchSpec",
    "Tuner",
    "_get_max_shared_mem_bytes",
    "_next_power_of_two",
]