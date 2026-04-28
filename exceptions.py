from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import triton
from triton.runtime.errors import OutOfResources


def _format_tune_args(tune_args: Any | None) -> str | None:
    if tune_args is None:
        return None
    if is_dataclass(tune_args):
        values = asdict(tune_args)
    elif isinstance(tune_args, Mapping):
        values = dict(tune_args)
    else:
        values = {
            key: getattr(tune_args, key)
            for key in dir(tune_args)
            if key.isupper() and not key.startswith("_")
        }
    formatted = ", ".join(f"{key}={value}" for key, value in values.items() if value is not None)
    return formatted or None


def _format_input_shapes(input_shapes: Any | None) -> str | None:
    if input_shapes is None:
        return None
    if isinstance(input_shapes, Mapping):
        values = dict(input_shapes)
    else:
        values = {"input_shapes": input_shapes}
    formatted = ", ".join(f"{key}={value}" for key, value in values.items() if value is not None)
    return formatted or None


def _resource_usage_summary(
    name: str,
    launch_context: Mapping[str, Any] | None,
    tune_args: Any | None = None,
    input_shapes: Any | None = None,
) -> str:
    launch_context = launch_context or {}
    shared_used = int(
        launch_context.get(
            "compiled_shared_memory_bytes_per_block",
            launch_context.get("attempted_shared_memory_bytes_per_block", 0),
        )
    )
    shared_total = int(launch_context.get("available_shared_memory_bytes_per_sm", 0))
    register_file_used = int(launch_context.get("attempted_register_file_size_bytes_per_block", 0))
    register_file_total = int(launch_context.get("available_register_file_size_bytes_per_sm", 0))
    registers_used = int(launch_context.get("attempted_registers_per_block", 0))
    registers_total = int(launch_context.get("available_registers_per_sm", 0))
    summary = (
        f"{name} tuner resources: \n"
        f"shared={shared_used}/{shared_total} bytes per block/SM, \n"
        f"register_file={register_file_used}/{register_file_total} bytes per block/SM, \n"
        f"registers={registers_used}/{registers_total} per block/SM "
    )
    formatted_tune_args = _format_tune_args(tune_args)
    if formatted_tune_args:
        summary += f", \ntune_args={formatted_tune_args}"
    formatted_input_shapes = _format_input_shapes(input_shapes)
    if formatted_input_shapes:
        summary += f", \ninput_shapes={formatted_input_shapes}"
    return summary


def _resource_failure_text(prefix: str, exc: "OutOfResourcesWithDetail") -> str:
    launch_context = getattr(exc, "launch_context", {}) or {}
    return f"{prefix}: {exc} [{_resource_usage_summary('failed', launch_context)}]"




class OutOfResourcesWithDetail(OutOfResources):
    """Richer Triton resource error with kernel and block-launch context."""

    def __init__(
        self,
        required: int,
        limit: int,
        name: str,
        *,
        kernel_name: str | None = None,
        launch_context: Mapping[str, Any] | None = None,
        extra_detail: str | None = None,
        original_exception: BaseException | None = None,
    ) -> None:
        super().__init__(required, limit, name)
        self.kernel_name = kernel_name
        self.launch_context = dict(launch_context or {})
        self.extra_detail = extra_detail
        self.original_exception = original_exception

    @classmethod
    def from_out_of_resources(
        cls,
        exc: OutOfResources,
        *,
        kernel_name: str | None = None,
        launch_context: Mapping[str, Any] | None = None,
        extra_detail: str | None = None,
    ) -> "OutOfResourcesWithDetail":
        return cls(
            exc.required,
            exc.limit,
            exc.name,
            kernel_name=kernel_name,
            launch_context=launch_context,
            extra_detail=extra_detail,
            original_exception=exc,
        )

    def _resource_hint(self) -> str:
        resource = str(self.name).lower()
        if "register" in resource:
            return "The block likely needs more register file capacity per SM than the hardware can provide."
        if "shared" in resource:
            return "The block likely needs more shared memory per SM than the hardware can provide."
        return "The block likely exceeds a per-SM hardware resource limit for this launch configuration."

    def _format_context(self) -> str | None:
        if not self.launch_context:
            return None
        parts = [f"{key}={value}" for key, value in self.launch_context.items()]
        return "\n".join(parts)

    def __str__(self) -> str:
        parts = [super().__str__(), self._resource_hint()]
        if self.kernel_name:
            parts.append(f"Kernel: {self.kernel_name}.")
        context = self._format_context()
        if context:
            parts.append(f"Launch context: {context}.")
        if self.extra_detail:
            parts.append(self.extra_detail.rstrip(".") + ".")
        return " ".join(parts)


def get_sm_resource_limits() -> dict[str, Any]:
    device = triton.runtime.driver.active.get_current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device)
    max_num_regs = props["max_num_regs"]
    return {
        "available_shared_memory_bytes_per_sm": props["max_shared_mem"],
        "available_registers_per_sm": max_num_regs,
        "available_register_file_size_bytes_per_sm": max_num_regs * 4,
        "warp_size": props["warpSize"],
        "multiprocessor_count": props["multiprocessor_count"],
    }


def estimate_flash_fwd_shared_bytes(
    *,
    block_m: int,
    block_n_pad: int,
    block_dmodel: int,
    element_size: int,
) -> int:
    # Triton stages q, k, and v tiles in shared memory for the two tl.dot calls.
    return element_size * block_dmodel * (block_m + 2 * block_n_pad)