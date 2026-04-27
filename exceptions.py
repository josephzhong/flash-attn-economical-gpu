from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import triton
from triton.runtime.errors import OutOfResources


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