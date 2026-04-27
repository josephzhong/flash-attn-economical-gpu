from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


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