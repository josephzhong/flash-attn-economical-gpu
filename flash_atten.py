import logging
from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra import cuda as tl_cuda
from triton.runtime.errors import OutOfResources

from exceptions import (
    OutOfResourcesWithDetail,
    _resource_usage_summary,
    estimate_flash_fwd_shared_bytes,
    get_sm_resource_limits,
)
from tuner import LaunchSpec, Tuner
from tuner import (
    FlashAttenTuneArguments,
    FlashAttentionLaunchSpec,
    FlashAttentionTuner,
)

logger = logging.getLogger(__name__)


@triton.jit
def _flash_attn_fwd_kernel_basic(
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


@triton.jit
def _flash_attn_fwd_kernel_optimize_shared_mem(
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
    H,
    M,
    N,
    D,
    sm_scale,
    causal: tl.constexpr,
    HAS_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_PAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    load_q_mask = True
    
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N_PAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    q_load_start_t = 0
    
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        cur_n = start_n + offs_n
        block_n_mask = cur_n < (start_n + BLOCK_N)

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

        kv_mask = block_n_mask[:, None] & (cur_n[:, None] < N) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        in_bounds = (offs_m[:, None] < M) & block_n_mask[None, :] & (cur_n[None, :] < N)
        qk = tl.where(in_bounds, qk, float("-inf"))

        if causal:
            causal_mask = offs_m[:, None] >= cur_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        row_max = tl.max(qk, axis=1)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        m_ij = tl.maximum(m_i, row_max)
        p = tl.exp(qk - m_ij[:, None])
        
        pv = tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc = acc + pv
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

flash_attention_kernel_mapping = {
    "_flash_attn_fwd_kernel_basic": _flash_attn_fwd_kernel_basic,
    "_flash_attn_fwd_kernel_optimize_shared_mem": _flash_attn_fwd_kernel_optimize_shared_mem
}

def flash_attention(
    launch_spec: FlashAttentionLaunchSpec,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    FlashAttention-2 style forward kernel implemented with OpenAI Triton.
    """
    if launch_spec.function_name in flash_attention_kernel_mapping:
        flash_attention_kernel_mapping[launch_spec.function_name][launch_spec.grid](*launch_spec.kernel_args, **launch_spec.kernel_kwargs)
    else:
        raise NameError(f"No such kernel named {launch_spec.function_name}.")

    o = launch_spec.kernel_args[3]
    lse = launch_spec.kernel_args[4] if isinstance(launch_spec.kernel_args[4], torch.Tensor) else None
    return (o, lse) if lse is not None else o


__all__ = [
    "flash_attention_kernel_mapping",
    "flash_attention",
]