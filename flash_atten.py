import logging
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from tuner import (
    FlashAttentionLaunchSpec
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
    BLOCK_N_PAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

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


@triton.jit
def _flash_attn_fwd_kernel_block_d_nomask_contiguous(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_OUTER: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh - b * H
    bh_offset = b * H + h

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_outer_d = tl.arange(0, BLOCK_DMODEL_OUTER)
    
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_M, BLOCK_DMODEL_OUTER), dtype=tl.float32)
    if BLOCK_DMODEL_OUTER <= D / 2:
        acc1 = tl.zeros((BLOCK_M, BLOCK_DMODEL_OUTER), dtype=tl.float32)
        out_d1 = BLOCK_DMODEL_OUTER + offs_outer_d
    if BLOCK_DMODEL_OUTER <= D / 4:
        acc2 = tl.zeros((BLOCK_M, BLOCK_DMODEL_OUTER), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_M, BLOCK_DMODEL_OUTER), dtype=tl.float32)
        out_d2 = BLOCK_DMODEL_OUTER * 2 + offs_outer_d
        out_d3 = BLOCK_DMODEL_OUTER * 3 + offs_outer_d

    for start_n in range(0, N, BLOCK_N):
        cur_n = start_n + offs_n
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for qk_start_d in range(0, D, BLOCK_DMODEL):
            cur_d = qk_start_d + offs_d
            q_ptrs = q_ptr + (bh_offset * M + offs_m[:, None]) * D + cur_d[None, :]
            k_ptrs = k_ptr + (bh_offset * N + cur_n[:, None]) * D + cur_d[None, :]
            q = tl.load(q_ptrs)
            k = tl.load(k_ptrs)
            qk += tl.dot(q, tl.trans(k), out_dtype=tl.float32)

        qk *= sm_scale
        row_max = tl.max(qk, axis=1)
        m_ij = tl.maximum(m_i, row_max)
        p = tl.exp(qk - m_ij[:, None])
        v0_ptrs = v_ptr + (bh_offset * N + cur_n[:, None]) * D + offs_outer_d[None, :]
        v0 = tl.load(v0_ptrs)
        if BLOCK_DMODEL_OUTER <= D / 2:
            v1_ptrs = v_ptr + (bh_offset * N + cur_n[:, None]) * D + out_d1[None, :]
            v1 = tl.load(v1_ptrs)
        if BLOCK_DMODEL_OUTER <= D / 4:
            v2_ptrs = v_ptr + (bh_offset * N + cur_n[:, None]) * D + out_d2[None, :]
            v3_ptrs = v_ptr + (bh_offset * N + cur_n[:, None]) * D + out_d3[None, :]
            v2 = tl.load(v2_ptrs)
            v3 = tl.load(v3_ptrs)
        
        pv0 = tl.dot(p.to(v0.dtype), v0, out_dtype=tl.float32)
        if BLOCK_DMODEL_OUTER <= D / 2:
            pv1 = tl.dot(p.to(v1.dtype), v1, out_dtype=tl.float32)
        if BLOCK_DMODEL_OUTER <= D / 4:
            pv2 = tl.dot(p.to(v2.dtype), v2, out_dtype=tl.float32)
            pv3 = tl.dot(p.to(v3.dtype), v3, out_dtype=tl.float32)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc0 = acc0 * alpha[:, None] + pv0
        if BLOCK_DMODEL_OUTER <= D / 2:
            acc1 = acc1 * alpha[:, None] + pv1
        if BLOCK_DMODEL_OUTER <= D / 4:
            acc2 = acc2 * alpha[:, None] + pv2
            acc3 = acc3 * alpha[:, None] + pv3
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    out0 = acc0 / l_i[:, None]
    if BLOCK_DMODEL_OUTER <= D / 2:
        out1 = acc1 / l_i[:, None]
    if BLOCK_DMODEL_OUTER <= D / 4:
        out2 = acc2 / l_i[:, None]
        out3 = acc3 / l_i[:, None]
    o0_ptrs = o_ptr + (bh_offset * M + offs_m[:, None]) * D + offs_outer_d[None, :]
    if BLOCK_DMODEL_OUTER <= D / 2:
        o1_ptrs = o_ptr + (bh_offset * M + offs_m[:, None]) * D + out_d1[None, :]
    if BLOCK_DMODEL_OUTER <= D / 4:
        o2_ptrs = o_ptr + (bh_offset * M + offs_m[:, None]) * D + out_d2[None, :]
        o3_ptrs = o_ptr + (bh_offset * M + offs_m[:, None]) * D + out_d3[None, :]
    tl.store(o0_ptrs, out0)
    if BLOCK_DMODEL_OUTER <= D / 2:
        tl.store(o1_ptrs, out1)
    if BLOCK_DMODEL_OUTER <= D / 4:
        tl.store(o2_ptrs, out2)
        tl.store(o3_ptrs, out3)


flash_attention_kernel_mapping = {
    "_flash_attn_fwd_kernel_basic": _flash_attn_fwd_kernel_basic,
    "_flash_attn_fwd_kernel_optimize_shared_mem": _flash_attn_fwd_kernel_optimize_shared_mem,
    "_flash_attn_fwd_kernel_block_d_nomask_contiguous": _flash_attn_fwd_kernel_block_d_nomask_contiguous
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