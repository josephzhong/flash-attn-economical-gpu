import torch
import triton
import triton.language as tl


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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    FlashAttention-2 style forward kernel implemented with OpenAI Triton.

    Expected tensor layout: [B, H, S, D]
    - q: [B, H, M, D]
    - k: [B, H, N, D]
    - v: [B, H, N, D]
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, v must be CUDA tensors")

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must all be 4D tensors shaped [B, H, S, D]")

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

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = _select_block_dmodel(D)

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    _flash_attn_fwd_kernel[grid](
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
        B,
        H,
        M,
        N,
        D,
        sm_scale,
        causal=causal,
        HAS_LSE=lse is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4 if D <= 64 else 8,
        num_stages=2,
    )

    return (o, lse) if return_lse else o


__all__ = ["flash_attention_2"]