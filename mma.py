import logging

import triton
import triton.language as tl

@triton.jit
def _matmul_contiguous_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, cache_modifier=".ca", eviction_policy="evict_first")
        b = tl.load(b_ptrs, cache_modifier=".ca", eviction_policy="evict_last")
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, accumulator.to(tl.float16), cache_modifier=".cg")


def _select_2d_contiguous_config(m: int, n: int, k: int) -> tuple[int, int, int, int, int, int]:
    if k % 32 != 0:
        raise ValueError("fast 2D contiguous matmul requires K % 32 == 0")
    if m >= 2048 and n >= 2048 and m % 128 == 0 and n % 128 == 0:
        return 128, 128, 32, 8, 4, 3
    if m % 128 == 0 and n % 64 == 0:
        return 128, 64, 32, 8, 4, 3
    if m % 128 == 0 and n % 128 == 0:
        return 128, 128, 32, 8, 4, 3
    if m % 64 == 0 and n % 256 == 0:
        return 64, 256, 32, 2, 8, 5
    raise ValueError(
        "fast 2D contiguous matmul requires a supported divisible tile: "
        "(M % 128 == 0 and N % 64 == 0), "
        "(M % 128 == 0 and N % 128 == 0), or "
        "(M % 64 == 0 and N % 256 == 0), with K % 32 == 0"
    )

__all__ = [
    "_select_2d_contiguous_config",
    "_matmul_contiguous_kernel",
]
