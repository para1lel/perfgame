"""BF16 矩阵乘法 — TileLang 模板。

只改 `matmul` 的 kernel 体与 block 参数；`solve` 由评测器约定，不要改。
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M: int, N: int, K: int, block_M: int = 128, block_N: int = 128, block_K: int = 64):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((K, N), "bfloat16"),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "bfloat16")
            B_shared = T.alloc_shared((block_K, block_N), "bfloat16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def solve(a, b):
    m, k = a.shape
    _, n = b.shape
    return matmul(m, n, k)(a, b)
