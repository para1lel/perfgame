"""BF16 矩阵乘法 — TIRx 模板。

注意：TIRx 附带的高性能 kernel 库面向 sm_100a，在 H100 上只能走通用
lowering，拿不到 Hopper 专用的 WGMMA/TMA 优化。同题分数会明显低于
TileLang，这是预期行为。
"""

import tvm
from tvm import tirx as T


@T.jit(target="cuda", target_format="cubin")
def matmul(A: T.Tensor((1024, 1024), "bfloat16"),
           B: T.Tensor((1024, 1024), "bfloat16"),
           C: T.Tensor((1024, 1024), "bfloat16"),
           block_M: T.int32 = 128, block_N: T.int32 = 128, block_K: T.int32 = 64):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
        A_s = T.alloc_shared((block_M, block_K), "bfloat16")
        B_s = T.alloc_shared((block_K, block_N), "bfloat16")
        C_l = T.alloc_fragment((block_M, block_N), "float32")

        T.clear(C_l)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_s)
            T.copy(B[ko * block_K, bx * block_N], B_s)
            T.gemm(A_s, B_s, C_l)

        T.copy(C_l, C[by * block_M, bx * block_N])


def solve(a, b):
    # 占位：适配器会按实际 shape 重新 JIT。见 docs/languages.md。
    raise NotImplementedError("由 evaluator.adapters.tirx 按 shape 注入")
