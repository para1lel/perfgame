"""BF16 矩阵乘法 — CuTe DSL 模板。

编译产物只有 PTX 与 SASS；CuTe DSL 无 CUDA C++ 中间产物，因此不提供
"CUDA 源码"标签。
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def matmul_kernel(A, B, C, M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32):
    # TODO: 参考 CUTLASS 的 Hopper GEMM 示例补全
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    row = by * 16 + ty
    col = bx * 16 + tx
    if row >= M or col >= N:
        return
    acc = cutlass.Float32(0.0)
    for p in range(K):
        acc += A[row, p].to(cutlass.Float32) * B[p, col].to(cutlass.Float32)
    C[row, col] = acc.to(cutlass.BFloat16)


@cute.jit
def solve(a, b, c):
    m, k = a.shape
    _, n = b.shape
    matmul_kernel(a, b, c, m, n, k).launch(
        grid=((n + 15) // 16, (m + 15) // 16, 1), block=(16, 16, 1)
    )
