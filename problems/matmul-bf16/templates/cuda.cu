// BF16 矩阵乘法 — CUDA 模板
//
// 入口签名由评测器约定，不要修改：
//   args[0] = A, args[1] = B, args[2] = C（裸指针）
//   shape = {M, K, N}，顺序与 problem.yaml 的 signature 一致
//
// 允许 include CUTLASS 头文件。编译选项见 docs/languages.md。

#include <cuda_bf16.h>
#include <cstdint>

__global__ void matmul_kernel(const __nv_bfloat16* __restrict__ A,
                              const __nv_bfloat16* __restrict__ B,
                              __nv_bfloat16* __restrict__ C,
                              int M, int N, int K) {
    // TODO: 在这里写你的 kernel
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int p = 0; p < K; ++p) {
        acc += __bfloat162float(A[row * K + p]) * __bfloat162float(B[p * N + col]);
    }
    C[row * N + col] = __float2bfloat16(acc);
}

extern "C" void run_kernel(const void* const* args, const int64_t* shape, int nargs) {
    const auto* A = static_cast<const __nv_bfloat16*>(args[0]);
    const auto* B = static_cast<const __nv_bfloat16*>(args[1]);
    auto* C = static_cast<__nv_bfloat16*>(args[2]);
    int M = static_cast<int>(shape[0]);
    int K = static_cast<int>(shape[1]);
    int N = static_cast<int>(shape[2]);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
