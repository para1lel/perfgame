"""理论下界：与算法无关，只与 shape 有关。

写清楚每个数字怎么来的，比写对更重要——题目作者日后要能核对。
"""


def compute_bounds(shape: dict[str, int]) -> dict[str, float]:
    n = shape["N"]
    return {
        # 读 A(4N) + 读 B(4N) + 写 C(4N)
        "hbm_bytes": 12.0 * n,
        # 一次加法走 CUDA core 的 FP32 管线；不用张量核
        "flops_fp32": float(n),
    }
