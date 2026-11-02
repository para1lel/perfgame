"""理论下界：与算法无关，只与 shape 有关。"""


def compute_bounds(shape: dict[str, int]) -> dict[str, float]:
    m, n, k = shape["M"], shape["N"], shape["K"]
    return {
        # 读 A(MK) + 读 B(KN) + 写 C(MN)，BF16 每元素 2 字节
        "hbm_bytes": 2.0 * (m * k + k * n + m * n),
        # 每个输出元素 K 次乘加，乘法与加法各计 1 FLOP
        "flops_bf16": 2.0 * m * n * k,
    }
