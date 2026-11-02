"""参考实现：C = A @ B（BF16 输入输出，FP64 内部累加）。

FP64 累加是允许的——ground truth 需要比被测 kernel 更准，
否则参考实现自身的误差会吃掉容差。输出以题目声明的 dtype 返回。
"""

import torch


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.double() @ b.double()).float().to(torch.bfloat16)
