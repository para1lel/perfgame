"""参考实现：C = A + B（FP32）。

语义基准。评测时不调用它——部署时由 `manage generate-groundtruth` 算出并落盘。
"""

import torch


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b
