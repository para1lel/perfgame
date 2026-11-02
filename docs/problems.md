# 题目

一道题目是一个目录，位于 `problems/<slug>/`。题目是代码，走 Git，由项目维护者管理，没有后台管理界面。

## 目录结构

```
problems/matmul-bf16/
├── problem.yaml          # 元数据、签名、测试点、容差
├── reference.py          # PyTorch 参考实现（语义依据）
├── compute_bounds.py     # 理论下界函数
├── editorial.md          # 可选：题解、提示
└── templates/
    ├── cuda.cu
    ├── tilelang.py
    ├── tirx.py
    └── cutedsl.py
```

`templates/` 可省略，此时用 `evaluator/contracts/` 里的通用骨架。

## `problem.yaml`

```yaml
slug: matmul-bf16
title: BF16 矩阵乘法
category: matmul          # matmul | elementwise | reduction | attention | conv | other
difficulty: medium        # easy | medium | hard

# 一句话说明，显示在列表页
summary: 计算 C = A @ B，BF16 输入输出。

# 题面正文，Markdown + LaTeX（$...$ 与 $$...$$）
description: |
  给定 A (M×K) 与 B (K×N)，计算 C = A @ B。
  输出 C 为 (M×N)，BF16。

  语义以 `reference.py` 为准。

# 输入输出签名。shape 里用占位符，具体值由测试点给出
signature:
  inputs:
    - { name: A, shape: [M, K], dtype: bfloat16, layout: contiguous }
    - { name: B, shape: [K, N], dtype: bfloat16, layout: contiguous }
  outputs:
    - { name: C, shape: [M, N], dtype: bfloat16 }

# 工作负载
test_cases:
  - { name: small,  shape: { M: 256,  K: 256,  N: 256 },  seed: 11 }
  - { name: medium, shape: { M: 1024, K: 1024, N: 1024 }, seed: 12 }
  - { name: large,  shape: { M: 4096, K: 4096, N: 4096 }, seed: 13 }

# 影子测试点：shape 与上述全不同，用于挡编译期特化 shape 的作弊
shadow_shape: { M: 1536, K: 768, N: 2048 }
shadow_seed: 99

# 容差，见 evaluation.md
tolerance:
  elementwise: { atol: 1.0e-2, rtol: 1.0e-2, min_correct_ratio: 0.99 }
  cosine_similarity: { min: 0.999 }

# 输入数据生成方式：uniform（默认）| normal | integer | custom
input_generator: normal
```

### 字段说明

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `slug` | 是 | 与目录名一致；只允许 `[a-z0-9-]` |
| `category` | 是 | 前端分类过滤用；枚举值见上 |
| `difficulty` | 是 | 只影响列表页的排序权重 |
| `description` | 是 | Markdown；会被渲染进题面左栏 |
| `signature` | 是 | shape 占位符必须能在每个 `test_cases[].shape` 与 `shadow_shape` 里找到值 |
| `test_cases` | 是 | 至少 1 个；`name` 在同一题内唯一 |
| `shadow_shape` | 是 | 与任何测试点不同；不计分，仅做正确性校验 |
| `tolerance` | 是 | |
| `input_generator` | 否 | `normal` 默认（`N(0,1)`）；`uniform` 为 `U(-1,1)`；`integer` 给量化题 |

### 输入数据

评测时不重新随机。部署时 `generate-groundtruth` 用 `torch.Generator().manual_seed(seed)` 生成每个张量，存成 `.npy`。同一部署下所有提交看到的输入逐位相同。

生成顺序固定：按 `signature.inputs` 顺序，每个张量用 `seed + index` 作为自己的 generator seed。这样给题目加一个输入不会打乱已有张量的值。

## `compute_bounds.py`

```python
def compute_bounds(shape: dict[str, int]) -> dict[str, float]:
    M, K, N = shape["M"], shape["K"], shape["N"]
    return {
        "hbm_bytes": (M * K + K * N + M * N) * 2,
        "flops_bf16": 2.0 * M * N * K,
    }
```

### 契约

- 入参是占位符到整数的映射，键集合等于 `signature` 里出现的全部占位符。
- 返回值必须是 `dict[str, float]`，必须含 `hbm_bytes`，且至少含一个 `flops_<precision>`。
- 精度键的合法集合：`flops_fp4`、`flops_fp8`、`flops_bf16`、`flops_fp16`、`flops_tf32`、`flops_fp32`、`flops_fp64`、`flops_int8`。
- 没有该精度计算时省略该键，或给 `0.0`（两者等价，推荐省略）。
- 必须是纯函数，不得读文件、不得访问 GPU。

### HBM 字节数怎么算

只数**必然发生的**全局内存流量：所有输入各读一次，所有输出各写一次。

```python
dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4, "float8_e4m3fn": 1, "float4_e2m1fn_x2": 0.5}
b = dtype_bytes
hbm = (
    M * K * b["bfloat16"]      # 读 A
  + K * N * b["bfloat16"]      # 读 B
  + M * N * b["bfloat16"]      # 写 C
)
```

不把"重复读"计入下界——那是算法相关的，而下界要与算法无关。这正是 roofline 下界的意义：**任何实现都不可能比这更快**。

对融合算子（attention），中间量不落 HBM，因此只数最外层输入输出。这会让下界更紧，也是融合优化的奖励来源。

### FLOPs 怎么算

数数学意义上的必要运算次数，不数实际执行次数（不数 padding、不数 tile 边界浪费）。

matmul 用 `2*M*N*K`（乘加各算一次）。softmax 的 exp 算 1 FLOP。题目作者要在 `description` 里写清所用口径，否则分数没有可比性。

### 多精度

一个 kernel 里不同阶段用不同精度时，各精度分别给出 FLOPs，评分时用 $\max_p(\text{flops}_p / F_{p,\text{peak}})$ 取瓶颈：

```python
# flash attention：QK^T 与 PV 用 BF16 张量核，softmax 用 FP32
return {
    "hbm_bytes": ...,
    "flops_bf16": 2 * 2 * B * H * S * S * D,
    "flops_fp32": 5 * B * H * S * S,      # exp / max / sum / div / sub
}
```

## 生成与校验

```bash
python -m manage new-problem <slug>          # 建目录与骨架
python -m manage validate-problem <slug>     # 校验 schema、跑参考实现、检查下界键
python -m manage generate-groundtruth <slug> # 生成输入与输出 .npy
python -m manage generate-groundtruth --all
```

`validate-problem` 做的检查：

1. YAML 能被 schema 校验通过，占位符闭合。
2. `reference.py` 有 `reference` 函数，能在 `shadow_shape` 上跑通，输出 shape 与 dtype 符合 `signature`。
3. `compute_bounds` 在 `shadow_shape` 上返回合法键，`hbm_bytes > 0`。
4. 用同一份 ground truth，FP32 参考实现的误差自身在容差内（跑两遍参考实现比对，确认确定性）。
5. 每个测试点的 `t_lb` 不为 0 且不为 NaN。

第 4 条容易漏：如果参考实现本身在容差边缘，题目就是不可通过的。

## ground truth 文件

```
data/groundtruth/matmul-bf16/
├── small/
│   ├── input_0.npy        # A
│   ├── input_1.npy        # B
│   ├── output_0.npy       # C（FP32，比对基准）
│   └── meta.json          # shapes, dtypes, seed, 生成时的 git commit
├── medium/ ...
└── large/ ...
```

`output_0.npy` 存 FP32 而非 BF16：BF16 只有 8 位尾数，存 BF16 会把参考值本身舍入到和用户输出同一量级，容差就没意义了。比对时用户的 BF16 输出先 upcast 到 FP32 再算误差。

不同步进 Git。体积上，4096³ 的 BF16 matmul 输出约 32 MB，三道题十个测试点就是几百 MB。

## 编写题目的经验

- **测试点要覆盖不同的瓶颈形态。** 只放一个大 shape，题目就退化成"谁能写一个快 kernel"；放一个 batch=1 的 shape 和一个小 K 的 shape，才能区分"会不会处理 tail"。
- **`atol` 与 dtype 匹配。** BF16 的有效精度约 3 位十进制，`atol=1e-2` 是合理的起点；FP8 要放宽到 `1e-1`。
- **`min_correct_ratio` 不要设成 1.0。** 归约算子的最后几个元素出现 ulp 级差异是正常的，设 1.0 会让所有提交都失败。
- **`description` 里写清 FLOPs 口径。** 否则用户无法理解自己的分数。
- **难题也要给 template。** 空文件会让大部分用户直接放弃。
