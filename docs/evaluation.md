# 评测

这一章定义"一次提交如何被判分"。所有数字都是规格，实现必须与之一致。

## 1. 评测契约

每个语言交出同一个形状的东西：一个能在给定输入上产生输出的入口。

### 运行时入口（Python）

评测器不关心用户用什么语言写 kernel，只调用一个固定签名：

```python
def solve(*inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
    ...
```

- 参数顺序与 `problem.yaml` 的 `signature.inputs` 一致。
- 返回值顺序与 `signature.outputs` 一致。单输出可直接返回 tensor。
- 评测器负责分配输入、比对输出、计时。用户代码不得自行分配输入，也不得读取 `sys.argv` 或环境变量。
- 输入张量是 contiguous 的；如果题目需要非连续布局，在 `signature` 里显式声明，评测器生成时遵守。

### CUDA 的额外一层

CUDA 用户写的是 `.cu`，但契约仍是上面的 `solve`。模板里提供一个薄封装：

```cpp
// 用户只写这一段
extern "C" void run_kernel(const void** args, int64_t* shape, int nargs) { ... }
```

评测器把输入张量的裸指针、shape 数组按顺序传进去，再用一个自动生成的 `solve`（`evaluator/contracts/cuda_wrapper.py`）把它包成 Python 函数。用户不需要写 pybind11、不需要碰 `torch.utils.cpp_extension`。

**编译次数**：CUDA 只编译一次，shape 在运行期作为参数传入。用户代码里不允许用宏或模板把 shape 在编译期写死（模板里出现具体数字会导致不同测试点结果不一致，属于作弊路径，见 §8）。DSL 语言天然按 shape 特化，每个测试点单独编译。

## 2. 参考实现与 ground truth

### 参考实现用 FP32

`reference.py` 里的参考实现以 FP32 语义为准。允许在内部用 FP64 累加、最后 cast 回 FP32——对长归约（softmax、layernorm、大 K matmul）这是必要的，否则参考实现自身的误差会和容差同量级。

```python
def reference(x: torch.Tensor, ...) -> torch.Tensor:
    x = x.double()          # 允许
    ...
    return out.float()      # 必须以题目声明的输出 dtype 返回
```

### ground truth 预计算

**评测时不跑参考实现。** 部署时用 `python -m manage generate-groundtruth <slug>` 一次性算好，按测试点分目录存成 `.npy`（完整布局见 [`problems.md`](problems.md#ground-truth-文件)）：

```
data/groundtruth/matmul-bf16/large/
├── input_0.npy      # 每个输入一份
├── input_1.npy
├── output_0.npy     # FP32 比对基准
└── meta.json        # shapes、dtypes、seed、生成时的 git commit
```

输入数据同样预生成（固定种子 → 同一次部署内所有提交看到的输入完全相同）。运行时只加载，不重新随机。

这么做的原因有三：参考实现是 FP32 甚至 FP64 的，每次评测重算出几秒到几十秒的时间，会淹没被测 kernel 的毫秒级耗时；预计算后可以离线检查题目是否合理；用户输入固定后，两次提交的分数才可比。

`.npy` 不进 Git（体积大且可重算），部署后必须执行生成命令。生成脚本用固定 seed + `torch.Generator`，保证跨机器可复现。

## 3. 容差

一个测试点通过，需要**同时**满足下面两条。

### 3.1 逐元素误差

设参考输出为 $y$，用户输出为 $x$（均在 FP32 下比对）。定义单元素"合格"：

$$
|x_i - y_i| \le \text{atol} \quad \text{或} \quad \frac{|x_i - y_i|}{|y_i| + 10^{-8}} \le \text{rtol}
$$

要求合格元素占比不低于 $R$：

$$
\frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\big[(x_i,y_i)\ \text{合格}\big] \ge R
$$

`atol`、`rtol`、`R` 逐题配置。默认 `atol=1e-2, rtol=1e-2, R=0.99`。

NaN 与 Inf：任何位置出现 NaN 即判该测试点失败，不参与比例统计。用户输出出现 Inf 而参考为有限值时同样失败。

### 3.2 余弦相似度

把两个输出展平为向量：

$$
\cos(x, y) = \frac{\langle x, y \rangle}{\|x\|_2 \|y\|_2} > \tau
$$

$\tau$ 逐题配置，默认 `0.999`。它兜住逐元素容差放过的"整体缩放错误"——比如输出被缩放 0.99 时逐元素误差可能落在 atol 内，但余弦值会明显偏离。

### 3.3 为什么两条都要

逐元素比例能容忍个别离群点（归约边角、特殊值处理），这正是实际 kernel 的常态；余弦相似度约束全局结构。只有前者会放过"大部分对、整体偏"的结果；只有后者会放过"方向对、个别元素错得离谱"的结果。

## 4. 计时协议

### 4.1 参数

| 项 | 值 |
| --- | --- |
| 预热 | 2 次（不计时） |
| 计时轮数 | 5 次 |
| 每轮取值 | 单次调用的 GPU 时间 |
| 汇总 | 5 轮的中位数 |

### 4.2 L2 flush

每轮计时前必须把 L2 缓存冲掉，否则第 2 轮起命中 L2，分数虚高。做法是写一块大于 L2 容量的缓冲区（H100 L2 为 50 MB，缓冲区取 256 MB）：

```python
flush = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device="cuda")  # 256 MB
for _ in range(5):
    flush.zero_()
    torch.cuda.synchronize()
    start.record()
    out = solve(*inputs)
    end.record()
    torch.cuda.synchronize()
    samples.append(start.elapsed_time(end))   # ms
median = statistics.median(samples)
```

### 4.3 测什么

用 CUDA Event 包住**一次调用**，不是 CPU wall clock。这样测的是设备上的实际执行时间，排除 Python 侧开销——但包括 launch 开销，这是用户 kernel 真实成本的一部分。

小 shape 的测试点如果只有几微秒，launch 开销占主导是正常现象，**不做 K 次循环摊薄**。摊薄会掩盖 launch 配置问题（grid 太小、每个 block 只干一点活），而这恰恰是要暴露的东西。

### 4.4 分配开销的归属

用户 `solve` 内部自行分配的临时张量，其分配时间计入被测时间（`torch.empty` 走 caching allocator，通常可忽略）。评测器放在计时窗口外的只有自己的 buffer 管理。

### 4.5 抖动处理

若 5 次采样的中位数与最小值相差超过 30%，说明该卡当时不稳定（别人跑了大作业、时钟降频）。处理方式：整轮重测一次；仍不稳定则标记 `noisy`，分数照给但在结果页标注。

## 5. 理论下界与评分

### 5.1 下界时间

roofline 模型：一个 kernel 的时间不可能同时低于它的访存时间和计算时间。

$$
t_{\text{mem}} = \frac{\text{hbm\_bytes}}{B_{\text{peak}}}, \qquad
t_{\text{compute}} = \max_{p \in \text{precisions}} \frac{\text{flops}_p}{F_{p,\text{peak}}}
$$

$$
t_{\text{lb}} = \max(t_{\text{mem}}, t_{\text{compute}})
$$

`hbm_bytes` 与各 `flops_p` 全部由题目的 `compute_bounds.py` 从 shape 算出（见 [`problems.md`](problems.md)），与算法无关。$B_{\text{peak}}$ 和 $F_{p,\text{peak}}$ 来自 `hw/<profile>.yaml`（见 [`deployment.md`](deployment.md#硬件-profile)）。

`t_compute` 取所有精度中的**最大值**：一个用 BF16 主算、FP32 做 softmax 的题，两个精度各有时长，瓶颈是更长的那条。

### 5.2 分数

单个测试点：

$$
s_i = \frac{t_{\text{lb},i}}{t_i} \times \frac{1000}{7}
$$

其中 $t_i$ 是实测中位数。$t_{\text{lb}}/t$ 就是 roofline efficiency。系数 $1000/7$ 使得效率 70% 时该测试点得 100 分——这正是"70% 效率即满分"的来历。

题目总分按测试点数归一，让"所有测试点都达到 70% 效率"恰好等于 1000 分：

$$
S = \frac{1000}{N} \sum_{i=1}^{N} \frac{t_{\text{lb},i}}{t_i} \cdot \frac{1}{0.7}
= \frac{1000}{0.7}\,\overline{\text{eff}}
$$

- 未通过容差的测试点 $s_i = 0$，直接拉低均值。
- 效率上限截到 1.5：现实中只有小 shape 的题能靠 L2 命中超过 100%，但 150 已足够表达"超过峰值"，再高就是测量错误。
- 总分不设额外封顶，理论上限 2143。

### 5.3 为什么用均值而不是求和

求和会让 3 个测试点的题和 10 个测试点的题不可比。均值让总分始终有"1000 = 全部达到 70%"这一共同含义。

### 5.4 边界情况

| 情况 | 处理 |
| --- | --- |
| `t_lb = 0`（空 shape） | 题目创建时校验时拒绝 |
| 实测 `t_i = 0` | 视为测量错误，该测试点记 0，标注 `invalid_timing` |
| 输出形状不符 | 该测试点 0 分，`shape_mismatch` |
| 输出 dtype 不符 | 自动 cast 到题目声明 dtype 后再比对，但记一条 warning |
| kernel 崩溃（段错误 / illegal memory access） | 该测试点 0 分，`runtime_error`，其余测试点继续 |

## 6. 错误分类

错误必须是结构化的，前端按类型展示不同 UI：

| `error.kind` | 触发 | 前端展示 |
| --- | --- | --- |
| `compile_error` | nvcc / DSL 编译失败 | 内联错误块 + 原始日志（等宽、保留原始行号） |
| `import_error` | Python 侧 import 失败 | 内联错误块 |
| `runtime_error` | 进程非零退出、CUDA 错误 | 内联错误块 + 崩溃时的 stderr |
| `timeout` | 超过时限 | 提示时限值 |
| `shape_mismatch` | 输出 shape 不符 | 显示期望与实际 |
| `tolerance_failed` | 容差不过 | 显示最大误差、不合格元素比例、余弦值 |
| `invalid_timing` | 计时结果不可信 | 显示原始采样 |

时限：编译 180 s；单测试点运行 60 s；profiling 300 s。超时立即 SIGKILL 该测试点的子进程。

## 7. 重测

对已完成的提交点"重测"，走这条路径：

1. 校验编译产物存在且哈希与提交代码匹配（`sha256(code) + language + testcase`）。
2. 匹配则复用（跳过编译），否则重新编译。
3. 重新跑全部测试点，保留历史结果（`submission_results` 增加行，或用 `attempt` 字段区分），结果页显示最新一次，可切换到历史。

重测的典型动机是"分数抖动"或"当时卡很忙"，这两种都不需要重编译，所以复用是常态。

## 8. 作弊与误用

不做防抄袭（提交记录对所有人公开，这是设计选择），但下面几条会影响分数可信度，必须挡：

- **编译期写死 shape**：CUDA 代码里出现与测试点完全一致的常量并跳过参数读取。检测方式：用一个"影子测试点"，shape 与任何正式测试点都不同，仍在容差内必须通过。
- **缓存结果**：第一次调用计算、后续调用直接返回缓存。检测方式：每次计时前输入被重新随机化的概率不为零（评测器在计时轮次后会用一个新 seed 的输入再验一次正确性）。
- **在计时窗口外计算**：比如第一次调用时做全部工作，后续调用空转。上述随机输入校验同样覆盖。
- **修改评测器**：用户代码在子进程中运行，只能访问自己的 workdir 和只读的输入目录；评测器的路径不在 `sys.path` 上。
