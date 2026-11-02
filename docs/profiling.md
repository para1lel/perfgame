# Profiling

用 ncu 采集，把结果转成"下一步该改什么"的判断。

## 采集

### 分层

ncu 全量采集（`--set full`）要 10–30 秒，且指标上百个，读不过来。分两层：

| 层 | `--set` | 耗时 | 内容 | 触发 |
| --- | --- | --- | --- | --- |
| 快速 | `basic` | 1–2 s | SOL 两维、occupancy、launch 配置 | 默认 |
| 详细 | `detailed` | 5–10 s | 上面的 + 访存层级、计算管线、warp stall | 用户点"深入分析" |

第一层回答"我是内存瓶颈还是计算瓶颈"，多数情况够了。

### 命令

```bash
ncu --set basic \
    --target-processes all \
    --kernel-name-base demangled \
    --launch-skip 2 --launch-count 1 \
    --csv --page raw \
    -o profile \
    python run_once.py
```

- `--launch-skip 2`：跳过预热的两轮，只采第三轮。预热调用里可能含 JIT、内存池首次分配等噪声。
- `--launch-count 1`：只采一个 kernel。用户的 `solve` 里如果有多个 kernel（比如 preprocess + matmul + epilogue），只采主 kernel——通过 `--kernel-name` 限定用户模板里约定的主 kernel 名。
- `-o` 落一份 `.ncu-rep`，同时用 `--csv` 拿结构化数据。`.ncu-rep` 供用户下载后用 `ncu-ui` 打开。

### 与计时互斥

ncu 会 replay kernel、锁定时钟，采集期间的上任何其他工作都会污染数据。因此 profiling 与评测**共用同一条队列**，串行执行（见 [`architecture.md`](architecture.md#为什么评测任务串行)）。

### 前置条件

需要 GPU 性能计数器权限。检查：

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly   # 应为 0
```

本机已确认为 0。部署文档里要把这条写进部署检查单，否则 ncu 会以 `ERR_NVGPUCTRPERM` 失败。

## 指标与优先级

按"对下一步优化的指导价值"排序。面板上按这个顺序排，前面的永远默认可见。

### 第一层：定位瓶颈

| 指标 | 含义 | 怎么用 |
| --- | --- | --- |
| **SOL Memory %** | 访存带宽利用率占峰值比例 | ≥80% 说明已经贴住带宽墙，再优化访存没意义 |
| **SOL Compute %** | 计算利用率占峰值比例 | ≥80% 说明计算饱和，要考虑减少运算或换精度 |
| **Achieved Occupancy** | 实际活跃 warp / 最大 warp | 低于 30% 通常是寄存器或 shared memory 限制 |
| **Launch 配置** | grid / block / 寄存器 / 静态+动态 shared mem | 决定能不能提高 occupancy |

**判断规则的完整表述**：

- 两者都低（<50%）→ 延迟问题，查 stall 与 occupancy。
- 访存高、计算低 → 内存 bound，查 L2 命中率、事务效率、是否有多余的重复读。
- 计算高、访存低 → 计算 bound，查指令 mix（是不是在跑非张量核的路径）、是否用上了 fp8/bf16。
- 两者都高（≥80%）→ 已经很好，剩下的收益在算法层面（减少 FLOPs 或 bytes），不是 kernel 层面。

### 第二层：访存

| 指标 | 阈值 / 解读 |
| --- | --- |
| L1/TEX Hit Rate | 低说明访问模式不合并或没有复用 |
| L2 Hit Rate | 对 tiled matmul 应 >80%；低说明 tile 尺寸或调度顺序有问题 |
| Global Load/Store 事务数 vs 理想值 | 比值 >1.2 说明合并不理想（非连续访问、stride 不对） |
| Shared Memory Bank Conflicts | 任何非 0 都值得看，>1.5 波次需要重排 swizzle |
| Shared Memory 静态/动态用量 | 超过 48 KB 需 opt-in，且会压 occupancy |

### 第三层：计算与调度

| 指标 | 解读 |
| --- | --- |
| FP64 / FP32 / FP16 / Tensor 管线利用率 | 判断是不是跑在错误的管线上（比如 BF16 数据却走了 CUDA core 的 float 路径） |
| 平均 warp 发射间隔（issue interval） | >2 说明发射不饱 |
| Warp Stall 原因分布 | 见下 |
| Branch Efficiency | 低说明 warp 内分支发散 |

**Stall 原因到动作的映射**（这是面板上最有用的部分）：

| 主要 stall | 通常意味着 | 建议动作 |
| --- | --- | --- |
| Long Scoreboard | 等全局内存 | 增加并行度、预取、改善合并 |
| Short Scoreboard | 等 shared memory | 减少 bank conflict、检查同步 |
| Wait | 等固定延迟指令（如 MMA 之后的依赖） | 加深流水、software pipelining |
| Barrier | 等 `__syncthreads` | 减少不必要的同步、检查负载均衡 |
| MIO Throttle | shared/特殊功能单元队列满 | 减少 shared 访问、调整指令 mix |
| LG Throttle | 局部/全局内存队列满 | 减少访存指令数（向量化） |
| Not Selected | 可发射的 warp 太多 | 通常是好事，说明延迟被隐藏了 |

## 数据转换

`ncu --csv --page raw` 输出扁平 CSV，一行一个指标。需要转成面板要的结构：

```python
{
  "kernel": "matmul_kernel",
  "grid": [128, 64, 1], "block": [256, 1, 1],
  "registers_per_thread": 168, "shared_mem_bytes": 49152,
  "sol": {"memory_pct": 71.2, "compute_pct": 43.8, "duration_ms": 1.83},
  "occupancy": {"achieved_pct": 46.9, "theoretical_pct": 50.0,
                "limiter": "registers"},
  "memory": {"l1_hit_pct": 62.1, "l2_hit_pct": 88.4,
             "dram_read_bytes": 100663296, "dram_write_bytes": 33554432,
             "sectors_per_request": 3.92, "bank_conflicts": 0.12},
  "compute": {"tensor_pipe_pct": 41.0, "fp32_pipe_pct": 4.2,
              "instructions": 1.2e6},
  "stalls": [{"reason": "Long Scoreboard", "pct": 38.1},
             {"reason": "Wait", "pct": 22.4}, ...]
}
```

解析器放 `evaluator/runner/ncu_parse.py`，纯函数，输入 CSV 文本、输出上面的 dict，必须有单测（用一份存下来的真实 CSV 做 fixture）。

## 面板布局

侧滑抽屉（Drawer），宽 560px，从右侧滑入，不遮挡编辑器（编辑器自动缩窄）。理由：看指标时经常要同时看代码。

```
┌───────────────────────────────────────────────┐
│ matmul_kernel              [basic ▾]  [关闭]  │
│ 1.83 ms · grid (128,64,1) · block (256,1,1)   │
├───────────────────────────────────────────────┤
│                                               │
│   71.2 %            43.8 %          46.9 %    │
│   访存 SOL          计算 SOL        占用率     │
│   ███████████░░     ██████░░░░░░░   ███████░  │
│                                               │
│   ┌───────────────────────────────────────┐   │
│   │ 访存 bound，尚有 29% 带宽余量          │   │
│   └───────────────────────────────────────┘   │
│                                               │
├─ Warp 停顿 ───────────────────────────────┬──┤
│                                           │  │
│  Long Scoreboard  ███████████████ 38.1%   │  │
│  Wait             █████████ 22.4%         │  │
│  Not Selected     ███████ 17.8%           │  │
│  MIO Throttle     ████ 9.2%               │  │
│  其他             █████ 12.5%             │  │
│                                           │  │
├─ 访存 ────────────────────────────────────┴──┤
│  L2 命中 88.4% · 合并度 3.92 扇区/请求        │
│  bank conflict 0.12 波次                      │
│  DRAM 读 96 MB · 写 32 MB                     │
├─ 计算 ───────────────────────────────────────┤
│  Tensor 41.0% · FP32 4.2% · 指令 1.2 M        │
├──────────────────────────────────────────────┤
│ [深入分析：detailed]  [下载 .ncu-rep]  [重测]  │
└──────────────────────────────────────────────┘
```

设计要点：

1. **三个大数字在最上方，带横向条。** 条不是装饰——它把"71.2%"这个抽象数变成"还剩多少"的空间直觉。阈值上色：≥80% `--danger`（到顶了，别在这上面花力气），50–80% `--signal`，<50% `--primary`。
2. **一句结论紧随其后。** 由规则生成（见上面的判断规则），不是大模型写的。这是整个面板最有价值的一行：它把三组数字变成一个判断。
3. **stall 是横向堆叠条**，不是饼图。饼图读不出排序，横条能。
4. **所有条形图共用一套比例尺。** 不让视觉面积说谎。
5. **详细指标折叠。** 第一层默认展开，二三层折叠，标题右侧显示一级摘要（如"L2 命中 88.4%"），不展开也能扫到。
6. **每个数值带单位与基准。** 写"96 MB"不写"100663296"；写"3.92 扇区/请求"不写"3.92"。

## 与其他页面的关系

- 提交详情页直接内嵌这个抽屉（可展开为整页路由 `/submissions/:id/profile`）。
- 题目页的"Profile"按钮触发一次新采集，走队列，完成后抽屉自动打开。
- 同一提交的多次 profiling 都保留，抽屉顶部可切换（`第 2 次 · detailed`）。

## 不做的事

- 不做时间轴（那是 nsys 的领域）。这里只关心单个 kernel 内部的效率。
- 不做自动调优建议。规则化的结论（"访存 bound，还有 29% 余量"）有；"把 block 改成 512"这类没有——那需要真的试一遍。
- 不做跨提交的指标趋势图。第一版没有足够数据量支撑，等有真实使用再考虑。
