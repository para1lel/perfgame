# 语言支持

支持四种：**CUDA C++**（含 CUTLASS）、**TileLang**、**TIRx**、**CuTe DSL**。

- CUTLASS 不是独立语言，只是 CUDA 编译时的 include 路径与 `-dc` 选项。前端语言下拉里不出现它。
- cuTile 不支持：`cuda-tile` 的 tileiras 编译器只覆盖 Blackwell 与 Ampere/Ada，Hopper（sm_90）尚未支持，而本项目的目标硬件是 H100 及更新。等官方 release notes 明确写了 sm_90 再加回来。

## 环境归属

```
perfgame-main   (Python 3.12)  →  CUDA / TileLang / CuTe DSL
perfgame-tirx   (Python 3.12)  →  TIRx
```

TileLang 与 TIRx 依赖冲突，不可合并（`apache-tvm-ffi` 的版本区间不相交）。worker 按语言选择用哪个解释器去 fork 子进程，其余流程完全一致。

## 适配器接口

每种语言实现同一个协议，worker 只认这个协议：

```python
# evaluator/adapters/base.py
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

@dataclass(frozen=True)
class Artifacts:
    cuda_source: str | None      # DSL 生成的 CUDA；CUDA 语言为 None（不展示）
    ptx: str | None
    sass: str | None
    log: str                     # 编译输出

@dataclass(frozen=True)
class Prepared:
    workdir: Path                # 含生成的 solve 封装与编译产物
    entry: str                   # 该语言实际执行的模块/函数标识
    artifacts: Artifacts

class LanguageAdapter(Protocol):
    name: str                    # "cuda" | "tilelang" | "tirx" | "cutedsl"
    conda_env: str               # "perfgame-main" | "perfgame-tirx"

    def compile(self, code: str, shape: dict[str, int], workdir: Path) -> Prepared:
        """编译用户代码。失败时抛 CompileError（带原始日志）。"""

    def extract(self, prepared: Prepared) -> Artifacts:
        """提取 PTX / SASS。可在 compile 时已顺手做完，这里允许直接返回缓存。"""
```

`shape` 参与编译，因为 DSL 是按 shape 特化的；CUDA 忽略它（但会校验用户代码没有把 shape 写死，见 [`evaluation.md`](evaluation.md#8-作弊与误用)）。

## 各语言细节

### CUDA C++

**用户写什么**：一个 `.cu` 文件，含 `__global__` kernel 和外部入口。

```cpp
extern "C" void run_kernel(const void* const* args, const int64_t* shape, int nargs);
```

`args` 依次是输入与输出的裸指针，`shape` 是题目 `signature` 中占位符对应的具体维度（顺序与 yaml 一致）。评测器生成一个 `solve.py`，用 `ctypes` 加载 `.so`，把 torch tensor 的 `data_ptr()` 与形状传进去。

**编译命令**：

```
nvcc -std=c++17 -O3 --use_fast_math=false \
     -arch=sm_90a -lineinfo \
     -I$CUTLASS_INCLUDE \
     -shared -Xcompiler -fPIC \
     kernel.cu -o kernel.so
```

`-arch` 取自 `hw/<profile>.yaml` 的 `arch_suffix`（H100 为 `sm_90a`，需要 `a` 才能用全速 TMA/WGMMA）。`-I` 指向 `environment-main.yml` 中安装的 CUTLASS 头文件目录。

**产物**：

| 项 | 方法 |
| --- | --- |
| CUDA 源码 | 不展示（就是用户所写） |
| PTX | `nvcc -ptx -arch=... kernel.cu -o kernel.ptx`（复用同一次前端解析，不重复编译） |
| SASS | `cuobjdump --dump-sass kernel.so` |

### TileLang

**用户写什么**：一个 Python 模块，暴露 `solve`。

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def matmul(M, K, N, block_M=128, block_K=32, block_N=128):
    ...

def solve(*inputs):
    return matmul(*[d for d in inputs_shapes])  # 由题目模板给出
```

模板里已经写好 shape 提取与 JIT 调用，用户只改 `matmul` 函数体。

**产物**：

| 项 | 方法 | 备注 |
| --- | --- | --- |
| CUDA 源码 | `kernel.get_kernel_source()` | 公开 API |
| PTX | `kernel.export_ptx(path)` | 公开 API；**不要用 `_get_ptx()`**，它是私有的 |
| SASS | `kernel.export_sass(path)` | 公开 API |

PTX 是"把生成的 CUDA 源码再喂给 nvcc"得到的，理论上可能与 JIT 缓存里实际加载的 cubin 有细微差异（编译 flag 不同时）。这是 TileLang 的设计，接受它，但要在"编译日志"标签里注明所用 flag。

### TIRx

**用户写什么**：Python 模块，用 `@T.jit` / `@T.prim_func` 风格写 kernel，暴露 `solve`。

**编译与产物**：

| 项 | 方法 |
| --- | --- |
| CUDA 源码 | `executable.mod.imports[0].inspect_source()` |
| PTX | 编译时 `target_format="ptx"`（或 `"cubin"` / `"fatbin"`） |
| SASS | 对 cubin 跑 `cuobjdump --dump-sass` |

`Target({"kind": "cuda", "arch": "sm_90"})` 必须显式传，否则默认回落到 `sm_50`，性能会低一个数量级且不报错。适配器里硬编码从 `hw/*.yaml` 读 arch。

注意：TIRx 附带的高性能 kernel 库面向 `sm_100a`，在 H100 上只能用通用的 TIR lowering，拿不到 Hopper 专属的 WGMMA/TMA 优化。这意味着**同一份 TIRx 代码在 H100 上的分数会明显低于 TileLang**，这是预期行为，不是 bug。题目页的说明里要写清楚。

### CuTe DSL

**用户写什么**：Python 模块，`@cute.jit` / `@cute.kernel`，暴露 `solve`。

**产物**：

| 项 | 方法 |
| --- | --- |
| CUDA 源码 | **不存在**。CuTe DSL 的路径是 Python AST → MLIR → PTX，没有 CUDA C++ 中间产物，因此不提供该标签 |
| PTX | `compiled.__ptx__`（返回字符串）；或编译前 `CUTE_DSL_KEEP=ptx` + `CUTE_DSL_DUMP_DIR` 落盘 |
| SASS | `compiled.__sass__`，需要编译时保留（`cute.compile[..., KeepSASS]` 或 `--keep-sass`） |

保留产物的程序化方式（4.8.0+）：

```python
from cutlass.cute import OptLevel, KeepPTX, KeepSASS
cute.compile[OptLevel(2), KeepPTX, KeepSASS](kernel, ...)
```

注意这些 flag 会进入 JIT 缓存 key——开着保留产物和不开是两份不同的缓存条目，不要混用。

### 产物矩阵（总表）

| | CUDA C++ | TileLang | TIRx | CuTe DSL |
| --- | --- | --- | --- | --- |
| CUDA 源码 | —（用户所写） | `get_kernel_source()` | `inspect_source()` | 无 |
| PTX | `nvcc -ptx` | `export_ptx()` | `target_format="ptx"` | `__ptx__` |
| SASS | `cuobjdump` | `export_sass()` | `cuobjdump` | `__sass__` |
| 编译次数 | 1 次 | 每 shape 1 次 | 每 shape 1 次 | 每 shape 1 次 |

## 前端约定

`frontend/src/languages.ts` 是唯一真相源，驱动：下拉选项、Monaco 语言注册、默认模板、产物标签可见性。

```ts
export const LANGUAGES = [
  { id: "cuda",    label: "CUDA C++", ext: ".cu",   showCuda: false, showPtx: true, showSass: true },
  { id: "tilelang",label: "TileLang", ext: ".py",   showCuda: true,  showPtx: true, showSass: true },
  { id: "tirx",    label: "TIRx",     ext: ".py",   showCuda: true,  showPtx: true, showSass: true },
  { id: "cutedsl", label: "CuTe DSL", ext: ".py",   showCuda: false, showPtx: true, showSass: true },
] as const;
```

切换语言时，各语言的草稿分开保存（key 里带语言 id），互不覆盖。

## 新增一门语言的检查清单

1. `evaluator/adapters/<name>.py` 实现 `LanguageAdapter`。
2. 在 `evaluator/contracts/` 补入口模板（用户新建时看到的骨架）。
3. 在 `environment-main.yml` 或 `environment-tirx.yml` 里加依赖，必要时新建第三个环境（并在 `deploy/` 加一个 worker 变体）。
4. 前端 `languages.ts` 加一行 + Monaco 语法注册。
5. 写一个端到端测试：向量加法，断言能编译、能跑、分数在合理区间。
6. 在本文档的产物矩阵补一列，在 [`decisions.md`](decisions.md) 记一条选型。
