# AGENTS.md

给在本仓库工作的 AI agent 的入口文档。**只读这一页不足以动手**：下面的"深入阅读"按任务类型指向真正的规格文档。

## 项目是什么

一个自托管的在线 IDE，供人练习 NVIDIA GPU kernel 优化。用户选一道题，用 CUDA / TileLang / TIRx / CuTe DSL 写 kernel，提交后系统自动验证正确性、测时、按 roofline 效率打分，并可用 ncu 采集 profiling 指标。

三个独立进程：**前端**（React，静态文件）、**API 后端**（FastAPI，无 GPU 依赖）、**评测 worker**（有 GPU，每个 GPU 一个进程）。前端与后端完全分离，只通过 HTTP/SSE 通信。

## 目录约定

```
perfgame/
├── frontend/                  # React + Vite + Monaco
├── backend/                   # FastAPI：认证、题目、提交、队列投递、SSE
├── evaluator/                 # GPU 侧：编译器适配、运行、计时、ncu
│   ├── contracts/             # 语言契约与 C ABI 定义
│   ├── adapters/              # 每种语言的编译与提取实现
│   ├── runner/                # 沙箱子进程、计时、判分
│   └── worker/                # 队列消费者 + GPU 空闲闸门
├── problems/<slug>/           # 题目：yaml + reference.py + compute_bounds.py
├── deploy/                    # systemd unit、nginx 模板、install 脚本
├── docs/                      # 规格文档，同时渲染进站点 /docs
├── hw/                        # 硬件 profile YAML（峰值带宽与各精度 FLOPS）
└── environment-*.yml          # 评测环境定义（两份，见下）
```

## 必须先知道的硬约束

1. **两个 conda 环境，不可合并。** `environment-main.yml` 装 TileLang + CuTe DSL；`environment-tirx.yml` 装 `apache-tvm`（TIRx）。原因：TileLang 钉死 `apache-tvm-ffi>=0.1.11,<0.1.13`，而 apache-tvm 0.27.x 要求 `>=0.1.14.post0`。任何"统一环境"的尝试都会失败，不要试。
2. **cuTile 不支持本项目。** `cuda-tile` 的 tileiras 编译器只覆盖 Blackwell 与 Ampere/Ada，Hopper（sm_90）尚不支持；本机是 H100。不要把它加回来，除非官方 release notes 明确写了 sm_90。
3. **CUTLASS 不是一门语言**，只是 CUDA 编译时的 include 路径与 `-dc` 选项。前端语言下拉里不应出现它。
4. **H100 无 FP4。** Hopper 的最低张量精度是 FP8。下界计算里出现 `fp4` 只在 Blackwell profile 中可能非零。
5. **评测环境的 Python 固定 3.12。** 四个技术栈都覆盖 3.12；3.14 是最新但不是目标。
6. **计时前必须 flush L2。** 不 flush 的话第二次迭代命中 L2，分数虚高。见 `docs/evaluation.md`。

## 深入阅读

| 你要做的事 | 先读 |
| --- | --- |
| 改评测逻辑、评分、容差 | `docs/evaluation.md` |
| 加一门语言 / 修某个 DSL 的产物提取 | `docs/languages.md` |
| 加题目、改 schema | `docs/problems.md` |
| 动 ncu 采集或指标面板 | `docs/profiling.md` |
| 加接口、改状态机 | `docs/api.md` |
| 改 UI、主题、字体 | `docs/frontend.md` |
| 部署、环境、运维 | `docs/deployment.md` |
| 想知道"为什么这么定" | `docs/decisions.md` |
| 不知道从哪下手 | `docs/implementation-plan.md` |

## 各语言产物提取速查

| | CUDA C++ | TileLang | TIRx | CuTe DSL |
| --- | --- | --- | --- | --- |
| CUDA 源码 | 用户所写，不展示 | `kernel.get_kernel_source()` | `exe.mod.imports[0].inspect_source()` | 无（走 MLIR→PTX） |
| PTX | `nvcc -ptx` | `kernel.export_ptx(path)` | `target_format="ptx"` | `compiled.__ptx__` |
| SASS | `cuobjdump --dump-sass` | `kernel.export_sass(path)` | `cuobjdump` on cubin | `compiled.__sass__` |

TileLang 的 `_get_ptx()` 是私有 API，**不要**在生产路径使用，用 `export_ptx()`。

## 常见坑

- **DSL 代码不等于 Python 代码。** 每个测试点单独编译，编译缓存 key 必须包含 shape，否则不同 shape 会互相覆盖。
- **CUDA 只编译一次**，shape 在运行期作为参数传入；不要为 CUDA 按 shape 生成模板。
- **子进程模型**：提交一个子进程，测试点各自再 fork。用户 kernel 崩掉是常态，一个测试点 segfault 只该让那一个测试点 0 分。
- **ncu 与计时互斥**：ncu replay 会改变时序。profiling 与评测共用同一队列，串行执行。
- **worker 空转不轮询**：任务队列阻塞等待；只有拿到任务后才开始按 3s 间隔检查 GPU 空闲。
- **前端断网可用**：编辑器的高亮、补全、格式化都在浏览器内完成，不要改成后端 API。部署链路不稳定是明确的设计前提。
- **不要记录用户代码到日志**，提交内容只在数据库与详情页出现。

## 验证

```bash
make check          # 全部：lint + 单测 + 契约测试
make test-eval      # 需要 GPU：跑基线 kernel，校验分数在预期区间
```

改评测逻辑后必须跑 `make test-eval`。分数漂移超过 ±3% 视为回归。

## 提交与文档同步

任何改变行为的设计变更，同步更新 `docs/` 对应章节和 `docs/decisions.md` 的决策记录。文档是规格，代码是它的实现。
