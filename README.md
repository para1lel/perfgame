# PerfGame

NVIDIA GPU 算子优化 Playground：在线编写、编译、评测并 profiling CUDA 与主流 Python DSL kernel，按 roofline 模型自动打分。

## 能力概览

- **在线 IDE**：Monaco 编辑器，CUDA C++（含 CUTLASS 头文件）、TileLang、TIRx、CuTe DSL 四种语言；行内语法检查、补全、格式化全部在浏览器内完成。
- **编译产物查看**：PTX 与 SASS 随点随看（CUDA 语言不重复展示源码本身）。
- **自动评分**：每个测试点先过容差（逐元素比例 + 余弦相似度），再取 5 次计时中位数，按理论下界换算出 roofline efficiency。70% 效率记 100 分。
- **ncu profiling**：层次化指标展示，先给 SOL 与 occupancy，再逐步深入 warp stall、缓存命中。
- **GPU 队列**：提交后等待一张真正空闲的 GPU（利用率 <3%、显存 <128 MB）才开跑，多卡环境下每卡一个 worker。
- **明暗双主题**：正文 LXGW WenKai，代码 FiraCode Nerd Font。

## 硬件与软件要求

| 项目 | 要求 |
| --- | --- |
| GPU | 至少 1 张 NVIDIA GPU，Hopper（sm_90）或更新；多卡可水平扩展评测吞吐 |
| 驱动 | r580+（cuTile 已不纳入，但 CuTe DSL / CUDA 13 需要较新驱动） |
| CUDA Toolkit | 13.0 及以上（本机为 13.0.88） |
| Python | 3.12（评测环境锁定版本，见 `docs/deployment.md`） |
| Node.js | 20+（仅构建前端时需要） |

## 快速开始

```bash
git clone <repo-url> perfgame && cd perfgame
make setup     # 创建 conda 环境、安装前后端依赖、生成 ground truth
make dev       # 前端 dev server + 后端 API
```

生产部署见 [`docs/deployment.md`](docs/deployment.md)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/README.md`](docs/README.md) | 文档索引与阅读路径 |
| [`docs/architecture.md`](docs/architecture.md) | 组件划分、请求生命周期、进程与 GPU 模型 |
| [`docs/evaluation.md`](docs/evaluation.md) | 容差、计时、评分公式、错误与重测语义 |
| [`docs/languages.md`](docs/languages.md) | 四种语言的调用契约与编译产物提取 |
| [`docs/problems.md`](docs/problems.md) | 题目目录结构、schema、下界函数 |
| [`docs/profiling.md`](docs/profiling.md) | ncu 采集策略与指标分层 |
| [`docs/api.md`](docs/api.md) | REST 与 SSE 接口 |
| [`docs/frontend.md`](docs/frontend.md) | 视觉规范、主题、字体、路由 |
| [`docs/deployment.md`](docs/deployment.md) | conda 环境、systemd、Nginx |
| [`docs/decisions.md`](docs/decisions.md) | 设计与取舍记录（含被否决的方案） |
| [`docs/implementation-plan.md`](docs/implementation-plan.md) | 分阶段实施计划与验收标准 |

贡献方式见 [`CONTRIBUTING.md`](CONTRIBUTING.md)；AI agent 请先读 [`AGENTS.md`](AGENTS.md)。

## 许可证

MIT
