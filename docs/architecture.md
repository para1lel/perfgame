# 架构

## 三个进程，一条队列

```
┌──────────────┐   HTTPS（REST + SSE）   ┌──────────────┐
│   前端 (SPA)  │ ──────────────────────▶ │  API 后端     │
│  React+Vite  │ ◀────────────────────── │  FastAPI     │
└──────────────┘                         └──────┬───────┘
                                                │ 写提交、投递任务
                                          ┌─────▼──────┐
                                          │  SQLite    │
                                          └─────┬──────┘
                                                │ 任务队列（Redis list 或进程内 asyncio.Queue）
                        ┌───────────────────────┴──────────────┐
                        │                                      │
                 ┌──────▼───────┐                      ┌───────▼──────┐
                 │ worker GPU:0 │  …                   │ worker GPU:N │
                 │ conda: main  │                      │ conda: main  │
                 │ + tirx 子环境 │                      │ + tirx 子环境 │
                 └──────────────┘                      └──────────────┘
```

### 前端

纯静态 SPA，构建产物是 `frontend/dist/`，由 Nginx 直接服务。不知道后端地址以外的任何事：不感知 GPU、不持有密钥、不做评测。所有编辑器侧能力（语法高亮、补全、格式化、自动保存）都在浏览器里跑，断网时编辑器仍可用。

### API 后端

FastAPI，**不 import torch，不碰 CUDA**。职责：

- GitHub OAuth 登录与 session
- 题目的读取与序列化（从 `problems/` 目录读 YAML，进程内缓存，文件 mtime 变化时失效）
- 提交的创建、查询、列表
- 把任务投递到队列，并消费 worker 回写的结果
- 通过 SSE 把状态变化推给前端
- 提供 `/docs` 路由的 Markdown 内容

把"评测"和"服务"分开的原因：评测环境需要 torch、CUDA、TileLang，依赖重且和 GPU 绑定；Web 服务要能独立重启、独立扩容、在没有 GPU 的机器上跑。两者混在一起会让一次 kernel 崩溃带走整个站点。

### 评测 worker

每个 GPU 一个进程，长期驻留。它是唯一持有 GPU 上下文的角色。内部按语言再分派到两个 conda 环境（见 [`deployment.md`](deployment.md)）。

## 一次提交的生命周期

1. 用户在编辑器里点"提交"。前端把 `{problem_slug, language, code}` POST 给后端。
2. 后端写入 `submissions`（`status='queued'`），返回提交 ID，并把任务投递到队列。
3. 某个 worker 阻塞在队列上，取到任务。它记录 `status='waiting_gpu'`。
4. worker 开始检查自己那张卡的 `nvidia-smi`：利用率 < 3% 且显存 < 128 MB 才算空闲。每 3 秒查一次。**没有任务时不查**（见下）。
5. 空闲后 `status='compiling'`：按语言编译。编译失败则 `status='compile_error'`，写回日志，结束。
6. 逐测试点执行（各自 fork 子进程）：
   - 分配输入（固定种子）、跑用户 kernel、与 ground truth 比对
   - 容差不过 → 该测试点 0 分，记录原因
   - 容差通过 → 计时（warmup 2、5 次带 L2 flush 的中位数）→ 换算分数
7. 全部完成：写 `submission_results`，算总分，`status='completed'`。
8. 每个状态变更通过 Redis pub/sub 或数据库轮询转成 SSE 事件推给前端。

用户随时可以对已完成的提交点"重测"（复用编译产物，只重跑计时）或"Profile"（入队，串行执行 ncu）。

## worker 与 GPU 的绑定

**每个 GPU 一个 worker 进程，worker 只认自己的卡。** 这消除了"两个任务抢同一张卡"的整类竞争问题：抢卡在物理上不可能发生，因为只有一个进程在用这张卡。

由此推出：

- **空闲闸门是 worker 自己的事**，不是调度器的事。`CUDA_VISIBLE_DEVICES` 在进程启动时固定，worker 用 `pynvml` 查自己那张卡的利用率与显存。
- **不轮询**。worker 阻塞在队列 `BRPOP`（或多进程 `Queue.get()`）上，CPU 占用为 0。只有拿到任务之后才开始 3 秒一次的 GPU 检查。你的服务器提交量不大，这样最省资源。
- **等待不设超时**。任务一直排队，前端显示队首位置。理由：设置超时只会让用户重试，反而加重排队。
- **多卡扩展是加进程**，不是改代码：在 `deploy/` 里多启一个 `perfgame-worker@1.service` 即可。

## 为什么评测任务串行

同一张卡上并发跑两个计时任务，会互相污染 wall time，而这套系统的全部价值就是可信的 wall time。所以：

- 一张卡同一时刻只跑一个评测或 profiling 任务。
- **profiling 与评测共用队列**。ncu 会 replay kernel，并发执行时采集到的指标毫无意义。用户点 Profile 时明确告知"已排队，前面还有 N 个任务"。
- 多张卡之间天然并行。

## 数据存在哪

| 数据 | 位置 | 是否进 Git |
| --- | --- | --- |
| 题目定义（yaml / 参考实现 / 下界函数） | `problems/` | 是 |
| 每测试点的 ground truth（`.npy`） | `data/groundtruth/<slug>/` | 否，部署时生成 |
| 提交代码、结果、分数 | SQLite `data/perfgame.db` | 否 |
| 编译产物（PTX/SASS/日志） | `data/artifacts/<submission_id>/` | 否 |
| ncu 报告（`.ncu-rep`） | `data/profiles/<submission_id>/` | 否 |
| 硬件峰值 profile | `hw/*.yaml` | 是（公开数据） |

代码与产物分开放：数据库存的是"发生了什么"，文件系统存的是"字节"。这样 SQLite 保持小体积，也便于单独备份（见 [`deployment.md`](deployment.md#备份)）。

## 组件不做什么

划清边界，避免实现时把职责搅在一起：

- 前端不做任何评测逻辑，也不计算分数——分数只由后端和 worker 产生。
- API 后端不执行用户代码，连语法检查都不做。
- worker 不生成 HTML，不写文档，只回写结构化数据。
- 数据库不存 ground truth，也不存 `.ncu-rep` 的字节，只存路径。

## 技术选型一览

| 层 | 选择 | 理由 |
| --- | --- | --- |
| 前端框架 | React 18 + Vite + TypeScript | Monaco 的 React 绑定最成熟 |
| 编辑器 | Monaco | 自带语法高亮、多光标、连字支持 |
| 后端 | FastAPI + Uvicorn | 异步、自带 OpenAPI、Python 生态与评测一致 |
| 数据库 | SQLite (WAL) | 单机、读多写少、无需运维 |
| 队列 | Redis list / 进程内队列 | 单机场景进程内即可；多机上 Redis |
| 实时推送 | SSE | 单向、Nginx 原生支持、断线自动重连 |
| 部署 | systemd + Nginx | 无容器运行时开销，日志走 journald |
