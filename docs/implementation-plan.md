# 实施计划

六个阶段。每阶段有可验证的终点，做完就能停下来。**顺序不可调换**——后一阶段依赖前一阶段产出的东西。

## 阶段 0：脚手架（0.5 天）

**产出**

- 仓库结构：`frontend/` `backend/` `evaluator/` `problems/` `deploy/` `docs/` `hw/`
- `Makefile`：`setup` `dev` `check` `test-eval` `build` `migrate`
- `environment-main.yml` / `environment-tirx.yml` + 两个 lock 文件
- `hw/h100-pcie.yaml`
- `.gitignore`、`.editorconfig`、`.clang-format`、`ruff.toml`、pre-commit
- 空的 `problems/`、`data/`（gitignored）

**验收**：`make setup` 在一台干净机器上跑完，两个环境都能 `import` 各自的包；`nvidia-smi` 与 `ncu --version` 正常。

---

## 阶段 1：题目与评分核心（1.5 天）

这是整个系统的地基。**UI 一行都还没有**，全部用 CLI 驱动。

**产出**

- `problems/<slug>/` 的 schema 校验（pydantic 模型）
- `manage new-problem` / `validate-problem` / `generate-groundtruth` / `check-groundtruth`
- `evaluator/runner/`：输入分配、ground truth 比对、容差两条、L2 flush 计时、评分公式
- `evaluator/adapters/cuda.py`（只做 CUDA，先把一条路走通）
- 两道题：`vector-add-fp32`（简单，验证链路）与 `matmul-bf16`（有计算也有访存）
- 单测：评分函数的边界（全 0 输出、NaN、`t=0`、`t_lb=0`）

**验收**：

```bash
python -m manage validate-problem matmul-bf16
python -m manage generate-groundtruth --all
python -m manage run-local matmul-bf16 --language cuda --file solutions/matmul_naive.cu
# → 打印每个测试点的效率与分数，naive 实现应落在 0.1–0.3 效率区间
```

---

## 阶段 2：评测引擎完整化（2 天）

**产出**

- `evaluator/adapters/{tilelang,tirx,cutedsl}.py`，按 [`languages.md`](languages.md) 的产物提取
- 两个 conda 环境的进程切换
- 沙箱子进程：编译/运行/超时/崩溃隔离，每测试点一次 fork
- `shadow_shape` 校验、随机输入复验（防作弊）
- 编译产物落盘 `data/artifacts/`
- `evaluator/worker/`：队列消费、GPU 空闲闸门（阻塞等待 + 拿到任务后 3s 轮询）
- `ncu` 采集与 CSV 解析（`--set basic`），指标转结构化
- `manage smoke-test`：向量加法端到端，断言分数在预期区间

**验收**：

```bash
python -m evaluator.worker --gpu 0 &
python -m manage enqueue matmul-bf16 --language tilelang --file t.py
# → 四个语言各跑通一次，PTX/SASS 都能取出
```

---

## 阶段 3：API 与数据层（1.5 天）

**产出**

- SQLite schema 与迁移（`manage migrate`）
- `backend/`：题目、提交、认证、SSE
- 提交队列投递与结果回写
- GitHub OAuth
- 限流、请求体校验、错误形状统一
- OpenAPI 自动文档

**验收**：`pytest tests/api` 全绿；用 `curl` 走完"登录 → 提交 → 收 SSE → 拿结果"。

---

## 阶段 4：前端（4 天）

按 [`frontend.md`](frontend.md) 的组件清单顺序做。

**产出**

- 设计 token、字体自托管、双主题、`ThemeProvider`
- `AppShell`、`ProblemTable`（含 sparkline）、`MarkdownView`（含 LaTeX）
- `CodeEditor`：四种语言注册、两套 Monaco 主题、自动保存、静态补全、WASM 格式化
- `ArtifactTabs`、`SubmitBar`、`EfficiencyRail`、`ResultTray`
- `ProfilingPanel`（抽屉 + 分层指标 + 规则化结论）
- 文档路由 `/docs/*`，直接渲染 `docs/` 下的 Markdown
- 无障碍：焦点环、键盘操作、对比度校验

**验收**：

- 亮暗两套都截图检查
- 断网时编辑器仍可写、可格式化
- 从题目列表到提交出结果，全程不需要刷新页面

---

## 阶段 5：部署与打磨（1 天）

**产出**

- `deploy/`：三个 systemd unit、nginx 模板、`install.sh`、`env.example`
- 部署检查单脚本化（`manage doctor`）
- 备份（`manage backup`）
- README 定稿，AGENTS/CONTRIBUTING 与实际代码对齐
- `make test-eval` 作为 CI 的 GPU 任务（可选）

**验收**：在一台干净的 GPU 机器上从零部署，跑通检查单全部六项。

---

## 依赖关系

```
0 ──▶ 1 ──▶ 2 ──▶ 3 ──▶ 4
              └──────────▶ 5
```

阶段 4 可以在阶段 3 的接口确定之后立刻并行开始（前端对着 OpenAPI 写），但 SSE 与提交状态机要等 3 完成才能联调。

## 风险与预留

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| TIRx 在 sm_90 上无优化 kernel，分数偏低且不报错 | 用户困惑 | 题目页与文档明示；阶段 2 实测后记录基线分数 |
| CuTe DSL 的 `__ptx__` 属性名在 4.x 后续版本变化 | 产物提取失败 | 适配器里做属性存在性检查，降级为"仅 SASS"并告警 |
| WASM clang-format 初始化慢 | 格式化首次卡顿 | 首次进入编辑器时后台预热 Worker |
| 多卡部署时 worker 内存（每进程 1.5 GB torch） | 显存无关，但宿主机 RSS 涨 | 文档注明；必要时 worker 改为按需 fork 解释器 |
| 题目下界公式写错 | 分数失真但不报错 | `validate-problem` 检查正负与量级；新题上线前跑一份 naive 实现确认效率 < 0.5 |

## 工期合计

约 10.5 人日（阶段 0–5）。前端占大头，但其中相当一部分是既有组件的拼装。

## 不做

明确排除，避免范围蔓延：

- 排行榜页面（题目列表里有个人最好成绩与分布，够了）
- 用户代码的私有可见性
- 自动调优建议
- 跨提交的指标趋势图
- 移动端编辑（移动端只做只读浏览）
- 多机部署的调度器（多卡用多进程，暂不做跨机器）
- 题目导入/导出格式
