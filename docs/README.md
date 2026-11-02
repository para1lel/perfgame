# 文档索引

这组文档是**规格**，不是实现笔记。代码应当符合这里写的内容；两者不一致时，先确认哪一边错了，再同步修改。

## 按角色阅读

**想部署一套自己的实例**
1. [`architecture.md`](architecture.md) — 先弄清有哪几个进程
2. [`deployment.md`](deployment.md) — 环境、systemd、Nginx
3. [`problems.md`](problems.md#生成与校验) — 生成题目数据

**想加题目**
1. [`problems.md`](problems.md)
2. [`evaluation.md`](evaluation.md#容差) — 容差怎么定
3. [`evaluation.md`](evaluation.md#5-理论下界与评分) — 下界函数怎么写

**想改评测引擎**
1. [`evaluation.md`](evaluation.md)
2. [`languages.md`](languages.md)
3. [`api.md`](api.md#状态机)

**想改界面**
1. [`frontend.md`](frontend.md)
2. [`profiling.md`](profiling.md#面板布局) — profiling 面板的指标层级

**想知道某个决定为什么这么做**
→ [`decisions.md`](decisions.md)

## 文档清单

| 文件 | 内容 |
| --- | --- |
| [`architecture.md`](architecture.md) | 进程模型、组件边界、一次提交的完整生命周期、多 GPU 调度 |
| [`evaluation.md`](evaluation.md) | 评测契约、参考实现与 ground truth、容差两条、计时协议、评分公式、错误处理 |
| [`languages.md`](languages.md) | 四种语言的调用契约、编译器适配接口、产物提取方式、环境归属 |
| [`problems.md`](problems.md) | 题目目录布局、`problem.yaml` 全字段、`compute_bounds.py` 契约、生成与校验流程 |
| [`profiling.md`](profiling.md) | ncu 采集分层、指标清单与优先级、面板布局、数据转换 |
| [`api.md`](api.md) | REST 端点、SSE 事件、认证、提交状态机、错误码 |
| [`frontend.md`](frontend.md) | 设计语言、色彩与排版 token、字体、主题切换、路由、编辑器行为 |
| [`deployment.md`](deployment.md) | 两个 conda 环境、硬件 profile、systemd unit、Nginx、备份 |
| [`decisions.md`](decisions.md) | 决策记录表：每条含背景、选项、结论、代价 |
| [`implementation-plan.md`](implementation-plan.md) | 分六阶段的实施顺序、每阶段的验收标准 |

## 文档约定

- 涉及数字的地方给具体值，不写"若干"。
- 公式用 LaTeX 写，渲染由前端 Markdown 管线负责。
- 引用代码时写路径，如 `evaluator/runner/timing.py`。
- 新增一节就要在本文档清单里加一行。
