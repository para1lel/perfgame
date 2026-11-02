# Contributing

## 环境准备

```bash
git clone <repo-url> perfgame && cd perfgame
make setup
```

`make setup` 会做四件事：创建两个 conda 环境（`perfgame-main`、`perfgame-tirx`）、安装前端依赖、安装 pre-commit hooks、为 `problems/` 下的题目生成 ground truth。

只有一个环境是不够的——TileLang 与 TIRx 的依赖冲突见 [`docs/deployment.md`](docs/deployment.md#为什么需要两个环境)。

## 代码规范

| 范围 | 工具 |
| --- | --- |
| Python | `ruff format` + `ruff check`（含 import 排序、未使用变量） |
| TypeScript / React | `prettier` + `eslint` |
| CUDA C++ | `clang-format`，配置在仓库根的 `.clang-format` |
| Commit message | Conventional Commits：`feat:` `fix:` `docs:` `refactor:` `test:` `chore:` |

提交前跑 `make check`。CI 会跑同一套。

## 常见贡献路径

### 添加一道题目

见 [`docs/problems.md`](docs/problems.md)。最短路径：

```bash
python -m manage new-problem attention-flash-bf16
# 编辑 problems/attention-flash-bf16/{problem.yaml,reference.py,compute_bounds.py}
python -m manage validate-problem attention-flash-bf16
python -m manage generate-groundtruth attention-flash-bf16
```

`validate-problem` 会检查 schema、跑一遍参考实现、确认下界函数返回的键齐全。题目合并前必须通过。

### 添加一门语言支持

1. 在 `evaluator/adapters/` 下新建模块，实现 `docs/languages.md` 里的 `LanguageAdapter` 协议：`compile()`、`artifacts()`、`entry_symbol()`。
2. 在 `evaluator/contracts/` 里补这个语言的入口模板（用户新建文件时的骨架）。
3. 前端 `frontend/src/languages.ts` 注册语言元数据（显示名、文件扩展名、默认模板、展示哪些产物标签）。
4. 在 `docs/languages.md` 的对照表补一行，并在 `docs/decisions.md` 记录选型。
5. 加一个端到端测试：用该语言写一个向量加法，断言能编译、能跑、分数在合理区间。

### 修改评分或容差规则

评分逻辑是 `evaluator/runner/scoring.py` 里的纯函数，不碰 GPU。改动前先读 [`docs/evaluation.md`](docs/evaluation.md)。任何改动都要：

- 补单元测试（含边界：全 0 输出、NaN 输出、时间为 0）
- 跑 `make test-eval` 确认既有题目的基线分数漂移小于 ±3%
- 在 `docs/decisions.md` 记一条

### 修改前端视觉

先读 [`docs/frontend.md`](docs/frontend.md) 的 token 表。颜色、间距、字号只从 token 取值，不要在组件里写魔法数。改完同时检查亮色与暗色两套，`make dev` 后按 `frontend/scripts/a11y-check` 跑一遍对比度校验。

## 测试

```bash
make check          # lint + 单测 + 契约测试（不需要 GPU）
make test-eval      # 需要 GPU：端到端跑基线 kernel
make test-frontend  # vitest + playwright
```

`make test-eval` 会占住一张 GPU 几分钟，本地跑之前确认没有别人的任务。

## PR 流程

1. Fork 并建分支：`feat/problem-flash-attention` 这类命名。
2. 改动带测试。只改文档的 PR 可以不附测试。
3. PR 描述里写清：改了什么、为什么、如何验证。涉及评分的改动附上分数前后对比。
4. 至少一次 review。

## 提交内容不设隐私

本项目的提交记录、代码、分数对所有访客可见，包括未登录用户。不要往提交里粘贴密钥或私有代码。
