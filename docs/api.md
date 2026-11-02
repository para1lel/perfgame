# API

REST + SSE。所有路径前缀 `/api`。响应体一律 JSON，时间戳为带时区的 ISO 8601。

## 认证

GitHub OAuth（Authorization Code）。后端不存密码。

```
GET  /api/auth/github/login     → 302 到 GitHub
GET  /api/auth/github/callback  → 建 session，302 回前端
POST /api/auth/logout           → 清 session
GET  /api/auth/me               → 当前用户或 401
```

- Session 用 HttpOnly + SameSite=Lax 的 cookie，30 天滚动过期。
- `users` 表存 `github_id`、`username`、`display_name`、`avatar_url`，不存 email。
- **不需要登录也能浏览**：题目、提交列表、提交详情、文档全部公开。只有提交与 Profile 需要登录。
- 没有权限系统：不存在"管理员"角色。题目由维护者改文件。

## 题目

```
GET /api/problems
→ [{ slug, title, category, difficulty, summary, test_case_count }]

GET /api/problems/{slug}
→ {
    slug, title, category, difficulty, description_md,
    signature, test_cases: [{name, shape}],   // 不含 seed 与输入值
    tolerance,
    bounds: {                                  // 每个测试点的下界，已换算
      small: { hbm_bytes, flops: {...}, t_lb_ms, bits_per_element },
      ...
    },
    templates: { cuda, tilelang, tirx, cutedsl },
    best_submission: { id, score, user } | null,
    submission_count: 312,
    score_histogram: [ ... ]                   // 20 个桶
  }
```

`bound.t_lb_ms` 已经用本机 hardware profile 换算过——前端不自己算分数。

## 提交

```
POST /api/submissions
  body: { problem_slug, language, code }
  → 201 { id, status: "queued", queue_position }

GET /api/submissions?problem_slug=&user=&language=&limit=50&cursor=
→ { items: [{ id, problem_slug, language, status, total_score,
               user, created_at }], next_cursor }

GET /api/submissions/{id}
→ {
    id, problem_slug, language, code, status, total_score,
    user, created_at, updated_at,
    results: [{
      test_case, passed, score, median_ms, efficiency,
      max_abs_error, correct_ratio, cosine, samples_ms: [...],
      error: { kind, message, log } | null
    }],
    attempts: [ { attempt: 1, created_at, total_score } ]
  }

POST /api/submissions/{id}/retest
  → 202 { status: "queued" }

POST /api/submissions/{id}/profile
  body: { test_case, level: "basic" | "detailed" }
  → 202 { status: "queued", queue_position }

GET /api/submissions/{id}/artifacts?test_case=&kind=ptx|sass|log
  → text/plain（不含 cuda_source 的降级标签由前端控制）

GET /api/submissions/{id}/profile?test_case=&attempt=
→ 结构化指标（见 profiling.md 的数据转换）+ ncu_rep_url
```

`code` 是公开字段：本项目的全部提交都对所有人可见，不需要隐私过滤。

## SSE

```
GET /api/events?submission_id={id}
```

事件流，`text/event-stream`。每条事件是 `event: <name>` + `data: <json>`：

| event | data | 说明 |
| --- | --- | --- |
| `status` | `{status, queue_position?, message?, progress?}` | 状态变更 |
| `result` | `{test_case, passed, score, ...}` | 单个测试点出结果 |
| `done` | `{total_score, status}` | 整次提交结束 |
| `error` | `{kind, message}` | 评测侧致命错误 |
| `heartbeat` | `{}` | 每 20 秒，保活 |

连接建立时先补发当前状态，避免"连上时已经过了中间态"。断线由浏览器自动重连，前端按 `submission_id` 重放即可，不需要续传游标。

## 状态机

```
queued ──▶ waiting_gpu ──▶ compiling ──▶ running ──▶ completed
                              │              │
                              ▼              ▼
                        compile_error    partial          （部分测试点失败，总分已出）
                              │
                              ▼
                           failed
```

`retest` 从 `completed` 回到 `queued`，`attempt` 加 1。`profile` 是独立任务，不改提交状态。

## 错误

统一形状：

```json
{ "error": { "code": "problem_not_found", "message": "题目 matmul-x 不存在" } }
```

| HTTP | `code` | 场景 |
| --- | --- | --- |
| 400 | `invalid_language` | language 不在枚举内 |
| 401 | `unauthenticated` | 未登录 |
| 403 | `forbidden` | 越权（当前无此类场景，保留） |
| 404 | `problem_not_found` / `submission_not_found` | |
| 409 | `already_running` | 对同一提交重复 retest |
| 413 | `code_too_large` | 超过 256 KB |
| 422 | `shape_mismatch` 等 | 请求体校验失败 |
| 429 | `rate_limited` | 同一用户 1 分钟内超过 6 次提交 |
| 503 | `no_worker` | 队列存在但没有 worker 在线 |

限流按用户，未登录者不能提交，所以不限。

## 实现要点

- 状态写入数据库与发 SSE 之间有窗口期。做法：写库后向 Redis channel `sub:{id}` publish，API 进程订阅并转发给该提交的 EventSource 连接；若没有 Redis（单机最简单部署），API 进程用 500ms 轮询数据库 `updated_at` 变化兜底。
- `/api/submissions` 用游标分页（`id < cursor`），不用 offset——提交是持续流入的，offset 会漏数据。
- 所有写接口要求 `Content-Type: application/json`，防 CSRF 的第一道；再加 `Origin` 白名单校验。
- 代码长度上限 256 KB。超过通常意味着粘贴了整个 cutlass 头文件。
