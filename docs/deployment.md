# 部署

## 为什么需要两个环境

TileLang 与 TIRx 对 `apache-tvm-ffi` 的版本区间不相交：

| 包 | 约束 |
| --- | --- |
| TileLang 0.1.15 | `apache-tvm-ffi>=0.1.11,<0.1.13` |
| apache-tvm 0.27.0.post1（TIRx） | `apache-tvm-ffi>=0.1.14.post0` |

pip 无法同时满足。任何"装到一个环境里"的尝试都会失败，不要试。

因此两份环境定义，都进仓库：

```
environment-main.yml   # Python 3.12 + torch + tilelang + nvidia-cutlass-dsl + cutlass 头文件
environment-tirx.yml   # Python 3.12 + apache-tvm + torch
```

两份都锁到具体的 `pip freeze` 输出（`requirements-main.lock`、`requirements-tirx.lock`），YAML 只做顶层声明与渠道配置。

**Python 版本固定 3.12。** 四种技术栈都覆盖 3.12；3.14 虽然多数也支持，但最新 ABI 上的边缘问题不值得赌。

## 环境内容

### `environment-main.yml`

```yaml
name: perfgame-main
channels: [conda-forge]
dependencies:
  - python=3.12
  - pip
  - cuda-toolkit=13.0.*        # 提供 nvcc / cuobjdump / nvdisasm
  - pip:
      - torch==2.13.0
      - tilelang==0.1.15
      - nvidia-cutlass-dsl==4.8.1
      - nvidia-cutlass-dsl[sass]
      - cuda-python>=11.8
      - nvidia-ml-py             # pynvml
      - numpy
      - fastapi
      - uvicorn[standard]
      - pydantic>=2
      - sqlalchemy
      - redis
      - httpx
      - python-jose               # OAuth state 校验
```

CUTLASS C++ 头文件从 release tarball 解到 `third_party/cutlass/`（不进 Git，由 `make setup` 下载并校验 SHA256）。CUDA 适配器用 `-I$(pwd)/third_party/cutlass/include`。

### `environment-tirx.yml`

```yaml
name: perfgame-tirx
channels: [conda-forge]
dependencies:
  - python=3.12
  - pip
  - cuda-toolkit=13.0.*
  - pip:
      - apache-tvm==0.27.0.post1
      - torch==2.13.0
      - numpy
      - nvidia-ml-py
```

TIRx 环境不需要 Web 依赖——它只跑 worker 的子进程。

## 硬件 profile

`hw/h100-pcie.yaml`：

```yaml
name: H100 PCIe
arch: sm_90
arch_suffix: sm_90a          # nvcc -arch
device_name_match: "NVIDIA H100 PCIe"   # pynvml 匹配用

# 标称峰值，密集（非稀疏）。来源：NVIDIA H100 数据表
bandwidth:
  hbm_bytes_per_s: 2.0e12

flops:                        # 单位：FLOP/s（张量核为密集）
  fp32: 48.0e12               # CUDA core
  fp64: 24.0e12
  tf32: 400.0e12
  bf16: 800.0e12
  fp16: 800.0e12
  fp8:  1600.0e12
  int8: 1600.0e12
  # fp4: Hopper 不支持，不要写

l2_bytes: 50_000_000
sm_count: 114
```

- `fp4` 键的缺失即表示该精度不可用；`compute_bounds` 若返回 `flops_fp4` 而 profile 无此键，评分时该项被忽略并在结果页告警。
- 峰值取**标称值**，不取实测值。理由：实测值因时钟、温度而异，会让同一份代码在不同时间得到不同分数。标称值是稳定的、公开可查的。
- worker 启动时用 `pynvml` 读设备名，匹配 `device_name_match`，不匹配则拒绝启动并打印提示——防止在 A100 机器上套用 H100 的峰值配置文件。
- 新增硬件：加一个 `hw/<name>.yaml`，用 `--hardware` 参数指定。

## 安装

```bash
make setup
```

等价于：

```bash
conda env create -f environment-main.yml
conda env create -f environment-tirx.yml
npm ci --prefix frontend
mkdir -p data/{groundtruth,artifacts,profiles}
curl -fsSL <cutlass-tarball-url> | tar -xz -C third_party/    # 并校验 sha256
conda run -n perfgame-main python -m manage generate-groundtruth --all
```

最后一步需要 GPU。没有 GPU 的机器上可以部署 API 与前端，worker 单独部署在有卡的机器（见下）。

## 服务

三个 systemd unit，`deploy/` 下提供模板与 `install.sh`。

### `perfgame-api.service`

```ini
[Unit]
Description=PerfGame API
After=network.target

[Service]
Type=simple
User=perfgame
WorkingDirectory=/opt/perfgame
EnvironmentFile=/etc/perfgame/env
ExecStart=/opt/perfgame/.conda/envs/perfgame-main/bin/uvicorn \
          backend.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### `perfgame-worker@.service`

模板 unit，实例号即 GPU 序号：

```ini
[Service]
Type=simple
User=perfgame
WorkingDirectory=/opt/perfgame
Environment=CUDA_VISIBLE_DEVICES=%i
Environment=PERFGAME_CONDA_MAIN=/opt/perfgame/.conda/envs/perfgame-main
Environment=PERFGAME_CONDA_TIRX=/opt/perfgame/.conda/envs/perfgame-tirx
ExecStart=/opt/perfgame/.conda/envs/perfgame-main/bin/python -m evaluator.worker --gpu %i
Restart=on-failure
RestartSec=5
```

每个 GPU 一个实例：

```bash
systemctl enable --now perfgame-worker@0
systemctl enable --now perfgame-worker@1   # 多卡时
```

`CUDA_VISIBLE_DEVICES=%i` 让进程只看见那一张卡，worker 内部的设备序号恒为 0。这样"抢卡"在物理上不可能发生。

### `perfgame-frontend.service`

前端构建产物由 Nginx 直接服务，不需要独立进程。可选地用一个 unit 只负责在部署时 `npm run build`。

## Nginx

```nginx
server {
    listen 80;
    server_name _;
    root /opt/perfgame/frontend/dist;
    index index.html;

    # 字体自托管，长缓存
    location /fonts/ {
        add_header Cache-Control "public, max-age=31536000, immutable";
    }

    # 带 hash 的构建产物
    location /assets/ {
        add_header Cache-Control "public, max-age=31536000, immutable";
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE 必须关缓冲并放长超时
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
    }

    # SPA 回退
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

几个必须的点：

- `location /api/` 里的 `proxy_buffering off` 是 SSE 能不能实时到达的前提。忘记这条，事件会攒到缓冲区满才吐出。
- `proxy_read_timeout` 必须大于心跳间隔（20 s）很多，否则空闲连接被掐断反复重连。
- 不改 `X-Accel-Buffering` 也行，`proxy_buffering off` 已足够。

## 配置

`/etc/perfgame/env`（不进 Git，`deploy/env.example` 提供模板）：

```
PERFGAME_DB=sqlite:////opt/perfgame/data/perfgame.db
PERFGAME_DATA_DIR=/opt/perfgame/data
PERFGAME_HW_PROFILE=/opt/perfgame/hw/h100-pcie.yaml
PERFGAME_QUEUE=redis://127.0.0.1:6379/0
PERFGAME_GITHUB_CLIENT_ID=...
PERFGAME_GITHUB_CLIENT_SECRET=...
PERFGAME_SESSION_SECRET=...
PERFGAME_PUBLIC_URL=https://perfgame.example.com
PERFGAME_ALLOWED_ORIGINS=https://perfgame.example.com
```

`PERFGAME_QUEUE=memory://` 时用进程内队列——单机、单 worker 的最简部署可以这样，省掉 Redis。多 worker 必须用 Redis。

## 部署检查单

```bash
# 1. GPU 与驱动
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# 2. profiling 权限（必须是 0）
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly

# 3. 工具链
nvcc --version && which cuobjdump nvdisasm ncu

# 4. 两个环境都能 import
conda run -n perfgame-main python -c "import torch, tilelang, cutlass; print('ok')"
conda run -n perfgame-tirx python -c "import tvm; print(tvm.__version__)"

# 5. ground truth 齐备
conda run -n perfgame-main python -m manage check-groundtruth

# 6. 端到端
conda run -n perfgame-main python -m manage smoke-test --gpu 0
```

`smoke-test` 会用一份已知正确的 vector-add kernel 走完整链路，断言分数落在 0.9–1.1 倍预期。它失败意味着部署有问题，不要上线。

## 备份

| 对象 | 方法 | 频率 |
| --- | --- | --- |
| SQLite | `sqlite3 data/perfgame.db ".backup data/backup.db"` | 每日 |
| ground truth | 可重算，不备份 | — |
| 编译产物与 profile | 可选；丢了重测即可 | — |

SQLite 用 WAL 模式，备份必须用 `.backup` 而不是 `cp`（`cp` 会漏掉 `-wal` 里的已提交事务）。

## 升级

```bash
git pull
conda env update -f environment-main.yml --prune
conda run -n perfgame-main python -m manage migrate
systemctl restart perfgame-api perfgame-worker@*
npm ci --prefix frontend && npm run build --prefix frontend
```

题目 schema 变化时 `migrate` 会校验并提示需要重新生成 ground truth 的题目。加题不需要重启服务（题目按 mtime 缓存失效）。
