# 部署

本文件是速查。完整说明见 [`../docs/deployment.md`](../docs/deployment.md)。

## 三个 unit

```bash
sudo cp deploy/*.service /etc/systemd/system/
sudo cp deploy/nginx.conf /etc/nginx/sites-available/perfgame
sudo ln -s /etc/nginx/sites-available/perfgame /etc/nginx/sites-enabled/

sudo systemctl daemon-reload
sudo systemctl enable --now perfgame-api nginx
sudo systemctl enable --now perfgame-worker@0
```

多卡：`systemctl enable --now perfgame-worker@1` 以此类推。

## 检查

```bash
/opt/perfgame/.conda/envs/perfgame-main/bin/python -m manage doctor
```

六项检查全通过才能上线：GPU 与驱动、profiling 权限、工具链、两个环境可 import、ground truth 齐备、端到端 smoke test。
