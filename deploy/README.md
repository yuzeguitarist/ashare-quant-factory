# Deploy to DigitalOcean (4C/8GB)

推荐路径：把仓库放到 `/opt/ashare-quant-factory`，并用 systemd 常驻运行。

## 1) 拉取代码
```bash
sudo mkdir -p /opt
sudo chown $USER:$USER /opt
cd /opt
git clone https://github.com/yourname/ashare-quant-factory.git
cd ashare-quant-factory
```

## 2) 安装依赖
```bash
bash scripts/bootstrap_ubuntu.sh
```

## 3) 配置
```bash
cp config.example.yaml config.yaml
cp .env.example .env
nano config.yaml
nano .env
```

## 4) 试跑
```bash
. .venv/bin/activate
aqf doctor
aqf run-once --force --skip-poll
```

## 5) systemd
```bash
sudo bash scripts/install_systemd.sh
sudo systemctl status aqf
sudo journalctl -u aqf -f
```
