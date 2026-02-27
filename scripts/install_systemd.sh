#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${ROOT_DIR}/config.yaml" ]]; then
  echo "[AQF] config.yaml not found. Please create it from config.example.yaml first."
  exit 1
fi

echo "[AQF] Installing systemd service..."

sudo mkdir -p /etc/aqf
sudo cp "${ROOT_DIR}/config.yaml" /etc/aqf/config.yaml

# .env is optional but strongly recommended
if [[ -f "${ROOT_DIR}/.env" ]]; then
  sudo cp "${ROOT_DIR}/.env" /etc/aqf/aqf.env
  sudo chmod 600 /etc/aqf/aqf.env
else
  echo "[AQF] Warning: .env not found. You can create /etc/aqf/aqf.env later."
fi

sudo sed "s|/opt/ashare-quant-factory|${ROOT_DIR}|g" "${ROOT_DIR}/deploy/systemd/aqf.service" | sudo tee /etc/systemd/system/aqf.service >/dev/null

sudo systemctl daemon-reload
sudo systemctl enable --now aqf.service

echo "[AQF] systemd service installed: aqf.service"
echo "Check status: sudo systemctl status aqf"
echo "Follow logs : sudo journalctl -u aqf -f"
