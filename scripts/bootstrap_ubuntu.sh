#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.11+ first."
  exit 1
fi

if ! python3 -c "import venv" >/dev/null 2>&1; then
  echo "[AQF] python3 venv module is missing. Installing python3-venv..."
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3-venv
  else
    apt-get update
    apt-get install -y python3-venv
  fi
fi

echo "[AQF] Creating venv..."
python3 -m venv .venv
source .venv/bin/activate

echo "[AQF] Upgrading pip..."
pip install -U pip wheel

echo "[AQF] Installing dependencies..."
pip install -r requirements.txt
pip install -e .

mkdir -p data/reports

echo ""
echo "[AQF] Done."
echo "Next steps:"
echo "  1) cp config.example.yaml config.yaml"
echo "  2) cp .env.example .env"
echo "  3) edit config.yaml and .env"
echo "  4) source .venv/bin/activate && aqf doctor && aqf run-once"
