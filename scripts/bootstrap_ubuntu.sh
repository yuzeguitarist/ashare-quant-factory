#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

install_venv_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[AQF] apt-get not found. Please install python3-venv manually."
    return 1
  fi

  local py_minor
  py_minor="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

  local -a apt_cmd
  if command -v sudo >/dev/null 2>&1 && [ "$(id -u)" -ne 0 ]; then
    apt_cmd=(sudo apt-get)
  else
    apt_cmd=(apt-get)
  fi

  echo "[AQF] Installing venv related packages..."
  "${apt_cmd[@]}" update
  if ! "${apt_cmd[@]}" install -y "python${py_minor}-venv" python3-venv python3-full; then
    echo "[AQF] Fallback install: python3-venv python3-full"
    "${apt_cmd[@]}" install -y python3-venv python3-full
  fi
}

create_venv() {
  if [ -x ".venv/bin/python" ]; then
    echo "[AQF] Reusing existing .venv"
    return 0
  fi

  echo "[AQF] Creating venv..."
  if python3 -m venv .venv; then
    return 0
  fi

  echo "[AQF] python3 venv failed. Trying to install missing packages..."
  install_venv_packages

  rm -rf .venv
  echo "[AQF] Retrying venv creation..."
  python3 -m venv .venv
}

install_global_aqf_command() {
  local source_cmd="${ROOT_DIR}/.venv/bin/aqf"
  local target_cmd="/usr/local/bin/aqf"

  if [ ! -x "${source_cmd}" ]; then
    return 0
  fi

  if ln -sf "${source_cmd}" "${target_cmd}" >/dev/null 2>&1; then
    echo "[AQF] Installed command: ${target_cmd} -> ${source_cmd}"
    return 0
  fi

  if command -v sudo >/dev/null 2>&1 && [ "$(id -u)" -ne 0 ]; then
    if sudo ln -sf "${source_cmd}" "${target_cmd}" >/dev/null 2>&1; then
      echo "[AQF] Installed command with sudo: ${target_cmd} -> ${source_cmd}"
      return 0
    fi
  fi

  echo "[AQF] Cannot write ${target_cmd}. You can still run: ${source_cmd}"
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.11+ first."
  exit 1
fi

create_venv
source .venv/bin/activate

echo "[AQF] Upgrading pip..."
pip install -U pip wheel

echo "[AQF] Installing dependencies..."
pip install -r requirements.txt
pip install -e .
install_global_aqf_command

mkdir -p data/reports

echo ""
echo "[AQF] Done."
echo "Next steps:"
echo "  1) ./.venv/bin/aqf setup --open-gmail-guide"
echo "  2) ./.venv/bin/aqf doctor"
echo "  3) ./.venv/bin/aqf run-once"
echo "  4) (optional) if /usr/local/bin/aqf exists, you can use: aqf ..."
