#!/bin/bash
cd "$(dirname "$0")"

VENV_DIR="$(pwd)/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

# 优先使用 Homebrew Python 创建项目独立环境。
if [ -x /opt/homebrew/bin/python3 ]; then
  BASE_PYTHON=/opt/homebrew/bin/python3
elif [ -x /usr/local/bin/python3 ]; then
  BASE_PYTHON=/usr/local/bin/python3
elif command -v python3 >/dev/null 2>&1; then
  BASE_PYTHON="$(command -v python3)"
else
  echo "[ERROR] 找不到 Python 3，请先安装 Python 3.10+"
  echo "https://www.python.org/downloads/"
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "首次启动：正在创建项目 Python 环境..."
  "$BASE_PYTHON" -m venv "$VENV_DIR" || exit 1
fi

if ! "$PYTHON_BIN" -c 'import fastapi, uvicorn, requests, pydantic, multipart, httpx, PIL, websockets' >/dev/null 2>&1; then
  echo "正在安装项目依赖..."
  "$PYTHON_BIN" -m pip install --no-index --find-links=packages -r requirements.txt >/dev/null 2>&1 || \
    "$PYTHON_BIN" -m pip install -r requirements.txt || exit 1
fi

LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)"
if [ -z "$LAN_IP" ]; then
  LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
if [ -z "$LAN_IP" ]; then
  LAN_IP="127.0.0.1"
fi
APP_URL="http://${LAN_IP}:3000/"

echo "Starting ComfyUI-API-Modelscope..."
echo "Visit: ${APP_URL}"
echo "Local: http://127.0.0.1:3000/"
echo "Press Ctrl+C to stop."
echo ""

# Open browser after 3 seconds
sleep 3 && open "${APP_URL}" &

"$PYTHON_BIN" main.py

echo ""
echo "Server stopped."
