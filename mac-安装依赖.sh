#!/bin/bash
cd "$(dirname "$0")"
echo "============================================"
echo "   Installing dependencies"
echo "============================================"
echo ""

# Check Python
if [ -x /opt/homebrew/bin/python3 ]; then
    BASE_PYTHON=/opt/homebrew/bin/python3
elif [ -x /usr/local/bin/python3 ]; then
    BASE_PYTHON=/usr/local/bin/python3
elif command -v python3 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3)"
else
    echo "[ERROR] Python not found. Please install Python 3.10+"
    echo "Download: https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
fi

"$BASE_PYTHON" --version

VENV_DIR="$(pwd)/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Creating isolated project environment..."
    "$BASE_PYTHON" -m venv "$VENV_DIR" || exit 1
fi

echo ""
echo "[1/2] Checking pip..."
"$PYTHON_BIN" -m pip --version &> /dev/null
if [ $? -ne 0 ]; then
    echo "pip not found, bootstrapping..."
    "$PYTHON_BIN" -m ensurepip --upgrade
fi

echo "[2/2] Installing from local packages folder..."
"$PYTHON_BIN" -m pip install --no-index --find-links=packages -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[WARN] Offline install failed, trying online..."
    "$PYTHON_BIN" -m pip install -r requirements.txt
fi

echo ""
echo "Done. Run './mac-启动服务.sh' to start."
read -p "Press Enter to exit..."
