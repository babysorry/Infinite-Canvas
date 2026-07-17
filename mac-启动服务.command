#!/bin/bash
# 修复权限并启动服务
# 双击运行即可

cd "$(dirname "$0")"

echo "============================================"
echo "   ComfyUI-API-Modelscope"
echo "============================================"
echo ""
echo "修复权限中..."

# 移除安全限制（只针对实际存在的文件类型）
xattr -r -d com.apple.quarantine mac-启动服务.command mac-修复权限.command 2>/dev/null
xattr -r -d com.apple.quarantine main.py 2>/dev/null

# 只修复启动脚本权限，避免每次启动都改变其他受 Git 管理文件的模式。
chmod +x mac-启动服务.command mac-修复权限.command mac-启动服务.sh mac-安装依赖.sh 2>/dev/null

echo "权限已修复！"
echo ""

# 清理占用 3000 端口的旧进程，避免 address already in use
OLD_PID=$(lsof -ti :3000 2>/dev/null)
if [ -n "$OLD_PID" ]; then
    echo "检测到 3000 端口被占用，正在停止旧进程 (PID: $OLD_PID)..."
    kill $OLD_PID 2>/dev/null
    sleep 1
    # 仍未退出则强制结束
    if lsof -ti :3000 >/dev/null 2>&1; then
        kill -9 $(lsof -ti :3000) 2>/dev/null
    fi
    echo "旧进程已停止。"
    echo ""
fi

echo "正在启动服务..."
echo "本机访问： http://127.0.0.1:3000/"
echo "============================================"
echo ""

# 统一交给项目启动脚本；它会创建独立 Python 环境并检查依赖。
exec /bin/bash ./mac-启动服务.sh
