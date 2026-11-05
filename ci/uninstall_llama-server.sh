#!/bin/bash

available() { command -v $1 >/dev/null; }
SUDO=
if [ "$(id -u)" -ne 0 ]; then
    # Running as root, no need for sudo
    if ! available sudo; then
        error "This script requires superuser permissions. Please re-run as root."
    fi

    SUDO="sudo"
fi

# 获取当前执行脚本的用户
if [ -n "$SUDO_USER" ]; then
    CURRENT_USER=$SUDO_USER
else
    CURRENT_USER=$USER
fi

echo "--------------------------------------------------------"
echo "正在卸载 llama-server 服务..."
echo "当前用户: ${CURRENT_USER}"
echo "--------------------------------------------------------"

# 停止服务
$SUDO systemctl stop llama-server.service
echo "✓ 服务已停止"

# 禁用服务
$SUDO systemctl disable llama-server.service
echo "✓ 服务已禁用"

# 删除服务文件
$SUDO rm -f /etc/systemd/system/llama-server.service
echo "✓ 服务文件已删除"

# 重新加载systemd配置
$SUDO systemctl daemon-reload
echo "✓ systemd配置已重新加载"

# 清理.bashrc中的环境变量配置
MARKER_START="# llama-server start"
MARKER_END="# llama-server end"
$SUDO sed -i "/${MARKER_START}/,/${MARKER_END}/d" /home/${CURRENT_USER}/.bashrc
echo "✓ 环境变量配置已从 .bashrc 中移除"

echo ""
echo "--------------------------------------------------------"
echo "llama-server 服务卸载完成！"
echo "--------------------------------------------------------"
