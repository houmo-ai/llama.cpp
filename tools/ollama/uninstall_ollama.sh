#! /bin/bash

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
$SUDO systemctl stop HLAWOllama.service
$SUDO systemctl disable HLAWOllama.service
$SUDO rm -f /etc/systemd/system/HLAWOllama.service
$SUDO systemctl daemon-reload

MARKER_START="# HLAWOllama start"
MARKER_END="# HLAWOllama end"
$SUDO sed -i "/${MARKER_START}/,/${MARKER_END}/d" /home/${CURRENT_USER}/.bashrc