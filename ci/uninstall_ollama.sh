#! /bin/bash
available() { command -v $1 >/dev/null; }
error() { echo "错误: $*" >&2; exit 1; }
SUDO=
if [ "$(id -u)" -ne 0 ]; then
    # Running as non-root, sudo is required
    if ! available sudo; then
        error "This script requires superuser permissions. Please re-run as root."
    fi
    SUDO="sudo"
fi
# 获取当前执行脚本的用户
if [ -n "$SUDO_USER" ]; then
    CURRENT_USER=$SUDO_USER
else
    CURRENT_USER=${USER:-$(id -un)}
fi

CURRENT_HOME="$(getent passwd "$CURRENT_USER" | cut -d: -f6)"
if [ -z "$CURRENT_HOME" ]; then
    if [ "$CURRENT_USER" = "root" ]; then
        CURRENT_HOME="/root"
    else
        CURRENT_HOME="/home/${CURRENT_USER}"
    fi
fi

BASHRC_PATH="${CURRENT_HOME}/.bashrc"

$SUDO systemctl stop HLAWOllama.service
$SUDO systemctl disable HLAWOllama.service
$SUDO rm -f /etc/systemd/system/HLAWOllama.service
$SUDO systemctl daemon-reload
MARKER_START="# HLAWOllama start"
MARKER_END="# HLAWOllama end"
$SUDO touch "${BASHRC_PATH}"
$SUDO sed -i "/${MARKER_START}/,/${MARKER_END}/d" "${BASHRC_PATH}"
