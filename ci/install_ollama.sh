#!/bin/bash
set -euo pipefail  # 开启严格模式，遇到错误立即退出

# 定义错误处理函数
error() {
    echo -e "\033[31m错误: $1\033[0m" >&2
    exit 1
}

# 检查命令是否存在
available() { 
    command -v "$1" >/dev/null 2>&1 
}

# ===================== 1. 权限判断 & SUDO 变量设置 =====================
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if available sudo; then
        SUDO="sudo"
    else
        error "此脚本需要超级用户权限，请以 root 身份重新运行（或安装 sudo）"
    fi
else
    echo "当前已以 root 用户运行，无需 sudo"
fi

# ===================== 2. 正确计算路径（适配 service 子目录） =====================
# 使用 BASH_SOURCE[0] 获取脚本实际路径（处理软链接）
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done

# 脚本所在目录（service）
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
# 项目根目录（service 的上级目录，即包含 bin/lib 的目录）
INSTALL_DIR="$(cd -P "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

# 验证核心目录/文件是否存在
[ -d "$INSTALL_DIR/bin" ] || error "bin 目录不存在: $INSTALL_DIR/bin"
[ -f "$INSTALL_DIR/bin/hlaw_ollama" ] || error "可执行文件不存在: $INSTALL_DIR/bin/hlaw_ollama"
[ -d "$INSTALL_DIR/lib" ] || error "lib 目录不存在: $INSTALL_DIR/lib"

# ===================== 3. 正确获取当前用户 =====================
if [ -n "$SUDO_USER" ]; then
    CURRENT_USER="$SUDO_USER"
    CURRENT_USER_HOME="/home/$CURRENT_USER"
    echo "当前通过 sudo 执行，原始用户：$CURRENT_USER（家目录：$CURRENT_USER_HOME）"
else
    CURRENT_USER="$USER"
    CURRENT_USER_HOME="$HOME"
    echo "未通过 sudo 执行，当前用户：$CURRENT_USER（家目录：$CURRENT_USER_HOME）"
fi

# 验证用户家目录存在
[ -d "$CURRENT_USER_HOME" ] || error "用户家目录不存在: $CURRENT_USER_HOME"

echo "--------------------------------------------------------"
echo "项目根目录（启动目录）: ${INSTALL_DIR}"
echo "脚本所在目录: ${SCRIPT_DIR}"
echo "服务将使用用户运行: ${CURRENT_USER}"
echo "--------------------------------------------------------"

# ===================== 4. 定义 systemd 服务文件 =====================
SERVICE_NAME="HLAWOllama.service"
SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}"
# 关键优化：环境变量使用绝对路径，避免相对路径问题；添加 WorkingDirectory
SERVICE_FILE_CONTENT=$(cat <<EOF
[Unit]
Description=HLAWOllama Service
After=network-online.target
Wants=network-online.target

[Service]
# 工作目录设为项目根目录
WorkingDirectory=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/bin/hlaw_ollama serve
# 执行用户/组
User=${CURRENT_USER}
Group=${CURRENT_USER}
# 重启策略
Restart=always
RestartSec=3
# 环境变量（使用绝对路径，避免依赖 .bashrc）
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="LD_LIBRARY_PATH=${INSTALL_DIR}/lib:\${LD_LIBRARY_PATH:-}"
Environment="PATH=${INSTALL_DIR}/bin:\${PATH:-}"

[Install]
WantedBy=multi-user.target
EOF
)

# ===================== 5. 生成并验证服务文件 =====================
echo "将要生成的 systemd 服务文件内容如下："
echo "----------------------------------------"
echo "${SERVICE_FILE_CONTENT}"
echo "----------------------------------------"

# 写入服务文件（强制覆盖）
echo "${SERVICE_FILE_CONTENT}" | $SUDO tee "${SERVICE_FILE_PATH}" >/dev/null || error "写入服务文件失败"

# 验证服务文件存在
[ -f "${SERVICE_FILE_PATH}" ] || error "服务文件创建失败: ${SERVICE_FILE_PATH}"

# ===================== 6. 设置目录/文件权限 =====================
echo "设置目录/文件权限..."
# 项目根目录权限（755 保证用户可读写，其他用户可读）
$SUDO chmod 755 "${INSTALL_DIR}"
# 递归设置项目目录归属（避免权限不足）
$SUDO chown -R "${CURRENT_USER}:${CURRENT_USER}" "${INSTALL_DIR}"
# 确保可执行文件有执行权限
$SUDO chmod +x "${INSTALL_DIR}/bin/hlaw_ollama"
# systemd 服务文件权限（必须 644，否则 systemd 识别失败）
$SUDO chmod 644 "${SERVICE_FILE_PATH}"

echo "服务文件已成功创建：${SERVICE_FILE_PATH}"

# ===================== 7. 安全更新 .bashrc（环境变量） =====================
MARKER_START="# HLAWOllama start"
MARKER_END="# HLAWOllama end"
BASHRC_FILE="${CURRENT_USER_HOME}/.bashrc"

echo "更新用户 .bashrc 环境变量（文件：${BASHRC_FILE}）..."
# 先删除旧的区块（避免重复）
$SUDO sed -i "/^${MARKER_START}/,/^${MARKER_END}/d" "${BASHRC_FILE}" || error "清理旧环境变量失败"

# 写入新的环境变量（使用 \${变量:-} 避免空值）
cat <<EOF | $SUDO tee -a "${BASHRC_FILE}" >/dev/null
${MARKER_START}
# HLAWOllama 环境变量（自动生成，请勿手动修改）
export PATH=${INSTALL_DIR}/bin:\${PATH:-}
${MARKER_END}
EOF

# 生效环境变量（仅对当前脚本生效，用户登录后自动生效）
echo "加载最新的 .bashrc 环境变量..."
source "${BASHRC_FILE}" || echo "提示：加载 .bashrc 时出现非致命错误，可忽略"

# ===================== 8. 启动/启用服务 =====================
echo "重新加载 systemd 配置并启动服务..."
$SUDO systemctl daemon-reload || error "daemon-reload 失败"
$SUDO systemctl enable --now "${SERVICE_NAME}" || error "启用/启动服务失败"

# 检查服务状态（仅输出，不中断脚本）
echo -e "\n服务状态检查："
$SUDO systemctl status "${SERVICE_NAME}" --no-pager || echo "提示：服务状态检查失败，可手动执行 sudo systemctl status ${SERVICE_NAME}"

# ===================== 9. 输出使用提示 =====================
echo -e "\n========================================================"
echo "HLAWOllama 服务安装并启动成功！"
echo "========================================================"
echo "核心信息："
echo "  项目根目录：${INSTALL_DIR}"
echo "  服务文件路径：${SERVICE_FILE_PATH}"
echo "  运行用户：${CURRENT_USER}"
echo "  服务端口：11434（0.0.0.0）"
echo "--------------------------------------------------------"
echo "常用管理命令："
echo "  sudo systemctl status ${SERVICE_NAME}  # 查看服务状态"
echo "  sudo systemctl start ${SERVICE_NAME}   # 启动服务"
echo "  sudo systemctl stop ${SERVICE_NAME}    # 停止服务"
echo "  sudo systemctl restart ${SERVICE_NAME}  # 重启服务"
echo "  sudo journalctl -u ${SERVICE_NAME} -f  # 实时查看服务日志"
echo "========================================================"

exit 0