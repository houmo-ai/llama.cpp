
#!/bin/bash
# start_ollama.sh - 动态生成并启动 ollama 服务
available() { command -v $1 >/dev/null; }
error() { echo "错误: $*" >&2; exit 1; }
SUDO=
if [ "$(id -u)" -ne 0 ]; then
    # Running as non-root, sudo is required for privileged operations
    if ! available sudo; then
        error "This script requires superuser permissions. Please re-run as root."
    fi
    SUDO="sudo"
fi
# 使用 BASH_SOURCE[0] 而不是 $0 來處理 symbolic links
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # 循环，直到找到源头文件
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # 如果是相对路径，则转换为绝对路径
done
# 获取脚本所在的目录，也就是安装目录
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

# 兼容两种目录布局：
# 1) install_ollama.sh 与 bin/ 同级
# 2) install_ollama.sh 位于 service/ 子目录，bin/ 位于其上级目录
if [ -x "${SCRIPT_DIR}/bin/hlaw_ollama" ]; then
    INSTALL_DIR="${SCRIPT_DIR}"
elif [ -x "${SCRIPT_DIR}/../bin/hlaw_ollama" ]; then
    INSTALL_DIR="$( cd -P "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd )"
else
    error "未找到 hlaw_ollama，可执行文件应位于 ${SCRIPT_DIR}/bin 或 ${SCRIPT_DIR}/../bin"
fi
# 获取当前执行脚本的用户
EFFECTIVE_USER="$(id -un)"
if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
    echo "当前通过 sudo 执行，原始用户：$SUDO_USER"
    CURRENT_USER="$SUDO_USER"
elif [ "$EFFECTIVE_USER" = "root" ]; then
    echo "当前以 root 用户执行"
    CURRENT_USER="root"
else
    echo "未通过 sudo 执行，当前用户：${USER:-$EFFECTIVE_USER}"
    CURRENT_USER="${USER:-$EFFECTIVE_USER}"
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
echo "--------------------------------------------------------"
echo "正在从以下目录启动服务: ${INSTALL_DIR}"
echo "服务将使用当前用户运行: ${CURRENT_USER}"
echo "--------------------------------------------------------"
SERVICE_NAME="HLAWOllama.service"
SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}"
SERVICE_FILE_CONTENT=$(cat <<EOF
[Unit]
Description=HLAWOllama Service
After=network-online.target
[Service]
ExecStart=${INSTALL_DIR}/bin/hlaw_ollama serve
User=${CURRENT_USER}
Group=${CURRENT_USER}
Restart=always
RestartSec=3
# environment variables
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
Environment="PATH=${INSTALL_DIR}/bin:${PATH}"
[Install]
WantedBy=multi-user.target
EOF
)
# 打印生成的服务文件内容到控制台
echo "将要生成的 systemd 服务文件内容如下："
echo "----------------------------------------"
echo "${SERVICE_FILE_CONTENT}"
echo "----------------------------------------"
# 将动态生成的内容写入服务文件
echo "${SERVICE_FILE_CONTENT}" | $SUDO tee "${SERVICE_FILE_PATH}" > /dev/null
# 检查文件是否成功创建
if [ ! -f "${SERVICE_FILE_PATH}" ]; then
    echo "错误: 无法创建服务文件 ${SERVICE_FILE_PATH}。请检查权限。"
    exit 1
fi
$SUDO chmod 755 ${INSTALL_DIR}
$SUDO chown ${CURRENT_USER}:${CURRENT_USER} ${INSTALL_DIR} -R
$SUDO chmod +x ${INSTALL_DIR}/bin/hlaw_ollama
echo "服务文件已成功创建：${SERVICE_FILE_PATH}"
# 追加到用户的.bashrc 文件中，以便在登录时自动启动服务
# 定义特征注释（用于标记需要清理的区块）
MARKER_START="# HLAWOllama start"
MARKER_END="# HLAWOllama end"

# 在需要时以实际用户身份修改其 .bashrc，避免 root 拥有者污染
RUN_AS_CURRENT_USER=()
if [ "$(id -un)" != "$CURRENT_USER" ]; then
    RUN_AS_CURRENT_USER=(sudo -u "$CURRENT_USER")
fi

# 先删除 .bashrc 中已有的该区块（如果存在）
# 使用 sed 命令删除标记之间的内容（包括标记本身）
"${RUN_AS_CURRENT_USER[@]}" touch "${BASHRC_PATH}"
"${RUN_AS_CURRENT_USER[@]}" sed -i "/${MARKER_START}/,/${MARKER_END}/d" "${BASHRC_PATH}"
# 再写入新的环境变量（带标记，方便下次清理）
cat <<EOF | "${RUN_AS_CURRENT_USER[@]}" tee -a "${BASHRC_PATH}" > /dev/null
${MARKER_START}
export PATH=${INSTALL_DIR}/bin:\${PATH}
${MARKER_END}
EOF
# 生效配置
echo "source ${BASHRC_PATH}  -- 生效环境变量"
echo "${SERVICE_NAME} 服务已安装！您可以使用以下命令加入系统启动并启动服务："
$SUDO systemctl daemon-reload
$SUDO systemctl enable ${SERVICE_NAME}
$SUDO systemctl start ${SERVICE_NAME}
echo -e "\n\n"
echo "${SERVICE_NAME} 服务已启动！您可以使用以下命令管理服务："
echo "sudo systemctl status ${SERVICE_NAME} - 查看服务状态"
echo "sudo systemctl start ${SERVICE_NAME}  - 启动服务"
echo "sudo systemctl stop ${SERVICE_NAME}   - 停止服务"
echo "sudo systemctl restart ${SERVICE_NAME} - 重启服务"
echo "sudo journalctl -u ${SERVICE_NAME} -f  # 实时查看服务日志"
exit 0
