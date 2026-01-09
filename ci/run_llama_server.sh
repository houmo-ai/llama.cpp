#!/bin/bash

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时退出
set -o pipefail  # 管道命令中任何一个失败都算失败

# ===================== 核心函数定义 =====================
# 图案化打印警告信息（黄底黑字+方块边框）
# 用法：print_warning [警告文字]
# 参数：
#   $1 - 自定义警告文字（可选，默认值：WARNING）
print_warning() {
    # 1. 参数处理：默认警告文字为 WARNING
    local WARN_TEXT=${1:-WARNING}

    # 2. 兼容性处理：终端不支持彩色则禁用颜色，不支持⚠️则替换为!
    local YELLOW_BG=""
    local RESET=""
    local WARN_ICON="⚠️"

    # 判断终端是否支持颜色（兼容无颜色终端）
    if [ -t 1 ] && command -v tput >/dev/null && [ $(tput colors) -ge 8 ]; then
        YELLOW_BG="\033[43;30m"  # 黄底黑字
        RESET="\033[0m"          # 重置样式
    fi

    # 可选：终端不支持emoji时，替换⚠️为!
    # WARN_ICON="!"

    # 3. 构造警告字符串和边框
    local WARNING_STR=" ${WARN_ICON}  ${WARN_TEXT} ${WARN_ICON}  "
    local BORDER_LEN=${#WARNING_STR}
    # 生成等长的方块边框（兼容bash/zsh，避免seq兼容性问题）
    local BORDER=$(printf "■%.0s" $(seq 1 "${BORDER_LEN}"))

    # 4. 打印图案（带颜色）
    echo -e "${YELLOW_BG}${BORDER}${RESET}"
    echo -e "${YELLOW_BG}${WARNING_STR}${RESET}"
    echo -e "${YELLOW_BG}${BORDER}${RESET}"

    # 可选：打印空行分隔，提升可读性
    echo ""
}

print_warning  "不支持旧版本(0.7.0之前）模型，请下载0.7.0版本GGUF, 本脚本后续版本废弃，直接./bin/llama-server运行"

show_help() {
    echo "usage: run_llama_server.sh [env] run model_path [extra_args...]"
    echo ""
    echo "usage 示例0: <多卡环境绑卡操作>：export HOUMO_DEVICE_ID=0 或者双卡export HOUMO_DEVICE_ID=0,1"
    echo ""
    echo "usage 示例1: <指定端口号>: run_llama_server.sh run model_path --port 17701"
    echo ""
    echo "usage 示例2: <关闭Qwen3的Think机制>: run_llama_server.sh run model_path --reasoning-budget 0"
    echo ""
    echo "usage 示例3: <非多batch模型性能会下降，且受限显存大小，最大4>: run_llama_server.sh run model_path -np 2"
    echo ""
    echo "usage 示例4: <后台运行>: run_llama_server.sh rund  model_path --port 17701"
    echo ""
    echo "usage 示例5: <多模态模型>: run_llama_server.sh run model_path --mmproj mmproj_name.gguf"
    echo ""
    echo "usage 示例6: <指定模型目录>: run_llama_server.sh run /path/to/ 可以配合model_cli 工具使用"
    echo ""
    echo "更多参数，执行 ./bin/llama-server --help 查看"
}
# 初始化额外参数数组（避免未定义错误）
extra_args=()
default_port_args=("--port" "17701")  # 默认端口号


# 检查是否没有提供任何参数
if [[ "$#" -eq 0 ]]; then
    show_help
    exit 1
fi
rund=false
# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        run | rund)
            if [ "$1" == "rund" ]; then
                rund=true
            fi
            if [ -z "$2" ]; then
                echo "错误: 请指定要运行的模型名称"
                exit 1
            fi
            model_name=$2
            if [ -f $model_name ]; then
                model_path=$model_name
            elif [ -d $model_name ]; then
                model_path=$model_name
            else
                echo "错误: 模型 $model_name 不存在"
                exit 1
            fi

            # 收集剩余参数作为透传参数
            shift 2
            extra_args=("$@")
            break
            ;;
        *)
            show_help
            exit 1
            ;;
    esac
done

if [ -d "$model_path" ]; then
    launch_cmd=(
        ./bin/llama-server
        --models-dir "$model_path"
        -c 0
        --no-context-shift
        --host 0.0.0.0
        --jinja
        --reasoning-format none
        --log-prefix --log-timestamps
    )
else
    launch_cmd=(
        ./bin/llama-server
        -m "$model_path"
        -c 0
        --no-context-shift
        --host 0.0.0.0
        --jinja
        --reasoning-format none
        --log-prefix --log-timestamps
    )
fi

# 如果是VL模型，添加mmproj参数
if [ -n "${mmproj_model_path:-}" ]; then
    launch_cmd+=(--mmproj "$mmproj_model_path")
fi

# 添加透传参数
launch_cmd+=("${extra_args[@]}")

# 启动服务
# 检查所有参数中是否已包含--port
if ! (printf "%s\n" "${launch_cmd[@]}" | grep -q -- "--port"); then
    # 不存在--port参数，添加默认端口
    launch_cmd+=("${default_port_args[@]}")
fi

# 显示透传参数（如果有）
if [ ${#extra_args[@]} -gt 0 ]; then
    echo "透传参数: ${extra_args[*]}"
fi

echo "启动命令: ${launch_cmd[*]}"
if [ "$rund" = false ]; then
    "${launch_cmd[@]}"
else

    nohup "${launch_cmd[@]}" > /dev/null 2>&1 &
    server_pid=$!
    echo "llama-server已在后台启动，PID: ${server_pid}"

    # 等待服务启动
    echo "等待llama-server启动..."
    start_time=$(date +%s)
    timeout=30  # 超时时间30秒

    while true; do
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))

        if [ $elapsed_time -ge $timeout ]; then
            echo "错误: llama-server在${timeout}秒内未能启动"
            exit 1
        fi

        if pgrep -f "llama-server -m ${model_path}" > /dev/null; then
            echo "llama-server已成功启动！"
            break
        fi

        echo "等待中...(${elapsed_time}/${timeout}秒)"
        sleep 1
    done

    # 健康检查
    echo "执行llama-server健康检查..."
    check_count=5
    check_interval=10

    for ((i=1; i<=check_count; i++)); do
        echo "健康检查 ${i}/${check_count}..."
        if ! pgrep -f "llama-server -m ${model_path}" > /dev/null; then
            echo "错误: llama-server已停止运行"
            exit 1
        fi
        remaining=$((check_count - i))
        if [ $remaining -gt 0 ]; then
            echo "llama-server运行正常，${check_interval}秒后进行下一次检查..."
            sleep $check_interval
        fi
    done

    echo "llama-server已成功启动并通过健康检查！"
    echo "服务信息:"
    if [ ${#extra_args[@]} -gt 0 ]; then
        echo "- 透传参数: ${extra_args[*]}"
    fi
fi
