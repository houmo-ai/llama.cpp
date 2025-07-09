#!/bin/bash

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时退出
set -o pipefail  # 管道命令中任何一个失败都算失败

# 设置默认值
default_port=31808
default_device_id=0
default_model_path="/data/xh1/model/HiModel-Qwen3-8B-F16-2k-4core-xh1-v0.0.1.gguf"
if [ $1 == "xh2" ]; then
    default_model_path="/data/xh2/model/HiModel-Qwen3-8B-F16-2k-2core-xh2-v0.0.1.gguf"
fi
# 参数处理
case $# in
    1)
        xh_type="$1"
        port="$default_port"
        device_id="$default_device_id"
        model_path="$default_model_path"
        ;;
    2)
        xh_type="$1"
        model_path="$2"
        port="$default_port"
        device_id="$default_device_id"
        ;;
    3)
        xh_type="$1"
        model_path="$2"
        port="$3"
        device_id="$default_device_id"
        ;;
    4)
        xh_type="$1"
        model_path="$2"
        port="$3"
        device_id="$4"
        ;;
    *)
        echo "Usage: $0 <xh1|xh2> [model_path] [port] [device_id]"
        echo "  port:        默认 $default_port"
        echo "  device_id:   默认 $default_device_id"
        echo "  model_path:  默认 $default_model_path"
        exit 1
        ;;
esac

# 验证xh_type参数
if [ "$xh_type" != "xh1" ] && [ "$xh_type" != "xh2" ]; then
    echo "错误: xh_type必须是xh1或xh2"
    exit 1
fi

NP=1
export HOUMO_DEVICE_ID=${device_id}
export LD_LIBRARY_PATH=$(pwd)/lib

# 设置模型路径和环境变量
if [ "${xh_type}" = "xh2" ]; then
    export TCIM_BACKEND=Xh2HalBackend
    export TCIM_RUNTIME_PATH=$(pwd)
    # NP=4
fi

# 验证模型文件是否存在
if [ ! -f "${model_path}" ]; then
    echo "错误: 模型文件不存在: ${model_path}"
    exit 1
fi

# 停止可能正在运行的llama-server实例
echo "检查是否存在正在运行的llama-server进程..."
if pgrep -f "llama-server -m ${model_path}" > /dev/null; then
    echo "找到正在运行的llama-server进程，正在停止..."
    pkill -f "llama-server -m ${model_path}"
    # 等待进程停止
    timeout 10 sh -c "while pgrep -f 'llama-server -m ${model_path}' > /dev/null; do sleep 1; done" || {
        echo "警告: 无法停止现有llama-server进程，尝试强制终止..."
        pkill -9 -f "llama-server -m ${model_path}"
        sleep 2
        if pgrep -f "llama-server -m ${model_path}" > /dev/null; then
            echo "错误: 无法终止现有llama-server进程，请手动终止。"
            exit 1
        fi
    }
    echo "现有llama-server进程已停止。"
else
    echo "未找到正在运行的llama-server进程。"
fi

# 启动服务
echo "正在端口${port}上启动llama-server..."
echo "使用模型: ${model_path}"
echo "使用NP: ${NP}"
echo "使用device_id: ${device_id}"

nohup ./bin/llama-server -m ${model_path} --port ${port} --host 0.0.0.0 -np ${NP} > /dev/null 2>&1 &
server_pid=$!
echo "llama-server已在后台启动，PID: ${server_pid}"

# 等待服务启动
echo "等待llama-server启动..."
start_time=$(date +%s)
timeout=30  # 设置更长的超时时间为30秒

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

# 健康检查 - 确保服务在一段时间内保持运行状态
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
echo "- 模型: ${model_path}"
echo "- 端口: ${port}"
echo "- 设备ID: ${device_id}"
echo "- NP: ${NP}"