#!/bin/bash

# 设置环境变量
export TCIM_RUNTIME_PATH=.
export LD_LIBRARY_PATH=.
export TCIM_BACKEND=Xh2HalBackend

# 检查可执行文件是否存在
if [ ! -f "llama-server.exe" ]; then
    echo "错误: 找不到 llama-server 可执行文件!"
    echo "请确保文件位于 $(dirname "$0")/bin/ 目录下。"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "/d/dev/cicd.models/qwen2.5-7B-instruct-hf-F16.gguf" ]; then
    echo "错误: 找不到模型文件!"
    echo "请确保模型文件位于 /dev/cici.models/ 目录下。"
    exit 1
fi


# 后台启动服务并记录日志
echo "Starting server in the background..."
nohup.exe ./llama-server.exe \
    -m /d/dev/cicd.models/qwen2.5-7B-instruct-hf-F16.gguf \
    -np 1 \
    -n 2048 \
    --host 0.0.0.0 > /dev/null 2>&1 &


# 保存进程ID以便后续停止服务
echo $! > "$(dirname "$0")/server.pid"

echo "Server started successfully."
