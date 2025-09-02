@echo off
REM 设置环境变量
set "TCIM_RUNTIME_PATH=."
set "LD_LIBRARY_PATH=."
set "TCIM_BACKEND=Xh2HalBackend"

REM 设置默认模型路径
set "DEFAULT_MODEL=./models/default_model.gguf"

REM 检查是否提供了命令行参数
if "%~1"=="" (
    echo 未指定模型文件，将使用默认模型: %DEFAULT_MODEL%

    REM 检查默认模型是否存在
    if not exist "%DEFAULT_MODEL%" (
        echo 错误: 默认模型文件不存在!
        echo 路径: %DEFAULT_MODEL%
        exit /b 1
    )

    set "MODEL_PATH=%DEFAULT_MODEL%"
) else (
    set "MODEL_PATH=%~1"
)

REM 检查可执行文件是否存在
if not exist "llama-server.exe" (
    echo 错误: 找不到 llama-server 可执行文件!
    echo 请确保文件位于当前目录下。
    exit /b 1
)

REM 检查模型文件是否存在
if not exist "%MODEL_PATH%" (
    echo 错误: 找不到模型文件!
    echo 路径: %MODEL_PATH%
    exit /b 1
)

REM 后台启动服务并记录日志
echo Starting server in the background...
start "" /b llama-server.exe ^
    -m "%MODEL_PATH%" ^
    -np 1 ^
    -c 32000 ^
    --no-context-shift  ^
    --host 0.0.0.0