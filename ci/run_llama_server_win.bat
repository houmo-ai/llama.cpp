@echo off
:: Set environment variables
set "TCIM_RUNTIME_PATH=."
set "LD_LIBRARY_PATH=."
set "TCIM_BACKEND=Xh2HalBackend"

:: Set default model path
set "DEFAULT_MODEL=.\qwen2.5-7B-instruct-hf-F16.gguf"

:: Check if command line argument is provided
if "%~1"=="" (
    echo No model file specified, will use default model: %DEFAULT_MODEL%

    :: Check if default model exists
    if not exist "%DEFAULT_MODEL%" (
        echo Error: Default model file does not exist!
        echo Path: %DEFAULT_MODEL%
        exit /b 1
    )

    set "MODEL_PATH=%DEFAULT_MODEL%"
) else (
    set "MODEL_PATH=%~1"
)

:: Check if executable exists
if not exist "llama-server.exe" (
    echo Error: Cannot find llama-server executable!
    echo Please ensure the file is in the current directory.
    exit /b 1
)

:: Check if model file exists
if not exist "%MODEL_PATH%" (
    echo Error: Cannot find model file!
    echo Path: %MODEL_PATH%
    exit /b 1
)

:: Start server in background and log
echo Starting server in the background...
start "" /b llama-server.exe ^
    -m "%MODEL_PATH%" ^
    -np 1 ^
    -c 32000 ^
    --no-context-shift  ^
    --host 0.0.0.0