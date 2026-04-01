#!/bin/bash

# 检查参数数量
if [ $# -ne 1 ]; then
    echo "Usage: $0 <xh2>"
    exit 1
fi
PLATFORM=$1
ARCH=x86_64

# SDK URLs
xh2_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v1.1.0/2026/03/12/2005/windows11/houmo_tcim_runtime_xh2_windows11_x86_64-1.1.0.tar.gz"
xh2_sw_sdk_url=""
modelcli_url="http://10.10.1.53:8082/artifactory/application_software/release/llama.cpp/apps/model-cli.exe"

# 下载文件并检查是否成功
download_file() {
    local url=$1
    local output=$2
    echo "Downloading $url..."
    wget "$url" -O "$output" || { echo "Failed to download $url"; return 1; }
    echo "Download completed: $output"
}
deploy_torch() {
    # 检查torch.zip是否存在
    if [ ! -f "third_party/torch.zip" ]; then
        echo "未找到torch.zip，开始下载..."
        download_file "$libtorch_url" "third_party/torch.zip"
    else
        echo "找到torch.zip文件，跳过下载"
    fi

    # 检查目标目录是否存在，若存在则删除
    if [ -d "third_party/libtorch" ]; then
        echo "删除已存在的 third_party/libtorch 目录..."
        rm -rf third_party/libtorch
    fi

    # 解压文件
    unzip -o third_party/torch.zip -d third_party/
}

# 部署 Dadao SDK
deploy_dadao_sdk() {
    local dadao_url=$1
    local sw_url=$2

    # 创建或清理第三方库目录
    rm -rf third_party/houmo*
    mkdir -p third_party/houmo

    # 下载并解压 Dadao SDK
    download_file "$dadao_url" "dadao_sdk.tar.gz"
    tar -xzf dadao_sdk.tar.gz -C third_party/houmo --strip-components=1
    cp third_party/houmo/bin/*.dll third_party/houmo/lib/
    rm dadao_sdk.tar.gz
}

# 编译 LLAMA
build_llama() {
    echo "Building for ARCH: $ARCH"
    git config --global --add safe.directory `pwd`
    # 创建或清理构建目录
    rm -rf build/*
    if [ $PLATFORM == "xh2" ]; then
        cmake -B build  -DLLAMA_CURL=OFF -DENABLE_HOUMO_XH2=ON
    else
        cmake -B build -DLLAMA_CURL=OFF
    fi
    if [ $? -ne 0 ]; then
        echo "cmake failed."
        exit 1
    else
        echo "cmake completed."
    fi
    cmake  --build build --config Release -j 8
    if [ $? -ne 0 ]; then
        echo "make build failed."
        exit 1
    else
        echo "make build completed."
    fi
    echo "Build completed."
}

# 打包 LLAMA
package_llama() {
    local release_dir="release/$PLATFORM/$ARCH"
    echo "Packaging LLAMA for $PLATFORM/$ARCH..."

    # 清理并创建目录结构
    rm -rf "$release_dir"/*
    mkdir -p "$release_dir/include" "$release_dir/lib" "$release_dir/bin" "$release_dir/scripts"
    mkdir -p "$release_dir/include/server"

    # 复制头文件
    cp -r include/* "$release_dir/include/" || true
    cp -r ggml/include/* "$release_dir/include/" || true
    cp -r common/*.h* "$release_dir/include/" || true
    cp -r common/jinja "$release_dir/include/jinja" || true
    cp -r tools/mtmd/mtmd*.h "$release_dir/include/" || true
    cp -r vendor/nlohmann "$release_dir/include/" || true
    cp -r src/llama*.h* "$release_dir/include/" || true
    cp -r tools/server/*.h* "$release_dir/include/server/" || true
    cp -r build/tools/server/*.hpp "$release_dir/include/server/" || true
    # 复制库文件
    cp build/src/Release/*.lib "$release_dir/lib/" || true
    cp build/common/Release/*.lib "$release_dir/lib/" || true
    cp build/tools/mtmd/Release/*.lib "$release_dir/lib/" || true
    cp build/ggml/src/Release/*.lib "$release_dir/lib/" || true
    cp build/tools/server/Release/*.lib "$release_dir/lib/" || true
    cp build/bin/Release/*.dll "$release_dir/bin/" || true

    cp third_party/houmo/lib/*.dll "$release_dir/bin/" || true
    # not check.
    cp -r third_party/houmo/lib/bin "$release_dir/bin/"
    cp third_party/libtorch/lib/*.dll* "$release_dir/bin/" || true

    # 复制可执行文件
    cp build/bin/Release/llama-server* "$release_dir/bin/" || true
    cp build/bin/Release/llama-gguf-split* "$release_dir/bin/" || true
    cp build/bin/Release/llama-cli* "$release_dir/bin/" || true
    cp tools/server/qwen3_nonthinking.jinja "$release_dir/bin/" || true
    cp build/bin/Release/llama-gguf* "$release_dir/bin/" || true
    cp build/bin/Release/llama-bench* "$release_dir/bin/" || true
    cp ci/run_llama_server_win.* "$release_dir/bin/" || true
    # 复制脚本
    cp convert_hm_to_gguf.py "$release_dir/scripts/" || true
    cp requirements/requirements-convert_hm_to_gguf.txt "$release_dir/scripts/" || true

    echo "Release files have been organized under: $release_dir"
    # 显示打包内容摘要
    echo "Package contents summary:"
    find "$release_dir" -type f | wc -l
}

# 主函数
main() {
    # 确定使用的SDK URL
    case "$PLATFORM" in
        xh2)
            deploy_dadao_sdk "$xh2_dadao_sdk_url" "$xh2_sw_sdk_url"
            ;;
    esac

    echo "Deployed Dadao SDK and System SDK for $PLATFORM/$ARCH."

    # 编译和打包
    build_llama
    package_llama
    download_file "$modelcli_url" "release/$PLATFORM/$ARCH/bin/model-cli.exe"
    echo "LLAMA build and packaging completed for $PLATFORM/$ARCH."
}
# 执行主函数
main
