#!/bin/bash

# 检查参数数量
if [ $# -ne 1 ]; then
    echo "Usage: $0 <xh1|xh2>"
    exit 1
fi
PLATFORM=$1
ARCH=x86_64

# SDK URLs
xh1_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_v2.4.0/2025/06/25/0200/windows11/houmo_tcim_runtime_xh1_windows11_x86_64-2.4.1.tar.gz"
xh2_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.2.0/2025/06/30/1110/windows11/houmo_tcim_runtime_xh2_windows11_x86_64-2.4.0.tar.gz"
xh1_sw_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_v2.4.0/2025/06/25/0200/windows11/houmo_drv_xh1_v2.4.1_windows11_x86_64.zip"
xh2_sw_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.2.0/2025/06/30/1110/windows11/houmo_drv_xh2_v0.0.0_windows11_x86_64.zip"

# libtorch 下载地址
libtorch_url="https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.1%2Bcpu.zip"
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

    # 下载并解压 SystemWare SDK
    download_file "$sw_url" "third_party/sw_sdk.zip"

    # 提取SDK
    unzip -o third_party/sw_sdk.zip -d third_party

    if [ $PLATFORM == "xh2" ]; then
        cp third_party/houmo_drv_xh2/hal/lib/*.dll third_party/houmo/lib/
        mkdir -p third_party/houmo/lib/bin/
        cp third_party/houmo_drv_xh2/hal/lib/*_hal.dll third_party/houmo/lib/bin/cygxh2a_hal.dll

        rm -rf third_party/sw_sdk.zip third_party/houmo_drv*
        # FIXME: 临时解决方案，等待SystemWare SDK更新
        download_file "$xh1_sw_sdk_url" "third_party/sw_sdk.zip"
        unzip third_party/sw_sdk.zip -d third_party/
    fi

    # 移动关键库文件
    if [ -f third_party/houmo_drv_xh1/hal/lib/*hdplrt_asic.dll ]; then
        cp third_party/houmo_drv_xh1/hal/lib/*hdplrt_asic.dll third_party/houmo/lib/
    else
        echo "Warning: libhdplrt_asic.dll not found in expected location"
    fi

    if [ -f third_party/houmo_drv_xh1/hal/lib/*hm800_hal.dll ]; then
        cp third_party/houmo_drv_xh1/hal/lib/*hm800_hal.dll third_party/houmo/lib/
    else
        echo "Warning: libhm800_hal.dll not found in expected location"
    fi

    cp third_party/houmo_drv_xh1/hal/lib/libwinpthread-1.dll third_party/houmo/lib/

    # 清理临时文件
    rm -rf third_party/sw_sdk.zip third_party/houmo_drv*
}

# 编译 LLAMA
build_llama() {
    echo "Building for ARCH: $ARCH"

    # 创建或清理构建目录
    rm -rf build/*
    cmake -B build  -DLLAMA_CURL=OFF
    cmake  --build build --config Release -j 8
    echo "Build completed."
}

# 打包 LLAMA
package_llama() {
    local release_dir="release/$PLATFORM/$ARCH"
    echo "Packaging LLAMA for $PLATFORM/$ARCH..."

    # 清理并创建目录结构
    rm -rf "$release_dir"/*
    mkdir -p "$release_dir/include" "$release_dir/lib" "$release_dir/bin" "$release_dir/scripts"

    # 复制头文件
    cp -r include/* "$release_dir/include/" || true
    cp -r ggml/include/* "$release_dir/include/" || true
    cp -r common/*.h* "$release_dir/include/" || true

    # 复制库文件
    cp build/src/Release/*.lib "$release_dir/lib/" || true
    cp build/common/Release/*.lib "$release_dir/lib/" || true

    cp build/bin/Release/*.dll "$release_dir/bin/" || true

    cp third_party/houmo/lib/*.dll "$release_dir/bin/" || true
    # not check.
    cp -r third_party/houmo/lib/bin "$release_dir/bin/"
    cp third_party/libtorch/lib/*.dll* "$release_dir/bin/" || true

    # 复制可执行文件
    cp build/bin/Release/llama-server* "$release_dir/bin/" || true
    cp build/bin/Release/llama-simple* "$release_dir/bin/" || true
    cp build/bin/Release/llama*-batched* "$release_dir/bin/" || true
    cp build/bin/Release/llama-gguf* "$release_dir/bin/" || true
    cp build/bin/Release/llama-cli* "$release_dir/bin/" || true
    cp build/bin/Release/llama-bench* "$release_dir/bin/" || true
    cp build/bin/Release/llama-mtmd-cli* "$release_dir/bin/" || true
    cp ci/run_llama_server_win.sh "$release_dir/bin/" || true
    # 复制脚本
    cp convert_hm_to_gguf.py "$release_dir/scripts/" || true

    echo "Release files have been organized under: $release_dir"
    # 显示打包内容摘要
    echo "Package contents summary:"
    find "$release_dir" -type f | wc -l
}

# 主函数
main() {
    # 确定使用的SDK URL
    deploy_torch
    case "$PLATFORM" in
        xh1)
            deploy_dadao_sdk "$xh1_dadao_sdk_url" "$xh1_sw_sdk_url"
            ;;
        xh2)
            deploy_dadao_sdk "$xh2_dadao_sdk_url" "$xh2_sw_sdk_url"
            ;;
    esac

    echo "Deployed Dadao SDK and System SDK for $PLATFORM/$ARCH."

    # 编译和打包
    build_llama
    package_llama

    echo "LLAMA build and packaging completed for $PLATFORM/$ARCH."
}
# 执行主函数
main
