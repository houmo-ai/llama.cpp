#!/bin/bash

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时退出
set -o pipefail  # 管道命令中任何一个失败都算失败

# 检查参数数量
if [ $# -ne 2 ]; then
    echo "Usage: $0 <xh1|xh2> <aarch64|x86_64>"
    exit 1
fi

# 平台和架构参数
PLATFORM=$1
ARCH=$2

# 验证架构参数
case "$ARCH" in
    aarch64|x86_64)
        ;;
    *)
        echo "Invalid architecture: $ARCH"
        exit 1
        ;;
esac

# 验证平台参数
case "$PLATFORM" in
    xh1|xh2)
        ;;
    *)
        echo "Invalid platform: $PLATFORM"
        exit 1
        ;;
esac

# SDK URLs
xh1_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_v2.4.0/2025/06/25/0200/ubuntu2004/houmo_tcim_runtime_xh1_ubuntu2004_${ARCH}-2.4.1.tar.gz"
xh2_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.2.0/2025/06/30/1110/ubuntu2004/houmo_tcim_runtime_xh2_ubuntu2004_${ARCH}-2.4.0.tar.gz"
xh1_sw_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_v2.4.0/2025/06/25/0200/ubuntu2004/houmo_drv_xh1_v2.4.1_ubuntu2004_${ARCH}.run"
#FIXME:
xh2_sw_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v0.2.0/2025/06/30/1110/ubuntu2004/houmo_drv_xh2_v0.0.0_ubuntu2004_${ARCH}.run"
xh2_sw_sdk_url_aarch64="http://10.10.1.53:8082/artifactory/application_software/release/llama.cpp/daily/libxh2a_hal_box.so"
# 下载文件并检查是否成功
download_file() {
    local url=$1
    local output=$2
    echo "Downloading $url..."
    wget "$url" -O "$output" || { echo "Failed to download $url"; return 1; }
    echo "Download completed: $output"
}

# 部署 Dadao SDK
deploy_dadao_sdk() {
    local dadao_url=$1
    local sw_url=$2

    # 创建或清理第三方库目录
    mkdir -p third_party/houmo
    rm -rf third_party/houmo/*

    # 下载并解压 Dadao SDK
    download_file "$dadao_url" "dadao_sdk.tar.gz"
    tar -xzf dadao_sdk.tar.gz -C third_party/houmo --strip-components=1
    rm dadao_sdk.tar.gz


    # 提取SDK
    if [[ $PLATFORM == "xh2"  && "$ARCH" == "aarch64" ]]; then
        download_file "$xh2_sw_sdk_url_aarch64" third_party/houmo/lib/libxh2a_hal.so
    else
        # 下载并解压 SystemWare SDK
        download_file "$sw_url" "third_party/sw_sdk.run"
        bash third_party/sw_sdk.run extract --path third_party/
    fi

    if [[ $PLATFORM == "xh2" && "$ARCH" == "x86_64" ]]; then
        cp third_party/houmo-sdk/hal/lib/*.so third_party/houmo/lib/
        rm -rf third_party/houmo-sdk third_party/sw_sdk.run third_party/houmo_drv*
    fi

    if [ $PLATFORM == "xh2" ]; then
        # FIXME: xh2 还需下载xh1的包，临时解决方案，等待SystemWare SDK更新
        download_file "$xh1_sw_sdk_url" "third_party/sw_sdk.run"
        bash third_party/sw_sdk.run extract --path third_party/
    fi

    # 移动关键库文件
    if [ -f third_party/houmo-sdk/hal/lib/libhdplrt_asic.so ]; then
        cp third_party/houmo-sdk/hal/lib/libhdplrt_asic.so third_party/houmo/lib/
    else
        echo "Warning: libhdplrt_asic.so not found in expected location"
    fi

    if [ -f third_party/houmo-sdk/hal/lib/libhm800_hal.so ]; then
        cp third_party/houmo-sdk/hal/lib/libhm800_hal.so third_party/houmo/lib/
    else
        echo "Warning: libhm800_hal.so not found in expected location"
    fi

    # 清理临时文件
    rm -rf third_party/houmo-sdk third_party/sw_sdk.run third_party/houmo_drv*
}

# 编译 LLAMA
build_llama() {
    echo "Building for ARCH: $ARCH"

    # 创建或清理构建目录
    mkdir -p build
    rm -rf build/*

    cd build
    if [ "$ARCH" == "aarch64" ]; then
        cmake .. -DCMAKE_TOOLCHAIN_FILE=../ci/toolchain-aarch64.cmake -DLLAMA_CURL=OFF
    else
        cmake .. -DLLAMA_CURL=OFF
    fi
    make -j$(nproc)
    cd ..

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
    cp build/bin/*.so "$release_dir/lib/" || true
    cp third_party/houmo/lib/*.so* "$release_dir/lib/" || true
    cp /opt/libtorch/$ARCH/libtorch/lib/*.so* "$release_dir/lib/" || true

    # 复制系统库(如果存在)
    if [[ -f /lib/x86_64-linux-gnu/libstdc++.so.6 && "$ARCH" == "x86_64" ]]; then
        cp /lib/x86_64-linux-gnu/libstdc++.so.6 "$release_dir/lib/"
    fi
    if [[ -f /usr/aarch64-linux-gnu/lib/libstdc++.so.6 && "$ARCH" == "aarch64" ]]; then
        cp /usr/aarch64-linux-gnu/lib/libstdc++.so.6 "$release_dir/lib/"
    fi
    # 复制可执行文件
    cp build/bin/llama-server "$release_dir/bin/" || true
    cp build/bin/llama-simple* "$release_dir/bin/" || true
    cp build/bin/llama*-batched "$release_dir/bin/" || true
    cp build/bin/llama-gguf "$release_dir/bin/" || true
    cp build/bin/llama-cli "$release_dir/bin/" || true
    cp build/bin/llama-bench* "$release_dir/bin/" || true
    cp build/bin/llama-mtmd-cli "$release_dir/bin/" || true
    cp ci/run_llama_server.sh "$release_dir/" || true
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