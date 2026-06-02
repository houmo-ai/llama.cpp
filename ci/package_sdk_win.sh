#!/bin/bash

# 检查参数数量
if [ $# -ne 1 ]; then
    echo "Usage: $0 <xh2>"
    exit 1
fi
PLATFORM=$1
ARCH=x86_64

# SDK URLs
xh2_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v1.3.0/2026/05/19/0015/windows11/houmo_tcim_runtime_xh2_windows11_x86_64-1.3.0.tar.gz"
xh2_sw_sdk_url=""
modelcli_url="http://10.10.1.53:8082/artifactory/application_software/release/llama.cpp/apps/model-cli.exe"
boringssl_url="http://artifactory.houmo.ai:80/artifactory/thirdparty_local/boringssl/boringssl-20260508.tar.gz"

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

# 部署 BoringSSL
deploy_boringssl() {
    local boringssl_dir="vendor/boringssl"
    local boringssl_archive="vendor/boringssl.tar.gz"

    if [ -f "${boringssl_dir}/CMakeLists.txt" ]; then
        echo "BoringSSL already exists: ${boringssl_dir}, skipping download."
        return 0
    fi

    rm -rf "${boringssl_dir}"

    download_file "${boringssl_url}" "${boringssl_archive}"
    tar -xzf "${boringssl_archive}" -C vendor

    if [ ! -f "${boringssl_dir}/CMakeLists.txt" ]; then
        echo "Failed to locate BoringSSL CMakeLists.txt after extracting ${boringssl_archive} to vendor"
        exit 1
    fi

    rm -f "${boringssl_archive}"
    echo "BoringSSL deployed to ${boringssl_dir}"
}

build_ui_assets() {
    local ui_dir="tools/ui"
    local ui_dist="${ui_dir}/dist"
    local required_assets=(index.html bundle.js bundle.css loading.html)
    local missing=0

    for asset in "${required_assets[@]}"; do
        if [ ! -f "${ui_dist}/${asset}" ]; then
            missing=1
            break
        fi
    done

    if [ "$missing" = "0" ]; then
        echo "UI assets already exist in ${ui_dist}, skipping npm build."
        return 0
    fi

    if ! command -v npm >/dev/null 2>&1; then
        echo "npm is required to build UI assets into ${ui_dist}."
        exit 1
    fi

    echo "Building UI assets into ${ui_dist}..."
    (
        cd "${ui_dir}"
        if [ -f package-lock.json ]; then
            npm ci
        else
            npm install
        fi
        npm run build
    )

    for asset in "${required_assets[@]}"; do
        if [ ! -f "${ui_dist}/${asset}" ]; then
            echo "UI build failed: missing ${ui_dist}/${asset}"
            exit 1
        fi
    done
}

# 编译 LLAMA
build_llama() {
    echo "Building for ARCH: $ARCH"
    git config --global --add safe.directory `pwd`
    # 创建或清理构建目录
    rm -rf build/*
    if [ $PLATFORM == "xh2" ]; then
        cmake -B build  -DLLAMA_CURL=OFF -DENABLE_HOUMO_XH2=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_BORINGSSL=ON 
    else
        cmake -B build -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_BORINGSSL=ON
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
    cp -r build/tools/ui/*.hpp "$release_dir/include/server/" || true
    # 复制库文件
    cp build/src/Release/*.lib "$release_dir/lib/" || true
    cp build/common/Release/*.lib "$release_dir/lib/" || true
    cp build/tools/mtmd/Release/*.lib "$release_dir/lib/" || true
    cp build/ggml/src/Release/*.lib "$release_dir/lib/" || true
    cp build/tools/server/Release/*.lib "$release_dir/lib/" || true
    cp build/tools/ui/Release/*.lib "$release_dir/lib/" || true
    cp build/bin/Release/*.dll "$release_dir/bin/" || true

    cp third_party/houmo/lib/*.dll "$release_dir/bin/" || true
    # not check.
    cp third_party/houmo/bin/readhmm.exe "$release_dir/scripts/" || true
    # 复制可执行文件
    cp build/bin/Release/llama-server* "$release_dir/bin/" || true
    cp build/bin/Release/llama-gguf-split* "$release_dir/bin/" || true
    cp build/bin/Release/llama-cli* "$release_dir/bin/" || true
    cp tools/server/qwen3_nonthinking.jinja "$release_dir/bin/" || true
    cp build/bin/Release/llama-gguf* "$release_dir/bin/" || true
    cp build/bin/Release/llama-bench* "$release_dir/bin/" || true
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
    deploy_boringssl
    build_ui_assets
    build_llama
    package_llama
    download_file "$modelcli_url" "release/$PLATFORM/$ARCH/bin/model-cli.exe"
    echo "LLAMA build and packaging completed for $PLATFORM/$ARCH."
}
# 执行主函数
main
