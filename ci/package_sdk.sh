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
    xh2)
        ;;
    *)
        echo "Invalid platform: $PLATFORM"
        exit 1
        ;;
esac

# SDK URLs
xh2_dadao_sdk_url="http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v1.3.0/2026/05/11/2348/ubuntu2004/houmo_tcim_runtime_xh2_linux_${ARCH}-1.3.0.tar.gz"

ollama_url="http://10.10.1.53:8082/artifactory/application_software/release/llama.cpp/apps/hlaw_ollama.${ARCH}"
modelcli_url="http://10.10.1.53:8082/artifactory/application_software/release/llama.cpp/apps/model-cli.${ARCH}"

# 对齐策略说明：
# - auto(默认): aarch64 默认启用 64K 对齐；x86_64 默认不启用
# - 64k: 强制启用 64K 对齐
# - native: 不强制页对齐（使用工具链默认行为）
SDK_ALIGN_MODE="${SDK_ALIGN_MODE:-auto}"
PAGE_ALIGN_LDFLAGS="-Wl,-z,max-page-size=65536 -Wl,-z,common-page-size=65536"
PATCHELF_PAGE_SIZE=65536
ENABLE_64K_ALIGN=0

resolve_alignment_policy() {
    case "${SDK_ALIGN_MODE}" in
        auto)
            if [ "$ARCH" = "aarch64" ]; then
                ENABLE_64K_ALIGN=1
            else
                ENABLE_64K_ALIGN=0
            fi
            ;;
        64k)
            ENABLE_64K_ALIGN=1
            ;;
        native)
            ENABLE_64K_ALIGN=0
            ;;
        *)
            echo "Invalid SDK_ALIGN_MODE: ${SDK_ALIGN_MODE} (expected: auto|64k|native)"
            exit 1
            ;;
    esac
}

patchelf_supports_page_size() {
    patchelf --help 2>&1 | grep -q -- "--page-size"
}

patchelf_set_rpath() {
    local rpath="$1"
    local target="$2"

    if [ "$ENABLE_64K_ALIGN" = "1" ] && patchelf_supports_page_size; then
        patchelf --page-size "${PATCHELF_PAGE_SIZE}" --set-rpath "$rpath" "$target"
    elif [ "$ENABLE_64K_ALIGN" = "1" ]; then
        echo "Warning: patchelf does not support --page-size, fallback may reset alignment: $target"
        patchelf --set-rpath "$rpath" "$target"
    else
        patchelf --set-rpath "$rpath" "$target"
    fi
}

verify_elf_64k_alignment() {
    local target="$1"

    if [ "$ENABLE_64K_ALIGN" != "1" ]; then
        return 0
    fi

    if ! command -v readelf >/dev/null 2>&1; then
        echo "Warning: readelf not found, skip 64K alignment verification: $target"
        return 0
    fi

    if readelf -l "$target" 2>/dev/null | awk '$1=="LOAD"{ if (tolower($NF) != "0x10000") bad=1 } END { exit bad ? 1 : 0 }'; then
        echo "Verified 64K ELF LOAD alignment: $target"
    else
        echo "Warning: non-64K ELF LOAD alignment detected after patchelf: $target"
    fi
}

# 检测主机页大小（仅用于日志），具体对齐策略由 SDK_ALIGN_MODE 决定
detect_host_page_size() {
    local page_size=""

    if command -v getconf >/dev/null 2>&1; then
        page_size=$(getconf PAGE_SIZE 2>/dev/null || true)
        if [ -z "$page_size" ]; then
            page_size=$(getconf PAGESIZE 2>/dev/null || true)
        fi
    fi

    if [ -n "$page_size" ]; then
        echo "Detected host page size via getconf: ${page_size}"
    else
        echo "getconf not available or failed, skip host page size detection."
    fi
}

print_alignment_policy() {
    echo "SDK_ALIGN_MODE=${SDK_ALIGN_MODE}"
    if [ "$ENABLE_64K_ALIGN" = "1" ]; then
        echo "Effective alignment policy: FORCE_64K"
        echo "Using linker page alignment flags: ${PAGE_ALIGN_LDFLAGS}"
    else
        echo "Effective alignment policy: NATIVE"
    fi
}

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

    # 创建或清理第三方库目录
    mkdir -p third_party/houmo
    rm -rf third_party/houmo/*

    # 下载并解压 Dadao SDK
    download_file "$dadao_url" "dadao_sdk.tar.gz"
    tar -xzf dadao_sdk.tar.gz -C third_party/houmo --strip-components=1
    rm dadao_sdk.tar.gz

    # 清理临时文件
    rm -rf third_party/houmo-sdk
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
    resolve_alignment_policy
    detect_host_page_size
    print_alignment_policy
    local cmake_linker_args=()
    if [ "$ENABLE_64K_ALIGN" = "1" ]; then
        cmake_linker_args+=(
            -DCMAKE_EXE_LINKER_FLAGS="${PAGE_ALIGN_LDFLAGS}"
            -DCMAKE_SHARED_LINKER_FLAGS="${PAGE_ALIGN_LDFLAGS}"
            -DCMAKE_MODULE_LINKER_FLAGS="${PAGE_ALIGN_LDFLAGS}"
        )
    fi
    # 创建或清理构建目录
    mkdir -p build
    rm -rf build/*

    cd build
    if [ "$ARCH" == "aarch64" ]; then
        if [ "$PLATFORM" == "xh2" ]; then
            cmake .. -DCMAKE_TOOLCHAIN_FILE=../ci/toolchain-aarch64.cmake -DLLAMA_CURL=OFF -DENABLE_HOUMO_XH2=ON \
                "${cmake_linker_args[@]}"
        else
            cmake .. -DCMAKE_TOOLCHAIN_FILE=../ci/toolchain-aarch64.cmake -DLLAMA_CURL=OFF \
                "${cmake_linker_args[@]}"
        fi
    else
        if [ "$PLATFORM" == "xh2" ]; then
            cmake .. -DLLAMA_CURL=OFF -DENABLE_HOUMO_XH2=ON \
                "${cmake_linker_args[@]}"
        else
            # 默认使用x86_64架构
            cmake .. -DLLAMA_CURL=OFF \
                "${cmake_linker_args[@]}"
        fi
    fi
    if [ $? -ne 0 ]; then
        echo "cmake failed."
        exit 1
    else
        echo "cmake completed."
    fi
    make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "make build failed."
        exit 1
    else
        echo "make build completed."
    fi
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
    mkdir -p "$release_dir/include/server" "$release_dir/service" "$release_dir/models"

    # 复制头文件
    cp -r include/* "$release_dir/include/" || true
    cp -r src/llama*.h* "$release_dir/include/" || true
    cp -r ggml/include/* "$release_dir/include/" || true
    cp -r common/*.h* "$release_dir/include/" || true
    cp -r common/jinja "$release_dir/include/jinja" || true
    cp -r tools/mtmd/mtmd*.h "$release_dir/include/" || true
    cp -r vendor/nlohmann "$release_dir/include/" || true
    cp -r tools/server/*.h* "$release_dir/include/server/" || true
    cp -r build/tools/ui/*.hpp "$release_dir/include/server/" || true
    # 复制库文件
    cp build/bin/*.so "$release_dir/lib/" || true
    cp build/common/*.a "$release_dir/lib/"
    cp build/tools/ui/*.a "$release_dir/lib/" || true
    cp build/tools/server/libserver-context.a "$release_dir/lib/" || true
    cp third_party/houmo/lib/*.so* "$release_dir/lib/" || true
    cp version.cfg "$release_dir/version.txt"

    if [[ -f /lib/x86_64-linux-gnu/libstdc++.so.6 && "$ARCH" == "x86_64" ]]; then
        cp /lib/x86_64-linux-gnu/libstdc++.so.6 "$release_dir/lib/"
    fi
    if [[ -f /usr/aarch64-linux-gnu/lib/libstdc++.so.6 && "$ARCH" == "aarch64" ]]; then
        cp /usr/aarch64-linux-gnu/lib/libstdc++.so.6 "$release_dir/lib/"
    fi

    # 仅复制llama-server可执行文件供测试使用
    cp build/bin/llama-server "$release_dir/bin/" || true
    cp build/bin/llama-bench* "$release_dir/bin/" || true
    cp build/bin/llama-gguf "$release_dir/bin/" || true
    cp build/bin/llama-gguf-split "$release_dir/bin/" || true
    cp build/bin/llama-cli "$release_dir/bin/" || true
    for file in "$release_dir/bin/"*; do
        patchelf_set_rpath '$ORIGIN/../lib' "$file"
        verify_elf_64k_alignment "$file"
    done
    for so_file in "$release_dir"/lib/*.so; do
        if [ -f "$so_file" ]; then
            patchelf_set_rpath '$ORIGIN' "$so_file"
            verify_elf_64k_alignment "$so_file"
        fi
    done

    cp tools/server/qwen3_nonthinking.jinja "$release_dir/bin/" || true
    cp ci/install_*.sh "$release_dir/service/" || true
    cp ci/uninstall_*.sh "$release_dir/service/" || true


    # 复制脚本
    cp convert_hm_to_gguf.py "$release_dir/scripts/" || true
    cp third_party/houmo/bin/hmmstrip "$release_dir/scripts/" || true
    cp third_party/houmo/bin/readhmm "$release_dir/scripts/" || true
    cp requirements/requirements-convert_hm_to_gguf.txt "$release_dir/scripts/" || true
    cp -r gguf-py "$release_dir/scripts/"

    echo "Release files have been organized under: $release_dir"
    # 显示打包内容摘要
    echo "Package contents summary:"
    find "$release_dir" -type f | wc -l
}

function package_ollama() {
    echo "Packaging the HLAWOllama..."

    local llama_release_dir="release/$PLATFORM/$ARCH"

    if [ ! -d "release" ]; then
        mkdir release
    fi
    rm -rf release/${PLATFORM}_${ARCH}
    mkdir -p release/${PLATFORM}_${ARCH}/HLAWOllama
    mkdir -p release/${PLATFORM}_${ARCH}/HLAWOllama/bin
    mkdir -p release/${PLATFORM}_${ARCH}/HLAWOllama/lib
    download_file "$ollama_url" "release/${PLATFORM}_${ARCH}/HLAWOllama/bin/hlaw_ollama"
    cp -r $llama_release_dir/bin/llama-server release/${PLATFORM}_${ARCH}/HLAWOllama/bin/
    cp -r $llama_release_dir/lib/*.so* release/${PLATFORM}_${ARCH}/HLAWOllama/lib/
    cp ci/install_ollama.sh release/${PLATFORM}_${ARCH}/HLAWOllama/
    cp ci/uninstall_ollama.sh release/${PLATFORM}_${ARCH}/HLAWOllama/

    makeself --gzip --target "/opt/HLAWOllama"\
            "./release/${PLATFORM}_${ARCH}/HLAWOllama" \
            "./release/HLAWOllama_${PLATFORM}_${ARCH}.run" \
            "HLAWOllama Installer" \
            "./install_ollama.sh"
    rm -rf release/${PLATFORM}_${ARCH}
    echo "Package completed!"
}


# 主函数
main() {
    # 确定使用的SDK URL
    case "$PLATFORM" in
        xh2)
            if [ "$ARCH" == "aarch64" ]; then
                deploy_dadao_sdk "$xh2_dadao_sdk_url"
            else
                deploy_dadao_sdk "$xh2_dadao_sdk_url"
            fi
            ;;
    esac

    echo "Deployed Dadao SDK and System SDK for $PLATFORM/$ARCH."

    # 编译和打包
    build_ui_assets
    build_llama
    package_llama

    download_file "$modelcli_url" "release/$PLATFORM/$ARCH/bin/model-cli"
    download_file "$ollama_url" "release/$PLATFORM/$ARCH/bin/hlaw_ollama"
    patchelf_set_rpath '$ORIGIN/../lib' release/$PLATFORM/$ARCH/bin/hlaw_ollama
    verify_elf_64k_alignment release/$PLATFORM/$ARCH/bin/hlaw_ollama
    chmod 755 release/$PLATFORM/$ARCH/bin/hlaw_ollama
    chmod 755 release/$PLATFORM/$ARCH/bin/model-cli
    echo "LLAMA build and packaging completed for $PLATFORM/$ARCH."
}
# 执行主函数
main
