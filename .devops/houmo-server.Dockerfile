# syntax=docker/dockerfile:1.7

ARG BUILDER_IMAGE=harbor.houmo.ai/application_software/ci_hlie.cpp:v0.0.1-x86_64-ubuntu20.04
ARG RUNTIME_IMAGE=ubuntu:20.04

FROM --platform=${BUILDPLATFORM} ${BUILDER_IMAGE} AS build

ARG TARGETARCH
ARG TARGETPLATFORM
ARG GIT_COMMIT=unknown
ARG GIT_COUNT=0
ARG TCIM_VERSION=1.4.0
ARG TCIM_RELEASE_BASE_URL=http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v1.4.0/2026/07/13/2100/ubuntu2004
ARG TCIM_RUNTIME_URL=

WORKDIR /workspace

COPY . .

RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) tcim_arch=x86_64 ;; \
        arm64) tcim_arch=aarch64 ;; \
        *) echo "unsupported target architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    tcim_url="${TCIM_RUNTIME_URL:-${TCIM_RELEASE_BASE_URL}/houmo_tcim_runtime_xh2_linux_${tcim_arch}-${TCIM_VERSION}.tar.gz}"; \
    echo "Downloading TCIM runtime for ${TARGETPLATFORM}: ${tcim_url}"; \
    rm -rf third_party/houmo; \
    mkdir -p third_party/houmo; \
    wget -nv "${tcim_url}" -O /tmp/tcim-runtime.tar.gz; \
    tar -xzf /tmp/tcim-runtime.tar.gz -C third_party/houmo --strip-components=1; \
    rm -f /tmp/tcim-runtime.tar.gz; \
    test -f third_party/houmo/include/tcim/tcim_runtime.h; \
    test -f third_party/houmo/lib/libtcim_runtime_lite.so; \
    test -f third_party/houmo/lib/libtcim_dev_ctrl.so; \
    if [ "${TARGETARCH}" = arm64 ]; then \
        test -f third_party/houmo/lib/libtcimblas.so; \
    fi

RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) toolchain_arg= ;; \
        arm64) toolchain_arg="-DCMAKE_TOOLCHAIN_FILE=/workspace/ci/toolchain-aarch64.cmake" ;; \
        *) echo "unsupported target architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    cmake -S . -B build-houmo-server \
        ${toolchain_arg} \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON \
        -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
        -DLLAMA_BUILD_COMMIT="${GIT_COMMIT}" \
        -DLLAMA_BUILD_NUMBER="${GIT_COUNT}" \
        -DGGML_NATIVE=OFF \
        -DGGML_BACKEND_DL=OFF \
        -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_COMMON=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DENABLE_HOUMO_XH2=ON; \
    cmake --build build-houmo-server --target llama-server -j "$(nproc)"

COPY --from=houmo_hal / /out/hal/lib/

RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) \
            cc=gcc; \
            cxx=g++; \
            strip_tool=strip; \
            readelf_tool=readelf; \
            expected_machine='Advanced Micro Devices X86-64' \
            ;; \
        arm64) \
            cc=aarch64-linux-gnu-gcc; \
            cxx=aarch64-linux-gnu-g++; \
            strip_tool=aarch64-linux-gnu-strip; \
            readelf_tool=aarch64-linux-gnu-readelf; \
            expected_machine=AArch64 \
            ;; \
        *) echo "unsupported target architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    mkdir -p /out/bin /out/lib /out/models; \
    cp build-houmo-server/bin/llama-server /out/bin/; \
    cp -a build-houmo-server/bin/*.so* /out/lib/; \
    cp -a third_party/houmo/lib/*.so* /out/lib/; \
    cp -L "$("${cxx}" -print-file-name=libstdc++.so.6)" /out/lib/libstdc++.so.6; \
    for runtime_lib in libgomp.so.1 libatomic.so.1 libgcc_s.so.1; do \
        runtime_path="$("${cc}" -print-file-name="${runtime_lib}")"; \
        if [ -f "${runtime_path}" ]; then \
            cp -L "${runtime_path}" "/out/lib/${runtime_lib}"; \
        fi; \
    done; \
    rm -f /out/hal/lib/*.a; \
    test -f /out/hal/lib/libhal_xh2a.so; \
    "${strip_tool}" --strip-unneeded /out/bin/llama-server; \
    find /out/lib -maxdepth 1 -type f \
        \( -name 'libllama.so*' -o -name 'libmtmd.so*' -o -name 'libggml*.so*' \) \
        -exec "${strip_tool}" --strip-unneeded {} +; \
    required_elfs='/out/bin/llama-server /out/lib/libtcim_runtime_lite.so /out/lib/libtcim_dev_ctrl.so /out/hal/lib/libhal_xh2a.so'; \
    if [ -f /out/lib/libtcimblas.so ]; then \
        required_elfs="${required_elfs} /out/lib/libtcimblas.so"; \
    fi; \
    for elf_file in ${required_elfs}; do \
        machine="$("${readelf_tool}" -h "${elf_file}" | awk -F: '/Machine:/{gsub(/^[[:space:]]+/, "", $2); print $2}')"; \
        test "${machine}" = "${expected_machine}" || { \
            echo "wrong ELF architecture for ${elf_file}: ${machine}, expected ${expected_machine}" >&2; \
            exit 1; \
        }; \
    done; \
    "${readelf_tool}" -d /out/bin/llama-server | grep NEEDED || true

FROM --platform=${TARGETPLATFORM} ${RUNTIME_IMAGE} AS server

ARG TARGETARCH
ARG GIT_COMMIT=unknown
ARG GIT_COUNT=0
ARG TCIM_VERSION=1.4.0

LABEL org.opencontainers.image.title="Houmo XH2 llama-server" \
      org.opencontainers.image.description="Lightweight Houmo XH2 llama-server runtime" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="0.0.${GIT_COUNT}" \
      org.opencontainers.image.architecture="${TARGETARCH}" \
      com.houmo.tcim.version="${TCIM_VERSION}"

ENV LD_LIBRARY_PATH=/opt/llama/lib:/usr/local/houmo-sdk/hal/lib \
    TCIM_RUNTIME_PATH=/opt/llama/lib \
    HOUMO_SDK_PATH=/usr/local/houmo-sdk \
    HDPL_PLATFORM=ASIC \
    TCIM_BACKEND=Xh2HalBackend \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080 \
    LLAMA_ARG_MODELS_DIR=/models

COPY --from=build /out/bin/llama-server /opt/llama/bin/llama-server
COPY --from=build /out/lib/ /opt/llama/lib/
COPY --from=build /out/hal/lib/ /usr/local/houmo-sdk/hal/lib/
COPY --from=build /out/models/ /models/

WORKDIR /opt/llama

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["/bin/bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/${LLAMA_ARG_PORT:-8080} && printf 'GET /health HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n' >&3 && IFS= read -r line <&3 && [[ \"$line\" == *\" 200 \"* ]]"]

ENTRYPOINT ["/opt/llama/bin/llama-server"]
