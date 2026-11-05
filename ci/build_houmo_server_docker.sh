#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_ARCH="${2:-${TARGET_ARCH:-amd64}}"
case "${TARGET_ARCH}" in
    amd64|x86_64)
        TARGET_ARCH=amd64
        TCIM_ARCH=x86_64
        EXPECTED_HAL_MACHINE="Advanced Micro Devices X86-64"
        ;;
    arm64|aarch64)
        TARGET_ARCH=arm64
        TCIM_ARCH=aarch64
        EXPECTED_HAL_MACHINE=AArch64
        ;;
    *)
        echo "unsupported target architecture: ${TARGET_ARCH} (expected amd64 or arm64)" >&2
        exit 1
        ;;
esac

if [[ -n "${1:-}" ]]; then
    IMAGE_NAME="$1"
elif [[ "${TARGET_ARCH}" == "arm64" ]]; then
    IMAGE_NAME="llama-server-xh2:arm64-latest"
else
    IMAGE_NAME="llama-server-xh2:latest"
fi

BUILDER_IMAGE="${BUILDER_IMAGE:-harbor.houmo.ai/application_software/ci_hlie.cpp:v0.0.1-x86_64-ubuntu20.04}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-ubuntu:20.04}"
HOUMO_SDK_PATH="${HOUMO_SDK_PATH:-/usr/local/houmo-sdk}"
TCIM_VERSION="${TCIM_VERSION:-1.4.0}"
TCIM_RELEASE_BASE_URL="${TCIM_RELEASE_BASE_URL:-http://10.10.1.53:8082/artifactory/houmo_software/release_xh2_v1.4.0/2026/07/13/2100/ubuntu2004}"
TCIM_RUNTIME_URL="${TCIM_RUNTIME_URL:-${TCIM_RELEASE_BASE_URL}/houmo_tcim_runtime_xh2_linux_${TCIM_ARCH}-${TCIM_VERSION}.tar.gz}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required" >&2
    exit 1
fi

if ! command -v readelf >/dev/null 2>&1; then
    echo "readelf is required to validate the Houmo HAL architecture" >&2
    exit 1
fi

if ! docker buildx version >/dev/null 2>&1; then
    echo "docker buildx is required" >&2
    exit 1
fi

if [[ -n "${HOUMO_HAL_LIB_DIR:-}" ]]; then
    HAL_LIB_DIR="${HOUMO_HAL_LIB_DIR}"
else
    HAL_LIB_DIR="${HOUMO_SDK_PATH}/hal/lib"
fi

if [[ ! -d "${HAL_LIB_DIR}" ]]; then
    echo "missing Houmo HAL directory: ${HAL_LIB_DIR}" >&2
    exit 1
fi

HAL_LIB_DIR="$(readlink -f "${HAL_LIB_DIR}")"
HAL_LIBRARY="${HAL_LIB_DIR}/libhal_xh2a.so"

if [[ ! -f "${HAL_LIBRARY}" ]]; then
    echo "missing Houmo HAL library: ${HAL_LIBRARY}" >&2
    exit 1
fi

HAL_MACHINE="$(readelf -h "${HAL_LIBRARY}" | awk -F: '/Machine:/{gsub(/^[[:space:]]+/, "", $2); print $2}')"
if [[ "${HAL_MACHINE}" != "${EXPECTED_HAL_MACHINE}" ]]; then
    echo "Houmo HAL architecture mismatch: ${HAL_LIBRARY}" >&2
    echo "  found:    ${HAL_MACHINE}" >&2
    echo "  expected: ${EXPECTED_HAL_MACHINE}" >&2
    exit 1
fi

GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
GIT_COUNT="$(git -C "${REPO_ROOT}" rev-list --count HEAD)"

echo "Building ${IMAGE_NAME}"
echo "  source:        ${REPO_ROOT}"
echo "  commit:        ${GIT_COMMIT}"
echo "  build number:  ${GIT_COUNT}"
echo "  target:        linux/${TARGET_ARCH}"
echo "  builder:       ${BUILDER_IMAGE}"
echo "  runtime:       ${RUNTIME_IMAGE}"
echo "  Houmo HAL:     ${HAL_LIB_DIR} (${HAL_MACHINE})"
echo "  TCIM runtime:  ${TCIM_RUNTIME_URL}"

docker buildx build \
    --load \
    --platform "linux/${TARGET_ARCH}" \
    --build-context "houmo_hal=${HAL_LIB_DIR}" \
    --build-arg "BUILDER_IMAGE=${BUILDER_IMAGE}" \
    --build-arg "RUNTIME_IMAGE=${RUNTIME_IMAGE}" \
    --build-arg "TCIM_VERSION=${TCIM_VERSION}" \
    --build-arg "TCIM_RELEASE_BASE_URL=${TCIM_RELEASE_BASE_URL}" \
    --build-arg "TCIM_RUNTIME_URL=${TCIM_RUNTIME_URL}" \
    --build-arg "GIT_COMMIT=${GIT_COMMIT}" \
    --build-arg "GIT_COUNT=${GIT_COUNT}" \
    --tag "${IMAGE_NAME}" \
    --file "${REPO_ROOT}/.devops/houmo-server.Dockerfile" \
    "${REPO_ROOT}"

BUILT_ARCH="$(docker image inspect "${IMAGE_NAME}" --format '{{.Architecture}}')"
if [[ "${BUILT_ARCH}" != "${TARGET_ARCH}" ]]; then
    echo "built image architecture mismatch: found ${BUILT_ARCH}, expected ${TARGET_ARCH}" >&2
    exit 1
fi

docker image inspect "${IMAGE_NAME}" \
    --format 'Built {{.RepoTags}} ({{.Os}}/{{.Architecture}}, {{.Size}} bytes)'

echo
if [[ "${TARGET_ARCH}" == "amd64" ]]; then
    echo "Smoke test:"
    echo "  docker run --rm --privileged ${IMAGE_NAME} --list-devices"
    echo
    echo "Server example:"
    echo "  docker run --rm --privileged -p 8080:8080 -v /path/to/models:/models:ro ${IMAGE_NAME}"
else
    echo "The ARM64 image was cross-built on an amd64 host."
    echo "Run the device smoke test on an ARM64 XH2 machine:"
    echo "  docker run --rm --privileged ${IMAGE_NAME} --list-devices"
fi
