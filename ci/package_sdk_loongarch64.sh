#!/bin/bash

set -e
set -u
set -o pipefail


# Usage:
#   bash ci/package_sdk_loongarch64.sh
# Example:
#   bash ci/package_sdk_loongarch64.sh
ARCH="loongarch64"
PLATFORM="xh2"
# CURRENT_YEAR="$(date +%Y)"
# CURRENT_MONTH="$(date +%m)"
# CURRENT_DAY="$(date +%d)"
CURRENT_YEAR="2026"
CURRENT_MONTH="05"
CURRENT_DAY="06"
URL="http://artifactory.houmo.ai:80/artifactory/houmo_software/loongarch64/${CURRENT_YEAR}/${CURRENT_MONTH}/${CURRENT_DAY}/0200/loongnix/houmo-tcim-runtime-xh2_1.3.0_loongarch64.deb"

download_file() {
	local url="$1"
	local output="$2"
	echo "Downloading $url ..."
	wget "$url" -O "$output"
	echo "Downloaded to $output"
}

deploy_houmo_deb_to_third_party() {
	local deb_file="houmo_runtime_${ARCH}.deb"

	echo "Deploying Houmo .deb to third_party/houmo ..."
	mkdir -p third_party/houmo
	rm -rf third_party/houmo/*

	download_file "$URL" "$deb_file"
	dpkg-deb -x "$deb_file" third_party/houmo
	rm -f "$deb_file"

	if [ -d "third_party/houmo/usr/local/houmo" ]; then
		cp -a third_party/houmo/usr/local/houmo/. third_party/houmo/
		rm -rf third_party/houmo/usr
	fi

	echo "Houmo deployed under: third_party/houmo"
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


resolve_houmo_root() {
	if [ -d "third_party/houmo" ]; then
		echo "third_party/houmo"
	elif [ -d "/usr/local/houmo" ]; then
		echo "/usr/local/houmo"
	else
		echo ""
	fi
}

build_llama() {
	echo "Building for PLATFORM=$PLATFORM ARCH=$ARCH"
	git config --global --add safe.directory "$(pwd)"

	mkdir -p build
	rm -rf build/*

	cmake -S . -B build \
		-DCMAKE_BUILD_TYPE=Release \
		-DLLAMA_CURL=OFF \
		-DLLAMA_USE_PREBUILT_UI=OFF \
		-DENABLE_HOUMO_XH2=ON

	cmake --build build -j"$(nproc)"

	echo "Build completed."
}

package_llama() {
	local release_dir="release/$PLATFORM/$ARCH"
	local houmo_root

	houmo_root="$(resolve_houmo_root)"

	echo "Packaging LLAMA for $PLATFORM/$ARCH ..."
	rm -rf "$release_dir"
	mkdir -p "$release_dir/include" "$release_dir/lib" "$release_dir/bin" "$release_dir/scripts"
	mkdir -p "$release_dir/include/server" "$release_dir/service" "$release_dir/models"

	# headers
	cp -r include/* "$release_dir/include/" || true
	cp -r src/llama*.h* "$release_dir/include/" || true
	cp -r ggml/include/* "$release_dir/include/" || true
	cp -r common/*.h* "$release_dir/include/" || true
	cp -r common/jinja "$release_dir/include/jinja" || true
	cp -r tools/mtmd/mtmd*.h "$release_dir/include/" || true
	cp -r vendor/nlohmann "$release_dir/include/" || true
	cp -r tools/server/*.h* "$release_dir/include/server/" || true
	cp -r build/tools/server/*.hpp "$release_dir/include/server/" || true

	# libraries
	cp build/bin/*.so "$release_dir/lib/" || true
	cp build/common/libcommon.a "$release_dir/lib/" || true
	cp build/tools/server/libserver-context.a "$release_dir/lib/" || true
	if [ -n "$houmo_root" ] && [ -d "$houmo_root/lib" ]; then
		cp "$houmo_root"/lib/*.so* "$release_dir/lib/" || true
	fi
	cp version.cfg "$release_dir/version.txt"

	# binaries
	cp build/bin/llama-server "$release_dir/bin/" || true
	cp build/bin/llama-bench* "$release_dir/bin/" || true
	cp build/bin/llama-gguf "$release_dir/bin/" || true
	cp build/bin/llama-gguf-split "$release_dir/bin/" || true
	cp build/bin/llama-cli "$release_dir/bin/" || true
	cp tools/server/qwen3_nonthinking.jinja "$release_dir/bin/" || true

	# service scripts
	cp ci/install_llama-server.sh "$release_dir/service/" || true
	cp ci/uninstall_llama-server.sh "$release_dir/service/" || true

	# converter scripts
	cp convert_hm_to_gguf.py "$release_dir/scripts/" || true
	if [ -n "$houmo_root" ] && [ -f "$houmo_root/bin/hmmstrip" ]; then
		cp "$houmo_root/bin/hmmstrip" "$release_dir/scripts/" || true
	fi
	cp requirements/requirements-convert_hm_to_gguf.txt "$release_dir/scripts/" || true
	cp -r gguf-py "$release_dir/scripts/"

	# set rpath
	for file in "$release_dir/bin/"*; do
		[ -f "$file" ] || continue
		patchelf --set-rpath '$ORIGIN/../lib' "$file" || true
	done
	for so_file in "$release_dir"/lib/*.so; do
		[ -f "$so_file" ] || continue
		patchelf --set-rpath '$ORIGIN' "$so_file" || true
	done

	echo "Release files have been organized under: $release_dir"
	echo "Package contents summary:"
	find "$release_dir" -type f | wc -l
}

main() {
	deploy_houmo_deb_to_third_party
	build_ui_assets
	build_llama
	package_llama
	echo "LLAMA build and packaging completed for $PLATFORM/$ARCH."
}

main