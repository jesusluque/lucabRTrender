#!/usr/bin/env bash
# Builds Intel Open Image Denoise for lucabRTrender into ~/tools/oidn-<version>.
#
# From source, not the release binaries: those carry their own libtbb.12,
# which would be a second TBB beside OpenUSD's oneTBB (same soname). Only the
# GPU devices are built -- Metal on Apple Silicon, CUDA on Linux -- and never
# the CPU device: a denoiser that could fall back to the CPU is a CPU fallback,
# which the engine does not have. Whatever TBB the core wants is USD's.
#
# Usage: scripts/build-oidn.sh [version]   (default 2.5.1)
set -euo pipefail

VERSION="${1:-2.5.1}"
PREFIX="${LRT_OIDN_ROOT:-$HOME/tools/oidn-${VERSION}}"
SRC_ROOT="${LRT_OIDN_SRC:-$HOME/tools/src}"
USD_ROOT="${LRT_USD_ROOT:-$HOME/tools/usd-26.08-mx}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN)}"
SRC="$SRC_ROOT/oidn-${VERSION}"

if [[ ! -d "$SRC" ]]; then
    mkdir -p "$SRC_ROOT"
    curl -L --fail -o "$SRC_ROOT/oidn-${VERSION}.src.tar.gz" \
        "https://github.com/RenderKit/oidn/releases/download/v${VERSION}/oidn-${VERSION}.src.tar.gz"
    tar -xzf "$SRC_ROOT/oidn-${VERSION}.src.tar.gz" -C "$SRC_ROOT"
fi

DEVICES=(-DOIDN_DEVICE_CPU=OFF)
if [[ "$(uname)" == "Darwin" ]]; then
    DEVICES+=(-DOIDN_DEVICE_METAL=ON)
else
    DEVICES+=(-DOIDN_DEVICE_CUDA=ON)
    # OIDN's CUDA device wants CUDA 12.8 or later; distributions ship older
    # toolkits in /usr. LRT_CUDA_ROOT, else the newest NVIDIA toolkit in
    # /usr/local, is found first (its nvcc on PATH and as CUDAToolkit_ROOT,
    # which the device's own configure step inherits).
    CUDA_ROOT="${LRT_CUDA_ROOT:-$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)}"
    if [[ -n "$CUDA_ROOT" ]]; then
        export PATH="$CUDA_ROOT/bin:$PATH"
        export CUDAToolkit_ROOT="$CUDA_ROOT"
        DEVICES+=(-DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc")
        echo "OIDN: CUDA toolkit at $CUDA_ROOT"
    fi
fi

cmake -S "$SRC" -B "$SRC/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DTBB_ROOT="$USD_ROOT" \
    -DOIDN_APPS=OFF \
    "${DEVICES[@]}"
cmake --build "$SRC/build" -j "$JOBS"
cmake --install "$SRC/build"
echo "OIDN ${VERSION} installed at ${PREFIX}"
