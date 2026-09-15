#!/usr/bin/env bash
# Builds OpenColorIO for lucabRTrender into ~/tools/ocio-<version>.
#
# From source on both machines, so Metal and Linux run the same version
# (Ubuntu 24.04 ships 2.1, which has no ACES 2.0). OCIO's own dependencies
# (yaml-cpp, pystring, expat, Imath, minizip-ng, zlib) are fetched and linked
# into it statically, so nothing of theirs meets OpenUSD's copies in the
# process. No apps, Python, OpenFX or tests: the engine uses OCIO only as a
# compiler of shader text and LUTs (docs/decisions.md, "OCIO as a compiler").
#
# Usage: scripts/build-ocio.sh [version]   (default 2.5.2)
set -euo pipefail

VERSION="${1:-2.5.2}"
PREFIX="${LRT_OCIO_ROOT:-$HOME/tools/ocio-${VERSION}}"
SRC_ROOT="${LRT_OCIO_SRC:-$HOME/tools/src}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN)}"
SRC="$SRC_ROOT/OpenColorIO-${VERSION}"

if [[ ! -d "$SRC" ]]; then
    mkdir -p "$SRC_ROOT"
    curl -L --fail -o "$SRC_ROOT/ocio-${VERSION}.tar.gz" \
        "https://github.com/AcademySoftwareFoundation/OpenColorIO/archive/refs/tags/v${VERSION}.tar.gz"
    tar -xzf "$SRC_ROOT/ocio-${VERSION}.tar.gz" -C "$SRC_ROOT"
fi

cmake -S "$SRC" -B "$SRC/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DOCIO_INSTALL_EXT_PACKAGES=ALL \
    -DOCIO_BUILD_APPS=OFF \
    -DOCIO_BUILD_PYTHON=OFF \
    -DOCIO_BUILD_OPENFX=OFF \
    -DOCIO_BUILD_TESTS=OFF \
    -DOCIO_BUILD_GPU_TESTS=OFF \
    -DOCIO_BUILD_DOCS=OFF \
    -DOCIO_USE_OIIO_FOR_APPS=OFF
cmake --build "$SRC/build" -j "$JOBS"
cmake --install "$SRC/build"
echo "OpenColorIO ${VERSION} installed at ${PREFIX}"
