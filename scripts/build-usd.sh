#!/usr/bin/env bash
# Builds OpenUSD for lucabRTrender into ~/tools/usd-<version>.
#
# Why a script and not FetchContent: OpenUSD takes tens of minutes and brings
# its own oneTBB, OpenSubdiv and friends. Building it once per machine, next to
# ~/tools/slang, keeps the engine's configure step in seconds. The engine then
# links the oneTBB this build installs, so there is one TBB in the process.
#
# Why no Python: OpenUSD 26.08 does not yet support the Python 3.14 that
# Homebrew ships, and the engine needs none of it -- the PLY -> USD converter is
# `lrt convert`, in C++. usdview is therefore not built.
#
# Usage: scripts/build-usd.sh [version]   (default v26.08)
set -euo pipefail

VERSION="${1:-v26.08}"
PREFIX="${LRT_USD_ROOT:-$HOME/tools/usd-${VERSION#v}}"
SRC="${LRT_USD_SRC:-$HOME/tools/src/OpenUSD-${VERSION#v}}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN)}"

if [[ ! -d "$SRC/.git" ]]; then
    mkdir -p "$(dirname "$SRC")"
    git clone --depth 1 --branch "$VERSION" \
        https://github.com/PixarAnimationStudios/OpenUSD.git "$SRC"
fi

PYTHON="$(command -v python3)"
"$PYTHON" "$SRC/build_scripts/build_usd.py" \
    --build-variant release \
    --no-python \
    --no-docs \
    --no-tests \
    --no-tutorials \
    --no-materialx \
    --no-embree \
    --no-ptex \
    --no-openvdb \
    --no-openimageio \
    --no-opencolorio \
    --onetbb \
    --usd-imaging \
    --ignore-homebrew \
    --tools \
    --examples \
    -j "$JOBS" \
    "$PREFIX"

echo "OpenUSD ${VERSION} installed at ${PREFIX}"
echo "Configure lucabRTrender with -DLRT_USD_ROOT=${PREFIX} (the presets default to it)."
