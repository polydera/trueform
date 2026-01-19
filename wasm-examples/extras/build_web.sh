#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Path setup
# ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ----------------------------------------
# Build configuration
# ----------------------------------------
BUILD_DIR="${BUILD_DIR:-${WASM_REPO_ROOT}/build/web}"
DIST_DIR="${DIST_DIR:-${WASM_REPO_ROOT}/build/web/dist}"

CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
CMAKE_FLAGS="${CMAKE_FLAGS:--DTF_BUILD_EXAMPLES=OFF \
    -DTF_BUILD_PYTHON=OFF \
    -DDIST_DIR=${DIST_DIR} \
    -DCMAKE_BUILD_TYPE=Release}"

# Help CMake find TBB installation for Emscripten
export TBB_DIR="/usr/local/lib/cmake/TBB"
CMAKE_FLAGS+=" -DTBB_DIR=${TBB_DIR}"

# ----------------------------------------
# Configure + build
# ----------------------------------------
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

emcmake cmake \
    -S "${WASM_REPO_ROOT}" \
    -B "${BUILD_DIR}" \
    -G "${CMAKE_GENERATOR}" \
    ${CMAKE_FLAGS}

cmake --build "${BUILD_DIR}"

# ----------------------------------------
# Artifact handling
# ----------------------------------------
if [[ -n "${ARTIFACTS_DIR:-}" ]]; then
    mkdir -p "${ARTIFACTS_DIR}"
    cp -r "${DIST_DIR}/." "${ARTIFACTS_DIR}/"
    echo "Artifacts copied to ${ARTIFACTS_DIR}"
else
    echo "Build artifacts are located in ${DIST_DIR}"
fi
