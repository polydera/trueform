#!/usr/bin/env bash
set -euo pipefail

# Initial directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# CMake configuration variables with defaults
BUILD_DIR="${BUILD_DIR:-${WASM_REPO_ROOT}/build/web}"
DIST_DIR="${DIST_DIR:-${WASM_REPO_ROOT}/build/web/dist}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
CMAKE_FLAGS="${CMAKE_FLAGS:--DBUILD_EXAMPLES=OFF -DBUILD_PYTHON=OFF -DDIST_DIR=${DIST_DIR} -DCMAKE_BUILD_TYPE=Release}"

# Set TBB_DIR to help CMake find TBB installation for Emscripten
export TBB_DIR="/usr/local/lib/cmake/TBB"
CMAKE_FLAGS="${CMAKE_FLAGS} -DTBB_DIR=${TBB_DIR}"

rm -rf "${BUILD_DIR}" && mkdir -p "${BUILD_DIR}"
emcmake cmake -S "${WASM_REPO_ROOT}" -B "${BUILD_DIR}" -G "${CMAKE_GENERATOR}" ${CMAKE_FLAGS}
cmake --build "${BUILD_DIR}"

# In case ARTIFACTS_DIR env is set, copy from DIST_DIR to ARTIFACTS_DIR
if [[ -n "${ARTIFACTS_DIR:-}" ]]; then
    mkdir -p "${ARTIFACTS_DIR}"
    cp -r "${DIST_DIR}/." "${ARTIFACTS_DIR}/"
    echo "Artifacts copied to ${ARTIFACTS_DIR}"
else
    echo "Build artifacts are located in ${DIST_DIR}"
fi
