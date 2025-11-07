#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/web}"
TF_EXAMPLE_TARGET="${TF_EXAMPLE_TARGET:-web_example1}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
CMAKE_FLAGS="${CMAKE_FLAGS:--DBUILD_EXAMPLES=OFF -DBUILD_EXAMPLES_WEB=ON}"
SERVE="${SERVE:-1}"
TF_SERVE_PORT="${TF_SERVE_PORT:-8080}"
SERVE_ROOT="${SERVE_ROOT:-${BUILD_DIR}/examples/web/example1}"
TARGET_PATH="${TARGET_PATH:-${SERVE_ROOT}/${TF_EXAMPLE_TARGET}.html}"

# Set TBB_DIR to help CMake find TBB installation for Emscripten
export TBB_DIR="/usr/local/lib/cmake/TBB"
CMAKE_FLAGS="${CMAKE_FLAGS} -DTBB_DIR=${TBB_DIR}"

echo "[build_web] Configuring ${TF_EXAMPLE_TARGET} in ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
emcmake cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G "${CMAKE_GENERATOR}" ${CMAKE_FLAGS}

echo "[build_web] Building target ${TF_EXAMPLE_TARGET}"
cmake --build "${BUILD_DIR}" --target "${TF_EXAMPLE_TARGET}"

if [[ "${SERVE}" != "0" ]]; then
  if [[ ! -f "${TARGET_PATH}" ]]; then
    echo "[build_web] Expected output ${TARGET_PATH} not found" >&2
    exit 1
  fi

  echo "[build_web] Serving ${TARGET_PATH} on port ${TF_SERVE_PORT}"
  emrun --no_browser \
    --serve_root "${SERVE_ROOT}" \
    --port "${TF_SERVE_PORT}" \
    "${TARGET_PATH}"
fi
