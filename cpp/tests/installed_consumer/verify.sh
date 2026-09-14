#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd -- "${script_dir}/../../.." && pwd)"

if (( $# > 1 )); then
  echo "usage: $0 [build-root]" >&2
  exit 2
fi

build_root_input="${TF_INSTALLED_CONSUMER_BUILD_ROOT:-${1:-${TMPDIR:-/tmp}/trueform-installed-consumer-${UID:-user}}}"
build_config="${TF_INSTALLED_CONSUMER_CONFIG:-Release}"
cmake_command="${CMAKE_COMMAND:-cmake}"
ctest_command="${CTEST_COMMAND:-ctest}"

# Only the build root is configurable. Canonicalize it before deriving the three
# verifier-owned directories so legacy/malicious per-directory environment
# overrides cannot redirect cleanup.
if [[ -z "${build_root_input}" ]]; then
  echo "refusing empty installed-consumer build root" >&2
  exit 2
fi
mkdir -p -- "${build_root_input}"
build_root="$(cd -- "${build_root_input}" && pwd -P)"
source_dir="$(cd -- "${source_dir}" && pwd -P)"

path_is_under() {
  [[ "$1" == "$2" || "$1" == "$2/"* ]]
}

if [[ "${build_root}" == "/" ]] || path_is_under "${build_root}" "${source_dir}" || path_is_under "${source_dir}" "${build_root}"; then
  echo "refusing installed-consumer build root that overlaps the source tree: ${build_root}" >&2
  exit 2
fi

producer_build_dir="${build_root}/producer"
install_prefix="${build_root}/install"
consumer_build_dir="${build_root}/consumer"

for clean_dir in "${install_prefix}" "${consumer_build_dir}"; do
  if [[ -L "${clean_dir}" ]]; then
    echo "refusing verifier cleanup through symbolic link: ${clean_dir}" >&2
    exit 2
  fi
done

build_args=(--build "${producer_build_dir}" --config "${build_config}" --parallel)
consumer_build_args=(--build "${consumer_build_dir}" --config "${build_config}" --parallel)
if [[ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
  build_args+=("${CMAKE_BUILD_PARALLEL_LEVEL}")
  consumer_build_args+=("${CMAKE_BUILD_PARALLEL_LEVEL}")
fi

# Keep the producer build incremental, but always verify against a clean install
# tree and a newly configured/rebuilt tiny consumer.
"${cmake_command}" -E remove_directory "${install_prefix}"
"${cmake_command}" -E remove_directory "${consumer_build_dir}"

"${cmake_command}" -S "${source_dir}" -B "${producer_build_dir}" \
  -DCMAKE_BUILD_TYPE="${build_config}" \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DTF_BUILD_CPP=ON \
  -DTF_BUILD_TESTS=OFF \
  -DTF_USE_MIMALLOC=OFF
"${cmake_command}" "${build_args[@]}" --target trueform_cpp
"${cmake_command}" --install "${producer_build_dir}" --config "${build_config}"

"${cmake_command}" -S "${script_dir}" -B "${consumer_build_dir}" \
  -DCMAKE_BUILD_TYPE="${build_config}" \
  -DCMAKE_PREFIX_PATH="${install_prefix}" \
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF \
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF \
  -DTRUEFORM_EXPECTED_INSTALL_PREFIX="${install_prefix}" \
  -DTRUEFORM_FORBIDDEN_INCLUDE_ROOTS="${source_dir}/include;${source_dir}/cpp/include;${producer_build_dir}" \
  -DTRUEFORM_FORBIDDEN_PRODUCER_ROOTS="${source_dir};${producer_build_dir}"
"${cmake_command}" "${consumer_build_args[@]}"
(
  cd -- "${consumer_build_dir}"
  "${ctest_command}" --build-config "${build_config}" --output-on-failure
)
