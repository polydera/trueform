if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "TF_BUILD_TYPESCRIPT requires Emscripten. Configure with emcmake.")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmscriptenUtils.cmake)
EMSCRIPTEN_ENSURE_TSC("5.9.3")

# TBB must use wasm-exceptions (not SJLJ) to avoid undefined __resumeException.
# -Wno-unused-command-line-argument silences -fstack-clash-protection injected
# by the system's default CXXFLAGS, which Emscripten's clang doesn't support.
if(TARGET tbb)
  target_compile_options(tbb PRIVATE -fwasm-exceptions -Wno-unused-command-line-argument)
endif()

set(TF_TS_DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dist" CACHE PATH
    "Directory to place built TypeScript/WASM artifacts.")
file(MAKE_DIRECTORY ${TF_TS_DIST_DIR})

add_executable(trueform_wasm
    cpp/src/core/wasm_ndarray.cpp
    cpp/src/core/async_dispatcher.cpp
    cpp/src/core/reductions.cpp
    cpp/src/core/elementwise.cpp
    cpp/src/core/wasm_offset_blocked_buffer.cpp
    cpp/src/core/wasm_mesh.cpp
    cpp/src/core/wasm_point_cloud.cpp
    cpp/src/core/wasm_curves.cpp
    cpp/src/core/random.cpp
    cpp/src/core/wasm_index_map.cpp
    cpp/src/io/io.cpp
    cpp/src/cut/boolean.cpp
    cpp/src/cut/mesh_arrangement.cpp
    cpp/src/cut/polygon_arrangement.cpp
    cpp/src/cut/isobands.cpp
    cpp/src/cut/embedded.cpp
    cpp/src/intersect/intersect.cpp
    cpp/src/spatial/distance.cpp
    cpp/src/spatial/intersects.cpp
    cpp/src/spatial/closest_metric_point.cpp
    cpp/src/spatial/closest_metric_point_pair.cpp
    cpp/src/spatial/ray_cast.cpp
    cpp/src/spatial/neighbor_search.cpp
    cpp/src/geometry/triangulate.cpp
    cpp/src/geometry/mesh_primitives.cpp
    cpp/src/geometry/measurements.cpp
    cpp/src/geometry/orientation.cpp
    cpp/src/geometry/curvature.cpp
    cpp/src/geometry/smoothing.cpp
    cpp/src/geometry/registration.cpp
    cpp/src/clean/clean.cpp
    cpp/src/reindex/reindex.cpp
    cpp/src/topology/topology.cpp
    cpp/src/remesh/remesh.cpp
)
target_include_directories(trueform_wasm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
target_link_libraries(trueform_wasm PRIVATE trueform)
target_compile_options(trueform_wasm PRIVATE
  -O3
  -flto
  -msimd128
  -mbulk-memory
  -pthread
  -sSHARED_MEMORY=1
  -sUSE_PTHREADS=1
  -Wall
  -Wextra
  -Wpedantic
  -fwasm-exceptions
)
target_link_options(trueform_wasm PRIVATE
  -O3
  -pthread
  -fwasm-exceptions
  -lembind
  -sWASM=1
  -sSHARED_MEMORY=1
  -sMODULARIZE=1
  -sEXPORT_ES6=1
  -sENVIRONMENT=web,node
  -sALLOW_MEMORY_GROWTH=1
  -sMAXIMUM_MEMORY=4GB
  -sINITIAL_MEMORY=256MB
  -sUSE_PTHREADS=1
  -sPTHREAD_POOL_SIZE='navigator.hardwareConcurrency*2+1'
  -sALLOW_BLOCKING_ON_MAIN_THREAD=1
  "-sEXPORTED_RUNTIME_METHODS=['wasmMemory']"
  -sSTACK_SIZE=8MB
  -sMALLOC=mimalloc
  -sFORCE_FILESYSTEM=0
  --emit-tsd ${TF_TS_DIST_DIR}/trueform.d.ts
)
set_target_properties(trueform_wasm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TF_TS_DIST_DIR}
)
