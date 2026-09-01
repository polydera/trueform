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
    cpp/src/core/mesh_data.cpp
    cpp/src/core/wasm_mesh.cpp
    cpp/src/core/wasm_point_cloud.cpp
    cpp/src/core/wasm_curves.cpp
    cpp/src/core/random.cpp
    cpp/src/core/wasm_index_map.cpp
    cpp/src/core/obb.cpp
    cpp/src/core/histogram.cpp
    cpp/src/core/ensure.cpp
    cpp/src/io/io_float32.cpp
    cpp/src/io/io_float64.cpp
    cpp/src/csg/csg_graph_float32.cpp
    cpp/src/csg/csg_graph_float64.cpp
    cpp/src/csg/outer_shell_float32.cpp
    cpp/src/csg/outer_shell_float64.cpp
    cpp/src/csg/boolean_float32.cpp
    cpp/src/csg/boolean_float64.cpp
    cpp/src/arrangement/mesh_arrangement_float32.cpp
    cpp/src/arrangement/mesh_arrangement_float64.cpp
    cpp/src/arrangement/polygon_arrangement_float32.cpp
    cpp/src/arrangement/polygon_arrangement_float64.cpp
    cpp/src/iso/isobands_float32.cpp
    cpp/src/iso/isobands_float64.cpp
    cpp/src/iso/isocontours_float32.cpp
    cpp/src/iso/isocontours_float64.cpp
    cpp/src/intersect/intersect_float32.cpp
    cpp/src/intersect/intersect_float64.cpp
    cpp/src/spatial/distance_float32.cpp
    cpp/src/spatial/distance_float64.cpp
    cpp/src/spatial/intersects_float32.cpp
    cpp/src/spatial/intersects_float64.cpp
    cpp/src/spatial/closest_metric_point_float32.cpp
    cpp/src/spatial/closest_metric_point_float64.cpp
    cpp/src/spatial/closest_metric_point_pair_float32.cpp
    cpp/src/spatial/closest_metric_point_pair_float64.cpp
    cpp/src/spatial/ray_cast_float32.cpp
    cpp/src/spatial/ray_cast_float64.cpp
    cpp/src/spatial/neighbor_search_float32.cpp
    cpp/src/spatial/neighbor_search_float64.cpp
    cpp/src/geometry/triangulate_float32.cpp
    cpp/src/geometry/triangulate_float64.cpp
    cpp/src/geometry/mesh_primitives_float32.cpp
    cpp/src/geometry/mesh_primitives_float64.cpp
    cpp/src/geometry/measurements_float32.cpp
    cpp/src/geometry/measurements_float64.cpp
    cpp/src/geometry/orientation_float32.cpp
    cpp/src/geometry/orientation_float64.cpp
    cpp/src/geometry/curvature_float32.cpp
    cpp/src/geometry/curvature_float64.cpp
    cpp/src/geometry/smoothing_float32.cpp
    cpp/src/geometry/smoothing_float64.cpp
    cpp/src/geometry/registration_float32.cpp
    cpp/src/geometry/registration_float64.cpp
    cpp/src/geometry/edges_float32.cpp
    cpp/src/geometry/edges_float64.cpp
    cpp/src/clean/clean_float32.cpp
    cpp/src/clean/clean_float64.cpp
    cpp/src/reindex/reindex_float32.cpp
    cpp/src/reindex/reindex_float64.cpp
    cpp/src/reindex/split_into_domains_float32.cpp
    cpp/src/reindex/split_into_domains_float64.cpp
    cpp/src/topology/topology_float32.cpp
    cpp/src/topology/topology_float64.cpp
    cpp/src/topology/cdt_float32.cpp
    cpp/src/topology/cdt_float64.cpp
    cpp/src/topology/make_domain_labels_float32.cpp
    cpp/src/topology/make_domain_labels_float64.cpp
    cpp/src/remesh/remesh_float32.cpp
    cpp/src/remesh/remesh_float64.cpp
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
  # release strips the name section anyway, and at our symbol count
  # emitting it first overruns what the strip step can process
  -Wl,--strip-all
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
  -sPTHREAD_POOL_SIZE='navigator.hardwareConcurrency'
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
