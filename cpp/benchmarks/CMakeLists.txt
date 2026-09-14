if(EMSCRIPTEN)
  message(FATAL_ERROR "TF_BUILD_CPP_BENCHMARKS is a native-only benchmark option.")
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND
   NOT CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo)$")
  message(WARNING
    "trueform_cpp_ndarray_benchmarks is intended for Release or RelWithDebInfo builds."
  )
endif()

set(tf_cpp_benchmark_allocator "operator-new")
if(TF_USE_MIMALLOC)
  if(TF_USE_SYSTEM_MIMALLOC)
    set(tf_cpp_benchmark_allocator "mimalloc-${TF_MIMALLOC_VERSION}-system")
  else()
    set(tf_cpp_benchmark_allocator "mimalloc-${TF_MIMALLOC_VERSION}-bundled")
  endif()
endif()

# This target is deliberately a separate, opaque optimization boundary. Its
# pointer-and-size call makes complete result storage escape even when the
# benchmark and trueform_cpp are built with supported IPO/LTO.
add_library(trueform_cpp_benchmark_black_box STATIC
  benchmark_black_box.cpp
)
set_target_properties(trueform_cpp_benchmark_black_box PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION FALSE
)
target_compile_features(trueform_cpp_benchmark_black_box PRIVATE cxx_std_17)
target_compile_options(trueform_cpp_benchmark_black_box PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -fno-lto>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /GL->
)

add_executable(trueform_cpp_ndarray_benchmarks
  ndarray_baseline.cpp
)
target_link_libraries(trueform_cpp_ndarray_benchmarks PRIVATE
  tf::trueform_cpp
  trueform_cpp_benchmark_black_box
)
target_compile_features(trueform_cpp_ndarray_benchmarks PRIVATE cxx_std_17)
target_compile_options(trueform_cpp_ndarray_benchmarks PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
target_compile_definitions(trueform_cpp_ndarray_benchmarks PRIVATE
  TF_BENCHMARK_BUILD_TYPE="$<CONFIG>"
  TF_BENCHMARK_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
  TF_BENCHMARK_IPO="$<IF:$<BOOL:${TF_ENABLE_IPO}>,enabled,disabled>"
  TF_BENCHMARK_ALLOCATOR="${tf_cpp_benchmark_allocator}"
)
