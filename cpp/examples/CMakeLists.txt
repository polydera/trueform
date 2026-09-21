cmake_minimum_required(VERSION 3.16)
project(trueform_cpp_examples LANGUAGES CXX)

find_package(trueform CONFIG REQUIRED)
if(NOT TARGET tf::trueform_cpp)
  message(FATAL_ERROR
    "the installed trueform carries no compiled facade: install with "
    "-DTF_BUILD_CPP=ON to build these examples")
endif()
include(CTest)

function(add_trueform_example target source)
  add_executable(${target} ${source})
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_link_libraries(${target} PRIVATE tf::trueform_cpp)
  if(BUILD_TESTING)
    add_test(NAME ${target} COMMAND ${target})
  endif()
endfunction()

add_trueform_example(trueform_mesh_and_arrays 01_mesh_and_arrays.cpp)
add_trueform_example(trueform_async_future 02_async_future.cpp)
add_trueform_example(trueform_reusable_csg 03_reusable_csg.cpp)
add_trueform_example(trueform_custom_async_resolvers 04_custom_async_resolvers.cpp)
