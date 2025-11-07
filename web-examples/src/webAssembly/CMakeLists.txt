cmake_minimum_required(VERSION 3.20)

project(trueform_wasm LANGUAGES CXX)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Configure this project with emcmake so the Emscripten toolchain is active.")
endif()
set(TBB_STRICT OFF)
set(TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_subdirectory(../../../ trueform)

set(DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist)
file(MAKE_DIRECTORY ${DIST_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DIST_DIR})

add_executable(native src/native.cpp)
target_include_directories(native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(native PRIVATE trueform)
target_compile_options(native PRIVATE
  -O3
  -pthread
  -matomics
  -mbulk-memory
)
target_link_options(native PRIVATE
  -O3
  -pthread
  -lembind
  -sWASM=1
  -sSHARED_MEMORY=1
  -sFORCE_FILESYSTEM=1
  -sMODULARIZE=1
  -sEXPORT_ES6=1
  -sENVIRONMENT=web
  -sEXPORTED_RUNTIME_METHODS=FS,PATH
  -sALLOW_MEMORY_GROWTH=1
  -sUSE_PTHREADS=1
  #--emit-tsd ${DIST_DIR}/native.d.ts
)
