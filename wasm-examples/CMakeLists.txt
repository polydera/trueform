cmake_minimum_required(VERSION 3.16)

project(trueform_wasm LANGUAGES CXX)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Configure this project with emcmake or proper toolchain so that the Emscripten toolchain is active.")
endif()

# Ensure the TypeScript compiler that Emscripten relies on for --emit-tsd exists.
include(cmake/EmscriptenUtils.cmake)
EMSCRIPTEN_ENSURE_TSC("5.9.3")

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(../ trueform)

option(DIST_DIR "Directory to place built WebAssembly artifacts." ${CMAKE_CURRENT_SOURCE_DIR}/dist)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DIST_DIR})
file(MAKE_DIRECTORY ${DIST_DIR})

add_executable(native
    src/main.cpp
    src/utils/cursor_interactor_interface.cpp
)
target_include_directories(native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(native PRIVATE trueform)
target_compile_options(native PRIVATE
  -O3
  -flto
  -msimd128
  -pthread
  -sSHARED_MEMORY=1
  -sUSE_PTHREADS=1
  -Wall
  -Wextra
  -Wpedantic
  -fwasm-exceptions
)
target_link_options(native PRIVATE
  -O3
  -pthread
  -fwasm-exceptions
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
  -sPTHREAD_POOL_SIZE='navigator.hardwareConcurrency*2+1'
  -sALLOW_BLOCKING_ON_MAIN_THREAD=1
  -sSTACK_SIZE=8MB
  -sASSERTIONS=1
  --emit-symbol-map
  -sMALLOC=mimalloc
  --emit-tsd ${DIST_DIR}/native.d.ts
)
