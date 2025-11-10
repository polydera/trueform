cmake_minimum_required(VERSION 3.20)

project(trueform_wasm LANGUAGES CXX)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Configure this project with emcmake so the Emscripten toolchain is active.")
endif()

# Ensure the TypeScript compiler that Emscripten relies on for --emit-tsd exists.
get_filename_component(EMSCRIPTEN_ROOT "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(EMSCRIPTEN_ROOT "${EMSCRIPTEN_ROOT}" REALPATH)
set(BUNDLED_EMSCRIPTEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/emsdk/upstream/emscripten")
if(EXISTS "${BUNDLED_EMSCRIPTEN_ROOT}")
  get_filename_component(BUNDLED_EMSCRIPTEN_ROOT "${BUNDLED_EMSCRIPTEN_ROOT}" REALPATH)
else()
  unset(BUNDLED_EMSCRIPTEN_ROOT)
endif()
set(TSC_BIN "${EMSCRIPTEN_ROOT}/node_modules/.bin/tsc")
set(TSC_VERSION "5.9.3")
if(NOT EXISTS "${TSC_BIN}")
  set(TSC_INSTALL_HINT
    "TypeScript is required to emit native TypeScript declarations. Run `cd ${EMSCRIPTEN_ROOT} && npm install --no-save --package-lock false typescript@${TSC_VERSION}` then reconfigure.")
  if(DEFINED BUNDLED_EMSCRIPTEN_ROOT AND EMSCRIPTEN_ROOT STREQUAL BUNDLED_EMSCRIPTEN_ROOT)
    find_program(NPM_EXECUTABLE npm)
    if(NOT NPM_EXECUTABLE)
      message(FATAL_ERROR "npm was not found but is required to install TypeScript. ${TSC_INSTALL_HINT}")
    endif()
    execute_process(
      COMMAND ${NPM_EXECUTABLE} install --no-save --package-lock false typescript@${TSC_VERSION}
      WORKING_DIRECTORY "${EMSCRIPTEN_ROOT}"
      RESULT_VARIABLE NPM_TSC_RESULT
    )
    if(NPM_TSC_RESULT)
      message(FATAL_ERROR "npm failed to install TypeScript (exit code ${NPM_TSC_RESULT}). ${TSC_INSTALL_HINT}")
    endif()
    if(NOT EXISTS "${TSC_BIN}")
      message(FATAL_ERROR "TypeScript installation finished but tsc is still missing. ${TSC_INSTALL_HINT}")
    else()
      message(STATUS "Installed TypeScript ${TSC_VERSION} for the bundled Emscripten SDK.")
    endif()
  else()
    message(FATAL_ERROR "${TSC_INSTALL_HINT}")
  endif()
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

add_executable(native
        src/native.cpp
        src/boolean_web.cpp
        src/utils/bridge_web.cpp
)
target_include_directories(native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(native PRIVATE trueform)
target_compile_options(native PRIVATE
  -O3
  -pthread
  -sSHARED_MEMORY=1
  -sUSE_PTHREADS=1
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
  -sPTHREAD_POOL_SIZE='navigator.hardwareConcurrency*2+1'
  -sALLOW_BLOCKING_ON_MAIN_THREAD=1
  -sSTACK_SIZE=8MB
  -sASSERTIONS=1
  --emit-symbol-map
  -sNO_DISABLE_EXCEPTION_CATCHING=1
  -sMALLOC=mimalloc
  --emit-tsd ${DIST_DIR}/native.d.ts
)
