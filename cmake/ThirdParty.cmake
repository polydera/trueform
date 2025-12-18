# ==============================================================================
# Dependency Management (CPM)
# ==============================================================================

# Add the current directory to the module path to locate CPM.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}")
include(CPM)

# Allow the project to use system libraries if the global flag is set
set(CPM_USE_LOCAL_PACKAGES ${TF_USE_SYSTEM_LIBS})

# ------------------------------------------------------------------------------
# Dependency: oneTBB (Threading Building Blocks)
# ------------------------------------------------------------------------------
if(TF_USE_STATIC_TBB)
    set(TBB_BUILD_SHARED OFF)
else()
    set(TBB_BUILD_SHARED ON)
endif()

if(TF_USE_STATIC_TBB AND TF_USE_SYSTEM_LIBS)
    message(WARNING "TF_USE_STATIC_TBB is ON while TF_USE_SYSTEM_LIBS is also ON. Building custom TBB to allow static linkage.")
    set(CPM_USE_LOCAL_PACKAGES OFF)
endif()

CPMAddPackage(
        NAME                TBB
        GITHUB_REPOSITORY   oneapi-src/oneTBB
        GIT_TAG             v${TF_TBB_VERSION}
        OPTIONS
        "BUILD_SHARED_LIBS ${TBB_BUILD_SHARED}"    # Set static or shared linkage
        "TBB_BUILD ON"
        "TBB_INSTALL ON"
        "TBB_TEST OFF"                          # Disable tests for faster integration
        "TBBMALLOC_BUILD OFF"                   # Disable custom allocator
        "TBBMALLOC_PROXY_BUILD OFF"
        "TBBBIND_BUILD OFF"
        "TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON" # Prevent looking for hwloc system lib
        "CMAKE_POLICY_VERSION_MINIMUM 3.5"      # Support for CMake >= 4.0
)

if(TF_USE_STATIC_TBB AND TF_USE_SYSTEM_LIBS)
    # Reset CPM flag to allow other dependencies to use system libs
    set(CPM_USE_LOCAL_PACKAGES ON)
endif()

# ------------------------------------------------------------------------------
# Dependency: Nanobind (Python Bindings)
# ------------------------------------------------------------------------------
if(TF_BUILD_PYTHON)

  # Ensure a Python environment is available before fetching bindings
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  CPMAddPackage(
          NAME                nanobind
          GITHUB_REPOSITORY   wjakob/nanobind
          GIT_TAG             v${TF_NANOBIND_VERSION}
          OPTIONS
          "NB_USE_SUBMODULE_DEPS ON"
          "NB_TEST OFF"
          "NB_TEST_SHARED_BUILD OFF"
  )

  # Manual include workarounds:
  # Sometimes nanobind config is not immediately picked up by CMake scope
  # if fetched via CPM, so we manually check and include it.
  if(EXISTS "${CPM_PACKAGE_nanobind_SOURCE_DIR}/cmake/nanobind-config.cmake")
    include("${CPM_PACKAGE_nanobind_SOURCE_DIR}/cmake/nanobind-config.cmake")
  endif()

endif()