# TrueForm Benchmarks
# Compare TrueForm performance against other geometry libraries

cmake_minimum_required(VERSION 3.20)
project(trueform-benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "=== Configuring TrueForm Benchmarks ===")

# ------------------------------------------------------------------------------
# Copy benchmark data to build directory for relative path access
# ------------------------------------------------------------------------------

message(STATUS "Copying benchmark data to build directory")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Copy Python consolidation script to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/consolidate_results.py
    ${CMAKE_CURRENT_BINARY_DIR}/consolidate_results.py
    COPYONLY
)

# ------------------------------------------------------------------------------
# Fetch external libraries for comparison
# ------------------------------------------------------------------------------

include(FetchContent)

message(STATUS "Fetching external libraries via FetchContent...")

# ------------------------------------------------------------------------------
# Allow duplicate custom targets (uninstall, install) from dependencies
# ------------------------------------------------------------------------------

set_property(GLOBAL PROPERTY ALLOW_DUPLICATE_CUSTOM_TARGETS TRUE)

# ------------------------------------------------------------------------------
# nanoflann - Fast KD-tree library for point clouds
# ------------------------------------------------------------------------------

message(STATUS "  - nanoflann v1.5.5 (Fast KD-tree for point clouds)")
FetchContent_Declare(
    nanoflann
    GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
    GIT_TAG        v1.5.5
)
set(NANOFLANN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NANOFLANN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nanoflann)
message(STATUS "    nanoflann: Ready")

# Some versions only define 'nanoflann' (non-namespaced); create an alias if needed.
if (TARGET nanoflann AND NOT TARGET nanoflann::nanoflann)
    add_library(nanoflann::nanoflann INTERFACE IMPORTED)
    target_link_libraries(nanoflann::nanoflann INTERFACE nanoflann)
endif()

# ------------------------------------------------------------------------------
# Boost - use system Boost (install via: brew install boost)
# ------------------------------------------------------------------------------

message(STATUS "  - Boost (system, required by CGAL)")
find_package(Boost REQUIRED)
message(STATUS "    Boost: Found (version ${Boost_VERSION})")

# ------------------------------------------------------------------------------
# TBB - for CGAL parallelism
# ------------------------------------------------------------------------------

message(STATUS "  - TBB (for CGAL parallel algorithms)")
find_package(TBB REQUIRED)
message(STATUS "    TBB: found, version: ${TBB_VERSION}")

# ------------------------------------------------------------------------------
# OpenMP - for libigl parallelism (special handling for macOS)
# ------------------------------------------------------------------------------

message(STATUS "  - OpenMP (for libigl parallel algorithms)")
find_package(OpenMP QUIET)

if(NOT OpenMP_CXX_FOUND AND APPLE)
    # Homebrew libomp doesn't provide cmake config, need manual setup
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(LIBOMP_PREFIX)
        message(STATUS "    OpenMP: Found Homebrew libomp at ${LIBOMP_PREFIX}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib" CACHE STRING "" FORCE)

        find_package(OpenMP REQUIRED)

        # Ensure include path is on the imported target
        if(TARGET OpenMP::OpenMP_CXX)
            set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${LIBOMP_PREFIX}/include"
            )
        endif()
    else()
        message(WARNING "    OpenMP: Not found. Install with: brew install libomp")
    endif()
endif()

if(OpenMP_CXX_FOUND)
    message(STATUS "    OpenMP: Ready (version ${OpenMP_CXX_VERSION})")
else()
    message(STATUS "    OpenMP: Not found (libigl will run single-threaded)")
endif()

# ------------------------------------------------------------------------------
# CGAL - Computational Geometry Algorithms Library
# ------------------------------------------------------------------------------

message(STATUS "  - CGAL v6.0.1 (Computational Geometry Algorithms Library)")

FetchContent_Declare(
    cgal
    GIT_REPOSITORY https://github.com/CGAL/cgal.git
    GIT_TAG        v6.0.1
)

# Silence CGAL's build-type warning
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE ON CACHE BOOL "" FORCE)

FetchContent_GetProperties(cgal)
if (NOT cgal_POPULATED)
    FetchContent_Populate(cgal)

    # Bring CGAL into this build as a subproject.
    # It will detect TBB and Boost we configured above.
    add_subdirectory(${cgal_SOURCE_DIR} ${cgal_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Make find_package(CGAL) resolve ONLY to the fetched CGAL build tree
set(CGAL_DIR "${cgal_BINARY_DIR}" CACHE PATH "CGAL build directory" FORCE)

# Use CONFIG mode + NO_DEFAULT_PATH so we don't ever hit system CGAL
find_package(CGAL REQUIRED CONFIG
             PATHS "${CGAL_DIR}"
             NO_DEFAULT_PATH)

message(STATUS "    CGAL: Ready (from FetchContent, CGAL_VERSION=${CGAL_VERSION})")

# ------------------------------------------------------------------------------
# Ensure CGAL::CGAL_Core exists for libigl's copyleft/cgal module
# ------------------------------------------------------------------------------

if (TARGET CGAL::CGAL_Core)
    message(STATUS "    CGAL: CGAL_Core target available (CGAL::CGAL_Core)")
else()
    message(STATUS "    CGAL: CGAL_Core target not exported; creating interface CGAL::CGAL_Core")

    add_library(CGAL::CGAL_Core INTERFACE IMPORTED)

    # For our purposes, linking to CGAL::CGAL is sufficient.
    # This satisfies libigl's igl_copyleft_cgal target without needing a separate Core lib.
    target_link_libraries(CGAL::CGAL_Core INTERFACE CGAL::CGAL)
endif()

# ------------------------------------------------------------------------------
# Eigen3 - Required by FCL
# ------------------------------------------------------------------------------

message(STATUS "  - Eigen3 v3.4.0 (Linear algebra library for FCL)")
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(eigen)
message(STATUS "    Eigen3: Ready")

# Hint FCL's internal FindEigen3 to use this Eigen, not system Eigen
set(EIGEN3_INCLUDE_DIR "${eigen_SOURCE_DIR}" CACHE PATH "" FORCE)
set(Eigen3_DIR "${eigen_BINARY_DIR}" CACHE PATH "" FORCE)
set(CMAKE_PREFIX_PATH "${eigen_BINARY_DIR};${CMAKE_PREFIX_PATH}" CACHE PATH "" FORCE)

# VTK expects Eigen3::Eigen3 - create IMPORTED target
if(TARGET Eigen3::Eigen AND NOT TARGET Eigen3::Eigen3)
    add_library(Eigen3::Eigen3 INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen3 PROPERTIES
        INTERFACE_LINK_LIBRARIES Eigen3::Eigen
    )
endif()

# ------------------------------------------------------------------------------
# FCL - Flexible Collision Library (mesh-mesh distance baseline)
# ------------------------------------------------------------------------------

message(STATUS "  - FCL 0.7.0 (Flexible Collision Library)")
FetchContent_Declare(
    fcl
    GIT_REPOSITORY https://github.com/flexible-collision-library/fcl.git
    GIT_TAG 0.7.0
)

# FCL options
set(FCL_ENABLE_PROFILING OFF CACHE BOOL "" FORCE)
set(FCL_TREAT_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
set(FCL_HIDE_ALL_SYMBOLS OFF CACHE BOOL "" FORCE)
set(FCL_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
set(FCL_USE_X64_SSE OFF CACHE BOOL "" FORCE)
set(FCL_USE_HOST_NATIVE_ARCH OFF CACHE BOOL "" FORCE)
set(FCL_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_GetProperties(fcl)
if(NOT fcl_POPULATED)
    FetchContent_Populate(fcl)

    # Add FCL as subdirectory
    add_subdirectory(${fcl_SOURCE_DIR} ${fcl_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Create fcl::fcl alias
if(TARGET fcl AND NOT TARGET fcl::fcl)
    add_library(fcl::fcl ALIAS fcl)
endif()

message(STATUS "    FCL: Ready (FetchContent)")

# ------------------------------------------------------------------------------
# Coal - High-performance collision library (FCL fork)
# ------------------------------------------------------------------------------

message(STATUS "  - Coal (High-performance collision library, FCL fork)")
FetchContent_Declare(
    coal
    GIT_REPOSITORY https://github.com/coal-library/coal.git
    GIT_TAG devel
)

# Coal options - disable optional dependencies we don't need
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(COAL_BUILD_WITH_TRACY OFF CACHE BOOL "" FORCE)
set(COAL_ENABLE_LOGGING OFF CACHE BOOL "" FORCE)
set(COAL_HAS_OCTOMAP OFF CACHE BOOL "" FORCE)
set(COAL_HAS_QHULL OFF CACHE BOOL "" FORCE)
set(HPP_FCL_HAS_OCTOMAP OFF CACHE BOOL "" FORCE)
set(HPP_FCL_HAS_QHULL OFF CACHE BOOL "" FORCE)
# Disable Python bindings - we only need C++ library
set(BUILD_PYTHON_INTERFACE OFF CACHE BOOL "" FORCE)
set(COAL_BUILD_PYTHON_INTERFACE OFF CACHE BOOL "" FORCE)

FetchContent_GetProperties(coal)
if(NOT coal_POPULATED)
    FetchContent_Populate(coal)

    # Add Coal as subdirectory
    add_subdirectory(${coal_SOURCE_DIR} ${coal_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

message(STATUS "    Coal: Ready (FetchContent)")

# ------------------------------------------------------------------------------
# libigl - Geometry Processing Library (header-only)
# ------------------------------------------------------------------------------

message(STATUS "  - libigl v2.5.0 (Geometry Processing Library)")


FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.5.0
)

# Force libigl to use our Eigen and CGAL
set(FETCHCONTENT_SOURCE_DIR_EIGEN ${eigen_SOURCE_DIR} CACHE PATH "" FORCE)
set(FETCHCONTENT_SOURCE_DIR_CGAL ${cgal_SOURCE_DIR} CACHE PATH "" FORCE)

# Disable all optional dependencies - we only need core functionality
set(LIBIGL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LIBIGL_BUILD_TUTORIALS OFF CACHE BOOL "" FORCE)
set(LIBIGL_INSTALL OFF CACHE BOOL "" FORCE)
set(LIBIGL_USE_STATIC_LIBRARY OFF CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_CGAL ON CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_COMISO OFF CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_CORE OFF CACHE BOOL "" FORCE)
set(LIBIGL_COPYLEFT_TETGEN OFF CACHE BOOL "" FORCE)
set(LIBIGL_EMBREE OFF CACHE BOOL "" FORCE)
set(LIBIGL_GLFW OFF CACHE BOOL "" FORCE)
set(LIBIGL_IMGUI OFF CACHE BOOL "" FORCE)
set(LIBIGL_OPENGL OFF CACHE BOOL "" FORCE)
set(LIBIGL_PNG OFF CACHE BOOL "" FORCE)
set(LIBIGL_PREDICATES OFF CACHE BOOL "" FORCE)
set(LIBIGL_RESTRICTED_MATLAB OFF CACHE BOOL "" FORCE)
set(LIBIGL_RESTRICTED_MOSEK OFF CACHE BOOL "" FORCE)
set(LIBIGL_RESTRICTED_TRIANGLE OFF CACHE BOOL "" FORCE)
set(LIBIGL_XML OFF CACHE BOOL "" FORCE)
set(LIBIGL_USE_OPENMP  ON CACHE BOOL "" FORCE)
set(LIBIGL_WITH_OPENMP ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(libigl)
message(STATUS "    libigl: Ready")

# ------------------------------------------------------------------------------
# VTK - Visualization Toolkit (optional, system only)
# ------------------------------------------------------------------------------

message(STATUS "  - VTK (optional, system)")
find_package(VTK QUIET)
if(VTK_FOUND)
    message(STATUS "    VTK: Found (version ${VTK_VERSION})")
    set(HAVE_VTK TRUE)
else()
    message(STATUS "    VTK: Not found (VTK benchmarks will be skipped)")
    set(HAVE_VTK FALSE)
endif()

# ------------------------------------------------------------------------------
# TBB support for CGAL: ensure CGAL::TBB_support exists
# ------------------------------------------------------------------------------

# Try to detect which TBB target name we actually have
set(_TBB_TARGET "")
if (TARGET TBB::tbb)
    set(_TBB_TARGET TBB::tbb)
elseif (TARGET tbb)
    # Some older FindTBB modules use plain 'tbb'
    set(_TBB_TARGET tbb)
endif()

if (TARGET CGAL::TBB_support)
    message(STATUS "    CGAL: TBB support available (CGAL::TBB_support)")
else()
    message(STATUS "    CGAL: TBB support not exported; creating interface CGAL::TBB_support")

    add_library(CGAL::TBB_support INTERFACE IMPORTED)

    # Enable CGAL's TBB-aware code paths in headers
    target_compile_definitions(CGAL::TBB_support INTERFACE
        CGAL_LINKED_WITH_TBB
    )

    # Link TBB if we identified a target; otherwise just warn
    if (_TBB_TARGET)
        target_link_libraries(CGAL::TBB_support INTERFACE ${_TBB_TARGET})
    else()
        message(WARNING "    CGAL: TBB was found but no TBB::tbb or tbb target exists. "
                        "CGAL::TBB_support will only set CGAL_LINKED_WITH_TBB. "
                        "Make sure to link your benchmarks to TBB manually.")
    endif()
endif()

# ------------------------------------------------------------------------------
# Common benchmark utilities
# ------------------------------------------------------------------------------

message(STATUS "Adding benchmark utilities:")
message(STATUS "  - common (timing, test_meshes)")
add_subdirectory(common)

message(STATUS "  - cgal-common (CGAL conversions)")
add_subdirectory(cgal-common)

message(STATUS "  - fcl-common (FCL conversions)")
add_subdirectory(fcl-common)

message(STATUS "  - coal-common (Coal conversions)")
add_subdirectory(coal-common)

message(STATUS "  - igl-common (libigl conversions)")
add_subdirectory(igl-common)

if(HAVE_VTK)
    message(STATUS "  - vtk-common (VTK conversions)")
    add_subdirectory(vtk-common)
endif()

# ------------------------------------------------------------------------------
# Module benchmarks
# ------------------------------------------------------------------------------

message(STATUS "Adding module benchmarks:")
message(STATUS "  - spatial")
add_subdirectory(spatial EXCLUDE_FROM_ALL)

message(STATUS "  - topology")
add_subdirectory(topology EXCLUDE_FROM_ALL)

message(STATUS "  - intersect")
add_subdirectory(intersect EXCLUDE_FROM_ALL)

message(STATUS "  - cut")
add_subdirectory(cut EXCLUDE_FROM_ALL)

message(STATUS "  - geometry")
add_subdirectory(geometry EXCLUDE_FROM_ALL)

message(STATUS "  - io")
add_subdirectory(io EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------------------
# Unified benchmark runner
# ------------------------------------------------------------------------------

message(STATUS "Creating unified benchmark runner: run-benchmarks")

# Collect all object library implementations
set(ALL_BENCHMARK_IMPLS
    # Cut module
    cut-boolean-tf-impl
    cut-boolean-cgal-impl
    cut-boolean-igl-impl
    cut-embedded_self_intersection_curves-tf-impl
    cut-embedded_self_intersection_curves-igl-impl
    cut-embedded_isocurves-tf-impl

    # Topology module
    topology-connected_components-cgal-impl
    topology-connected_components-tf-impl
    topology-connected_components-igl-impl
    topology-boundary_paths-cgal-impl
    topology-boundary_paths-tf-impl
    topology-boundary_paths-igl-impl

    # Intersect module
    intersect-mesh_mesh_curves-tf-impl
    intersect-mesh_mesh_curves-cgal-impl
    intersect-isocontours-tf-impl
    intersect-isocontours-igl-impl

    # Spatial module
    spatial-point_cloud-build_tree-tf-impl
    spatial-point_cloud-build_tree-nanoflann-impl
    spatial-point_cloud-knn-tf-impl
    spatial-point_cloud-knn-nanoflann-impl
    spatial-polygons-build_tree-tf-impl
    spatial-polygons-build_tree-cgal-impl
    spatial-polygons-build_tree-fcl-impl
    spatial-polygons-closest_point-tf-impl
    spatial-polygons-closest_point-cgal-impl
    spatial-polygons_to_polygons-closest_point-tf-impl
    spatial-polygons_to_polygons-closest_point-fcl-impl
    spatial-polygons_to_polygons-collision-tf-impl
    spatial-polygons_to_polygons-collision-fcl-impl
    spatial-polygons-build_tree-coal-impl
    spatial-polygons_to_polygons-closest_point-coal-impl
    spatial-polygons_to_polygons-collision-coal-impl

    # Geometry module
    geometry-principal_curvatures-tf-impl
    geometry-principal_curvatures-igl-impl
    geometry-point_normals-tf-impl
    geometry-point_normals-igl-impl

    # IO module
    io-read_stl-tf-impl
    io-read_stl-igl-impl
)

# Add VTK implementations if available
if(HAVE_VTK)
    list(APPEND ALL_BENCHMARK_IMPLS
        cut-embedded_isocurves-vtk-impl
        topology-connected_components-vtk-impl
        topology-boundary_paths-vtk-impl
        intersect-mesh_mesh_curves-vtk-impl
        intersect-isocontours-vtk-impl
        geometry-point_normals-vtk-impl
        io-read_stl-vtk-impl
    )

    # Define HAVE_VTK for run_benchmarks.cpp
    set(VTK_COMPILE_DEFS "HAVE_VTK")
endif()

# Create the unified runner executable
add_executable(run-benchmarks run_benchmarks.cpp)
target_link_libraries(run-benchmarks ${ALL_BENCHMARK_IMPLS})

if(HAVE_VTK)
    target_compile_definitions(run-benchmarks PRIVATE ${VTK_COMPILE_DEFS})
endif()

# ------------------------------------------------------------------------------
# Custom target to build all benchmarks
# ------------------------------------------------------------------------------

set(BENCHMARK_TARGETS
    spatial-point_cloud-build_tree-tf
    spatial-point_cloud-build_tree-nanoflann
    spatial-point_cloud-knn-tf
    spatial-point_cloud-knn-nanoflann
    spatial-polygons-build_tree-tf
    spatial-polygons-build_tree-cgal
    spatial-polygons-build_tree-fcl
    spatial-polygons-closest_point-tf
    spatial-polygons-closest_point-cgal
    spatial-polygons_to_polygons-closest_point-tf
    spatial-polygons_to_polygons-closest_point-fcl
    spatial-polygons_to_polygons-collision-tf
    spatial-polygons_to_polygons-collision-fcl
    spatial-polygons-build_tree-coal
    spatial-polygons_to_polygons-closest_point-coal
    spatial-polygons_to_polygons-collision-coal
    topology-connected_components-cgal
    topology-connected_components-tf
    topology-boundary_paths-cgal
    topology-boundary_paths-tf
    topology-connected_components-igl
    topology-boundary_paths-igl
    intersect-mesh_mesh_curves-tf
    intersect-mesh_mesh_curves-cgal
    intersect-isocontours-tf
    intersect-isocontours-igl
    cut-boolean-tf
    cut-boolean-cgal
    cut-boolean-igl
    cut-embedded_self_intersection_curves-tf
    cut-embedded_self_intersection_curves-igl
    cut-embedded_isocurves-tf
    geometry-principal_curvatures-tf
    geometry-principal_curvatures-igl
    geometry-point_normals-tf
    geometry-point_normals-igl
    io-read_stl-tf
    io-read_stl-igl
    run-benchmarks
)

# Add VTK targets if available
if(HAVE_VTK)
    list(APPEND BENCHMARK_TARGETS
        topology-connected_components-vtk
        topology-boundary_paths-vtk
        intersect-mesh_mesh_curves-vtk
        intersect-isocontours-vtk
        cut-embedded_isocurves-vtk
        geometry-point_normals-vtk
        io-read_stl-vtk
    )
endif()

list(LENGTH BENCHMARK_TARGETS NUM_BENCHMARKS)
message(STATUS "Creating 'benchmarks' target with ${NUM_BENCHMARKS} executables")
add_custom_target(benchmarks)
add_dependencies(benchmarks ${BENCHMARK_TARGETS})

# ------------------------------------------------------------------------------
# Link OpenMP into libigl benchmark targets only
# ------------------------------------------------------------------------------

if(OpenMP_CXX_FOUND)
    set(IGL_BENCHMARK_TARGETS
        topology-connected_components-igl
        topology-boundary_paths-igl
        intersect-isocontours-igl
        cut-boolean-igl
        cut-embedded_self_intersection_curves-igl
        geometry-principal_curvatures-igl
        geometry-point_normals-igl
        io-read_stl-igl
        run-benchmarks
    )
    foreach(target ${IGL_BENCHMARK_TARGETS})
        if (TARGET ${target})
            target_link_libraries(${target} OpenMP::OpenMP_CXX)
        endif()
    endforeach()
endif()

# ------------------------------------------------------------------------------
# Compiler optimizations for benchmark targets
# ------------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckCXXCompilerFlag)

    check_cxx_compiler_flag("-march=native" SUPPORTS_MARCH_NATIVE)
    if(SUPPORTS_MARCH_NATIVE)
        message(STATUS "Applying -march=native to benchmark targets")
        foreach(target ${BENCHMARK_TARGETS})
            if (TARGET ${target})
                target_compile_options(${target} PRIVATE -march=native)
            endif()
        endforeach()
    else()
        message(WARNING "Compiler does NOT support -march=native")
    endif()
endif()
