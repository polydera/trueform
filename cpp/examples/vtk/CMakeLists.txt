if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.16)
  project(trueform_cpp_vtk_examples LANGUAGES CXX)
  find_package(trueform CONFIG REQUIRED)
endif()

if(NOT TARGET tf::trueform_cpp)
  message(FATAL_ERROR
    "The Trueform C++ facade target is unavailable. Build or install Trueform "
    "with -DTF_BUILD_CPP=ON.")
endif()

get_filename_component(TRUEFORM_CPP_VTK_DEFAULT_DATA_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(TRUEFORM_CPP_VTK_DATA_DIR "${TRUEFORM_CPP_VTK_DEFAULT_DATA_DIR}" CACHE PATH
  "Trueform source root containing benchmarks/data for default example inputs")

find_package(VTK 9.0 CONFIG REQUIRED COMPONENTS
  CommonCore
  CommonDataModel
  CommonMath
  FiltersCore
  InteractionStyle
  IOImage
  RenderingCore
  RenderingOpenGL2
  RenderingAnnotation
  RenderingFreeType
)

set(TRUEFORM_CPP_VTK_MODULES
  VTK::CommonCore
  VTK::CommonDataModel
  VTK::CommonMath
  VTK::FiltersCore
  VTK::InteractionStyle
  VTK::IOImage
  VTK::RenderingCore
  VTK::RenderingOpenGL2
  VTK::RenderingAnnotation
  VTK::RenderingFreeType
)

set(TRUEFORM_CPP_VTK_BOUNDARY_CHECKER
  "${CMAKE_CURRENT_LIST_DIR}/../../../cmake/CheckCppFacadeTestBoundary.cmake")
if(EXISTS "${TRUEFORM_CPP_VTK_BOUNDARY_CHECKER}")
  add_custom_target(trueform_cpp_vtk_example_boundary
    COMMAND "${CMAKE_COMMAND}"
      -DTRUEFORM_SOURCE_DIR=${TRUEFORM_CPP_VTK_DEFAULT_DATA_DIR}
      -DTRUEFORM_CPP_TEST_ROOT=${CMAKE_CURRENT_LIST_DIR}
      -P "${TRUEFORM_CPP_VTK_BOUNDARY_CHECKER}"
    VERBATIM
  )
endif()

add_library(trueform_cpp_vtk_example_util STATIC
  util/mesh_interactor.cpp
)
target_include_directories(trueform_cpp_vtk_example_util
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(trueform_cpp_vtk_example_util PUBLIC cxx_std_17)
target_link_libraries(trueform_cpp_vtk_example_util
  PUBLIC tf::trueform_cpp ${TRUEFORM_CPP_VTK_MODULES}
)

function(add_trueform_cpp_vtk_example name)
  set(target trueform_cpp_vtk_${name})
  add_executable(${target} ${name}.cpp)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_link_libraries(${target} PRIVATE trueform_cpp_vtk_example_util)
  target_compile_definitions(${target} PRIVATE
    TRUEFORM_DATA_DIR="${TRUEFORM_CPP_VTK_DATA_DIR}"
  )
  vtk_module_autoinit(TARGETS ${target} MODULES ${TRUEFORM_CPP_VTK_MODULES})
  if(TARGET trueform_cpp_vtk_example_boundary)
    add_dependencies(${target} trueform_cpp_vtk_example_boundary)
  endif()
  set_property(GLOBAL APPEND PROPERTY TRUEFORM_CPP_VTK_EXAMPLE_TARGETS ${target})
endfunction()

add_trueform_cpp_vtk_example(alignment)
add_trueform_cpp_vtk_example(boolean_difference)
add_trueform_cpp_vtk_example(collision)
add_trueform_cpp_vtk_example(cross_section)
add_trueform_cpp_vtk_example(csg_fracture)
add_trueform_cpp_vtk_example(domains)
add_trueform_cpp_vtk_example(domains_from_files)
add_trueform_cpp_vtk_example(intersection_curves)
add_trueform_cpp_vtk_example(isobands)
add_trueform_cpp_vtk_example(isocontours)
add_trueform_cpp_vtk_example(remeshing)
add_trueform_cpp_vtk_example(simplify)

get_property(TRUEFORM_CPP_VTK_EXAMPLE_TARGETS GLOBAL
  PROPERTY TRUEFORM_CPP_VTK_EXAMPLE_TARGETS)
add_custom_target(trueform_cpp_vtk_examples
  DEPENDS ${TRUEFORM_CPP_VTK_EXAMPLE_TARGETS}
)
