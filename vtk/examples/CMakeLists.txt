find_package(Freetype QUIET)
find_package(VTK CONFIG REQUIRED COMPONENTS
  CommonCore
  CommonDataModel
  FiltersCore
  FiltersSources
  InteractionStyle
  RenderingCore
  RenderingOpenGL2
  RenderingAnnotation
  RenderingFreeType
)

# Utility library for examples
add_library(vtk_example_util STATIC
  util/drag_interactor.cpp
)
target_include_directories(vtk_example_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vtk_example_util PUBLIC tf::vtk ${VTK_LIBRARIES})

# Helper function to create VTK examples
function(add_vtk_example name)
  add_executable(vtk_example_${name} ${name}.cpp)
  target_link_libraries(vtk_example_${name} PRIVATE vtk_example_util)
  target_compile_definitions(vtk_example_${name} PRIVATE
    TRUEFORM_DATA_DIR="${CMAKE_SOURCE_DIR}"
  )
  vtk_module_autoinit(
    TARGETS vtk_example_${name}
    MODULES ${VTK_LIBRARIES}
  )
endfunction()

# Individual examples
add_vtk_example(intersection_curves)
add_vtk_example(boolean)
add_vtk_example(isocontours)
add_vtk_example(isobands)
add_vtk_example(cross_section)
add_vtk_example(boundary_paths)
add_vtk_example(connected_components)
add_vtk_example(collision)
add_vtk_example(closest_pair)

# Target to build all examples
add_custom_target(trueform_vtk_examples DEPENDS
  vtk_example_intersection_curves
  vtk_example_boolean
  vtk_example_isocontours
  vtk_example_isobands
  vtk_example_cross_section
  vtk_example_boundary_paths
  vtk_example_connected_components
  vtk_example_collision
  vtk_example_closest_pair
)
