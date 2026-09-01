# Intersect binding sources
# Add new files here when creating new bindings
set(MODULE_INTERSECT_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/intersection_curves.cpp
  ${CMAKE_CURRENT_LIST_DIR}/intersection_curves_int64int64_double3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/intersection_curves_int64int64_float3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/intersection_curves_intint_double3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/intersection_curves_intint_float3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/intersection_curves_intint64_double3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/intersection_curves_intint64_float3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/self_intersection_curves.cpp
  ${CMAKE_CURRENT_LIST_DIR}/self_intersection_curves_int3double3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/self_intersection_curves_int3float3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/self_intersection_curves_int643double3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/self_intersection_curves_int643float3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/self_intersection_curves_int64dyndouble3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/self_intersection_curves_int64dynfloat3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/self_intersection_curves_intdyndouble3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/self_intersection_curves_intdynfloat3d.cpp
)
