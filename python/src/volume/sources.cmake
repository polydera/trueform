# Volume binding sources
# Add new files here when creating new bindings
set(MODULE_VOLUME_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/volume_float.cpp
  ${CMAKE_CURRENT_LIST_DIR}/volume_double.cpp
  ${CMAKE_CURRENT_LIST_DIR}/volume_int16.cpp
  ${CMAKE_CURRENT_LIST_DIR}/volume_uint16.cpp
  ${CMAKE_CURRENT_LIST_DIR}/volume_uint8.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mesh_sdf.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mesh_sdf_int3float3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mesh_sdf_int3double3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mesh_sdf_int643float3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mesh_sdf_int643double3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mesh_sdf_intdynfloat3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mesh_sdf_intdyndouble3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mesh_sdf_int64dynfloat3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/mesh_sdf_int64dyndouble3d.cpp
)
