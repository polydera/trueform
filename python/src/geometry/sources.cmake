# Geometry binding sources
# Add new files here when creating new bindings
set(MODULE_GEOMETRY_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/chamfer_error.cpp
  ${CMAKE_CURRENT_LIST_DIR}/compute_principal_curvatures.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ensure_positive_orientation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/fit_icp_alignment.cpp
  ${CMAKE_CURRENT_LIST_DIR}/fit_knn_alignment.cpp
  ${CMAKE_CURRENT_LIST_DIR}/fit_obb_alignment.cpp
  ${CMAKE_CURRENT_LIST_DIR}/fit_rigid_alignment.cpp
  ${CMAKE_CURRENT_LIST_DIR}/laplacian_smoothed.cpp
  ${CMAKE_CURRENT_LIST_DIR}/make_mesh_primitives.cpp
  ${CMAKE_CURRENT_LIST_DIR}/measurements.cpp
  ${CMAKE_CURRENT_LIST_DIR}/taubin_smoothed.cpp
  ${CMAKE_CURRENT_LIST_DIR}/triangulated.cpp
)
