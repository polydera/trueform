# Topology binding sources
# Add new files here when creating new bindings
set(MODULE_TOPOLOGY_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/boundary_edges.cpp
  ${CMAKE_CURRENT_LIST_DIR}/boundary_paths.cpp
  ${CMAKE_CURRENT_LIST_DIR}/compute_cell_membership.cpp
  ${CMAKE_CURRENT_LIST_DIR}/compute_face_link.cpp
  ${CMAKE_CURRENT_LIST_DIR}/compute_manifold_edge_link.cpp
  ${CMAKE_CURRENT_LIST_DIR}/compute_vertex_link.cpp
  ${CMAKE_CURRENT_LIST_DIR}/connect_edges_to_paths.cpp
  ${CMAKE_CURRENT_LIST_DIR}/is_closed.cpp
  ${CMAKE_CURRENT_LIST_DIR}/is_manifold.cpp
  ${CMAKE_CURRENT_LIST_DIR}/label_connected_components.cpp
  ${CMAKE_CURRENT_LIST_DIR}/make_cdt.cpp
  ${CMAKE_CURRENT_LIST_DIR}/make_k_rings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/make_neighborhoods.cpp
  ${CMAKE_CURRENT_LIST_DIR}/non_manifold_edges.cpp
  ${CMAKE_CURRENT_LIST_DIR}/orient_faces_consistently.cpp
  ${CMAKE_CURRENT_LIST_DIR}/orient_faces_consistently_double2d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/orient_faces_consistently_double3d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/orient_faces_consistently_float2d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/orient_faces_consistently_float3d.cpp
)
