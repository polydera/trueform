# Topology binding headers
set(HEADERS_TOPOLOGY
  ${CMAKE_CURRENT_LIST_DIR}/boundary_edges.hpp
  ${CMAKE_CURRENT_LIST_DIR}/boundary_paths.hpp
  ${CMAKE_CURRENT_LIST_DIR}/compute_cell_membership.hpp
  ${CMAKE_CURRENT_LIST_DIR}/compute_face_link.hpp
  ${CMAKE_CURRENT_LIST_DIR}/compute_manifold_edge_link.hpp
  ${CMAKE_CURRENT_LIST_DIR}/compute_vertex_link.hpp
  ${CMAKE_CURRENT_LIST_DIR}/connect_edges_to_paths.hpp
  ${CMAKE_CURRENT_LIST_DIR}/is_closed.hpp
  ${CMAKE_CURRENT_LIST_DIR}/is_manifold.hpp
  ${CMAKE_CURRENT_LIST_DIR}/label_connected_components_impl.hpp
  ${CMAKE_CURRENT_LIST_DIR}/make_k_rings.hpp
  ${CMAKE_CURRENT_LIST_DIR}/make_neighborhoods.hpp
  ${CMAKE_CURRENT_LIST_DIR}/non_manifold_edges.hpp
  ${CMAKE_CURRENT_LIST_DIR}/orient_faces_consistently.hpp
)
