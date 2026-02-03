/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/

#include "trueform/python/topology.hpp"

namespace tf::py {

auto register_topology(nanobind::module_ &m) -> void {
  // Create topology submodule
  auto topology_module = m.def_submodule("topology", "Topology operations");

  // Register topology components to submodule
  register_topology_label_connected_components(topology_module);
  register_topology_compute_cell_membership(topology_module);
  register_topology_compute_manifold_edge_link(topology_module);
  register_topology_compute_face_link(topology_module);
  register_topology_compute_vertex_link(topology_module);
  register_topology_boundary_edges(topology_module);
  register_topology_non_manifold_edges(topology_module);
  register_topology_boundary_paths(topology_module);
  register_topology_orient_faces_consistently(topology_module);
  register_topology_connect_edges_to_paths(topology_module);
  register_topology_make_k_rings(topology_module);
  register_topology_make_neighborhoods(topology_module);
  register_topology_is_closed(topology_module);
  register_topology_is_manifold(topology_module);
}

} // namespace tf::py
