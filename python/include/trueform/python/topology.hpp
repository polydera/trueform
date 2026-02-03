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
#pragma once

#include <nanobind/nanobind.h>

namespace tf::py {

// Unified topology module registration
auto register_topology(nanobind::module_ &m) -> void;

// Forward declarations for topology module registration (internal)
auto register_topology_label_connected_components(nanobind::module_ &m) -> void;
auto register_topology_compute_cell_membership(nanobind::module_ &m) -> void;
auto register_topology_compute_manifold_edge_link(nanobind::module_ &m) -> void;
auto register_topology_compute_face_link(nanobind::module_ &m) -> void;
auto register_topology_compute_vertex_link(nanobind::module_ &m) -> void;
auto register_topology_boundary_edges(nanobind::module_ &m) -> void;
auto register_topology_non_manifold_edges(nanobind::module_ &m) -> void;
auto register_topology_boundary_paths(nanobind::module_ &m) -> void;
auto register_topology_orient_faces_consistently(nanobind::module_ &m) -> void;
auto register_topology_connect_edges_to_paths(nanobind::module_ &m) -> void;
auto register_topology_make_k_rings(nanobind::module_ &m) -> void;
auto register_topology_make_neighborhoods(nanobind::module_ &m) -> void;
auto register_topology_is_closed(nanobind::module_ &m) -> void;
auto register_topology_is_manifold(nanobind::module_ &m) -> void;

} // namespace tf::py
