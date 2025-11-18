/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
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

} // namespace tf::py
