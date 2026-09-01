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
#include "./compact_half_edges.hpp"
#include "./retained_half_edges.hpp"

namespace tf::topology::cdt {

template <typename Index> struct triangle_only_topology_policy {
  using edge_buffer = tf::topology::cdt::compact_half_edges<Index>;
  static auto finish_delaunay_edges(edge_buffer &, Index) -> void {}
};

template <typename Index> struct retained_topology_policy {
  using half_edge = retained_delaunay_half_edge<Index>;
  using edge_buffer = retained_half_edges<Index>;

  /// Retained lanes become authoritative only after topology construction.
  /// Initialize them once at that phase boundary instead of in the hot edge
  /// creation and deletion paths.
  static auto finish_delaunay_edges(edge_buffer &edges, Index none) -> void {
    edges.initialize_retained_state(none);
  }
};

} // namespace tf::topology::cdt
