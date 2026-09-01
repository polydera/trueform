/*
 * Copyright (c) 2026 XLAB
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

#include <cstddef>

namespace tf::intersect::graph {

/// What one local arrangement counted, stage by stage: the detection's
/// work and its findings, the closure's classes, the gate's collapses,
/// and the definition tables in and out.
struct plane_census {
  std::size_t canonical_edges = 0;
  std::size_t pairs_tested = 0;
  std::size_t crossings_ee = 0;
  std::size_t crossings_ve = 0;
  std::size_t collinear_pairs = 0;
  std::size_t names_point = 0;
  std::size_t names_vertex = 0;
  std::size_t names_triple = 0;
  std::size_t names_edge_plane = 0;
  std::size_t names_edge_edge = 0;
  std::size_t names_truncated = 0;
  std::size_t degenerate_exact = 0;
  std::size_t degenerate_materialized = 0;
  std::size_t undecided_order = 0;
  std::size_t classes = 0;
  std::size_t same_root_collapses = 0;
  std::size_t cross_root_merges = 0;
  std::size_t splits_out_of_span = 0;
  std::size_t splits_on_endpoint = 0;
  std::size_t defs_in = 0;
  std::size_t defs_out = 0;

  auto operator+=(const plane_census &o) -> plane_census & {
    canonical_edges += o.canonical_edges;
    pairs_tested += o.pairs_tested;
    crossings_ee += o.crossings_ee;
    crossings_ve += o.crossings_ve;
    collinear_pairs += o.collinear_pairs;
    names_point += o.names_point;
    names_vertex += o.names_vertex;
    names_triple += o.names_triple;
    names_edge_plane += o.names_edge_plane;
    names_edge_edge += o.names_edge_edge;
    names_truncated += o.names_truncated;
    degenerate_exact += o.degenerate_exact;
    degenerate_materialized += o.degenerate_materialized;
    undecided_order += o.undecided_order;
    classes += o.classes;
    same_root_collapses += o.same_root_collapses;
    cross_root_merges += o.cross_root_merges;
    splits_out_of_span += o.splits_out_of_span;
    splits_on_endpoint += o.splits_on_endpoint;
    defs_in += o.defs_in;
    defs_out += o.defs_out;
    return *this;
  }
};

} // namespace tf::intersect::graph
