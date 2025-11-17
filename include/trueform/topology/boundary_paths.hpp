/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/edges.hpp"
#include "../core/faces.hpp"
#include "../core/offset_block_buffer.hpp"
#include "./boundary_edges.hpp"
#include "./face_membership_like.hpp"
#include "./find_eulerian_paths.hpp"

namespace tf {

template <typename Policy>
auto make_boundary_paths(const tf::edges<Policy> &edges,
                         std::size_t max_vertex_id) {
  using Index = std::decay_t<decltype(edges[0][0])>;
  tf::vertex_link<Index> vl;
  vl.build(edges, max_vertex_id, tf::edge_orientation::forward);
  tf::offset_block_buffer<Index, Index> buffer;
  buffer.data_buffer().reserve(max_vertex_id);
  buffer.offsets_buffer().reserve(3);
  tf::find_eulerian_paths(vl, buffer.offsets_buffer(), buffer.data_buffer());
  return buffer;
}

template <typename Policy, typename Policy1>
auto make_boundary_paths(const tf::faces<Policy> &faces,
                         const tf::face_membership_like<Policy1> &fm) {
  auto edges = tf::make_boundary_edges(faces, fm);
  return tf::make_boundary_paths(tf::make_edges(edges), fm.size());
}

template <typename Policy>
auto make_boundary_paths(const tf::polygons<Policy> &polygons) {
  auto edges = tf::make_boundary_edges(polygons);
  return tf::make_boundary_paths(tf::make_edges(edges),
                                 polygons.points().size());
}
} // namespace tf
