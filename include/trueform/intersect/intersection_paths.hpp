/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/edges.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../topology/find_eulerian_paths.hpp"
#include "../topology/vertex_link.hpp"
#include "./intersection_edges.hpp"

namespace tf {
template <typename Policy>
auto make_intersection_paths(const tf::edges<Policy> &edges,
                             std::size_t n_unique_vertices) {
  using Index = std::decay_t<decltype(edges[0][0])>;
  tf::vertex_link<Index> vl;
  vl.build(edges, n_unique_vertices, tf::edge_orientation::forward);
  tf::offset_block_buffer<Index, Index> buffer;
  buffer.data_buffer().reserve(n_unique_vertices);
  buffer.offsets_buffer().reserve(3);
  tf::find_eulerian_paths(vl, buffer.offsets_buffer(), buffer.data_buffer());
  return buffer;
}

template <typename Index, typename RealT, std::size_t Dims, typename Policy,
          typename Range>
auto make_intersection_paths(
    const tf::scalar_field_intersections<Index, RealT, Dims> &sfi,
    const tf::polygons<Policy> &polygons, const Range &scalar_field,
    typename Range::value_type cut_value = {}) {
  auto ie = tf::make_intersection_edges(sfi, polygons, scalar_field, cut_value);
  return tf::make_intersection_paths(tf::make_edges(ie),
                                     sfi.intersection_points().size());
}
} // namespace tf
