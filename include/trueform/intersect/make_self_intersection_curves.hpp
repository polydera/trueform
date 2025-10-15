/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/curves_buffer.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./intersections_within_polygons.hpp"
#include "./make_intersection_edges.hpp"

namespace tf {
template <std::size_t Dims, typename Policy>
auto make_self_intersection_curves(const tf::form<Dims, Policy> &form) {
  using Index = typename Policy::index_t;
  tf::intersections_within_polygons<Index, double,
                                    tf::coordinate_dims_v<Policy>>
      iwp;
  iwp.build(form);
  auto ie = tf::make_intersection_edges(iwp);
  auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
  tf::curves_buffer<Index, tf::coordinate_type<Policy>,
                    tf::coordinate_dims_v<Policy>>
      cb;
  cb.paths_buffer() = std::move(paths);
  cb.points_buffer().allocate(iwp.intersection_points().size());
  tf::parallel_copy(iwp.intersection_points(), cb.points());
  return cb;
}
} // namespace tf
