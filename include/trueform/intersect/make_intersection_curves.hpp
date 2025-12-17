/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/curves_buffer.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./intersections_between_polygons.hpp"
#include "./make_intersection_edges.hpp"

namespace tf {
template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_intersection_curves(const tf::form<Dims, Policy0> &form0,
                              const tf::form<Dims, Policy1> &form1) {
  using Index =
      std::common_type_t<typename Policy0::index_type, typename Policy1::index_type>;
  tf::intersections_between_polygons<Index, double,
                                     tf::coordinate_dims_v<Policy0>>
      ibp;
  ibp.build(form0, form1);
  auto ie = tf::make_intersection_edges(ibp);
  auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
  tf::curves_buffer<Index, tf::coordinate_type<Policy0, Policy1>,
                    tf::coordinate_dims_v<Policy0>>
      cb;
  cb.paths_buffer() = std::move(paths);
  cb.points_buffer().allocate(ibp.intersection_points().size());
  tf::parallel_copy(ibp.intersection_points(), cb.points());
  return cb;
}
} // namespace tf
