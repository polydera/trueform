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
#include "../../exact/dyadic_ratio.hpp"
#include "../../exact/snap_parameter_to_edge.hpp"

#include "../../core/buffer.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "./has_adjacent_region_split.hpp"
#include <array>

namespace tf::cut {

template <typename Index, typename Int, typename GetPoint>
auto register_region_split(
    Index n_tags, Index tag,
    const std::array<tf::intersect::graph::vertex<Index>, 2> &edge,
    const tf::intersect::graph::vertex<Index> &vertex,
    const GetPoint &get_point,
    const tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>>
        &splits,
    tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>> &fresh,
    bool named = false) -> void {
  const auto key_0 =
      tf::cut::detail::region_vertex_key(n_tags, tag, edge[0]);
  const auto key_1 =
      tf::cut::detail::region_vertex_key(n_tags, tag, edge[1]);
  const auto key = tf::cut::detail::region_edge_key(key_0, key_1);
  const auto &low = key_0 < key_1 ? edge[0] : edge[1];
  const auto &high = key_0 < key_1 ? edge[1] : edge[0];
  const auto vertex_key =
      tf::cut::detail::region_vertex_key(n_tags, tag, vertex);
  // A position at either end of the edge is not a split of it: placing one
  // there puts a second identity on a vertex that already exists and leaves
  // a constraint with no length.
  if (vertex_key == key_0 || vertex_key == key_1)
    return;

  const auto a = get_point(tag, low);
  const auto b = get_point(tag, high);
  const auto point = get_point(tag, vertex);
  typename tf::exact::meta<Int>::T2 parameter(0);
  for (int coordinate = 0; coordinate < 3; ++coordinate)
    parameter +=
        typename tf::exact::meta<Int>::T2(
            typename tf::exact::meta<Int>::T1(point[coordinate]) -
            typename tf::exact::meta<Int>::T1(a[coordinate])) *
        typename tf::exact::meta<Int>::T2(
            typename tf::exact::meta<Int>::T1(b[coordinate]) -
            typename tf::exact::meta<Int>::T1(a[coordinate]));

  typename tf::exact::meta<Int>::T2 maximum(0);
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    const typename tf::exact::meta<Int>::T1 distance =
        typename tf::exact::meta<Int>::T1(b[coordinate]) -
        typename tf::exact::meta<Int>::T1(a[coordinate]);
    maximum += typename tf::exact::meta<Int>::T2(distance) *
               typename tf::exact::meta<Int>::T2(distance);
  }
  const auto numerator = tf::exact::snap_parameter_to_edge<Int>(
      tf::exact::dyadic_ratio<Int>(parameter, maximum,
                                   tf::exact::meta<Int>::split_grid_bits),
      a, b);

  if (tf::cut::has_adjacent_region_split<Index>(key, numerator, splits, fresh))
    return;
  fresh.push_back({key, numerator, vertex_key, vertex, char(named)});
}

} // namespace tf::cut
