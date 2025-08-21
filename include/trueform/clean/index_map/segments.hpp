/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/make_unique_index_map.hpp"
#include "../../core/algorithm/parallel_replace.hpp"
#include "../../core/algorithm/update_by_mask.hpp"
#include "../../core/base/segments.hpp"
#include "../../core/index_map.hpp"
#include "./points.hpp"

namespace tf {
namespace clean {
template <typename Range0, typename Range1, typename Index>
auto make_clean_index_map(const tf::core::segments<Range0, Range1> &segments,
                          tf::index_map_buffer<Index> &edge_map,
                          tf::index_map_buffer<Index> &point_map) {
  auto make_edge = [&](const auto &edge) {
    auto out = std::make_pair(point_map.f()[edge[0]], point_map.f()[edge[1]]);
    if (out.second < out.first)
      std::swap(out.first, out.second);
    else if (out.first == out.second) {
      out = std::make_pair(std::numeric_limits<Index>::max(),
                           std::numeric_limits<Index>::max());
    }
    return out;
  };
  tf::make_unique_index_map(
      segments.edges(), edge_map,
      [&](const auto &x0, const auto &x1) {
        return make_edge(x0) == make_edge(x1);
      },
      [](const auto &x0, const auto &x1) {
        return make_edge(x0) < make_edge(x1);
      });

  // all edges with equal vertices are mapped to the
  // last edge, if they exist
  const auto &last_edge = segments.edges()[edge_map.kept_ids().back()];
  if (point_map.f()[last_edge[0]] == point_map.f()[last_edge[1]]) {
    Index id = edge_map.kept_ids().back();
    edge_map.kept_ids().pop_back();
    tf::parallel_replace(edge_map.f(), id, Index(edge_map.f().size()));
  }

  // now remove all uncontained points from the map
  tf::buffer<bool> contained_points;
  contained_points.allocate(segments.points().size());
  tf::parallel_fill(contained_points, false);
  tf::parallel_apply(
      tf::make_indirect_range(edge_map.kept_ids(), segments.edges()),
      [&](const auto &edge) {
        contained_points[edge[0]] = true;
        contained_points[edge[1]] = true;
      },
      tf::checked);
  tf::update_by_mask(point_map, contained_points);
}
} // namespace clean

template <typename Range0, typename Range1, typename Index>
auto make_clean_index_map(const tf::core::segments<Range0, Range1> &segments,
                          tf::index_map_buffer<Index> &edge_map,
                          tf::index_map_buffer<Index> &point_map) {
  if(!segments.size())
    return;
  make_clean_index_map(segments.points(), point_map);
  clean::make_clean_index_map(segments, edge_map, point_map);
}

template <typename Range0, typename Range1, typename RealType, typename Index>
auto make_clean_index_map(
    const tf::core::segments<Range0, Range1> &segments,
    tf::coordinate_type<decltype(segments.points())> tolerance,
    tf::index_map_buffer<Index> &edge_map,
    tf::index_map_buffer<Index> &point_map) {
  if(!segments.size())
    return;
  make_clean_index_map(segments.points(), tolerance, point_map);
  clean::make_clean_index_map(segments, edge_map, point_map);
}
} // namespace tf
