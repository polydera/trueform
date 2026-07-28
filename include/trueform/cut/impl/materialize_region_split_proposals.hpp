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
#include "../../exact/dyadic_blend.hpp"

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../../exact/snap_parameter_to_edge.hpp"
#include "./has_adjacent_region_split.hpp"
#include "./region_triangulation_point.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int, typename IntersectionPoints,
          typename GetMeshPoint>
auto materialize_region_split_proposals(
    Index n_tags, Index n_intersection_points,
    const tf::buffer<tf::cut::detail::region_split_proposal<Index, Int>>
        &proposals,
    const IntersectionPoints &intersection_points,
    const GetMeshPoint &get_mesh_point,
    tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>> &splits)
    -> std::size_t {
  struct record {
    std::array<std::array<Index, 2>, 2> edge;
    typename tf::exact::meta<Int>::param_type parameter;
    Index tag;
    tf::intersect::graph::vertex<Index> low;
    tf::intersect::graph::vertex<Index> high;
  };

  const bool check_existing = splits.size() != 0;
  auto point = [&](Index tag,
                   const tf::intersect::graph::vertex<Index> &vertex) {
    return tf::cut::region_triangulation_point(
        n_intersection_points, extra_points, tag, vertex,
        intersection_points, get_mesh_point);
  };
  tf::buffer<record> records;
  records.allocate(proposals.size());
  tf::parallel_for_each(
      tf::enumerate(proposals),
      [&](auto indexed_proposal) {
        auto &&[index, proposal] = indexed_proposal;
        const auto key_0 = tf::cut::detail::region_vertex_key(
            n_tags, proposal.tag, proposal.edge[0]);
        const auto key_1 = tf::cut::detail::region_vertex_key(
            n_tags, proposal.tag, proposal.edge[1]);
        const bool forward = key_0 < key_1;
        const auto &low = forward ? proposal.edge[0] : proposal.edge[1];
        const auto &high = forward ? proposal.edge[1] : proposal.edge[0];
        // The same quantisation the crossing producer applies. Without it
        // sameness is asked in parameter units and answered in lattice
        // units: two positions closer than one ulp along the edge are two
        // parameters and one coordinate, so they survive the dedup below
        // and materialize as two ids on one point.
        records[std::size_t(index)] = {
            tf::cut::detail::region_edge_key(key_0, key_1),
            tf::exact::snap_parameter_to_edge<Int>(
                proposal.param, point(proposal.tag, low),
                point(proposal.tag, high)),
            proposal.tag,
            low,
            high};
      },
      tf::checked);
  tbb::parallel_sort(records.begin(), records.end(),
                     [](const record &a, const record &b) {
                       if (a.edge != b.edge)
                         return a.edge < b.edge;
                       return a.parameter < b.parameter;
                     });

  tf::buffer<std::size_t> unique;
  unique.reserve(records.size());
  // Same-split tolerance as tf::cut::has_adjacent_region_split: one
  // quantum of the grid the transported parameter defines.
  using param_t = typename tf::exact::meta<Int>::param_type;
  param_t previous_parameter = param_t(0);
  bool has_previous = false;
  const param_t maximum = param_t(1) << tf::exact::meta<Int>::split_grid_bits;
  for (std::size_t index = 0; index < records.size(); ++index) {
    if (index != 0 && records[index].edge != records[index - 1].edge)
      has_previous = false;
    // One snap step is one ulp of the real the lattice came from, so a
    // position within half a step of an end IS that end. Materializing it
    // would put a second identity on a vertex that already exists — the
    // rejection the crossing producer states on its own endpoints.
    if (records[index].parameter <= param_t(0) ||
        maximum <= records[index].parameter)
      continue;
    if (has_previous &&
        records[index].parameter - previous_parameter <= param_t(1))
      continue;
    if (check_existing &&
        tf::cut::has_adjacent_region_split<Index>(
            records[index].edge, records[index].parameter, splits))
      continue;
    unique.push_back(index);
    previous_parameter = records[index].parameter;
    has_previous = true;
  }

  const Index id_base = n_intersection_points + Index(extra_points.size());
  const std::size_t split_base = splits.size();
  extra_points.reallocate(extra_points.size() + unique.size());
  splits.reallocate(splits.size() + unique.size());
  tf::parallel_for_each(
      tf::enumerate(tf::make_range(unique)),
      [&](auto indexed_record) {
        auto &&[offset, record_index] = indexed_record;
        const auto &input = records[record_index];
        const auto a = point(input.tag, input.low);
        const auto b = point(input.tag, input.high);
        tf::point<Int, 3> materialized;
        for (int coordinate = 0; coordinate < 3; ++coordinate)
          materialized[coordinate] = tf::exact::dyadic_blend(
              a[coordinate], b[coordinate], input.parameter,
              tf::exact::meta<Int>::split_grid_bits);
        const Index id = id_base + Index(offset);
        extra_points[std::size_t(id - n_intersection_points)] =
            materialized;
        const tf::intersect::graph::vertex<Index> vertex{
            tf::intersect::graph::vertex_source::created,
            id,
            {0, tf::topo_type::edge}};
        splits[split_base + std::size_t(offset)] = {
            input.edge,
            input.parameter,
            tf::cut::detail::region_vertex_key(
                n_tags, input.tag, vertex),
            vertex};
      },
      tf::checked);
  if (unique.size() != 0)
    std::inplace_merge(
        splits.begin(), splits.begin() + std::ptrdiff_t(split_base),
        splits.end(), [](const auto &a, const auto &b) {
          if (a.edge != b.edge)
            return a.edge < b.edge;
          return a.param < b.param;
        });
  return unique.size();
}

} // namespace tf::cut
