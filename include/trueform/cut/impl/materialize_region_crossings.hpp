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

#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../exact/edge_edge_closest_point.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../../exact/dyadic_blend.hpp"
#include "./created_point_index.hpp"
#include "./register_region_split.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int, typename GetPoint>
auto materialize_region_crossings(
    Index n_tags, Index n_intersection_points,
    const tf::buffer<tf::cut::detail::region_crossing_proposal<Index>>
        &proposals,
    const tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 2>>
        &parent_edges,
    const tf::buffer<typename tf::exact::meta<Int>::param_type> &parameters,
    const GetPoint &get_point,
    tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>> &splits)
    -> void {
  if (proposals.size() == 0)
    return;

  // One lattice coordinate is one created vertex. The endpoint fuse below
  // settles an original vertex sitting on the crossing; this settles a
  // created one another carrier already materialized there.
  // Splits this wave produces are queried before they are ordered, so they
  // accumulate apart from the table that answers by binary search and join
  // it once, below.
  tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>> fresh;
  tf::buffer<tf::cut::created_point_record<Index, Int>> created_order;
  tf::cut::build_created_point_index<Index, Int>(extra_points, created_order);
  const Index n_indexed = Index(extra_points.size());

  std::size_t cursor = 0;
  std::size_t proposal_index = 0;
  for (const auto &proposal : proposals) {
    const std::size_t first_parent = cursor;
    cursor += std::size_t(proposal.n_parents);
    auto vertex = proposal.v;
    bool named = bool(proposal.t_junction);
    if (!proposal.t_junction) {
      const auto &first_edge = parent_edges[first_parent];
      const auto first_key = tf::cut::detail::region_edge_key(
          tf::cut::detail::region_vertex_key(
              n_tags, proposal.tag, first_edge[0]),
          tf::cut::detail::region_vertex_key(
              n_tags, proposal.tag, first_edge[1]));
      std::size_t second_parent = first_parent;
      for (std::size_t parent = first_parent + 1; parent < cursor;
           ++parent) {
        const auto &candidate = parent_edges[parent];
        if (tf::cut::detail::region_edge_key(
                tf::cut::detail::region_vertex_key(
                    n_tags, proposal.tag, candidate[0]),
                tf::cut::detail::region_vertex_key(
                    n_tags, proposal.tag, candidate[1])) != first_key) {
          second_parent = parent;
          break;
        }
      }
      const auto &second_edge = parent_edges[second_parent];
      // The resolution reported where it placed this crossing on the first
      // parent, so that is where the split goes. Constructing the position
      // a second way here puts it where the two constraints do not meet,
      // and they still cross once it is applied.
      using param_t = typename tf::exact::meta<Int>::param_type;
      const param_t parameter = proposal_index < parameters.size()
                                    ? parameters[proposal_index]
                                    : param_t(-1);
      tf::point<Int, 3> point{};
      if (parameter >= param_t(0)) {
        const auto key_0 = tf::cut::detail::region_vertex_key(
            n_tags, proposal.tag, first_edge[0]);
        const auto key_1 = tf::cut::detail::region_vertex_key(
            n_tags, proposal.tag, first_edge[1]);
        const bool forward = key_0 < key_1;
        const auto a =
            get_point(proposal.tag, forward ? first_edge[0] : first_edge[1]);
        const auto b =
            get_point(proposal.tag, forward ? first_edge[1] : first_edge[0]);
        for (int coordinate = 0; coordinate < 3; ++coordinate)
          point[coordinate] = tf::exact::dyadic_blend(
              a[coordinate], b[coordinate], parameter,
              tf::exact::meta<Int>::split_grid_bits);
      } else {
        point = tf::exact::edge_edge_closest_point(
            get_point(proposal.tag, first_edge[0]),
            get_point(proposal.tag, first_edge[1]),
            get_point(proposal.tag, second_edge[0]),
            get_point(proposal.tag, second_edge[1]));
      }
      auto sub_ulp =
          [&](const tf::intersect::graph::vertex<Index> &candidate) {
        const auto candidate_point =
            get_point(proposal.tag, candidate);
        typename tf::exact::meta<Int>::T2 distance(0);
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          const typename tf::exact::meta<Int>::T1 delta =
              typename tf::exact::meta<Int>::T1(
                  candidate_point[coordinate]) -
              typename tf::exact::meta<Int>::T1(point[coordinate]);
          distance += typename tf::exact::meta<Int>::T2(delta) *
                      typename tf::exact::meta<Int>::T2(delta);
        }
        return !(typename tf::exact::meta<Int>::T2(8) < distance);
      };
      const std::array<tf::intersect::graph::vertex<Index>, 4> candidates{
          first_edge[0], first_edge[1], second_edge[0], second_edge[1]};
      bool fused = false;
      for (const auto &candidate : candidates)
        if (sub_ulp(candidate) &&
            (!fused ||
             tf::cut::detail::region_vertex_key(
                 n_tags, proposal.tag, candidate) <
                 tf::cut::detail::region_vertex_key(
                     n_tags, proposal.tag, vertex))) {
          vertex = candidate;
          fused = true;
        }
      if (!fused) {
        Index occupant =
            tf::cut::find_created_point<Index, Int>(created_order, point);
        for (Index added = n_indexed;
             occupant == Index(-1) && added < Index(extra_points.size());
             ++added) {
          const auto &candidate = extra_points[std::size_t(added)];
          if (candidate[0] == point[0] && candidate[1] == point[1] &&
              candidate[2] == point[2])
            occupant = added;
        }
        if (occupant != Index(-1)) {
          vertex = {tf::intersect::graph::vertex_source::created,
                    n_intersection_points + occupant,
                    {0, tf::topo_type::edge}};
          named = false;
          fused = true;
        }
      }
      if (!fused) {
        vertex = {
            tf::intersect::graph::vertex_source::created,
            n_intersection_points + Index(extra_points.size()),
            {0, tf::topo_type::edge}};
        extra_points.push_back(point);
      }
      named = fused;
    }
    for (std::size_t parent = first_parent; parent < cursor; ++parent)
      tf::cut::register_region_split<Index, Int>(
          n_tags, proposal.tag, parent_edges[parent], vertex, get_point,
          splits, fresh, named);
    ++proposal_index;
  }
  splits.reallocate(splits.size() + fresh.size());
  std::copy(fresh.begin(), fresh.end(), splits.end() - fresh.size());
}

} // namespace tf::cut
