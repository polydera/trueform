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
#include "../../exact/rebase_parameter.hpp"
#include "./constraint_parent_span.hpp"
#include "../../exact/meta.hpp"

#include "../../core/buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/range.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/constant.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int>
auto discover_region_crossings(
    tf::cut::detail::region_triangulation_workspace<Index, Int> &workspace,
    Index n_tags, Index tag,
    tf::buffer<tf::cut::detail::region_crossing_proposal<Index>>
        &proposals,
    tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 2>>
        &parent_edges,
    tf::buffer<typename tf::exact::meta<Int>::param_type> &parameters)
    -> void {
  auto edges = tf::make_edges(tf::make_blocked_range<2>(
      tf::make_range(workspace.cons.data(), workspace.cons.size())));
  workspace.tri.clear();
  if (!workspace.tri.build(tf::make_points(workspace.pts), edges,
                           tf::make_constant_range(true, edges.size()), true))
    return;

  workspace.tri.for_each_constraint_crossing(
      [&](Index point, const auto &parents) {
        const Index first = parents[0];
        Index second = Index(-1);
        const auto &first_edge =
            workspace.cons_verts[std::size_t(first)];
        const auto first_key = tf::cut::detail::region_edge_key(
            tf::cut::detail::region_vertex_key(
                n_tags, tag, first_edge[0]),
            tf::cut::detail::region_vertex_key(
                n_tags, tag, first_edge[1]));
        for (std::size_t parent = 1; parent < parents.size(); ++parent) {
          const auto &candidate =
              workspace.cons_verts[std::size_t(parents[parent])];
          if (tf::cut::detail::region_edge_key(
                  tf::cut::detail::region_vertex_key(
                      n_tags, tag, candidate[0]),
                  tf::cut::detail::region_vertex_key(
                      n_tags, tag, candidate[1])) != first_key) {
            second = parents[parent];
            break;
          }
        }
        if (second == Index(-1) && parents.size() > 1)
          return;

        tf::cut::detail::region_crossing_proposal<Index> proposal{
            tag,
            Index(parents.size()),
            char(0),
            tf::intersect::graph::vertex<Index>{}};
        if (second == Index(-1)) {
          const auto &kept = workspace.tri.index_map().kept_ids();
          const Index n_input =
              Index(workspace.tri.index_map().f().size());
          if (kept[std::size_t(point)] == n_input)
            return;
          proposal.t_junction = char(1);
          proposal.v =
              workspace.ihm.kept_ids()[std::size_t(
                  kept[std::size_t(point)])];
        }
        // The parameter the resolution actually placed this crossing at.
        // Recomputing the position downstream by another construction puts
        // the split somewhere the constraints do not meet, and they keep
        // crossing after it is applied.
        using param_t = typename tf::exact::meta<Int>::param_type;
        param_t parameter(-1);
        for (const auto &record : workspace.tri.parameterized_crossings()) {
          if (record.point != point)
            continue;
          const param_t raw =
              record.id_a == first
                  ? record.t_a
                  : (record.id_b == first ? record.t_b : param_t(-1));
          if (raw < param_t(0))
            continue;
          // The triangulator reports on its own parameter width; carriers
          // share splits on the grid. Rounding to nearest here rather than
          // truncating keeps the placement centred when the grid is the
          // coarser of the two.
          constexpr int shift = tf::exact::meta<Int>::param_bits -
                                tf::exact::meta<Int>::split_grid_bits;
          if constexpr (shift == 0)
            parameter = raw;
          else
            parameter =
                param_t((raw + (param_t(1) << (shift - 1))) >> shift);
          break;
        }
        // A split belongs to the edge every carrier of it shares, never to
        // the piece this build happened to be handed: a record keyed by a
        // piece's endpoints is one no walk ever forms, so nothing can find
        // it — not even the region that reported it.
        const auto self_span = tf::cut::constraint_parent_span<Index, Int>(
            workspace, n_tags, tag, std::size_t(first));
        parameter = tf::exact::rebase_parameter<Int>(
            parameter, self_span.t0, self_span.t1,
            typename tf::exact::meta<Int>::T2(1)
                << tf::exact::meta<Int>::split_grid_bits);
        for (std::size_t parent = 0; parent < parents.size(); ++parent)
          parent_edges.push_back(
              tf::cut::constraint_parent_span<Index, Int>(
                  workspace, n_tags, tag, std::size_t(parents[parent]))
                  .edge);
        parameters.push_back(parameter);
        proposals.push_back(proposal);
      });
}

} // namespace tf::cut
