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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/elect_identity_components.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "./plane_arrangement_arena.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// A Steiner point the arena holds and the standing direct identity of its
/// plane at its position, if one stands there, are one coincidence of THE
/// ELECTION (@ref tf::intersect::graph::elect_identity_component); a site no
/// standing identity holds keeps its place in the range, where
/// materialization mints it. `standing_triangles(plane)` states the plane's
/// standing triangles in flat identities; an identity `merges` names on either
/// side stands for nothing here. The elected corners become flat, and the
/// plane's Steiner range keeps only the positions that still mint.
template <typename Index, typename Int, typename StandingTriangles>
auto elect_standing_plane_steiners(
    plane_arrangement_arena<Index, Int> &arena,
    const StandingTriangles &standing_triangles,
    const tf::buffer<Index> &created_class,
    const tf::buffer<tf::exact::pt3<Int>> &direct_created_points,
    const tf::buffer<std::array<Index, 3>> &merges, Index n_flat,
    Index base_created) -> void {
  if (arena.steiners.size() == 0)
    return;
  tf::buffer<Index> targets;
  targets.allocate(merges.size());
  for (std::size_t at = 0; at < merges.size(); ++at)
    targets[at] = merges[at][2];
  std::sort(targets.begin(), targets.end());
  struct standing_t {
    tf::exact::pt3<Int> position;
    Index flat;
  };
  struct local_t {
    tf::buffer<standing_t> standing;
    tf::buffer<Index> codes;
    tf::buffer<tf::intersect::graph::identity_key<Index>> coincidence;
  };
  const auto created_flat = n_flat + base_created;
  tf::parallel_for_each(
      tf::make_sequence_range(Index(arena.stn_range.size())),
      [&](Index plane, local_t &local) {
        auto &stn = arena.stn_range[std::size_t(plane)];
        if (stn[0] == stn[1])
          return;
        local.standing.clear();
        for (const auto &triangle : standing_triangles(plane))
          for (const auto corner : triangle) {
            if (corner < created_flat)
              continue;
            const auto id = corner - n_flat;
            const auto cls = created_class[std::size_t(id - base_created)];
            if (cls >= Index(0) ||
                tf::intersect::graph::plane_merge_target(merges, Index(1),
                                                         id) != Index(-1) ||
                std::binary_search(targets.begin(), targets.end(), id))
              continue;
            local.standing.push_back(
                {direct_created_points[std::size_t(-cls - Index(1))], corner});
          }
        if (local.standing.size() == 0)
          return;
        const auto less = [](const standing_t &x, const standing_t &y) {
          return x.position != y.position ? x.position < y.position
                                          : x.flat < y.flat;
        };
        std::sort(local.standing.begin(), local.standing.end(), less);
        local.codes.allocate(std::size_t(stn[1] - stn[0]));
        auto kept = stn[0];
        for (auto s = stn[0]; s < stn[1]; ++s) {
          const auto position = arena.steiners[std::size_t(s)];
          const auto at = std::lower_bound(
              local.standing.begin(), local.standing.end(),
              standing_t{position, Index(-1)}, less);
          local.coincidence.clear();
          if (at != local.standing.end() && at->position == position)
            // a standing key here speaks the flat corner `codes` holds
            local.coincidence.push_back(
                tf::intersect::graph::standing_identity_key<Index>(at->flat));
          local.coincidence.push_back(
              tf::intersect::graph::fresh_identity_key<Index>(s));
          local.codes[std::size_t(s - stn[0])] =
              tf::intersect::graph::elect_identity_component<Index>(
                  tf::make_range(local.coincidence),
                  [&](const tf::intersect::graph::identity_key<Index> &) {
                    arena.steiners[std::size_t(kept)] = position;
                    return Index(-2) - (kept++ - stn[0]);
                  });
        }
        if (kept == stn[1])
          return;
        stn[1] = kept;
        const auto span = arena.range[std::size_t(plane)];
        for (auto t = span[0]; t < span[1]; ++t)
          for (auto &corner : arena.tris[std::size_t(t)])
            if (corner < Index(0))
              corner = local.codes[std::size_t(Index(-2) - corner)];
      },
      local_t{});
}

} // namespace tf::arrangement
