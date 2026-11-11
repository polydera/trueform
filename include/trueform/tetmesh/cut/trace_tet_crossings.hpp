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
#include "../../topology/topo_type.hpp"
#include "./classify_tet_pair.hpp"
#include "./tet_pair_workspace.hpp"
#include "./tet_slot_holders.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// Trace the intersection curves through the crossing pairs `seeds` of the
/// surface's `triangles` over the lattice `points`, `faces` the same
/// triangles in site slots and `live(t)` the triangles the surface holds:
/// each pair is one call of the intersect module's pair kernel
/// (@ref classify_tet_pair) on the positions the entrance's lattice placed,
/// and a pair that meets puts both
/// triangles into `work.cut`, ascending. A contact on an edge of one of the
/// two continues the curve on every other live triangle holding that edge,
/// and one on a corner on every other live triangle holding it, paired with
/// the other of the two, through `work.holders`, the triangles at each slot
/// (@ref tet_slot_holders); two triangles of one facet, `facet_of(t)`, are a
/// stack and never a pair; a pair already traced is not stated again, so the
/// trace ends where its curves close or leave the surface. Nothing beyond the
/// pairs on the curves is read.
template <typename Index, typename Int, typename Points, typename Triangles,
          typename Faces, typename Live, typename FacetOf>
auto trace_tet_crossings(const Points &points, const Triangles &triangles,
                         const Faces &faces, const Live &live,
                         const FacetOf &facet_of,
                         const tf::buffer<std::array<Index, 2>> &seeds,
                         tet_pair_workspace<Index, Int> &work) -> void {
  auto &queue = work.queue;
  auto &traced = work.traced;
  auto &cut = work.cut;
  queue.clear();
  traced.clear();
  cut.clear();
  const auto push = [&queue, &facet_of](Index a, Index b) {
    if (a != b && facet_of(a) != facet_of(b))
      queue.push_back({std::min(a, b), std::max(a, b)});
  };
  for (const auto &seed : seeds)
    push(seed[0], seed[1]);
  const auto holds = [&faces](Index t, Index slot) {
    const auto &f = faces[std::size_t(t)];
    return f[0] == slot || f[1] == slot || f[2] == slot;
  };
  std::array<Index, 2> pair{};
  while (queue.size()) {
    pair = queue.back();
    queue.pop_back();
    if (!traced.insert(pair).second)
      continue;
    bool meets = false;
    const auto follow = [&](Index side, const tf::topo_id<Index> &target) {
      const Index t = pair[std::size_t(side)];
      const Index other = pair[std::size_t(1 - side)];
      if (target.label == tf::topo_type::face)
        return;
      const auto &slots = faces[std::size_t(t)];
      const Index a = slots[std::size_t(target.id)];
      const Index b = target.label == tf::topo_type::edge
                          ? slots[(std::size_t(target.id) + 1) % 3]
                          : a;
      for_each_tet_slot_holder(work.holders, a, [&](Index u) {
        if (u != t && live(std::size_t(u)) && holds(u, b))
          push(u, other);
      });
    };
    const auto follow_all = [&](const auto &records) {
      for (const auto &record : records) {
        if (record.object == record.object_other)
          continue;
        meets = true;
        follow(record.object, record.target);
        follow(record.object_other, record.target_other);
      }
    };
    classify_tet_pair(points, triangles, pair, work.pair_kernel);
    follow_all(work.pair_kernel.pair.intersections);
    if (meets) {
      cut.push_back(pair[0]);
      cut.push_back(pair[1]);
    }
  }
  std::sort(cut.begin(), cut.end());
  cut.erase_till_end(std::unique(cut.begin(), cut.end()));
}

} // namespace tf::tetmesh
