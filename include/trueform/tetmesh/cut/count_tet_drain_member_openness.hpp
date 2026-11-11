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
#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../exact/meta.hpp"
#include "../entrance/count_tet_open_edges.hpp"
#include "./tet_changed_triangle.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// The changed members a drain leaves open: a member's pieces run its own
/// boundary — its sides split at the points the drain states on them
/// (`points_on_edge`), read through `speaks` — once, and nothing else. A
/// changed member every piece of which dropped is a line and runs nothing
/// (the line law, @ref count_tet_drain_t_junctions).
/// `corners_of(t)` reads a member's corners and `point` their positions,
/// which only order a side. O(a member's pieces + its side points, with one
/// sort each).
template <typename Int, typename Index, typename CornersOf,
          typename PointsOnEdge, typename Point, typename Speaks>
auto count_tet_drain_member_openness(
    const tf::buffer<tet_changed_triangle<Index>> &changed,
    const tf::buffer<std::array<Index, 3>> &pieces, const CornersOf &corners_of,
    const PointsOnEdge &points_on_edge, const Point &point,
    const Speaks &speaks) -> std::size_t {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto distance2 = [&point](Index a, Index b) {
    const auto pa = point(a), pb = point(b);
    T2 sum(0);
    for (std::size_t d = 0; d < 3; ++d) {
      const T2 delta = T2(T1(pb[d]) - T1(pa[d]));
      sum = sum + delta * delta;
    }
    return sum;
  };
  struct local_t {
    tf::buffer<Index> side;
    tf::buffer<tet_edge_step<Index>> runs;
    std::size_t open = 0;
  };
  std::size_t open = 0;
  tf::blocked_reduce(
      changed, open, local_t{},
      [&](const auto &block, local_t &local) {
        for (const auto &member : block) {
          if (member.first == member.last)
            continue;
          local.runs.clear();
          const auto run = [&local](Index u, Index v, int step) {
            if (u != v)
              local.runs.push_back(
                  {{std::min(u, v), std::max(u, v)}, u < v ? step : -step});
          };
          const auto corners = corners_of(member.triangle);
          for (std::size_t k = 0; k < 3; ++k) {
            const Index from = corners[k], to = corners[(k + 1) % 3];
            local.side.clear();
            points_on_edge(member.triangle, k, [&](Index id) {
              if (id != corners[0] && id != corners[1] && id != corners[2])
                local.side.push_back(id);
            });
            std::sort(local.side.begin(), local.side.end());
            local.side.erase_till_end(
                std::unique(local.side.begin(), local.side.end()));
            std::sort(local.side.begin(), local.side.end(),
                      [&](Index x, Index y) {
                        const auto dx = distance2(from, x);
                        const auto dy = distance2(from, y);
                        return dx < dy || (!(dy < dx) && x < y);
                      });
            Index previous = speaks(from);
            for (auto id : local.side) {
              run(previous, speaks(id), -1);
              previous = speaks(id);
            }
            run(previous, speaks(to), -1);
          }
          for (Index i = member.first; i < member.last; ++i) {
            const auto &piece = pieces[std::size_t(i)];
            for (std::size_t k = 0; k < 3; ++k)
              run(piece[k], piece[(k + 1) % 3], 1);
          }
          std::sort(
              local.runs.begin(), local.runs.end(),
              [](const tet_edge_step<Index> &x, const tet_edge_step<Index> &y) {
                return x.key < y.key;
              });
          bool unbalanced = false;
          for (std::size_t r = 0; r < local.runs.size() && !unbalanced;) {
            std::size_t end = r;
            int balance = 0;
            for (; end < local.runs.size() &&
                   local.runs[end].key == local.runs[r].key;
                 ++end)
              balance += local.runs[end].step;
            unbalanced = balance != 0;
            r = end;
          }
          local.open += unbalanced;
        }
      },
      [](const local_t &local, std::size_t &total) { total += local.open; },
      tf::checked);
  return open;
}

} // namespace tf::tetmesh
