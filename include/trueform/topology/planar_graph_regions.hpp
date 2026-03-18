/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Ziga Sajovic
 */
#pragma once
#include "../core/coordinate_type.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/points.hpp"
#include "../core/views/mapped_range.hpp"
#include "../exact/int128.hpp"
#include "../exact/pt_converter.hpp"
#include "./edge_membership.hpp"
#include "./edge_orientation.hpp"

#include <algorithm>
#include <type_traits>

namespace tf {

/// @ingroup topology_planar
/// @brief Extracts closed regions from a planar graph.
///
/// Given directed edges and vertex positions, walks the graph to find
/// all minimal closed loops (regions). Uses exact int128 polar sort for
/// edge ordering at each vertex. Float points are converted to int32 via
/// pt_converter before ordering.
///
/// Walk uses the cp-algorithms approach: at each vertex, find the twin
/// of the incoming edge in the sorted adjacency, take the cyclic successor.
/// Inner faces are CW, outer face is CCW.
template <typename Index> class planar_graph_regions
    : public tf::offset_block_buffer<Index, Index> {
  using base_t = tf::offset_block_buffer<Index, Index>;

public:
  template <typename Policy0, typename Policy1>
  auto build(const tf::edges<Policy0> &directed_edges,
             const tf::points<Policy1> &points) {
    clear();
    _em.build(directed_edges, points.size(), tf::edge_orientation::forward);

    using coord_t = tf::coordinate_type<Policy1>;
    if constexpr (std::is_integral_v<coord_t>) {
      sort_adjacency(directed_edges, points);
    } else {
      auto conv = tf::exact::make_pt_converter(points);
      auto int_pts = tf::make_points(tf::make_mapped_range(
          points, [&](const auto &pt) { return conv(pt); }));
      sort_adjacency(directed_edges, int_pts);
    }

    walk_regions(directed_edges);
  }

  auto clear() {
    base_t::clear();
    _em.clear();
    _visited.clear();
  }

private:
  /// Sort outgoing edges at each vertex by polar angle (CCW).
  /// Uses int128 cross product — exact for int32 coordinates.
  template <typename Policy0, typename Policy1>
  auto sort_adjacency(const tf::edges<Policy0> &edges,
                      const tf::points<Policy1> &points) {
    using i128 = tf::exact::int128;

    for (std::size_t v = 0; v < _em.size(); ++v) {
      auto &&adj = _em[v];
      if (adj.size() < 2)
        continue;

      auto pv = points[v];

      std::sort(adj.begin(), adj.end(), [&](Index a, Index b) -> bool {
        auto pa = points[edges[a][1]];
        auto pb = points[edges[b][1]];

        i128 ax = i128(pa[0] - pv[0]), ay = i128(pa[1] - pv[1]);
        i128 bx = i128(pb[0] - pv[0]), by = i128(pb[1] - pv[1]);

        auto half = [](i128 x, i128 y) -> bool {
          return (y > 0) || (y == 0 && x > 0);
        };

        bool ha = half(ax, ay);
        bool hb = half(bx, by);
        if (ha != hb)
          return ha > hb;

        i128 cross = ax * by - ay * bx;
        if (cross != 0)
          return cross > 0;

        i128 da = ax * ax + ay * ay;
        i128 db = bx * bx + by * by;
        if (da != db)
          return da < db;

        return a < b;
      });
    }
  }

  template <typename Policy>
  auto make_walk(const tf::edges<Policy> &edges, Index start) {
    if (_visited[start])
      return Index(0);

    Index current = start;
    Index count = 0;
    base_t::offsets_buffer().push_back(base_t::data_buffer().size());

    for (;;) {
      _visited[current] = true;
      ++count;
      base_t::data_buffer().push_back(edges[current][0]);

      // Find twin: current = u->v, twin = v->u
      auto u = edges[current][0];
      auto v = edges[current][1];
      auto &&adj_v = _em[v];

      // Find twin v->u in adj_v
      Index twin_pos = -1;
      for (Index i = 0; i < Index(adj_v.size()); ++i)
        if (edges[adj_v[i]][1] == u) {
          twin_pos = i;
          break;
        }

      if (twin_pos < 0)
        break;

      // Cyclic predecessor of twin → CCW inner faces (positive area).
      // (Cyclic successor would give CW inner faces.)
      auto next_pos =
          (twin_pos + Index(adj_v.size()) - 1) % Index(adj_v.size());
      current = adj_v[next_pos];

      if (current == start)
        break;

      if (_visited[current])
        break;
    }

    return count;
  }

  template <typename Policy> auto walk_regions(const tf::edges<Policy> &edges) {
    _visited.allocate(edges.size());
    std::fill(_visited.begin(), _visited.end(), false);
    for (Index i = 0; i < Index(edges.size()); ++i) {
      if (_visited[i])
        continue;
      make_walk(edges, i);
    }
    if (base_t::data_buffer().size())
      base_t::offsets_buffer().push_back(base_t::data_buffer().size());
  }

  tf::edge_membership<Index> _em;
  tf::buffer<bool> _visited;
};
} // namespace tf
