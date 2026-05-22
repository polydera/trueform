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
#include "../exact/meta.hpp"
#include "../exact_coordinate_converter.hpp"
#include "./edge_membership.hpp"
#include "./edge_orientation.hpp"

#include <algorithm>
#include <type_traits>

namespace tf {

/// @ingroup topology_planar
/// @brief Extracts closed regions from a planar graph.
///
/// Given directed edges and vertex positions, walks the graph to find
/// all minimal closed loops (regions). Uses exact polar sort for
/// edge ordering at each vertex. Float points are converted to int32 via
/// pt_converter before ordering.
///
/// Walk uses the cp-algorithms approach: at each vertex, find the twin
/// of the incoming edge in the sorted adjacency, take the cyclic successor.
/// Inner faces are CW, outer face is CCW.
template <typename Index, typename Int = tf::exact::int32>
class planar_graph_regions : public tf::offset_block_buffer<Index, Index> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
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
      auto conv = tf::make_exact_coordinate_converter<Int>(points);
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
  /// Uses exact cross product via meta<Int>::T2.
  template <typename Policy0, typename Policy1>
  auto sort_adjacency(const tf::edges<Policy0> &edges,
                      const tf::points<Policy1> &points) {

    for (std::size_t v = 0; v < _em.size(); ++v) {
      auto &&adj = _em[v];
      if (adj.size() < 2)
        continue;

      auto pv = points[v];

      std::sort(adj.begin(), adj.end(), [&](Index a, Index b) -> bool {
        auto pa = points[edges[a][1]];
        auto pb = points[edges[b][1]];

        T1 ax = T1(pa[0]) - pv[0], ay = T1(pa[1]) - pv[1];
        T1 bx = T1(pb[0]) - pv[0], by = T1(pb[1]) - pv[1];

        auto half = [](T1 x, T1 y) -> bool {
          return (y > 0) || (y == 0 && x > 0);
        };

        bool ha = half(ax, ay);
        bool hb = half(bx, by);
        if (ha != hb)
          return ha > hb;

        T2 cross = T2(ax) * by - T2(ay) * bx;
        if (cross != 0)
          return cross > 0;

        T2 da = T2(ax) * ax + T2(ay) * ay;
        T2 db = T2(bx) * bx + T2(by) * by;
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
