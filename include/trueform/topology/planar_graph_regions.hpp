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
* Author: Žiga Sajovic
*/
#pragma once
#include "../core/constants.hpp"
#include "../core/points.hpp"
#include "../core/small_vector.hpp"
#include "./directed_edge_link.hpp"
#include "./edge_membership.hpp"
#include "./edge_orientation.hpp"

namespace tf {

/// @ingroup topology_planar
/// @brief Extracts closed regions from a planar graph.
///
/// Given directed edges and vertex positions, walks the graph to find
/// all minimal closed loops (regions). Uses angle-based ordering to
/// determine the next edge at each vertex.
///
/// @tparam Index The integer type for indices.
/// @tparam RealT The real type for angle computation (default: double).
template <typename Index, typename RealT = double>
class planar_graph_regions : public tf::offset_block_buffer<Index, Index> {
  using base_t = tf::offset_block_buffer<Index, Index>;

public:
  /// @brief Build regions from directed edges and points.
  /// @tparam Policy0 The edges policy type.
  /// @tparam Policy1 The points policy type.
  /// @param directed_edges The directed edges.
  /// @param points The vertex positions.
  template <typename Policy0, typename Policy1>
  auto build(const tf::edges<Policy0> &directed_edges,
             const tf::points<Policy1> &points) {
    clear();
    build_connectivities(directed_edges, points.size());
    compute_angles(directed_edges, points);
    order_link(directed_edges);
    walk_regions(directed_edges);
  }

  /// @brief Clear all internal state.
  auto clear() {
    base_t::clear();
    _em.clear();
    _dil.clear();
    _angles.clear();
    _work_buffer.clear();
    _visited.clear();
  }

private:
  template <typename Policy>
  auto build_connectivities(const tf::edges<Policy> &edges,
                            std::size_t n_vertices) {
    _em.build(edges, n_vertices, tf::edge_orientation::forward);
    _dil.build(edges, _em);
  }

  template <typename Policy0, typename Policy1>
  auto compute_angles(const tf::edges<Policy0> &edges,
                      const tf::points<Policy1> &points) {
    _angles.allocate(edges.size());
    for (auto &&[angle, edge] : tf::zip(_angles, edges)) {
      auto dir = points[edge[1]] - points[edge[0]];
      angle = std::atan2(RealT(dir[1]), RealT(dir[0]));
    }
  }

  template <typename Policy> auto order_link(const tf::edges<Policy> &edges) {
    for (auto &&[edge0, angle, link] : tf::zip(edges, _angles, _dil)) {
      if (link.size() < 2)
        continue;
      constexpr auto pi = tf::pi<std::decay_t<decltype(angle)>>;
      auto angle0 = angle - pi;
      angle0 += (angle0 <= 0) * 2 * pi;
      _work_buffer.clear();
      _work_buffer.resize(link.size());
      for (auto &&[next, val] : tf::zip(link, _work_buffer)) {
        const auto &edge1 = edges[next];
        if (edge0[0] == edge1[1]) {
          val = 2 * pi;
        } else {
          auto angle1 = _angles[next];
          angle1 += (angle1 <= 0) * 2 * pi;
          auto angle = angle0 - angle1;
          angle += (angle <= 0) * 2 * pi;
          val = angle;
        }
      }
      auto r_to_sort = tf::zip(_work_buffer, link);
      std::sort(r_to_sort.begin(), r_to_sort.end());
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
      ++count;
      _visited[current] = true;
      base_t::data_buffer().push_back(edges[current][0]);
      if (edges[current][1] == edges[start][0]) {
        break;
      }
      auto next_it = std::find_if(
          _dil[current].begin(), _dil[current].end(),
          [this](const auto &next_id) { return !_visited[next_id]; });
      if (next_it == _dil[current].end()) {
        break;
      }
      current = *next_it;
    }
    return count;
  }

  template <typename Policy> auto walk_regions(const tf::edges<Policy> &edges) {
    _visited.allocate(edges.size());
    std::fill(_visited.begin(), _visited.end(), false);
    Index n_edges_left = edges.size();
    for (Index current = 0; n_edges_left && current < Index(_dil.size());
         ++current) {
      n_edges_left -= make_walk(edges, current);
    }
    if (base_t::data_buffer().size())
      base_t::offsets_buffer().push_back(base_t::data_buffer().size());
  }

  tf::edge_membership<Index> _em;
  tf::directed_edge_link<Index> _dil;
  tf::buffer<RealT> _angles;
  tf::small_vector<RealT, 4> _work_buffer;
  tf::buffer<bool> _visited;
};
} // namespace tf
