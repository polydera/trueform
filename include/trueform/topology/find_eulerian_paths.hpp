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
#include "../core/buffer.hpp"
#include "../core/edges.hpp"
#include "../core/views/enumerate.hpp"
#include "./edge_membership_like.hpp"
#include "./vertex_link_like.hpp"
#include <algorithm>

namespace tf {

/// @ingroup topology_paths
/// @brief Find Eulerian paths through a graph of edges.
///
/// Decomposes a graph into a minimal set of edge-disjoint paths that
/// together cover all edges. Uses Hierholzer's algorithm. The result
/// is a collection of paths expressed as edge indices.
///
/// @tparam Policy The edges policy type.
/// @tparam Policy1 The edge membership policy type.
/// @tparam Index The index type.
/// @param edges The edges of the graph.
/// @param link Edge membership (edges incident on each vertex).
/// @param path_offsets Output: offsets into edge_ids for each path.
/// @param edge_ids Output: edge indices forming the paths.
template <typename Policy, typename Policy1, typename Index>
auto find_eulerian_paths(const tf::edges<Policy> &edges,
                         const tf::edge_membership_like<Policy1> &link,
                         tf::buffer<Index> &path_offsets,
                         tf::buffer<Index> &edge_ids) {

  tf::buffer<Index> stack;
  tf::buffer<Index> edge_counts;
  edge_counts.allocate(link.size());
  for (auto &&[c, r] : tf::zip(edge_counts, link)) {
    c = r.size();
  }
  tf::buffer<bool> is_second;
  is_second.allocate(link.size());
  std::fill(is_second.begin(), is_second.end(), false);
  for (const auto &e : edges)
    is_second[e[1]] = true;
  std::size_t edge_count = edges.size();
  auto old_size = edge_ids.size();
  auto run_f = [&](auto vertex_id, auto l_edges) {
    while (edge_counts[vertex_id]) {
      stack.push_back(l_edges[--edge_counts[vertex_id]]);
      path_offsets.push_back(edge_ids.size());
      while (stack.size()) {
        Index current = stack.back();
        const auto &edge = edges[current];
        const auto &next_vertex_id = edge[1];
        auto &free_edge_count = edge_counts[next_vertex_id];
        if (free_edge_count) {
          auto next_edge_id = link[next_vertex_id][--free_edge_count];
          stack.push_back(next_edge_id);
        } else {
          edge_ids.push_back(current);
          stack.pop_back();
          --edge_count;
        }
      }
      std::reverse(edge_ids.begin() + path_offsets.back(), edge_ids.end());
    }
  };

  // first exhaust start points
  for (auto [vertex_id, l_edges] : tf::enumerate(link)) {
    if (!edge_count)
      break;
    if (!is_second[vertex_id])
      run_f(vertex_id, l_edges);
  }
  // then the rest
  for (auto [vertex_id, l_edges] : tf::enumerate(link)) {
    if (!edge_count)
      break;
    run_f(vertex_id, l_edges);
  }
  auto new_size = edge_ids.size();
  if (new_size != old_size)
    path_offsets.push_back(new_size);
}

/// @ingroup topology_paths
/// @brief Find Eulerian paths through a graph defined by vertex links.
///
/// Decomposes a directed graph into a minimal set of edge-disjoint paths.
/// The result is a collection of paths expressed as vertex indices.
///
/// @tparam Policy The vertex link policy type.
/// @tparam Index The index type.
/// @param link The vertex link structure (neighbors of each vertex).
/// @param path_offsets Output: offsets into vertex_ids for each path.
/// @param vertex_ids Output: vertex indices forming the paths.
template <typename Policy, typename Index>
auto find_eulerian_paths(const tf::vertex_link_like<Policy> &link,
                         tf::buffer<Index> &path_offsets,
                         tf::buffer<Index> &vertex_ids) {

  tf::buffer<Index> stack;
  tf::buffer<Index> next_counts;
  next_counts.allocate(link.size());
  Index edge_count = 0;
  for (auto &&[c, r] : tf::zip(next_counts, link)) {
    c = r.size();
    edge_count += c;
  }
  tf::buffer<bool> is_second;
  is_second.allocate(link.size());
  std::fill(is_second.begin(), is_second.end(), false);
  for (const auto &l : link)
    for (auto e : l)
      is_second[e] = true;
  auto old_size = vertex_ids.size();

  auto run_f = [&](auto vertex_id, auto nexts) {
    while (next_counts[vertex_id]) {
      stack.push_back(nexts[--next_counts[vertex_id]]);
      path_offsets.push_back(vertex_ids.size());
      while (stack.size()) {
        Index current = stack.back();
        auto &free_edge_count = next_counts[current];
        if (free_edge_count) {
          auto next_id = link[current][--free_edge_count];
          stack.push_back(next_id);
        } else {
          vertex_ids.push_back(current);
          stack.pop_back();
          --edge_count;
        }
      }
      vertex_ids.push_back(vertex_id);
      std::reverse(vertex_ids.begin() + path_offsets.back(), vertex_ids.end());
    }
  };

  // first exhaust start points
  for (auto [vertex_id, nexts] : tf::enumerate(link)) {
    if (!edge_count)
      break;
    if (!is_second[vertex_id])
      run_f(vertex_id, nexts);
  }
  // then the rest
  for (auto [vertex_id, nexts] : tf::enumerate(link)) {
    if (!edge_count)
      break;
    run_f(vertex_id, nexts);
  }
  auto new_size = vertex_ids.size();
  if (new_size != old_size)
    path_offsets.push_back(new_size);
}
} // namespace tf
