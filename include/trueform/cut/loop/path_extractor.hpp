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
#include "../../core/buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../topology/vertex_link.hpp"
#include <algorithm>

namespace tf::loop {
template <typename Index> class path_extractor {
public:
  template <typename Range0, typename Range1>
  auto build(const Range0 &base_loop, const Range1 &edges,
             std::size_t n_vertices) {
    clear();
    mark_on_loop(base_loop, n_vertices);
    build_connectivity(edges, n_vertices);
    fill_paths();
  }

  auto paths() const {
    return tf::make_offset_block_range(_path_offsets, _paths);
  }

  auto paths() { return tf::make_offset_block_range(_path_offsets, _paths); }

  auto is_on_loop(Index v_id) const { return _id_on_base_loop[v_id] != -1; }

  auto id_on_loop(Index v_id) const { return _id_on_base_loop[v_id]; }

  auto is_cut_endpoint(Index v_id) const {
    return _v_link[v_id].size() == 1 && !is_on_loop(v_id);
  }

  auto is_endpoint(Index v_id) const {
    return _v_link[v_id].size() != 2 || is_on_loop(v_id);
  }

  auto clear() {
    _endpoints.clear();
    _v_link.clear();
    _id_on_base_loop.clear();
    _paths.clear();
    _path_offsets.clear();
  }

private:
  auto fill_paths() {
    for (auto id : _endpoints)
      exhaust_endpoint(id);
    // only loops that don't touch the base_loop left
    for (Index i = 0; i < Index(_v_link.size()); ++i)
      trace_path(i);
    if (_paths.size())
      _path_offsets.push_back(_paths.size());
  }

  auto trace_path(Index v_id) {
    Index vertex_count = 0;
    auto next = move_to_next(v_id);
    if (next == -1) {
      return vertex_count;
    }
    _path_offsets.push_back(_paths.size());
    _paths.push_back(v_id);
    vertex_count++;
    while (next != -1) {
      _paths.push_back(next);
      vertex_count++;
      if (next == v_id || is_endpoint(next))
        break;
      next = move_to_next(next);
    }
    return vertex_count;
  }

  auto exhaust_endpoint(Index v_id) {
    Index last_path = 0;
    do {
      last_path = trace_path(v_id);
    } while (last_path);
  }

  auto move_to_next(Index v_id) {
    auto next_iter = std::find_if(_v_link[v_id].begin(), _v_link[v_id].end(),
                                  [](const auto &x) { return x != -1; });
    if (next_iter == _v_link[v_id].end())
      return Index(-1);
    auto next_id = *next_iter;
    *next_iter = -1;
    std::replace(_v_link[next_id].begin(), _v_link[next_id].end(), v_id,
                 Index(-1));
    return next_id;
  }

  template <typename Range>
  auto mark_on_loop(const Range &base_loop, std::size_t n_vertices) {
    _id_on_base_loop.allocate(n_vertices);
    std::fill(_id_on_base_loop.begin(), _id_on_base_loop.end(), -1);
    for (auto [i, v] : tf::enumerate(base_loop))
      _id_on_base_loop[v] = i;
  }

  template <typename Range>
  auto build_connectivity(const Range &edges, std::size_t n_vertices) {
    _v_link.build(tf::make_edges(edges), n_vertices,
                  tf::edge_orientation::bidirectional);
    for (const auto &[i, l] : tf::enumerate(_v_link))
      if (l.size() != 0 && (is_on_loop(i) || l.size() != 2))
        _endpoints.push_back(i);
  }

  tf::buffer<Index> _endpoints;
  tf::vertex_link<Index> _v_link;
  tf::buffer<Index> _id_on_base_loop;
  tf::buffer<Index> _paths;
  tf::buffer<Index> _path_offsets;
};
} // namespace tf::loop
