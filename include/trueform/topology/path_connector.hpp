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
#include "../core/contiguous_index_hash_map.hpp"
#include "../core/edges.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/views/enumerate.hpp"
#include "./vertex_link.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>

namespace tf {

/// @ingroup topology_paths
/// @brief Connects edges into continuous vertex paths.
///
/// This class handles the mechanics of connecting edges into paths.
/// It deduplicates edges, builds connectivity, and traces paths from
/// endpoints. Handles both open paths (starting from degree-1 vertices)
/// and closed loops.
///
/// @tparam T The vertex identifier type.
/// @tparam Index The index type.
/// @tparam Hash Hash function for vertex identifiers.
template <typename T, typename Index, typename Hash = std::hash<T>>
class path_connector {
public:
  /// @brief Build paths from edges.
  /// @tparam Policy The edges policy type.
  /// @param edges The edges to connect into paths.
  template <typename Policy> auto build(const tf::edges<Policy> &edges) {
    build_edges(edges);
    build_connectivity();
    fill_paths();
  }

  /// @brief Get the resulting paths (const).
  /// @return Reference to the paths buffer.
  auto paths_buffer() const -> const tf::offset_block_buffer<Index, Index> & {
    return _paths_buffer;
  }

  /// @brief Get the resulting paths.
  /// @return Reference to the paths buffer.
  auto paths_buffer() -> tf::offset_block_buffer<Index, Index> & {
    return _paths_buffer;
  }

  /// @brief Clear all internal state for reuse.
  auto clear() {
    _ihm.f().clear();
    _ihm.kept_ids().clear();
    _work_edges.clear();
    _endpoints.clear();
    _v_link.clear();
    _paths_buffer.clear();
  }

private:
  template <typename Policy> auto build_edges(const tf::edges<Policy> &edges) {
    tf::make_contiguous_index_hash_map(edges, _ihm);
    _work_edges.allocate(edges.size());
    tf::parallel_for_each(tf::zip(edges, _work_edges), [&](auto pair) {
      auto &&[_in, _out] = pair;
      _out[0] = _ihm.f()[_in[0]];
      _out[1] = _ihm.f()[_in[1]];
      if (_out[1] < _out[0])
        std::swap(_out[0], _out[1]);
    });
    tbb::parallel_sort(_work_edges);
    _work_edges.erase_till_end(
        std::unique(_work_edges.begin(), _work_edges.end()));
  }

  auto fill_paths() {
    for (auto id : _endpoints)
      exhaust_endpoint(id);
    // only loops left
    for (Index i = 0; i < Index(_v_link.size()); ++i)
      trace_path(i);
    if (_paths_buffer.data_buffer().size())
      _paths_buffer.offsets_buffer().push_back(
          _paths_buffer.data_buffer().size());
  }

  auto trace_path(Index v_id) {
    Index vertex_count = 0;
    auto next = move_to_next(v_id);
    if (next == -1) {
      return vertex_count;
    }

    auto &_path_offsets = _paths_buffer.offsets_buffer();
    auto &_paths = _paths_buffer.data_buffer();
    _path_offsets.push_back(_paths.size());
    _paths.push_back(_ihm.kept_ids()[v_id]);
    vertex_count++;
    while (next != -1) {
      _paths.push_back(_ihm.kept_ids()[next]);
      vertex_count++;
      if (next == v_id || _v_link[next].size() != 2)
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

  auto move_to_next(Index v_id) -> Index {
    auto next_iter = std::find_if(_v_link[v_id].begin(), _v_link[v_id].end(),
                                  [](const auto &x) { return x != -1; });
    if (next_iter == _v_link[v_id].end())
      return -1;
    auto next_id = *next_iter;
    *next_iter = -1;
    std::replace(_v_link[next_id].begin(), _v_link[next_id].end(), v_id,
                 Index(-1));
    return next_id;
  }

  auto build_connectivity() {
    _v_link.build(tf::make_edges(_work_edges), _ihm.kept_ids().size(),
                  tf::edge_orientation::bidirectional);
    for (const auto &[i, l] : tf::enumerate(_v_link))
      if (l.size() != 2)
        _endpoints.push_back(i);
  }

  tf::index_hash_map<T, Index, Hash> _ihm;
  tf::buffer<std::array<Index, 2>> _work_edges;
  tf::buffer<Index> _endpoints;
  tf::vertex_link<Index> _v_link;
  tf::offset_block_buffer<Index, Index> _paths_buffer;
};
} // namespace tf
