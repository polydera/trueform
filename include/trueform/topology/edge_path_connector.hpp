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
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/contiguous_index_hash_map.hpp"
#include "../core/edges.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/views/zip.hpp"
#include "./edge_membership.hpp"
#include "./edge_orientation.hpp"
#include <array>

namespace tf {

/// @ingroup topology_paths
/// @brief Connects edges into continuous paths of EDGE indices.
///
/// Mirrors @ref tf::path_connector but each path is a sequence of EDGE
/// indices (not vertex indices). Two edges link when they share a vertex
/// whose degree in the edge-incidence graph is exactly 2. Paths split at
/// vertices of degree != 2 (boundaries with degree 1; branches with
/// degree >= 3).
///
/// @pre If the same logical edge appears in both directions (a, b) and
/// (b, a) in the input, both copies are treated as distinct edges with
/// distinct indices. The walker prefers any non-inverse neighbor at the
/// current vertex; it only steps to an inverse edge when no other
/// neighbor is available. Callers wanting (a, b) and (b, a) NOT
/// connected should pre-deduplicate edges before passing to `build`.
///
/// @tparam T The vertex identifier type.
/// @tparam Index The integer type for edge and path indices.
/// @tparam Hash Hash function for vertex identifiers.
template <typename T, typename Index, typename Hash = std::hash<T>>
class edge_path_connector {
public:
  /// @brief Build paths of edge indices from edges.
  template <typename Policy>
  auto build(const tf::edges<Policy> &edges) -> void {
    build_edges(edges);
    build_connectivity();
    fill_paths();
  }

  /// @brief Get the resulting paths as a lightweight range view (no copy
  /// when assigned via `auto`).
  auto paths() const { return tf::make_range(_paths_buffer); }

  /// @brief Per-position direction along each path. Same offsets as
  /// `paths()`. `true` means the edge's natural orientation
  /// (edge[0] -> edge[1]) matches the path's traversal direction at that
  /// position; `false` means it is reversed. Returned as a lightweight
  /// range view (no copy).
  auto directions() const {
    return tf::make_offset_block_range(_paths_buffer.offsets_buffer(),
                                       _directions);
  }

  /// @brief Clear all internal state for reuse.
  auto clear() -> void {
    _ihm.f().clear();
    _ihm.kept_ids().clear();
    _work_edges.clear();
    _consumed.clear();
    _endpoints.clear();
    _em.clear();
    _paths_buffer.clear();
    _directions.clear();
  }

private:
  template <typename Policy>
  auto build_edges(const tf::edges<Policy> &edges) -> void {
    tf::make_contiguous_index_hash_map(edges, _ihm);
    _work_edges.allocate(edges.size());
    tf::parallel_for_each(
        tf::zip(edges, _work_edges),
        [&](auto pair) {
          auto &&[_in, _out] = pair;
          _out[0] = _ihm.f()[_in[0]];
          _out[1] = _ihm.f()[_in[1]];
        },
        tf::checked);
    _consumed.allocate(edges.size());
    tf::parallel_fill(_consumed, false);
    // Total edge slots across all paths == edges.size(); reserve to
    // avoid reallocations during sequential push_back in walks.
    _paths_buffer.data_buffer().reserve(edges.size());
    _directions.reserve(edges.size());
    _paths_buffer.offsets_buffer().reserve(10);
  }

  auto build_connectivity() -> void {
    _em.build(tf::make_edges(_work_edges), _ihm.kept_ids().size(),
              tf::edge_orientation::bidirectional);
    for (Index e = Index(0); e < Index(_work_edges.size()); ++e) {
      auto edge = _work_edges[e];
      if (_em[edge[0]].size() != 2 || _em[edge[1]].size() != 2)
        _endpoints.push_back(e);
    }
  }

  auto fill_paths() -> void {
    for (auto e_id : _endpoints)
      if (!_consumed[e_id])
        trace_path_from_endpoint(e_id);
    // Closed loops remain — start at any unconsumed edge.
    for (Index e_id = Index(0); e_id < Index(_work_edges.size()); ++e_id) {
      if (!_consumed[e_id])
        trace_path_from(e_id, Index(0));
    }
    if (_paths_buffer.data_buffer().size())
      _paths_buffer.offsets_buffer().push_back(
          _paths_buffer.data_buffer().size());
  }

  // For an endpoint edge, pick the side whose vertex has degree != 2 as
  // the "boundary" side, and walk INTO the path from the other side.
  auto trace_path_from_endpoint(Index e_id) -> void {
    auto edge = _work_edges[e_id];
    Index start_v_side = (_em[edge[0]].size() != 2) ? Index(0) : Index(1);
    trace_path_from(e_id, start_v_side);
  }

  // Walk away from edges[start_edge][start_v_side]. Push edge ids in
  // traversal order; push per-position direction (true = edge[0]->edge[1]
  // matches path direction, false = reversed).
  auto trace_path_from(Index start_edge, Index start_v_side) -> Index {
    auto edge = _work_edges[start_edge];
    Index cur_v = edge[Index(1) - start_v_side];
    Index cur_edge = start_edge;

    auto &offsets = _paths_buffer.offsets_buffer();
    auto &data = _paths_buffer.data_buffer();
    offsets.push_back(data.size());
    data.push_back(cur_edge);
    _directions.push_back(start_v_side == Index(0));
    _consumed[cur_edge] = true;
    Index n = Index(1);

    while (_em[cur_v].size() == 2) {
      Index next_edge = move_to_next(cur_edge, cur_v);
      if (next_edge == Index(-1))
        break;
      auto next_e = _work_edges[next_edge];
      bool forward = (next_e[0] == cur_v);
      Index next_v = forward ? next_e[1] : next_e[0];
      data.push_back(next_edge);
      _directions.push_back(forward);
      _consumed[next_edge] = true;
      ++n;
      cur_edge = next_edge;
      cur_v = next_v;
    }
    return n;
  }

  // Pick the next edge at cur_v: prefer a non-inverse of cur_edge.
  // Fall back to inverse only if no other unconsumed neighbor exists.
  auto move_to_next(Index cur_edge, Index cur_v) -> Index {
    auto incident = _em[cur_v];
    auto cur = _work_edges[cur_edge];
    Index inverse_fallback = Index(-1);
    for (auto e : incident) {
      if (e == cur_edge || _consumed[e])
        continue;
      auto next = _work_edges[e];
      if (next[0] == cur[1] && next[1] == cur[0]) {
        if (inverse_fallback == Index(-1))
          inverse_fallback = e;
        continue;
      }
      return e;
    }
    return inverse_fallback;
  }

  tf::index_hash_map<T, Index, Hash> _ihm;
  tf::buffer<std::array<Index, 2>> _work_edges;
  tf::buffer<bool> _consumed;
  tf::buffer<Index> _endpoints;
  tf::edge_membership<Index> _em;
  tf::offset_block_buffer<Index, Index> _paths_buffer;
  tf::buffer<bool> _directions;
};

} // namespace tf
