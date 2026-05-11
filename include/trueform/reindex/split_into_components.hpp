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
#include "../core/algorithm/compute_offsets.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_copy_blocked.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_iota.hpp"
#include "../core/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/segments.hpp"
#include "../core/segments_buffer.hpp"
#include "../core/static_size.hpp"
#include "../core/views/block_indirect_range.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/indirect_range.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/offset_block_range.hpp"
#include "../core/views/zip.hpp"
#include "tbb/parallel_sort.h"

namespace tf {

/// @cond INTERNAL
namespace reindex {

/// Polygon-specialised split. Walks each component's faces once with a shared
/// point-remap buffer (sentinel-reset between components) and emits faces and
/// points with a single allocation each. Handles fixed-size
/// (`blocked_buffer`) and variable-size (`offset_block_buffer`) face layouts
/// via `static_size_v`.
template <typename Index, typename Policy, typename Range>
auto split_into_components_polygons(const tf::polygons<Policy> &polygons,
                                    const Range &labels) {
  using RealT = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  constexpr auto N = tf::static_size_v<decltype(polygons[0])>;
  using out_t = tf::polygons_buffer<Index, RealT, Dims, N>;
  using label_t = std::decay_t<decltype(labels[0])>;

  auto n_elements = static_cast<Index>(labels.size());
  tf::buffer<Index> ids;
  ids.allocate(n_elements);
  tf::parallel_iota(ids, Index(0));
  tbb::parallel_sort(ids.begin(), ids.end(),
                     [&](auto a, auto b) { return labels[a] < labels[b]; });
  tf::buffer<Index> offsets;
  tf::compute_offsets(ids, std::back_inserter(offsets), Index(0),
                      [&](auto a, auto b) { return labels[a] == labels[b]; });
  auto ids_r = tf::make_offset_block_range(offsets, ids);
  auto n_components = static_cast<Index>(ids_r.size());

  tf::buffer<label_t> out_labels;
  out_labels.allocate(n_components);

  tf::buffer<Index> point_map;
  point_map.allocate(polygons.points().size());
  tf::parallel_fill(point_map, Index(-1));
  tf::buffer<Index> pt_ids;
  Index point_sentinel = 0;

  std::vector<out_t> out_v(n_components);
  for (auto &&[comp_ids, out, l] : tf::zip(ids_r, out_v, out_labels)) {
    l = labels[comp_ids.front()];
    Index current = point_sentinel;
    pt_ids.clear();
    auto comp_faces = tf::make_indirect_range(comp_ids, polygons.faces());
    for (auto face : comp_faces) {
      for (auto v_id : face) {
        auto &v = point_map[v_id];
        if (v == -1 || v < point_sentinel) {
          pt_ids.push_back(v_id);
          v = current++;
        }
      }
    }

    if constexpr (N == tf::dynamic_size) {
      auto &offs = out.faces_buffer().offsets_buffer();
      offs.allocate(comp_ids.size() + 1);
      offs[0] = 0;
      std::size_t total = 0;
      for (auto [i, f] : tf::enumerate(comp_faces)) {
        total += f.size();
        offs[i + 1] = static_cast<Index>(total);
      }
      out.faces_buffer().data_buffer().allocate(total);
    } else {
      out.faces_buffer().data_buffer().allocate(N * comp_ids.size());
    }

    auto src = tf::make_block_indirect_range(
        comp_faces, tf::make_mapped_range(point_map, [point_sentinel](auto id) {
          return id - point_sentinel;
        }));
    if constexpr (N == tf::dynamic_size)
      tf::parallel_copy_blocked(src, out.faces());
    else
      tf::parallel_copy(src, out.faces_buffer());

    out.points_buffer().allocate(pt_ids.size());
    tf::parallel_copy(tf::make_indirect_range(pt_ids, polygons.points()),
                      out.points_buffer());
    point_sentinel = current;
  }
  return std::make_pair(std::move(out_v), std::move(out_labels));
}

/// Segment-specialised split. Same sentinel-based remap strategy as the
/// polygon path. Edges are always 2-elem (static), so no dynamic branch.
template <typename Index, typename Policy, typename Range>
auto split_into_components_segments(const tf::segments<Policy> &segments,
                                    const Range &labels) {
  using RealT = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  constexpr std::size_t N = 2;
  using out_t = tf::segments_buffer<Index, RealT, Dims>;
  using label_t = std::decay_t<decltype(labels[0])>;

  auto n_elements = static_cast<Index>(labels.size());
  tf::buffer<Index> ids;
  ids.allocate(n_elements);
  tf::parallel_iota(ids, Index(0));
  tbb::parallel_sort(ids.begin(), ids.end(),
                     [&](auto a, auto b) { return labels[a] < labels[b]; });
  tf::buffer<Index> offsets;
  tf::compute_offsets(ids, std::back_inserter(offsets), Index(0),
                      [&](auto a, auto b) { return labels[a] == labels[b]; });
  auto ids_r = tf::make_offset_block_range(offsets, ids);
  auto n_components = static_cast<Index>(ids_r.size());

  tf::buffer<label_t> out_labels;
  out_labels.allocate(n_components);

  tf::buffer<Index> point_map;
  point_map.allocate(segments.points().size());
  tf::parallel_fill(point_map, Index(-1));
  tf::buffer<Index> pt_ids;
  Index point_sentinel = 0;

  std::vector<out_t> out_v(n_components);
  for (auto &&[comp_ids, out, l] : tf::zip(ids_r, out_v, out_labels)) {
    l = labels[comp_ids.front()];
    Index current = point_sentinel;
    pt_ids.clear();
    auto comp_edges = tf::make_indirect_range(comp_ids, segments.edges());
    for (auto edge : comp_edges) {
      for (auto v_id : edge) {
        auto &v = point_map[v_id];
        if (v == -1 || v < point_sentinel) {
          pt_ids.push_back(v_id);
          v = current++;
        }
      }
    }

    out.edges_buffer().data_buffer().allocate(N * comp_ids.size());
    auto src = tf::make_block_indirect_range(
        comp_edges, tf::make_mapped_range(point_map, [point_sentinel](auto id) {
          return id - point_sentinel;
        }));
    tf::parallel_copy(src, out.edges_buffer());

    out.points_buffer().allocate(pt_ids.size());
    tf::parallel_copy(tf::make_indirect_range(pt_ids, segments.points()),
                      out.points_buffer());
    point_sentinel = current;
  }
  return std::make_pair(std::move(out_v), std::move(out_labels));
}

} // namespace reindex
/// @endcond

/// @ingroup reindex
/// @brief Split polygons into labeled components.
///
/// Groups faces by label and returns separate geometry for each.
/// Single-pass per component with a shared point-remap buffer; supports both
/// fixed-size and variable-size face layouts.
///
/// @tparam Policy The policy type of the polygons.
/// @tparam Range The label range type.
/// @param polygons The input @ref tf::polygons.
/// @param labels Per-face labels (same size as faces()).
/// @return Pair of (vector of @ref tf::polygons_buffer, vector of labels).
template <typename Policy, typename Range>
auto split_into_components(const tf::polygons<Policy> &polygons,
                           const Range &labels) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  return reindex::split_into_components_polygons<Index>(polygons, labels);
}

/// @ingroup reindex
/// @brief Split segments into labeled components.
///
/// Groups edges by label and returns separate geometry for each.
/// Single-pass per component with a shared point-remap buffer.
///
/// @tparam Policy The policy type of the segments.
/// @tparam Range The label range type.
/// @param segments The input @ref tf::segments.
/// @param labels Per-edge labels (same size as edges()).
/// @return Pair of (vector of @ref tf::segments_buffer, vector of labels).
template <typename Policy, typename Range>
auto split_into_components(const tf::segments<Policy> &segments,
                           const Range &labels) {
  using Index = std::decay_t<decltype(segments.edges()[0][0])>;
  return reindex::split_into_components_segments<Index>(segments, labels);
}
} // namespace tf
