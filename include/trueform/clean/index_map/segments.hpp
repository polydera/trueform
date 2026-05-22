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
#include "../../core/algorithm/make_unique_index_map.hpp"
#include "../../core/algorithm/mask_to_index_map.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_replace.hpp"
#include "../../core/algorithm/update_by_mask.hpp"
#include "../../core/base/segments.hpp"
#include "../../core/index_map.hpp"
#include "../../core/none.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../config.hpp"
#include "./points.hpp"

namespace tf {

namespace clean {
/// @cond INTERNAL
template <typename Range0, typename Range1, typename Index, typename Tolerance>
auto make_clean_index_map(const tf::core::segments<Range0, Range1> &segments,
                          const tf::clean_config_t<Tolerance> &cfg,
                          tf::index_map_buffer<Index> &edge_map,
                          tf::index_map_buffer<Index> &point_map) {
  auto make_edge = [&](const auto &edge) {
    auto out = std::make_pair(point_map.f()[edge[0]], point_map.f()[edge[1]]);
    if (out.second < out.first)
      std::swap(out.first, out.second);
    else if (out.first == out.second) {
      out = std::make_pair(std::numeric_limits<Index>::max(),
                           std::numeric_limits<Index>::max());
    }
    return out;
  };

  // Edge dedup + degenerate-edge removal:
  //   dup=on:  canonical sort dedups, then always drop the MAX bucket
  //            (zero-length edges canonicalize to {MAX, MAX}).
  //   dup=off: identity edge_map then mask-out zero-length edges directly.
  // Degenerate removal is always applied (zero-length edges are not legitimate
  // geometry).
  if (cfg.remove_duplicate_primitives) {
    tf::make_unique_index_map(
        segments.edges(), edge_map,
        [&](const auto &x0, const auto &x1) {
          return make_edge(x0) == make_edge(x1);
        },
        [&](const auto &x0, const auto &x1) {
          return make_edge(x0) < make_edge(x1);
        });
    if (edge_map.kept_ids().size() > 0) {
      const auto &last_edge = segments.edges()[edge_map.kept_ids().back()];
      if (point_map.f()[last_edge[0]] == point_map.f()[last_edge[1]]) {
        Index id = edge_map.kept_ids().back();
        edge_map.kept_ids().pop_back();
        tf::parallel_replace(edge_map.f(), id, Index(edge_map.f().size()));
      }
    }
  } else {
    auto n = static_cast<Index>(segments.edges().size());
    tf::buffer<bool> keep;
    keep.allocate(n);
    tf::parallel_for_each(tf::enumerate(segments.edges()), [&](auto pair) {
      auto &&[i, edge] = pair;
      keep[i] = (point_map.f()[edge[0]] != point_map.f()[edge[1]]);
    });
    tf::mask_to_index_map(keep, edge_map);
  }

  // Step 3: unreferenced-point removal.
  if (cfg.remove_unreferenced_points) {
    tf::buffer<bool> contained_points;
    contained_points.allocate(point_map.kept_ids().size());
    tf::parallel_fill(contained_points, false);
    tf::parallel_for_each(
        tf::make_indirect_range(
            edge_map.kept_ids(),
            tf::make_block_indirect_range(segments.edges(), point_map.f())),
        [&](const auto &edge) {
          contained_points[edge[0]] = true;
          contained_points[edge[1]] = true;
        },
        tf::checked);
    tf::update_by_mask(point_map, contained_points);
  }
}
/// @endcond
} // namespace clean

/// @ingroup clean
/// @brief Generate index maps for segment cleaning (output parameters).
///
/// Creates index maps for both edges and points.
/// Use @ref tf::reindexed to apply the maps to associated data.
///
/// @tparam Range0 The edge range type.
/// @tparam Range1 The point range type.
/// @tparam Index The index type.
/// @param segments The input @ref tf::segments.
/// @param edge_map Output edge @ref tf::index_map_buffer to populate.
/// @param point_map Output point @ref tf::index_map_buffer to populate.
template <typename Range0, typename Range1, typename Index, typename Tolerance>
auto make_clean_index_map(const tf::core::segments<Range0, Range1> &segments,
                          const tf::clean_config_t<Tolerance> &cfg,
                          tf::index_map_buffer<Index> &edge_map,
                          tf::index_map_buffer<Index> &point_map) {
  if (!segments.size())
    return;
  make_clean_index_map(segments.points(), cfg.tolerance, point_map);
  clean::make_clean_index_map(segments, cfg, edge_map, point_map);
}

template <typename Range0, typename Range1, typename Index>
auto make_clean_index_map(const tf::core::segments<Range0, Range1> &segments,
                          tf::index_map_buffer<Index> &edge_map,
                          tf::index_map_buffer<Index> &point_map) {
  using Real = tf::coordinate_type<Range1>;
  make_clean_index_map(segments, tf::clean_config_t<Real>{Real{0}}, edge_map,
                       point_map);
}

/// @ingroup clean
/// @brief Generate index maps for segment cleaning with tolerance (output parameters).
/// @overload
template <typename Range0, typename Range1, typename Index>
auto make_clean_index_map(
    const tf::core::segments<Range0, Range1> &segments,
    tf::coordinate_type<decltype(segments.points())> tolerance,
    tf::index_map_buffer<Index> &edge_map,
    tf::index_map_buffer<Index> &point_map) {
  using Real = tf::coordinate_type<Range1>;
  make_clean_index_map(segments, tf::clean_config_t<Real>{tolerance}, edge_map,
                       point_map);
}

/// @ingroup clean
/// @brief Generate index maps for exact segment deduplication.
///
/// Creates index maps for both edges and points.
/// Use @ref tf::reindexed to apply the maps to associated data.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Range0 The edge range type.
/// @tparam Range1 The point range type.
/// @param segments The input @ref tf::segments.
/// @return Tuple of (edge @ref tf::index_map_buffer, point @ref tf::index_map_buffer).
template <typename Index = tf::none_t, typename Range0, typename Range1>
auto make_clean_index_map(const tf::core::segments<Range0, Range1> &segments) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return make_clean_index_map<ActualIndex>(segments);
  } else {
    tf::index_map_buffer<Index> edge_map;
    tf::index_map_buffer<Index> point_map;
    make_clean_index_map(segments, edge_map, point_map);
    return std::make_pair(std::move(edge_map), std::move(point_map));
  }
}

/// @ingroup clean
/// @brief Generate index maps for segment cleaning with tolerance.
/// @overload
template <typename Index = tf::none_t, typename Range0, typename Range1>
auto make_clean_index_map(const tf::core::segments<Range0, Range1> &segments,
                          tf::coordinate_type<Range1> tolerance) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return make_clean_index_map<ActualIndex>(segments, tolerance);
  } else {
    tf::index_map_buffer<Index> edge_map;
    tf::index_map_buffer<Index> point_map;
    make_clean_index_map(segments, tolerance, edge_map, point_map);
    return std::make_pair(std::move(edge_map), std::move(point_map));
  }
}

/// @ingroup clean
/// @brief Generate index maps for segment cleaning with a config.
/// @overload
template <typename Index = tf::none_t, typename Range0, typename Range1,
          typename Tolerance>
auto make_clean_index_map(const tf::core::segments<Range0, Range1> &segments,
                          const tf::clean_config_t<Tolerance> &cfg) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return make_clean_index_map<ActualIndex>(segments, cfg);
  } else {
    tf::index_map_buffer<Index> edge_map;
    tf::index_map_buffer<Index> point_map;
    make_clean_index_map(segments, cfg, edge_map, point_map);
    return std::make_pair(std::move(edge_map), std::move(point_map));
  }
}
} // namespace tf
