/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/curves_buffer.hpp"
#include "../core/segments.hpp"
#include "../core/views/slide_range.hpp"
#include "../reindex/return_index_map.hpp"
#include "../reindex/segments.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./index_map/segments.hpp"

namespace tf {
namespace clean::detail {

template <typename Index, typename Policy>
auto extract_edges_from_curves(const tf::curves<Policy> &curves) {
  // Count total edges across all paths
  Index total_edges = 0;
  for (const auto &path : curves.paths()) {
    if (path.size() >= 2) {
      total_edges += static_cast<Index>(path.size() - 1);
    }
  }

  // Allocate edges buffer
  tf::blocked_buffer<Index, 2> edges_buf;
  if (total_edges > 0) {
    edges_buf.allocate(total_edges);

    // Copy edges from all paths
    Index edge_idx = 0;
    for (const auto &path : curves.paths()) {
      if (path.size() >= 2) {
        for (auto edge : tf::make_slide_range<2>(path)) {
          edges_buf[edge_idx][0] = static_cast<Index>(edge[0]);
          edges_buf[edge_idx][1] = static_cast<Index>(edge[1]);
          ++edge_idx;
        }
      }
    }
  }
  return edges_buf;
}

} // namespace clean::detail

template <typename Index, typename Range0, typename Range1>
auto cleaned(const tf::core::curves<Range0, Range1> &curves,
             tf::coordinate_type<Range1> tolerance, tf::return_index_map_t) {
  auto edges_buf = clean::detail::extract_edges_from_curves<Index>(
      tf::make_curves(curves.paths(), curves.points()));
  if (edges_buf.size() == 0) {
    tf::curves_buffer<Index, tf::coordinate_type<Range1>,
                      tf::coordinate_dims_v<Range1>>
        out;
    tf::index_map_buffer<Index> point_im;
    return std::make_tuple(std::move(out), std::move(point_im));
  }

  // Build segments from edges and points
  auto segments = tf::make_segments(tf::make_edges(edges_buf), curves.points());

  // Clean the segments
  auto [edge_im, point_im] =
      tf::make_clean_index_map<Index>(segments, tolerance);
  auto cleaned_segments = tf::reindexed(segments, edge_im, point_im);

  // Reconnect edges to paths
  auto paths = tf::connect_edges_to_paths(cleaned_segments.edges());

  // Build output curves_buffer
  tf::curves_buffer<Index, tf::coordinate_type<Range1>,
                    tf::coordinate_dims_v<Range1>>
      out;
  out.paths_buffer() = std::move(paths);
  out.points_buffer() = std::move(cleaned_segments.points_buffer());

  return std::make_tuple(std::move(out), std::move(point_im));
}

template <typename Index, typename Range0, typename Range1>
auto cleaned(const tf::core::curves<Range0, Range1> &curves,
             tf::return_index_map_t) {
  auto edges_buf = clean::detail::extract_edges_from_curves<Index>(
      tf::make_curves(curves.paths(), curves.points()));
  if (edges_buf.size() == 0) {
    tf::curves_buffer<Index, tf::coordinate_type<Range1>,
                      tf::coordinate_dims_v<Range1>>
        out;
    tf::index_map_buffer<Index> point_im;
    return std::make_tuple(std::move(out), std::move(point_im));
  }

  // Build segments from edges and points
  auto segments = tf::make_segments(tf::make_edges(edges_buf), curves.points());

  // Clean the segments (no tolerance = exact deduplication)
  auto [edge_im, point_im] = tf::make_clean_index_map<Index>(segments);
  auto cleaned_segments = tf::reindexed(segments, edge_im, point_im);

  // Reconnect edges to paths
  auto paths = tf::connect_edges_to_paths(cleaned_segments.edges());

  // Build output curves_buffer
  tf::curves_buffer<Index, tf::coordinate_type<Range1>,
                    tf::coordinate_dims_v<Range1>>
      out;
  out.paths_buffer() = std::move(paths);
  out.points_buffer() = std::move(cleaned_segments.points_buffer());

  return std::make_tuple(std::move(out), std::move(point_im));
}

template <typename Index, typename Policy>
auto cleaned(const tf::curves<Policy> &curves,
             tf::coordinate_type<Policy> tolerance) {
  return std::get<0>(
      cleaned<Index>(curves, tolerance, tf::return_index_map));
}

template <typename Index, typename Policy>
auto cleaned(const tf::curves<Policy> &curves) {
  return std::get<0>(cleaned<Index>(curves, tf::return_index_map));
}

} // namespace tf
