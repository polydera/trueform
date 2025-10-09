/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/is_soup.hpp"
#include "../reindex/return_index_map.hpp"
#include "../reindex/segments.hpp"
#include "./index_map/segments.hpp"
#include "./soup/segments.hpp"

namespace tf {
template <typename Index, typename Policy>
auto cleaned(const tf::segments<Policy> &segments,
             tf::coordinate_type<Policy> tolerance)
    -> tf::segments_buffer<Index, tf::coordinate_type<Policy>,
                           tf::coordinate_dims_v<Policy>> {
  if constexpr (tf::is_soup<Policy>) {
    tf::clean::segment_soup<Index, tf::coordinate_type<Policy>,
                            tf::coordinate_dims_v<Policy>>
        out;
    out.build(segments, tolerance);
    return out;
  } else {
    auto [edge_im, point_im] =
        tf::make_clean_index_map<Index>(segments, tolerance);
    return tf::reindexed(segments, edge_im, point_im);
  }
}

template <typename Index, typename Range0, typename Range1>
auto cleaned(const tf::core::segments<Range0, Range1> &segments,
             tf::coordinate_type<Range1> tolerance, tf::return_index_map_t) {
  auto [edge_im, point_im] =
      tf::make_clean_index_map<Index>(segments, tolerance);
  auto out = tf::reindexed(segments, edge_im, point_im);
  return std::make_tuple(std::move(out), std::move(edge_im),
                         std::move(point_im));
}

template <typename Index, typename Policy>
auto cleaned(const tf::segments<Policy> &segments)
    -> tf::segments_buffer<Index, tf::coordinate_type<Policy>,
                           tf::coordinate_dims_v<Policy>> {
  if constexpr (tf::is_soup<Policy>) {
    tf::clean::segment_soup<Index, tf::coordinate_type<Policy>,
                            tf::coordinate_dims_v<Policy>>
        out;
    out.build(segments);
    return out;
  } else {
    auto [edge_im, point_im] = tf::make_clean_index_map<Index>(segments);
    return tf::reindexed(segments, edge_im, point_im);
  }
}

template <typename Index, typename Range0, typename Range1>
auto cleaned(const tf::core::segments<Range0, Range1> &segments,
             tf::return_index_map_t) {
  auto [edge_im, point_im] = tf::make_clean_index_map<Index>(segments);
  auto out = tf::reindexed(segments, edge_im, point_im);
  return std::make_tuple(std::move(out), std::move(edge_im),
                         std::move(point_im));
}
} // namespace tf
