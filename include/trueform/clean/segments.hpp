/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/is_soup.hpp"
#include "../core/none.hpp"
#include "../reindex/return_index_map.hpp"
#include "../reindex/segments.hpp"
#include "./index_map/segments.hpp"
#include "./soup/segments.hpp"

namespace tf {
template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::segments<Policy> &segments,
             tf::coordinate_type<Policy> tolerance) {
  if constexpr (std::is_same_v<Index, tf::none_t> && tf::is_soup<Policy>) {
    return cleaned<int>(segments, tolerance);
  } else if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return cleaned<ActualIndex>(segments, tolerance);
  } else if constexpr (tf::is_soup<Policy>) {
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

template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::segments<Policy> &segments) {
  if constexpr (std::is_same_v<Index, tf::none_t> && tf::is_soup<Policy>) {
    return cleaned<int>(segments);
  } else if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return cleaned<ActualIndex>(segments);
  } else if constexpr (tf::is_soup<Policy>) {
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

template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::segments<Policy> &segments,
             tf::coordinate_type<Policy> tolerance, tf::return_index_map_t) {
  static_assert(!tf::is_soup<Policy>, "Soups cannot return index maps.");
  using ActualIndex =
      std::conditional_t<std::is_same_v<Index, tf::none_t>,
                         std::decay_t<decltype(segments.edges()[0][0])>, Index>;
  auto [edge_im, point_im] =
      tf::make_clean_index_map<ActualIndex>(segments, tolerance);
  auto out =
      tf::reindexed(tf::make_segments(segments.edges(), segments.points()),
                    edge_im, point_im);
  return std::make_tuple(std::move(out), std::move(edge_im),
                         std::move(point_im));
}

template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::segments<Policy> &segments, tf::return_index_map_t) {
  static_assert(!tf::is_soup<Policy>, "Soups cannot return index maps.");
  using ActualIndex =
      std::conditional_t<std::is_same_v<Index, tf::none_t>,
                         std::decay_t<decltype(segments.edges()[0][0])>, Index>;
  auto [edge_im, point_im] = tf::make_clean_index_map<ActualIndex>(segments);
  auto out =
      tf::reindexed(tf::make_segments(segments.edges(), segments.points()),
                    edge_im, point_im);
  return std::make_tuple(std::move(out), std::move(edge_im),
                         std::move(point_im));
}
} // namespace tf
