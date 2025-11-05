/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/is_soup.hpp"
#include "../reindex/polygons.hpp"
#include "../reindex/return_index_map.hpp"
#include "./index_map/polygons.hpp"
#include "./soup/polygons.hpp"

namespace tf {
template <typename Index, typename Policy>
auto cleaned(const tf::polygons<Policy> &polygons,
             tf::coordinate_type<Policy> tolerance)
    -> tf::polygons_buffer<Index, tf::coordinate_type<Policy>,
                           tf::coordinate_dims_v<Policy>,
                           tf::static_size_v<decltype(polygons[0])>> {
  if constexpr (tf::is_soup<Policy>) {
    tf::clean::polygon_soup<Index, tf::coordinate_type<Policy>,
                            tf::coordinate_dims_v<Policy>,
                            tf::static_size_v<decltype(polygons[0])>>
        out;
    out.build(polygons, tolerance);
    return out;
  } else {
    auto [face_im, point_im] =
        tf::make_clean_index_map<Index>(polygons, tolerance);
    return tf::reindexed(polygons, face_im, point_im);
  }
}

template <typename Index, typename Policy>
auto cleaned(const tf::polygons<Policy> &polygons)
    -> tf::polygons_buffer<Index, tf::coordinate_type<Policy>,
                           tf::coordinate_dims_v<Policy>,
                           tf::static_size_v<decltype(polygons[0])>> {
  if constexpr (tf::is_soup<Policy>) {
    tf::clean::polygon_soup<Index, tf::coordinate_type<Policy>,
                            tf::coordinate_dims_v<Policy>,
                            tf::static_size_v<decltype(polygons[0])>>
        out;
    out.build(polygons);
    return out;
  } else {
    auto [face_im, point_im] = tf::make_clean_index_map<Index>(polygons);
    return tf::reindexed(polygons, face_im, point_im);
  }
}

template <typename Index, typename Range0, typename Range1>
auto cleaned(const tf::core::polygons<Range0, Range1> &polygons,
             tf::coordinate_type<Range1> tolerance, tf::return_index_map_t) {
  auto [face_im, point_im] =
      tf::make_clean_index_map<Index>(polygons, tolerance);
  auto out = tf::reindexed(polygons, face_im, point_im);
  return std::make_tuple(std::move(out), std::move(face_im),
                         std::move(point_im));
}

template <typename Index, typename Range0, typename Range1>
auto cleaned(const tf::core::polygons<Range0, Range1> &polygons,
             tf::return_index_map_t) {
  auto [face_im, point_im] = tf::make_clean_index_map<Index>(polygons);
  auto out = tf::reindexed(polygons, face_im, point_im);
  return std::make_tuple(std::move(out), std::move(face_im),
                         std::move(point_im));
}
} // namespace tf
