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
#include "../core/curves_buffer.hpp"
#include "../core/small_vector.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./make_intersection_edges.hpp"
#include "./scalar_field_intersections.hpp"

namespace tf {

/// @ingroup intersect_curves
/// @brief Extract isocontour curves where a scalar field crosses a threshold.
///
/// Computes the isosurface intersection where a scalar field defined over
/// mesh vertices crosses the specified threshold value.
///
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input @ref tf::polygons.
/// @param scalars The scalar field values (one per vertex).
/// @param cut_value The threshold value.
/// @return A @ref tf::curves_buffer containing connected isocontour curves.
///
/// @see tf::scalar_field_intersections for low-level access.
template <typename Policy, typename Iterator0, std::size_t N0>
auto make_isocontours(const tf::polygons<Policy> &polygons,
                      const tf::range<Iterator0, N0> &scalars,
                      tf::coordinate_type<Policy> cut_value) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  tf::scalar_field_intersections<Index, tf::coordinate_type<Policy>,
                                 tf::coordinate_dims_v<Policy>>
      sfi;
  sfi.build(polygons, scalars, cut_value);
  auto ie = tf::make_intersection_edges(sfi);
  auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
  tf::curves_buffer<Index, tf::coordinate_type<Policy>,
                    tf::coordinate_dims_v<Policy>>
      cb;
  cb.paths_buffer() = std::move(paths);
  cb.points_buffer().allocate(sfi.intersection_points().size());
  tf::parallel_copy(sfi.intersection_points(), cb.points());
  return cb;
}

/// @ingroup intersect_curves
/// @brief Extract isocontour curves at multiple threshold values.
/// @overload
///
/// @param polygons The input @ref tf::polygons.
/// @param scalars The scalar field values (one per vertex).
/// @param cut_values Multiple threshold values.
/// @return A @ref tf::curves_buffer containing all isocontour curves.
template <typename Policy, typename Iterator0, std::size_t N0,
          typename Iterator1, std::size_t N1>
auto make_isocontours(const tf::polygons<Policy> &polygons,
                      const tf::range<Iterator0, N0> &scalars,
                      const tf::range<Iterator1, N1> &cut_values) {
  tf::small_vector<tf::coordinate_type<Policy>, 10> cut_values_;
  cut_values_.reserve(cut_values.size());
  std::copy(cut_values.begin(), cut_values.end(),
            std::back_inserter(cut_values_));
  std::sort(cut_values_.begin(), cut_values_.end());
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  tf::scalar_field_intersections<Index, tf::coordinate_type<Policy>,
                                 tf::coordinate_dims_v<Policy>>
      sfi;
  sfi.build_many(polygons, scalars, cut_values_);
  auto ie = tf::make_intersection_edges(sfi);
  auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
  tf::curves_buffer<Index, tf::coordinate_type<Policy>,
                    tf::coordinate_dims_v<Policy>>
      cb;
  cb.paths_buffer() = std::move(paths);
  cb.points_buffer().allocate(sfi.intersection_points().size());
  tf::parallel_copy(sfi.intersection_points(), cb.points());
  return cb;
}
} // namespace tf
