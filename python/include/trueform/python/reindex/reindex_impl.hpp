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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <trueform/core/points.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/reindex/by_ids.hpp>
#include <trueform/reindex/by_mask.hpp>

namespace tf::py {

// =============================================================================
// REINDEX_BY_IDS - Points
// =============================================================================

template <typename Index, typename RealT, std::size_t Dims>
auto reindexed_by_ids_impl(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>> ids) {
  // Create points range from numpy array
  std::size_t num_points = points.shape(0);
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points.data(), num_points * Dims));

  // Create ids range
  std::size_t num_ids = ids.shape(0);
  auto ids_range = tf::make_range(ids.data(), num_ids);

  // Always call with return_index_map
  auto [res, maps] = tf::reindexed_by_ids<Index>(points_range, ids_range,
                                                 tf::return_index_map);

  return nanobind::make_tuple(make_numpy_array(std::move(res)),
                              make_numpy_array(std::move(maps)));
}

// =============================================================================
// REINDEX_BY_IDS - Indexed Geometry (Polygons/Segments)
// =============================================================================

template <typename Index, std::size_t V, typename RealT, std::size_t Dims>
auto reindexed_by_ids_impl(
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, V>>
        indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>> ids) {
  // Create points range from numpy array
  std::size_t num_points = points.shape(0);
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points.data(), num_points * Dims));

  // Create indices range
  std::size_t num_indices = indices.shape(0);
  auto indices_range = tf::make_blocked_range<V>(
      tf::make_range(indices.data(), num_indices * V));

  // Create primitive range
  auto primitive_range = [&] {
    if constexpr (V == 2)
      return tf::make_segments(indices_range, points_range);
    else
      return tf::make_polygons(indices_range, points_range);
  }();

  // Create ids range
  std::size_t num_ids = ids.shape(0);
  auto ids_range = tf::make_range(ids.data(), num_ids);

  // Always call with return_index_map
  auto [res, i_map, p_map] = tf::reindexed_by_ids<Index>(
      primitive_range, ids_range, tf::return_index_map);

  return nanobind::make_tuple(make_numpy_array(std::move(res)),
                              make_numpy_array(std::move(i_map)),
                              make_numpy_array(std::move(p_map)));
}

// =============================================================================
// REINDEX_BY_IDS - Dynamic Indexed Geometry (Variable-sized Polygons)
// =============================================================================

template <typename Index, typename RealT, std::size_t Dims>
auto reindexed_by_ids_impl_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>> ids) {
  // Create points range from numpy array
  std::size_t num_points = points.shape(0);
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points.data(), num_points * Dims));

  // Create indices range from offset_blocked_array
  auto indices_range = indices.make_range();

  // Create primitive range
  auto primitive_range = tf::make_polygons(indices_range, points_range);

  // Create ids range
  std::size_t num_ids = ids.shape(0);
  auto ids_range = tf::make_range(ids.data(), num_ids);

  // Always call with return_index_map
  auto [res, i_map, p_map] = tf::reindexed_by_ids<Index>(
      primitive_range, ids_range, tf::return_index_map);

  return nanobind::make_tuple(make_numpy_array(std::move(res)),
                              make_numpy_array(std::move(i_map)),
                              make_numpy_array(std::move(p_map)));
}

// =============================================================================
// REINDEX_BY_MASK - Points
// =============================================================================

template <typename Index, typename RealT, std::size_t Dims>
auto reindexed_by_mask_impl(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    nanobind::ndarray<nanobind::numpy, const bool, nanobind::shape<-1>> mask) {
  // Create points range from numpy array
  std::size_t num_points = points.shape(0);
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points.data(), num_points * Dims));

  // Create mask range
  std::size_t mask_size = mask.shape(0);
  auto mask_range = tf::make_range(mask.data(), mask_size);

  // Always call with return_index_map
  auto [res, maps] = tf::reindexed_by_mask<Index>(points_range, mask_range,
                                                  tf::return_index_map);

  return nanobind::make_tuple(make_numpy_array(std::move(res)),
                              make_numpy_array(std::move(maps)));
}

// =============================================================================
// REINDEX_BY_MASK - Indexed Geometry (Polygons/Segments)
// =============================================================================

template <typename Index, std::size_t V, typename RealT, std::size_t Dims>
auto reindexed_by_mask_impl(
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, V>>
        indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    nanobind::ndarray<nanobind::numpy, const bool, nanobind::shape<-1>> mask) {
  // Create points range from numpy array
  std::size_t num_points = points.shape(0);
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points.data(), num_points * Dims));

  // Create indices range
  std::size_t num_indices = indices.shape(0);
  auto indices_range = tf::make_blocked_range<V>(
      tf::make_range(indices.data(), num_indices * V));

  // Create primitive range
  auto primitive_range = [&] {
    if constexpr (V == 2)
      return tf::make_segments(indices_range, points_range);
    else
      return tf::make_polygons(indices_range, points_range);
  }();

  // Create mask range
  std::size_t mask_size = mask.shape(0);
  auto mask_range = tf::make_range(mask.data(), mask_size);

  // Always call with return_index_map
  auto [res, i_map, p_map] = tf::reindexed_by_mask<Index>(
      primitive_range, mask_range, tf::return_index_map);

  return nanobind::make_tuple(make_numpy_array(std::move(res)),
                              make_numpy_array(std::move(i_map)),
                              make_numpy_array(std::move(p_map)));
}

// =============================================================================
// REINDEX_BY_MASK - Dynamic Indexed Geometry (Variable-sized Polygons)
// =============================================================================

template <typename Index, typename RealT, std::size_t Dims>
auto reindexed_by_mask_impl_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    nanobind::ndarray<nanobind::numpy, const bool, nanobind::shape<-1>> mask) {
  // Create points range from numpy array
  std::size_t num_points = points.shape(0);
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points.data(), num_points * Dims));

  // Create indices range from offset_blocked_array
  auto indices_range = indices.make_range();

  // Create primitive range
  auto primitive_range = tf::make_polygons(indices_range, points_range);

  // Create mask range
  std::size_t mask_size = mask.shape(0);
  auto mask_range = tf::make_range(mask.data(), mask_size);

  // Always call with return_index_map
  auto [res, i_map, p_map] = tf::reindexed_by_mask<Index>(
      primitive_range, mask_range, tf::return_index_map);

  return nanobind::make_tuple(make_numpy_array(std::move(res)),
                              make_numpy_array(std::move(i_map)),
                              make_numpy_array(std::move(p_map)));
}

} // namespace tf::py
