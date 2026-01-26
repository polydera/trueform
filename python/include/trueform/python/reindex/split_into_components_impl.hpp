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

#include "trueform/core/segments.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/reindex/split_into_components.hpp>

namespace tf::py {
template <typename Index, std::size_t V, typename RealT, std::size_t Dims>
auto split_into_components_impl(
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, V>>
        indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>> labels) {
  auto i_range =
      tf::make_blocked_range<V>(tf::make_range(indices.data(), indices.size()));
  auto p_range =
      tf::make_points<Dims>(tf::make_range(points.data(), points.size()));
  auto l_range = tf::make_range(labels.data(), labels.size());
  auto primitive_range = [&] {
    if constexpr (V == 2)
      return tf::make_segments(i_range, p_range);
    else
      return tf::make_polygons(i_range, p_range);
  }();
  auto [components, c_labels] =
      tf::split_into_components(primitive_range, l_range);

  auto make_pair = [&components = components](auto i) {
    auto [indices, pts] = make_numpy_array(std::move(components[i]));
    return nanobind::make_tuple(std::move(indices), std::move(pts));
  };
  std::vector<decltype(make_pair(0))> cs;
  cs.reserve(components.size());
  int *ls = new int[c_labels.size()];
  for (std::size_t i = 0; i < components.size(); ++i) {
    cs.push_back(make_pair(i));
    ls[i] = c_labels[i];
  }

  return nanobind::make_tuple(std::move(cs),
                              make_numpy_array(ls, {c_labels.size()}));
}

// =============================================================================
// SPLIT_INTO_COMPONENTS - Dynamic Indexed Geometry (Variable-sized Polygons)
// =============================================================================

template <typename Index, typename RealT, std::size_t Dims>
auto split_into_components_impl_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>> labels) {
  // Create points range from numpy array
  std::size_t num_points = points.shape(0);
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points.data(), num_points * Dims));

  // Create indices range from offset_blocked_array
  auto indices_range = indices.make_range();

  // Create primitive range
  auto primitive_range = tf::make_polygons(indices_range, points_range);

  // Create labels range
  std::size_t num_labels = labels.shape(0);
  auto labels_range = tf::make_range(labels.data(), num_labels);

  // Call the algorithm
  auto [components, c_labels] =
      tf::split_into_components(primitive_range, labels_range);

  // Convert results - each component is now an offset_blocked_buffer
  auto make_pair = [&components = components](auto i) {
    auto [indices_result, pts] = make_numpy_array(std::move(components[i]));
    return nanobind::make_tuple(std::move(indices_result), std::move(pts));
  };

  std::vector<decltype(make_pair(0))> cs;
  cs.reserve(components.size());
  int *ls = new int[c_labels.size()];
  for (std::size_t i = 0; i < components.size(); ++i) {
    cs.push_back(make_pair(i));
    ls[i] = c_labels[i];
  }

  return nanobind::make_tuple(std::move(cs),
                              make_numpy_array(ls, {c_labels.size()}));
}

} // namespace tf::py
