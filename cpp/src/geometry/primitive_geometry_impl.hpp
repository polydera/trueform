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

#include "trueform/cpp/geometry/area.hpp"
#include "trueform/cpp/geometry/mean_edge_length.hpp"
#include "trueform/cpp/geometry/normals.hpp"

#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/core/area.hpp"
#include "trueform/core/mean_edge_length.hpp"
#include "trueform/core/points.hpp"
#include "trueform/core/polygons.hpp"
#include "trueform/core/views/blocked_range.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/geometry/compute_normals.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace tf::cpp {
namespace detail {

template <typename Real, std::size_t Dims, typename Function>
auto with_primitive_polygons(const primitive<Real, Dims> &value,
                             const char *operation, Function &&function) {
  if (value.kind() != primitive_kind::triangle &&
      value.kind() != primitive_kind::polygon)
    throw std::invalid_argument(std::string(operation) +
                                ": only triangle and polygon are supported");

  const auto data = value.data();
  auto points = tf::make_points<Dims>(data.make_range());
  const auto point_ids = tf::make_sequence_range(points.size());
  if (value.kind() == primitive_kind::triangle) {
    auto faces = tf::make_blocked_range<3>(point_ids);
    return function(tf::make_polygons(faces, points));
  }

  auto faces = tf::make_blocked_range(
      point_ids, static_cast<std::size_t>(value.polygon_vertex_count()));
  return function(tf::make_polygons(faces, points));
}

} // namespace detail

template <typename Real, std::size_t Dims>
auto area(const primitive<Real, Dims> &value) -> area_result<Real> {
  const auto is_batch = value.is_batch();
  const auto count = value.count();
  return detail::with_primitive_polygons(
      value, "area", [is_batch, count](const auto &polygons) {
        if (!is_batch)
          return area_result<Real>(static_cast<Real>(tf::area(polygons[0])));

        tf::buffer<Real> output;
        output.allocate(static_cast<std::size_t>(count));
        tf::parallel_transform(
            polygons, output,
            [](const auto &polygon) -> Real { return tf::area(polygon); },
            tf::checked);
        return area_result<Real>(
            nd_array<Real>::from_buffer(std::move(output), {count}));
      });
}

template <typename Real, std::size_t Dims>
auto mean_edge_length(const primitive<Real, Dims> &value) -> Real {
  return detail::with_primitive_polygons(
      value, "mean_edge_length", [](const auto &polygons) -> Real {
        return tf::mean_edge_length(polygons);
      });
}

template <typename Real>
auto normals(const primitive<Real, 3> &value) -> nd_array<Real> {
  const auto is_batch = value.is_batch();
  const auto count = value.count();
  return detail::with_primitive_polygons(
      value, "normals", [is_batch, count](const auto &polygons) {
        auto output = tf::compute_normals(polygons);
        if (is_batch)
          return nd_array<Real>::from_buffer(std::move(output.data_buffer()),
                                             {count, 3});
        return nd_array<Real>::from_buffer(std::move(output.data_buffer()),
                                           {3});
      });
}

} // namespace tf::cpp
