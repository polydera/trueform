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

#include "trueform/cpp/spatial/primitive.hpp"

#include "trueform/core/aabb_like.hpp"
#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/line_like.hpp"
#include "trueform/core/plane_like.hpp"
#include "trueform/core/point_view.hpp"
#include "trueform/core/points.hpp"
#include "trueform/core/polygon.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/ray_like.hpp"
#include "trueform/core/segment.hpp"
#include "trueform/core/unit_vector_view.hpp"
#include "trueform/core/vector_view.hpp"
#include "trueform/core/views/sequence_range.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tf::cpp::detail {

template <std::size_t Dims = 3, typename Real> auto as_point(const Real *data) {
  return tf::make_point_view<Dims>(data);
}

template <std::size_t Dims = 3, typename Real>
auto as_segment(const Real *data) {
  return tf::make_segment_between_points(
      tf::make_point_view<Dims>(data), tf::make_point_view<Dims>(data + Dims));
}

template <std::size_t Dims = 3, typename Real>
auto as_triangle(const Real *data) {
  return tf::make_polygon<3>(
      tf::make_points<Dims>(tf::make_range(data, 3 * Dims)));
}

template <std::size_t Dims = 3, typename Real> auto as_ray(const Real *data) {
  return tf::make_ray_like(tf::make_point_view<Dims>(data),
                           tf::make_vector_view<Dims>(data + Dims));
}

template <std::size_t Dims = 3, typename Real> auto as_line(const Real *data) {
  return tf::make_line_like(tf::make_point_view<Dims>(data),
                            tf::make_vector_view<Dims>(data + Dims));
}

template <std::size_t Dims = 3, typename Real> auto as_plane(const Real *data) {
  static_assert(Dims == 3, "plane dispatch requires Dims=3");
  return tf::make_plane_like(
      tf::make_unit_vector_view(tf::unsafe, tf::make_vector_view<Dims>(data)),
      data[Dims]);
}

template <std::size_t Dims = 3, typename Real> auto as_aabb(const Real *data) {
  return tf::make_aabb_like(tf::make_point_view<Dims>(data),
                            tf::make_point_view<Dims>(data + Dims));
}

template <std::size_t Dims = 3, typename Real>
auto as_polygon(const Real *data, int vertex_count) {
  return tf::make_polygon(tf::make_points<Dims>(
      tf::make_range(data, static_cast<std::size_t>(vertex_count) * Dims)));
}

template <std::size_t Dims = 3, typename Real, typename Fn>
auto visit_spatial_primitive(Fn &&fn, const Real *data, primitive_kind kind,
                             int polygon_vertex_count)
    -> decltype(fn(as_point<Dims>(data))) {
  switch (kind) {
  case primitive_kind::point:
    return fn(as_point<Dims>(data));
  case primitive_kind::segment:
    return fn(as_segment<Dims>(data));
  case primitive_kind::triangle:
    return fn(as_triangle<Dims>(data));
  case primitive_kind::ray:
    return fn(as_ray<Dims>(data));
  case primitive_kind::line:
    return fn(as_line<Dims>(data));
  case primitive_kind::plane:
    if constexpr (Dims == 3)
      return fn(as_plane<Dims>(data));
    else
      throw std::invalid_argument("spatial operation: plane requires Dims=3");
  case primitive_kind::aabb:
    return fn(as_aabb<Dims>(data));
  case primitive_kind::polygon:
    return fn(as_polygon<Dims>(data, polygon_vertex_count));
  case primitive_kind::vector:
    throw std::invalid_argument("spatial operation: vector is not supported");
  }
  throw std::invalid_argument("spatial operation: unknown primitive kind");
}

template <std::size_t Dims = 3>
inline auto primitive_stride(primitive_kind kind, int polygon_vertex_count)
    -> int {
  switch (kind) {
  case primitive_kind::point:
  case primitive_kind::vector:
    return static_cast<int>(Dims);
  case primitive_kind::segment:
  case primitive_kind::ray:
  case primitive_kind::line:
  case primitive_kind::aabb:
    return static_cast<int>(2 * Dims);
  case primitive_kind::triangle:
    return static_cast<int>(3 * Dims);
  case primitive_kind::plane:
    if constexpr (Dims == 3)
      return 4;
    else
      throw std::invalid_argument("spatial operation: plane requires Dims=3");
  case primitive_kind::polygon:
    return polygon_vertex_count * static_cast<int>(Dims);
  }
  throw std::invalid_argument("spatial operation: unknown primitive kind");
}

inline auto require_spatial(primitive_kind kind) -> void {
  if (kind == primitive_kind::vector)
    throw std::invalid_argument("spatial operation: vector is not supported");
}

template <typename OutputReal, typename InputReal, std::size_t Dims>
auto cast_primitive(const primitive<InputReal, Dims> &input)
    -> primitive<OutputReal, Dims> {
  if constexpr (std::is_same<OutputReal, InputReal>::value) {
    return input.shallow_copy();
  } else {
    const auto source = input.data();
    tf::buffer<OutputReal> buffer;
    buffer.allocate(source.length());
    auto *output = buffer.data();
    const auto *values = source.raw_data();
    const auto length = static_cast<int>(source.length());
    tf::parallel_for_each(tf::make_sequence_range(length), [&](int index) {
      output[index] = static_cast<OutputReal>(values[index]);
    });
    return primitive<OutputReal, Dims>(
        input.kind(), nd_array<OutputReal>::from_buffer(std::move(buffer),
                                                        source.raw_shape()));
  }
}

} // namespace tf::cpp::detail
