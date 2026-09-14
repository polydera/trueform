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

#include "primitive_dispatch.hpp"
#include "trueform/cpp/spatial/closest_metric_point_pair.hpp"

#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/closest_metric_point_pair.hpp"
#include "trueform/core/views/sequence_range.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace tf::cpp {
namespace detail {

/// The batch cutoff this query states for itself: one primitive pair is a
/// handful of arithmetic, so the batch length is the work.
constexpr unsigned long closest_metric_point_pair_parallel_threshold = 1000;

template <typename Real, std::size_t Dims, typename PointLike>
auto closest_pair_point_array(const PointLike &point) -> nd_array<Real> {
  tf::buffer<Real> output;
  output.allocate(Dims);
  for (std::size_t dim = 0; dim < Dims; ++dim)
    output[dim] = static_cast<Real>(point[dim]);
  return nd_array<Real>::from_buffer(std::move(output),
                                     {static_cast<int>(Dims)});
}

template <std::size_t Dims, typename Real>
auto closest_metric_point_pair_single(const Real *a, primitive_kind kind_a,
                                      int polygon_vertices_a, const Real *b,
                                      primitive_kind kind_b,
                                      int polygon_vertices_b)
    -> tf::metric_point_pair<Real, Dims> {
  return visit_spatial_primitive<Dims>(
      [&](const auto &primitive_a) {
        return visit_spatial_primitive<Dims>(
            [&](const auto &primitive_b) {
              return tf::closest_metric_point_pair(primitive_a, primitive_b);
            },
            b, kind_b, polygon_vertices_b);
      },
      a, kind_a, polygon_vertices_a);
}

template <typename Real, std::size_t Dims>
auto closest_metric_point_pair_compute_same(const primitive<Real, Dims> &a,
                                            const primitive<Real, Dims> &b)
    -> std::variant<closest_metric_point_pair_result<Real>,
                    closest_metric_point_pair_batch_result<Real>> {
  const auto values_a = a.data();
  const auto values_b = b.data();
  const auto *data_a = values_a.raw_data();
  const auto *data_b = values_b.raw_data();
  const auto polygon_vertices_a = a.polygon_vertex_count();
  const auto polygon_vertices_b = b.polygon_vertex_count();

  if (!a.is_batch() && !b.is_batch()) {
    const auto result = closest_metric_point_pair_single<Dims>(
        data_a, a.kind(), polygon_vertices_a, data_b, b.kind(),
        polygon_vertices_b);
    return closest_metric_point_pair_result<Real>{
        closest_pair_point_array<Real, Dims>(result.first),
        closest_pair_point_array<Real, Dims>(result.second), result.metric};
  }

  const auto count = a.is_batch() ? a.count() : b.count();
  const auto stride_a = a.is_batch() ? a.element_stride() : 0;
  const auto stride_b = b.is_batch() ? b.element_stride() : 0;

  tf::buffer<Real> point0_buffer;
  point0_buffer.allocate(static_cast<std::size_t>(count) * Dims);
  tf::buffer<Real> point1_buffer;
  point1_buffer.allocate(static_cast<std::size_t>(count) * Dims);
  tf::buffer<Real> distance_buffer;
  distance_buffer.allocate(count);
  auto *points0 = point0_buffer.data();
  auto *points1 = point1_buffer.data();
  auto *distances = distance_buffer.data();

  const auto compute = [&](int index) {
    const auto offset_a = static_cast<std::size_t>(index) * stride_a;
    const auto offset_b = static_cast<std::size_t>(index) * stride_b;
    const auto result = closest_metric_point_pair_single<Dims>(
        data_a + offset_a, a.kind(), polygon_vertices_a, data_b + offset_b,
        b.kind(), polygon_vertices_b);
    auto *output0 = points0 + static_cast<std::size_t>(index) * Dims;
    auto *output1 = points1 + static_cast<std::size_t>(index) * Dims;
    for (std::size_t dim = 0; dim < Dims; ++dim) {
      output0[dim] = result.first[dim];
      output1[dim] = result.second[dim];
    }
    distances[index] = result.metric;
  };
  tf::parallel_for_each(tf::make_sequence_range(count), compute,
                        tf::checked(closest_metric_point_pair_parallel_threshold));

  return closest_metric_point_pair_batch_result<Real>{
      nd_array<Real>::from_buffer(std::move(point0_buffer),
                                  {count, static_cast<int>(Dims)}),
      nd_array<Real>::from_buffer(std::move(point1_buffer),
                                  {count, static_cast<int>(Dims)}),
      nd_array<Real>::from_buffer(std::move(distance_buffer), {count})};
}

template <typename Real0, typename Real1, std::size_t Dims>
auto closest_metric_point_pair_compute(const primitive<Real0, Dims> &a,
                                       const primitive<Real1, Dims> &b)
    -> std::variant<
        closest_metric_point_pair_result<std::common_type_t<Real0, Real1>>,
        closest_metric_point_pair_batch_result<
            std::common_type_t<Real0, Real1>>> {
  require_spatial(a.kind());
  require_spatial(b.kind());
  if (a.is_batch() && b.is_batch() && a.count() != b.count())
    throw std::invalid_argument(
        "closest_metric_point_pair: primitive batch lengths must be equal");

  using compute_real = std::common_type_t<Real0, Real1>;
  if constexpr (!std::is_same<Real0, compute_real>::value ||
                !std::is_same<Real1, compute_real>::value) {
    auto converted_a = cast_primitive<compute_real>(a);
    auto converted_b = cast_primitive<compute_real>(b);
    return closest_metric_point_pair_compute_same(converted_a, converted_b);
  } else {
    return closest_metric_point_pair_compute_same(a, b);
  }
}

} // namespace detail

template <typename Real0, typename Real1, std::size_t Dims>
auto closest_metric_point_pair(const primitive<Real0, Dims> &a,
                               const primitive<Real1, Dims> &b)
    -> std::variant<
        closest_metric_point_pair_result<std::common_type_t<Real0, Real1>>,
        closest_metric_point_pair_batch_result<
            std::common_type_t<Real0, Real1>>> {
  return detail::closest_metric_point_pair_compute<Real0, Real1, Dims>(a, b);
}

} // namespace tf::cpp
