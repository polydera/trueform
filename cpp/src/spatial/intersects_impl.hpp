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
#include "trueform/cpp/spatial/intersects.hpp"

#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/intersects.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/spatial/intersects.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace detail {

/// The batch cutoffs these queries state for themselves: a primitive against
/// one primitive is a handful of arithmetic, a primitive against a whole form
/// is a tree descent.
constexpr unsigned long intersects_primitive_parallel_threshold = 1000;
constexpr unsigned long intersects_form_parallel_threshold = 100;

template <std::size_t Dims, typename Real0, typename Real1>
auto primitive_pair_single(const Real0 *a, primitive_kind kind_a,
                           int polygon_vertices_a, const Real1 *b,
                           primitive_kind kind_b, int polygon_vertices_b)
    -> bool {
  return visit_spatial_primitive<Dims>(
      [&](const auto &primitive_a) {
        return visit_spatial_primitive<Dims>(
            [&](const auto &primitive_b) {
              return tf::intersects(primitive_a, primitive_b);
            },
            b, kind_b, polygon_vertices_b);
      },
      a, kind_a, polygon_vertices_a);
}

template <typename Form, typename Real, std::size_t Dims>
auto form_primitive_compute_same(const Form &form,
                                 const primitive<Real, Dims> &query)
    -> intersection_result {
  const auto values = query.data();
  const auto *data = values.raw_data();
  const auto kind = query.kind();
  const auto polygon_vertices = query.polygon_vertex_count();
  const auto native_form = form.form();
  if (!query.is_batch())
    return intersection_result(visit_spatial_primitive<Dims>(
        [&](const auto &primitive_value) {
          return tf::intersects(native_form, primitive_value);
        },
        data, kind, polygon_vertices));

  const auto count = query.count();
  const auto stride = query.element_stride();
  tf::buffer<std::int8_t> output_buffer;
  output_buffer.allocate(count);
  auto *output = output_buffer.data();

  const auto compute = [&](int index) {
    output[index] = visit_spatial_primitive<Dims>(
                        [&](const auto &primitive_value) {
                          return tf::intersects(native_form, primitive_value);
                        },
                        data + static_cast<std::size_t>(index) * stride, kind,
                        polygon_vertices)
                        ? 1
                        : 0;
  };
  tf::parallel_for_each(tf::make_sequence_range(count), compute,
                        tf::checked(intersects_form_parallel_threshold));

  return intersection_result(
      nd_array<std::int8_t>::from_buffer(std::move(output_buffer), {count}));
}

template <typename Form, typename PrimitiveReal, std::size_t Dims>
auto form_primitive_compute(const Form &form,
                            const primitive<PrimitiveReal, Dims> &query)
    -> intersection_result {
  require_spatial(query.kind());
  auto converted = cast_primitive<typename Form::real_type>(query);
  return form_primitive_compute_same(form, converted);
}

template <typename Form0, typename Form1>
auto form_form_compute(const Form0 &a, const Form1 &b) -> intersection_result {
  using real0 = typename Form0::real_type;
  using real1 = typename Form1::real_type;
  if constexpr (!std::is_same<real0, real1>::value)
    throw std::invalid_argument("intersects: form precision mismatch");
  else
    return intersection_result(tf::intersects(a.form(), b.form()));
}

template <typename Real0, typename Real1, std::size_t Dims>
auto primitive_pair_compute(const primitive<Real0, Dims> &a,
                            const primitive<Real1, Dims> &b)
    -> intersection_result {
  using compute_real = std::common_type_t<Real0, Real1>;
  if constexpr (!std::is_same<Real0, compute_real>::value ||
                !std::is_same<Real1, compute_real>::value) {
    auto converted_a = cast_primitive<compute_real>(a);
    auto converted_b = cast_primitive<compute_real>(b);
    return primitive_pair_compute<compute_real, compute_real, Dims>(
        converted_a, converted_b);
  }

  require_spatial(a.kind());
  require_spatial(b.kind());

  if (a.is_batch() && b.is_batch() && a.count() != b.count())
    throw std::invalid_argument(
        "intersects: primitive batch lengths must be equal");

  const auto values_a = a.data();
  const auto values_b = b.data();
  const auto *data_a = values_a.raw_data();
  const auto *data_b = values_b.raw_data();
  const auto polygon_vertices_a = a.polygon_vertex_count();
  const auto polygon_vertices_b = b.polygon_vertex_count();

  if (!a.is_batch() && !b.is_batch())
    return intersection_result(
        primitive_pair_single<Dims>(data_a, a.kind(), polygon_vertices_a,
                                    data_b, b.kind(), polygon_vertices_b));

  const auto count = a.is_batch() ? a.count() : b.count();
  const auto stride_a = a.is_batch() ? a.element_stride() : 0;
  const auto stride_b = b.is_batch() ? b.element_stride() : 0;

  tf::buffer<std::int8_t> output_buffer;
  output_buffer.allocate(count);
  auto *output = output_buffer.data();
  const auto compute = [&](int index) {
    output[index] = primitive_pair_single<Dims>(
                        data_a + static_cast<std::size_t>(index) * stride_a,
                        a.kind(), polygon_vertices_a,
                        data_b + static_cast<std::size_t>(index) * stride_b,
                        b.kind(), polygon_vertices_b)
                        ? 1
                        : 0;
  };
  tf::parallel_for_each(tf::make_sequence_range(count), compute,
                        tf::checked(intersects_primitive_parallel_threshold));

  return intersection_result(
      nd_array<std::int8_t>::from_buffer(std::move(output_buffer), {count}));
}

} // namespace detail

template <typename Real0, typename Real1, std::size_t Dims>
auto intersects(const primitive<Real0, Dims> &a,
                const primitive<Real1, Dims> &b) -> intersection_result {
  return detail::primitive_pair_compute<Real0, Real1, Dims>(a, b);
}

template <typename Index, typename FormReal, std::size_t Dims, std::size_t Ngon,
          typename PrimitiveReal>
auto intersects(const mesh<Index, FormReal, Dims, Ngon> &form,
                const primitive<PrimitiveReal, Dims> &query)
    -> intersection_result {
  return detail::form_primitive_compute(form, query);
}

template <typename Index, typename FormReal, std::size_t Dims,
          typename PrimitiveReal>
auto intersects(const edge_mesh<Index, FormReal, Dims> &form,
                const primitive<PrimitiveReal, Dims> &query)
    -> intersection_result {
  return detail::form_primitive_compute(form, query);
}

template <typename FormReal, std::size_t Dims, typename PrimitiveReal>
auto intersects(const point_cloud<FormReal, Dims> &form,
                const primitive<PrimitiveReal, Dims> &query)
    -> intersection_result {
  return detail::form_primitive_compute(form, query);
}

template <typename PrimitiveReal, typename Index, typename FormReal,
          std::size_t Dims, std::size_t Ngon>
auto intersects(const primitive<PrimitiveReal, Dims> &query,
                const mesh<Index, FormReal, Dims, Ngon> &form)
    -> intersection_result {
  return cpp::intersects(form, query);
}

template <typename PrimitiveReal, typename Index, typename FormReal,
          std::size_t Dims>
auto intersects(const primitive<PrimitiveReal, Dims> &query,
                const edge_mesh<Index, FormReal, Dims> &form)
    -> intersection_result {
  return cpp::intersects(form, query);
}

template <typename PrimitiveReal, typename FormReal, std::size_t Dims>
auto intersects(const primitive<PrimitiveReal, Dims> &query,
                const point_cloud<FormReal, Dims> &form)
    -> intersection_result {
  return cpp::intersects(form, query);
}

template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon0, std::size_t Ngon1>
auto intersects(const mesh<Index0, Real0, Dims, Ngon0> &a,
                const mesh<Index1, Real1, Dims, Ngon1> &b)
    -> intersection_result {
  return detail::form_form_compute(a, b);
}

template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon0>
auto intersects(const mesh<Index0, Real0, Dims, Ngon0> &a,
                const edge_mesh<Index1, Real1, Dims> &b)
    -> intersection_result {
  return detail::form_form_compute(a, b);
}

template <typename Index0, typename Real0, typename Real1, std::size_t Dims,
          std::size_t Ngon0>
auto intersects(const mesh<Index0, Real0, Dims, Ngon0> &a,
                const point_cloud<Real1, Dims> &b) -> intersection_result {
  return detail::form_form_compute(a, b);
}

template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon1>
auto intersects(const edge_mesh<Index0, Real0, Dims> &a,
                const mesh<Index1, Real1, Dims, Ngon1> &b)
    -> intersection_result {
  return detail::form_form_compute(a, b);
}

template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims>
auto intersects(const edge_mesh<Index0, Real0, Dims> &a,
                const edge_mesh<Index1, Real1, Dims> &b)
    -> intersection_result {
  return detail::form_form_compute(a, b);
}

template <typename Index0, typename Real0, typename Real1, std::size_t Dims>
auto intersects(const edge_mesh<Index0, Real0, Dims> &a,
                const point_cloud<Real1, Dims> &b) -> intersection_result {
  return detail::form_form_compute(a, b);
}

template <typename Real0, typename Index1, typename Real1, std::size_t Dims,
          std::size_t Ngon1>
auto intersects(const point_cloud<Real0, Dims> &a,
                const mesh<Index1, Real1, Dims, Ngon1> &b)
    -> intersection_result {
  return detail::form_form_compute(a, b);
}

template <typename Real0, typename Index1, typename Real1, std::size_t Dims>
auto intersects(const point_cloud<Real0, Dims> &a,
                const edge_mesh<Index1, Real1, Dims> &b)
    -> intersection_result {
  return detail::form_form_compute(a, b);
}

template <typename Real0, typename Real1, std::size_t Dims>
auto intersects(const point_cloud<Real0, Dims> &a,
                const point_cloud<Real1, Dims> &b) -> intersection_result {
  return detail::form_form_compute(a, b);
}

} // namespace tf::cpp
