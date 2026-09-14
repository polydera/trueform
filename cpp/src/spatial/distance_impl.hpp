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
#include "trueform/cpp/spatial/distance.hpp"

#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/distance.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/spatial/distance.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace detail {

/// The batch cutoffs these queries state for themselves: a distance between two
/// primitives is a handful of arithmetic, a distance to a whole form is a tree
/// descent.
constexpr unsigned long distance_primitive_parallel_threshold = 5000;
constexpr unsigned long distance_form_parallel_threshold = 100;

template <typename Real, std::size_t Dims>
auto require_distance_primitive(const primitive<Real, Dims> &value) -> void {
  require_spatial(value.kind());
  const auto data = value.data();
  if (!data.is_valid())
    throw std::invalid_argument("distance: primitive data is not valid");
  const auto expected =
      value.is_batch()
          ? static_cast<std::size_t>(value.count()) * value.element_stride()
          : value.element_stride();
  if (data.length() != expected)
    throw std::invalid_argument("distance: malformed primitive storage");
}

template <bool Squared, typename Primitive0, typename Primitive1>
auto primitive_distance(const Primitive0 &a, const Primitive1 &b) {
  if constexpr (Squared)
    return tf::distance2(a, b);
  else
    return tf::distance(a, b);
}

template <bool Squared, std::size_t Dims, typename Real>
auto primitive_pair_single(const Real *a, primitive_kind kind_a,
                           int polygon_vertices_a, const Real *b,
                           primitive_kind kind_b, int polygon_vertices_b)
    -> Real {
  return visit_spatial_primitive<Dims>(
      [&](const auto &primitive_a) -> Real {
        return visit_spatial_primitive<Dims>(
            [&](const auto &primitive_b) -> Real {
              return primitive_distance<Squared>(primitive_a, primitive_b);
            },
            b, kind_b, polygon_vertices_b);
      },
      a, kind_a, polygon_vertices_a);
}

template <bool Squared, typename Real, std::size_t Dims>
auto primitive_pair_compute_same(const primitive<Real, Dims> &a,
                                 const primitive<Real, Dims> &b)
    -> distance_result<Real> {
  const auto values_a = a.data();
  const auto values_b = b.data();
  const auto *data_a = values_a.raw_data();
  const auto *data_b = values_b.raw_data();
  const auto polygon_vertices_a = a.polygon_vertex_count();
  const auto polygon_vertices_b = b.polygon_vertex_count();

  if (!a.is_batch() && !b.is_batch())
    return distance_result<Real>(primitive_pair_single<Squared, Dims>(
        data_a, a.kind(), polygon_vertices_a, data_b, b.kind(),
        polygon_vertices_b));

  const auto count = a.is_batch() ? a.count() : b.count();
  const auto stride_a = a.is_batch() ? a.element_stride() : 0;
  const auto stride_b = b.is_batch() ? b.element_stride() : 0;
  tf::buffer<Real> output_buffer;
  output_buffer.allocate(count);
  auto *output = output_buffer.data();
  const auto compute = [&](int index) {
    output[index] = primitive_pair_single<Squared, Dims>(
        data_a + static_cast<std::size_t>(index) * stride_a, a.kind(),
        polygon_vertices_a, data_b + static_cast<std::size_t>(index) * stride_b,
        b.kind(), polygon_vertices_b);
  };
  tf::parallel_for_each(tf::make_sequence_range(count), compute,
                        tf::checked(distance_primitive_parallel_threshold));

  return distance_result<Real>(
      nd_array<Real>::from_buffer(std::move(output_buffer), {count}));
}

template <bool Squared, typename Real0, typename Real1, std::size_t Dims>
auto primitive_pair_compute(const primitive<Real0, Dims> &a,
                            const primitive<Real1, Dims> &b)
    -> distance_result<std::common_type_t<Real0, Real1>> {
  require_distance_primitive(a);
  require_distance_primitive(b);
  if (a.is_batch() && b.is_batch() && a.count() != b.count())
    throw std::invalid_argument(
        "distance: primitive batch lengths must be equal");

  using compute_real = std::common_type_t<Real0, Real1>;
  if constexpr (!std::is_same<Real0, compute_real>::value ||
                !std::is_same<Real1, compute_real>::value) {
    auto converted_a = cast_primitive<compute_real>(a);
    auto converted_b = cast_primitive<compute_real>(b);
    return primitive_pair_compute_same<Squared>(converted_a, converted_b);
  } else {
    return primitive_pair_compute_same<Squared>(a, b);
  }
}

/// The one refusal every distance states of a form it is about to read. The
/// caller names itself, because a user reads back the entry it spelled.
template <typename NativeForm>
auto require_nonempty_form(const NativeForm &form, const char *entry) -> void {
  if (form.empty())
    throw std::invalid_argument(std::string(entry) + ": form is empty");
}

template <bool Squared, typename Form, typename Real, std::size_t Dims>
auto form_primitive_compute_same(const Form &form,
                                 const primitive<Real, Dims> &query)
    -> distance_result<Real> {
  const auto values = query.data();
  const auto *data = values.raw_data();
  const auto kind = query.kind();
  const auto polygon_vertices = query.polygon_vertex_count();
  const auto native_form = form.form();
  require_nonempty_form(native_form, "distance");
  if (!query.is_batch())
    return distance_result<Real>(visit_spatial_primitive<Dims>(
        [&](const auto &primitive_value) -> Real {
          return primitive_distance<Squared>(native_form, primitive_value);
        },
        data, kind, polygon_vertices));

  const auto count = query.count();
  const auto stride = query.element_stride();
  tf::buffer<Real> output_buffer;
  output_buffer.allocate(count);
  auto *output = output_buffer.data();
  const auto compute = [&](int index) {
    output[index] = visit_spatial_primitive<Dims>(
        [&](const auto &primitive_value) -> Real {
          return primitive_distance<Squared>(native_form, primitive_value);
        },
        data + static_cast<std::size_t>(index) * stride, kind,
        polygon_vertices);
  };
  tf::parallel_for_each(tf::make_sequence_range(count), compute,
                        tf::checked(distance_form_parallel_threshold));

  return distance_result<Real>(
      nd_array<Real>::from_buffer(std::move(output_buffer), {count}));
}

/// Whether the call ends before the form is read at all. An empty query batch
/// answers itself, so the structure is never asked to build a tree for it.
template <typename PrimitiveReal, std::size_t Dims>
auto empty_batch_distance(const primitive<PrimitiveReal, Dims> &query) -> bool {
  require_distance_primitive(query);
  return query.is_batch() && query.count() == 0;
}

template <typename Real> auto empty_distance_batch() -> distance_result<Real> {
  tf::buffer<Real> output_buffer;
  output_buffer.allocate(0);
  return distance_result<Real>(
      nd_array<Real>::from_buffer(std::move(output_buffer), {0}));
}

template <bool Squared, typename Form, typename PrimitiveReal, std::size_t Dims>
auto form_primitive_compute(const Form &form,
                            const primitive<PrimitiveReal, Dims> &query)
    -> distance_result<typename Form::real_type> {
  using form_real = typename Form::real_type;
  if (empty_batch_distance(query))
    return empty_distance_batch<form_real>();
  return form_primitive_compute_same<Squared>(form,
                                              cast_primitive<form_real>(query));
}

/// One operand at a time: an empty form ends the call where it is read, so the
/// second structure is never asked for a tree the first refusal made pointless.
template <bool Squared, typename Form0, typename Form1>
auto form_form_compute(const Form0 &a, const Form1 &b)
    -> std::common_type_t<typename Form0::real_type,
                          typename Form1::real_type> {
  using real0 = typename Form0::real_type;
  using real1 = typename Form1::real_type;
  if constexpr (!std::is_same<real0, real1>::value) {
    throw std::invalid_argument("distance: form precision mismatch");
  } else {
    const auto form_a = a.form();
    require_nonempty_form(form_a, "distance");
    const auto form_b = b.form();
    require_nonempty_form(form_b, "distance");
    return primitive_distance<Squared>(form_a, form_b);
  }
}

} // namespace detail
template <typename Real0, typename Real1, std::size_t Dims>
auto distance2(const primitive<Real0, Dims> &a, const primitive<Real1, Dims> &b)
    -> distance_result<std::common_type_t<Real0, Real1>> {
  return detail::primitive_pair_compute<true>(a, b);
}

template <typename Real0, typename Real1, std::size_t Dims>
auto distance(const primitive<Real0, Dims> &a, const primitive<Real1, Dims> &b)
    -> distance_result<std::common_type_t<Real0, Real1>> {
  return detail::primitive_pair_compute<false>(a, b);
}

#define TF_CPP_DEFINE_MESH_PRIMITIVE_DISTANCE(Operation, Squared)              \
  template <typename Index, typename FormReal, std::size_t Dims,               \
            std::size_t Ngon, typename PrimitiveReal>                          \
  auto Operation(const mesh<Index, FormReal, Dims, Ngon> &form,                \
                 const primitive<PrimitiveReal, Dims> &query)                  \
      -> distance_result<FormReal> {                                           \
    return detail::form_primitive_compute<Squared>(form, query);               \
  }

#define TF_CPP_DEFINE_POINT_CLOUD_PRIMITIVE_DISTANCE(Operation, Squared)       \
  template <typename FormReal, std::size_t Dims, typename PrimitiveReal>       \
  auto Operation(const point_cloud<FormReal, Dims> &form,                      \
                 const primitive<PrimitiveReal, Dims> &query)                  \
      -> distance_result<FormReal> {                                           \
    return detail::form_primitive_compute<Squared>(form, query);               \
  }

TF_CPP_DEFINE_MESH_PRIMITIVE_DISTANCE(distance2, true)
TF_CPP_DEFINE_MESH_PRIMITIVE_DISTANCE(distance, false)
TF_CPP_DEFINE_POINT_CLOUD_PRIMITIVE_DISTANCE(distance2, true)
TF_CPP_DEFINE_POINT_CLOUD_PRIMITIVE_DISTANCE(distance, false)

#undef TF_CPP_DEFINE_POINT_CLOUD_PRIMITIVE_DISTANCE
#undef TF_CPP_DEFINE_MESH_PRIMITIVE_DISTANCE

#define TF_CPP_DEFINE_MESH_MESH_DISTANCE(Operation, Squared)                   \
  template <typename Index0, typename Real0, typename Index1, typename Real1,  \
            std::size_t Dims, std::size_t Ngon0, std::size_t Ngon1>            \
  auto Operation(const mesh<Index0, Real0, Dims, Ngon0> &a,                    \
                 const mesh<Index1, Real1, Dims, Ngon1> &b)                    \
      -> std::common_type_t<Real0, Real1> {                                    \
    return detail::form_form_compute<Squared>(a, b);                           \
  }

#define TF_CPP_DEFINE_MESH_POINT_CLOUD_DISTANCE(Operation, Squared)            \
  template <typename Index0, typename Real0, typename Real1, std::size_t Dims, \
            std::size_t Ngon0>                                                 \
  auto Operation(const mesh<Index0, Real0, Dims, Ngon0> &a,                    \
                 const point_cloud<Real1, Dims> &b)                            \
      -> std::common_type_t<Real0, Real1> {                                    \
    return detail::form_form_compute<Squared>(a, b);                           \
  }

#define TF_CPP_DEFINE_POINT_CLOUD_MESH_DISTANCE(Operation, Squared)            \
  template <typename Real0, typename Index1, typename Real1, std::size_t Dims, \
            std::size_t Ngon1>                                                 \
  auto Operation(const point_cloud<Real0, Dims> &a,                            \
                 const mesh<Index1, Real1, Dims, Ngon1> &b)                    \
      -> std::common_type_t<Real0, Real1> {                                    \
    return detail::form_form_compute<Squared>(a, b);                           \
  }

#define TF_CPP_DEFINE_POINT_CLOUD_POINT_CLOUD_DISTANCE(Operation, Squared)     \
  template <typename Real0, typename Real1, std::size_t Dims>                  \
  auto Operation(const point_cloud<Real0, Dims> &a,                            \
                 const point_cloud<Real1, Dims> &b)                            \
      -> std::common_type_t<Real0, Real1> {                                    \
    return detail::form_form_compute<Squared>(a, b);                           \
  }

#define TF_CPP_DEFINE_FORM_FORM_DISTANCE_MATRIX(Operation, Squared)            \
  TF_CPP_DEFINE_MESH_MESH_DISTANCE(Operation, Squared)                         \
  TF_CPP_DEFINE_MESH_POINT_CLOUD_DISTANCE(Operation, Squared)                  \
  TF_CPP_DEFINE_POINT_CLOUD_MESH_DISTANCE(Operation, Squared)                  \
  TF_CPP_DEFINE_POINT_CLOUD_POINT_CLOUD_DISTANCE(Operation, Squared)

TF_CPP_DEFINE_FORM_FORM_DISTANCE_MATRIX(distance2, true)
TF_CPP_DEFINE_FORM_FORM_DISTANCE_MATRIX(distance, false)

#undef TF_CPP_DEFINE_FORM_FORM_DISTANCE_MATRIX
#undef TF_CPP_DEFINE_POINT_CLOUD_POINT_CLOUD_DISTANCE
#undef TF_CPP_DEFINE_POINT_CLOUD_MESH_DISTANCE
#undef TF_CPP_DEFINE_MESH_POINT_CLOUD_DISTANCE
#undef TF_CPP_DEFINE_MESH_MESH_DISTANCE

} // namespace tf::cpp
