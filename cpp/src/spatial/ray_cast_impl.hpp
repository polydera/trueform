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

#include "./primitive_dispatch.hpp"
#include "trueform/cpp/spatial/ray_cast.hpp"

#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/ray_cast.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/spatial/ray_cast.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace detail {

/// The batch cutoffs these casts state for themselves: a ray against one
/// primitive is a handful of arithmetic, a ray against a whole form is a tree
/// descent, so the two are an order of magnitude apart in what one element
/// costs.
constexpr unsigned long ray_cast_primitive_parallel_threshold = 1000;
constexpr unsigned long ray_cast_form_parallel_threshold = 100;

template <typename Real> class ray_bounds {
  const Real *_min_ts = nullptr;
  const Real *_max_ts = nullptr;
  Real _min_t;
  Real _max_t;

  static auto validate(const nd_array<Real> &values, int count,
                       const char *name) -> void {
    if (!values.is_valid())
      return;
    if (values.ndim() != 1 || values.shape_at(0) != count)
      throw std::invalid_argument(std::string("ray_cast: ") + name +
                                  " must have shape [result_count]");
  }

public:
  ray_bounds(const ray_cast_options<Real> &options, int count)
      : _min_t(options.min_t), _max_t(options.max_t) {
    validate(options.min_ts, count, "min_ts");
    validate(options.max_ts, count, "max_ts");
    if (options.min_ts.is_valid())
      _min_ts = options.min_ts.raw_data();
    if (options.max_ts.is_valid())
      _max_ts = options.max_ts.raw_data();
  }

  auto min_t(int index) const -> Real {
    return _min_ts ? _min_ts[index] : _min_t;
  }
  auto max_t(int index) const -> Real {
    return _max_ts ? _max_ts[index] : _max_t;
  }
};

template <std::size_t Dims, typename Real>
auto primitive_single(const Real *ray_data, const Real *target_data,
                      primitive_kind target_kind, int polygon_vertex_count,
                      Real min_t, Real max_t)
    -> ray_cast_result<default_index_t, Real> {
  const auto ray = as_ray<Dims>(ray_data);
  const auto config = tf::make_ray_config(min_t, max_t);
  return visit_spatial_primitive<Dims>(
      [&](const auto &target) {
        const auto result = tf::ray_cast(ray, target, config);
        return ray_cast_result<default_index_t, Real>{
            min_t <= max_t && static_cast<bool>(result), result.t, -1};
      },
      target_data, target_kind, polygon_vertex_count);
}

template <typename Real, std::size_t Dims>
auto primitive_compute(const primitive<Real, Dims> &rays,
                       const primitive<Real, Dims> &target,
                       const ray_cast_options<Real> &options)
    -> ray_cast_primitive_result<Real> {
  if (rays.kind() != primitive_kind::ray)
    throw std::invalid_argument("ray_cast: first primitive must be a ray");
  require_spatial(target.kind());
  if (rays.is_batch() && target.is_batch() && rays.count() != target.count())
    throw std::invalid_argument(
        "ray_cast: primitive batch lengths must be equal");

  const auto count = rays.is_batch() ? rays.count() : target.count();
  const ray_bounds<Real> bounds(options, count);
  const auto ray_values = rays.data();
  const auto target_values = target.data();
  const auto *ray_data = ray_values.raw_data();
  const auto *target_data = target_values.raw_data();
  const auto target_vertices = target.polygon_vertex_count();

  if (!rays.is_batch() && !target.is_batch())
    return primitive_single<Dims>(ray_data, target_data, target.kind(),
                                  target_vertices, bounds.min_t(0),
                                  bounds.max_t(0));

  const auto ray_stride = rays.is_batch() ? rays.element_stride() : 0;
  const auto target_stride = target.is_batch() ? target.element_stride() : 0;
  tf::buffer<std::int8_t> hit_buffer;
  tf::buffer<Real> t_buffer;
  hit_buffer.allocate(count);
  t_buffer.allocate(count);
  auto *hits = hit_buffer.data();
  auto *ts = t_buffer.data();

  const auto compute = [&](int index) {
    const auto result = primitive_single<Dims>(
        ray_data + index * ray_stride, target_data + index * target_stride,
        target.kind(), target_vertices, bounds.min_t(index),
        bounds.max_t(index));
    hits[index] = result.hit ? 1 : 0;
    ts[index] = result.t;
  };
  tf::parallel_for_each(tf::make_sequence_range(count), compute,
                        tf::checked(ray_cast_primitive_parallel_threshold));

  return ray_cast_primitive_batch_result<Real>{
      nd_array<std::int8_t>::from_buffer(std::move(hit_buffer), {count}),
      nd_array<Real>::from_buffer(std::move(t_buffer), {count})};
}

template <typename Form> struct ray_cast_form_index;

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
struct ray_cast_form_index<cpp::mesh<Index, Real, Dims, Ngon>> {
  using type = Index;
};

template <typename Real, std::size_t Dims>
struct ray_cast_form_index<cpp::point_cloud<Real, Dims>> {
  using type = std::int32_t;
};

template <typename Index, typename Real, std::size_t Dims>
struct ray_cast_form_index<cpp::edge_mesh<Index, Real, Dims>> {
  using type = Index;
};

template <typename Real, std::size_t Dims, typename Form>
auto form_compute(const primitive<Real, Dims> &rays, const Form &form,
                  const ray_cast_options<Real> &options)
    -> ray_cast_form_result<typename ray_cast_form_index<Form>::type, Real> {
  using Index = typename ray_cast_form_index<Form>::type;
  if (rays.kind() != primitive_kind::ray)
    throw std::invalid_argument("ray_cast: first primitive must be a ray");

  const auto native_form = form.form();
  const auto count = rays.count();
  const ray_bounds<Real> bounds(options, count);
  const auto values = rays.data();
  const auto *data = values.raw_data();
  const auto ray_stride = rays.element_stride();

  if (!rays.is_batch()) {
    const auto min_t = bounds.min_t(0);
    const auto max_t = bounds.max_t(0);
    const auto result = tf::ray_cast(as_ray<Dims>(data), native_form,
                                     tf::make_ray_config(min_t, max_t));
    const auto hit = min_t <= max_t && static_cast<bool>(result);
    return ray_cast_result<Index, Real>{hit, result.info.t,
                                        hit ? static_cast<Index>(result.element)
                                            : Index{-1}};
  }

  tf::buffer<std::int8_t> hit_buffer;
  tf::buffer<Real> t_buffer;
  tf::buffer<Index> id_buffer;
  hit_buffer.allocate(count);
  t_buffer.allocate(count);
  id_buffer.allocate(count);
  auto *hits = hit_buffer.data();
  auto *ts = t_buffer.data();
  auto *element_ids = id_buffer.data();

  const auto compute = [&](int index) {
    const auto min_t = bounds.min_t(index);
    const auto max_t = bounds.max_t(index);
    const auto result =
        tf::ray_cast(as_ray<Dims>(data + index * ray_stride), native_form,
                     tf::make_ray_config(min_t, max_t));
    const auto hit = min_t <= max_t && static_cast<bool>(result);
    hits[index] = hit ? 1 : 0;
    ts[index] = result.info.t;
    element_ids[index] = hit ? static_cast<Index>(result.element) : Index{-1};
  };
  tf::parallel_for_each(tf::make_sequence_range(count), compute,
                        tf::checked(ray_cast_form_parallel_threshold));

  return ray_cast_form_batch_result<Index, Real>{
      nd_array<std::int8_t>::from_buffer(std::move(hit_buffer), {count}),
      nd_array<Real>::from_buffer(std::move(t_buffer), {count}),
      nd_array<Index>::from_buffer(std::move(id_buffer), {count})};
}

} // namespace detail

template <typename RayReal, typename TargetReal, std::size_t Dims>
auto ray_cast(
    const primitive<RayReal, Dims> &rays,
    const primitive<TargetReal, Dims> &target,
    const ray_cast_options<std::common_type_t<RayReal, TargetReal>> &options)
    -> ray_cast_primitive_result<std::common_type_t<RayReal, TargetReal>> {
  using compute_real = std::common_type_t<RayReal, TargetReal>;
  auto converted_rays = detail::cast_primitive<compute_real>(rays);
  auto converted_target = detail::cast_primitive<compute_real>(target);
  return detail::primitive_compute(converted_rays, converted_target, options);
}

template <typename RayReal, typename Index, typename FormReal, std::size_t Dims,
          std::size_t Ngon>
auto ray_cast(const primitive<RayReal, Dims> &rays,
              const mesh<Index, FormReal, Dims, Ngon> &form,
              const ray_cast_options<FormReal> &options)
    -> ray_cast_form_result<Index, FormReal> {
  return detail::form_compute(detail::cast_primitive<FormReal>(rays), form,
                              options);
}

template <typename RayReal, typename FormReal, std::size_t Dims>
auto ray_cast(const primitive<RayReal, Dims> &rays,
              const point_cloud<FormReal, Dims> &form,
              const ray_cast_options<FormReal> &options)
    -> ray_cast_form_result<std::int32_t, FormReal> {
  return detail::form_compute(detail::cast_primitive<FormReal>(rays), form,
                              options);
}

template <typename RayReal, typename Index, typename FormReal, std::size_t Dims>
auto ray_cast(const primitive<RayReal, Dims> &rays,
              const edge_mesh<Index, FormReal, Dims> &form,
              const ray_cast_options<FormReal> &options)
    -> ray_cast_form_result<Index, FormReal> {
  return detail::form_compute(detail::cast_primitive<FormReal>(rays), form,
                              options);
}

} // namespace tf::cpp
