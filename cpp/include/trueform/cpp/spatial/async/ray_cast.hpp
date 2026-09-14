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

#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/spatial/detail/spatial_carrier.hpp"
#include "trueform/cpp/spatial/primitive.hpp"
#include "trueform/cpp/spatial/ray_cast.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

/// @brief Resolve a primitive ray cast on the common executor.
template <typename Resolver, typename RayReal, typename TargetReal>
auto ray_cast(Resolver &&resolver, const primitive<RayReal> &rays,
              const primitive<TargetReal> &target,
              const ray_cast_options<std::common_type_t<RayReal, TargetReal>>
                  &options = {}) {
  using Real = std::common_type_t<RayReal, TargetReal>;
  auto owned_rays = rays;
  auto owned_target = target;
  auto owned_options = options;
  return submit<ray_cast_primitive_result<Real>>(
      std::forward<Resolver>(resolver),
      [rays = std::move(owned_rays), target = std::move(owned_target),
       options = std::move(owned_options)] {
        return cpp::ray_cast(rays, target, options);
      });
}

template <typename RayReal, typename TargetReal>
auto ray_cast(const primitive<RayReal> &rays,
              const primitive<TargetReal> &target,
              const ray_cast_options<std::common_type_t<RayReal, TargetReal>>
                  &options = {})
    -> std::future<
        ray_cast_primitive_result<std::common_type_t<RayReal, TargetReal>>> {
  return async::ray_cast(future_resolver{}, rays, target, options);
}

/// @brief Resolve a 2D primitive ray cast on the common executor.
template <typename Resolver, typename RayReal, typename TargetReal,
          std::size_t Dims, std::enable_if_t<Dims == 2, int> = 0>
auto ray_cast(Resolver &&resolver, const primitive<RayReal, Dims> &rays,
              const primitive<TargetReal, Dims> &target,
              const ray_cast_options<std::common_type_t<RayReal, TargetReal>>
                  &options = {}) {
  using Real = std::common_type_t<RayReal, TargetReal>;
  auto owned_rays = rays;
  auto owned_target = target;
  auto owned_options = options;
  return submit<ray_cast_primitive_result<Real>>(
      std::forward<Resolver>(resolver),
      [rays = std::move(owned_rays), target = std::move(owned_target),
       options = std::move(owned_options)] {
        return cpp::ray_cast(rays, target, options);
      });
}

template <typename RayReal, typename TargetReal, std::size_t Dims,
          std::enable_if_t<Dims == 2, int> = 0>
auto ray_cast(const primitive<RayReal, Dims> &rays,
              const primitive<TargetReal, Dims> &target,
              const ray_cast_options<std::common_type_t<RayReal, TargetReal>>
                  &options = {})
    -> std::future<
        ray_cast_primitive_result<std::common_type_t<RayReal, TargetReal>>> {
  return async::ray_cast(future_resolver{}, rays, target, options);
}

template <
    typename Resolver, typename RayReal, std::size_t Dims, typename Form,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto ray_cast(Resolver &&resolver, const primitive<RayReal, Dims> &rays,
              const Form &form,
              const ray_cast_options<typename Form::real_type> &options = {}) {
  using result_type = decltype(cpp::ray_cast(rays, form, options));
  return submit<result_type>(
      std::forward<Resolver>(resolver),
      [rays, form, options] { return cpp::ray_cast(rays, form, options); });
}

template <
    typename RayReal, std::size_t Dims, typename Form,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto ray_cast(const primitive<RayReal, Dims> &rays, const Form &form,
              const ray_cast_options<typename Form::real_type> &options = {})
    -> std::future<decltype(cpp::ray_cast(rays, form, options))> {
  return async::ray_cast(future_resolver{}, rays, form, options);
}

} // namespace tf::cpp::async
