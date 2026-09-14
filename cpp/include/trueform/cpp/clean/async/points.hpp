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

#include "trueform/cpp/clean/points.hpp"
#include "trueform/cpp/clean/supported.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/point_cloud.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

// An array overload requires explicit dimensions because nd_array stores rank
// and shape at runtime.
template <typename Real, std::size_t Dims = 3, typename Resolver,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points(Resolver &&resolver, const nd_array<Real> &points,
                    Real tolerance = Real{},
                    bool remove_duplicate_primitives = true,
                    bool remove_unreferenced_points = true)
    -> resolver_result_t<Resolver, nd_array<Real>> {
  return submit<nd_array<Real>>(std::forward<Resolver>(resolver),
                                [points, tolerance, remove_duplicate_primitives,
                                 remove_unreferenced_points] {
                                  return cpp::cleaned_points<Real, Dims>(
                                      points, tolerance,
                                      remove_duplicate_primitives,
                                      remove_unreferenced_points);
                                });
}

template <typename Real, std::size_t Dims = 3,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points(const nd_array<Real> &points, Real tolerance = Real{},
                    bool remove_duplicate_primitives = true,
                    bool remove_unreferenced_points = true)
    -> std::future<nd_array<Real>> {
  return async::cleaned_points<Real, Dims>(future_resolver{}, points, tolerance,
                                           remove_duplicate_primitives,
                                           remove_unreferenced_points);
}

template <typename Real, std::size_t Dims = 3, typename Resolver,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points_with_map(Resolver &&resolver, const nd_array<Real> &points,
                             Real tolerance = Real{},
                             bool remove_duplicate_primitives = true,
                             bool remove_unreferenced_points = true)
    -> resolver_result_t<Resolver,
                         cleaned_points_result<default_index_t, Real>> {
  return submit<cleaned_points_result<default_index_t, Real>>(
      std::forward<Resolver>(resolver),
      [points, tolerance, remove_duplicate_primitives,
       remove_unreferenced_points] {
        return cpp::cleaned_points_with_map<Real, Dims>(
            points, tolerance, remove_duplicate_primitives,
            remove_unreferenced_points);
      });
}

template <typename Real, std::size_t Dims = 3,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points_with_map(const nd_array<Real> &points,
                             Real tolerance = Real{},
                             bool remove_duplicate_primitives = true,
                             bool remove_unreferenced_points = true)
    -> std::future<cleaned_points_result<default_index_t, Real>> {
  return async::cleaned_points_with_map<Real, Dims>(
      future_resolver{}, points, tolerance, remove_duplicate_primitives,
      remove_unreferenced_points);
}

template <typename Resolver, typename Real, std::size_t Dims,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points(Resolver &&resolver, const point_cloud<Real, Dims> &value,
                    Real tolerance = Real{},
                    bool remove_duplicate_primitives = true,
                    bool remove_unreferenced_points = true)
    -> resolver_result_t<Resolver, nd_array<Real>> {
  return submit<nd_array<Real>>(std::forward<Resolver>(resolver),
                                [value, tolerance, remove_duplicate_primitives,
                                 remove_unreferenced_points] {
                                  return cpp::cleaned_points(
                                      value, tolerance,
                                      remove_duplicate_primitives,
                                      remove_unreferenced_points);
                                });
}

template <typename Real, std::size_t Dims,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points(const point_cloud<Real, Dims> &value,
                    Real tolerance = Real{},
                    bool remove_duplicate_primitives = true,
                    bool remove_unreferenced_points = true)
    -> std::future<nd_array<Real>> {
  return async::cleaned_points(future_resolver{}, value, tolerance,
                               remove_duplicate_primitives,
                               remove_unreferenced_points);
}

template <typename Resolver, typename Real, std::size_t Dims,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points_with_map(Resolver &&resolver,
                             const point_cloud<Real, Dims> &value,
                             Real tolerance = Real{},
                             bool remove_duplicate_primitives = true,
                             bool remove_unreferenced_points = true)
    -> resolver_result_t<Resolver,
                         cleaned_points_result<default_index_t, Real>> {
  return submit<cleaned_points_result<default_index_t, Real>>(
      std::forward<Resolver>(resolver),
      [value, tolerance, remove_duplicate_primitives,
       remove_unreferenced_points] {
        return cpp::cleaned_points_with_map(value, tolerance,
                                            remove_duplicate_primitives,
                                            remove_unreferenced_points);
      });
}

template <typename Real, std::size_t Dims,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points_with_map(const point_cloud<Real, Dims> &value,
                             Real tolerance = Real{},
                             bool remove_duplicate_primitives = true,
                             bool remove_unreferenced_points = true)
    -> std::future<cleaned_points_result<default_index_t, Real>> {
  return async::cleaned_points_with_map(future_resolver{}, value, tolerance,
                                        remove_duplicate_primitives,
                                        remove_unreferenced_points);
}

} // namespace tf::cpp::async
