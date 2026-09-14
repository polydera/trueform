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
#include "trueform/cpp/spatial/closest_metric_point.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <future>
#include <type_traits>
#include <utility>
#include <variant>

namespace tf::cpp::async {

/// @brief Resolve the closest point on the first runtime primitive on the
/// common executor.
template <typename Resolver, typename Real0, typename Real1>
auto closest_metric_point(Resolver &&resolver, const primitive<Real0> &a,
                          const primitive<Real1> &b)
    -> resolver_result_t<Resolver,
                         std::variant<closest_metric_point_result<
                                          std::common_type_t<Real0, Real1>>,
                                      closest_metric_point_batch_result<
                                          std::common_type_t<Real0, Real1>>>> {
  auto owned_a = a;
  auto owned_b = b;
  return submit<std::variant<
      closest_metric_point_result<std::common_type_t<Real0, Real1>>,
      closest_metric_point_batch_result<std::common_type_t<Real0, Real1>>>>(
      std::forward<Resolver>(resolver),
      [a = std::move(owned_a), b = std::move(owned_b)] {
        return cpp::closest_metric_point(a, b);
      });
}

template <typename Real0, typename Real1>
auto closest_metric_point(const primitive<Real0> &a, const primitive<Real1> &b)
    -> std::future<std::variant<
        closest_metric_point_result<std::common_type_t<Real0, Real1>>,
        closest_metric_point_batch_result<std::common_type_t<Real0, Real1>>>> {
  return async::closest_metric_point(future_resolver{}, a, b);
}

} // namespace tf::cpp::async
