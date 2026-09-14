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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/geometry/fit_obb.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Real, std::size_t Dims>
auto fit_obb(Resolver &&resolver, const point_cloud<Real, Dims> &source,
             const point_cloud<Real, Dims> &target,
             const fit_obb_options &options = {}) {
  auto owned_options = options;
  return submit<nd_array<Real>>(
      std::forward<Resolver>(resolver),
      [source, target, options = std::move(owned_options)] {
        return cpp::fit_obb(source, target, options);
      });
}

template <typename Real, std::size_t Dims>
auto fit_obb(const point_cloud<Real, Dims> &source,
             const point_cloud<Real, Dims> &target,
             const fit_obb_options &options = {})
    -> std::future<nd_array<Real>> {
  return async::fit_obb(future_resolver{}, source, target, options);
}

} // namespace tf::cpp::async
