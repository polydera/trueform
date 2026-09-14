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
#include "trueform/cpp/geometry/fit_rigid.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Real, std::size_t Dims>
auto fit_rigid(Resolver &&resolver, const point_cloud<Real, Dims> &source,
               const point_cloud<Real, Dims> &target) {
  return submit<nd_array<Real>>(
      std::forward<Resolver>(resolver),
      [source, target] { return cpp::fit_rigid(source, target); });
}

template <typename Real, std::size_t Dims>
auto fit_rigid(const point_cloud<Real, Dims> &source,
               const point_cloud<Real, Dims> &target)
    -> std::future<nd_array<Real>> {
  return async::fit_rigid(future_resolver{}, source, target);
}

} // namespace tf::cpp::async
