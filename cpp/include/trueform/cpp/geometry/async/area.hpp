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
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/geometry/area.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon>
auto area(Resolver &&resolver,
          const cpp::mesh<Index, Real, Dims, Ngon> &value) {
  return submit<Real>(std::forward<Resolver>(resolver),
                      [value] { return cpp::area(value); });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto area(const cpp::mesh<Index, Real, Dims, Ngon> &value)
    -> std::future<Real> {
  return async::area(future_resolver{}, value);
}

template <typename Resolver, typename Real, std::size_t Dims>
auto area(Resolver &&resolver, const primitive<Real, Dims> &value) {
  auto owned_value = value;
  return submit<area_result<Real>>(
      std::forward<Resolver>(resolver),
      [value = std::move(owned_value)] { return cpp::area(value); });
}

template <typename Real, std::size_t Dims>
auto area(const primitive<Real, Dims> &value)
    -> std::future<area_result<Real>> {
  return async::area(future_resolver{}, value);
}

} // namespace tf::cpp::async
