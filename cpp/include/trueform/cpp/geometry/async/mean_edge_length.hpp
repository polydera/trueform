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
#include "trueform/cpp/geometry/mean_edge_length.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon>
auto mean_edge_length(Resolver &&resolver,
                      const cpp::mesh<Index, Real, Dims, Ngon> &value) {
  return submit<Real>(std::forward<Resolver>(resolver),
                      [value] { return cpp::mean_edge_length(value); });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto mean_edge_length(const cpp::mesh<Index, Real, Dims, Ngon> &value)
    -> std::future<Real> {
  return async::mean_edge_length(future_resolver{}, value);
}

template <typename Resolver, typename Real, std::size_t Dims>
auto mean_edge_length(Resolver &&resolver, const primitive<Real, Dims> &value) {
  auto owned_value = value;
  return submit<Real>(std::forward<Resolver>(resolver),
                      [value = std::move(owned_value)] {
                        return cpp::mean_edge_length(value);
                      });
}

template <typename Real, std::size_t Dims>
auto mean_edge_length(const primitive<Real, Dims> &value) -> std::future<Real> {
  return async::mean_edge_length(future_resolver{}, value);
}

} // namespace tf::cpp::async
