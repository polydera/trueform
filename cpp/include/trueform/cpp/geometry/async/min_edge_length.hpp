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
#include "trueform/cpp/geometry/min_edge_length.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon>
auto min_edge_length(Resolver &&resolver,
                     const cpp::mesh<Index, Real, Dims, Ngon> &value) {
  return submit<Real>(std::forward<Resolver>(resolver),
                      [value] { return cpp::min_edge_length(value); });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto min_edge_length(const cpp::mesh<Index, Real, Dims, Ngon> &value)
    -> std::future<Real> {
  return async::min_edge_length(future_resolver{}, value);
}

} // namespace tf::cpp::async
