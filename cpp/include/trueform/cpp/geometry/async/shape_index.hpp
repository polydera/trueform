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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/geometry/shape_index.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon, std::enable_if_t<Dims == 3, int> = 0>
auto shape_index(Resolver &&resolver,
                 const cpp::mesh<Index, Real, Dims, Ngon> &value, int k = 2) {
  return submit<nd_array<Real>>(std::forward<Resolver>(resolver), [value, k] {
    return cpp::shape_index<Index, Real, Dims, Ngon>(value, k);
  });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto shape_index(const cpp::mesh<Index, Real, Dims, Ngon> &value, int k = 2)
    -> std::future<nd_array<Real>> {
  return async::shape_index<future_resolver, Index, Real, Dims, Ngon>(
      future_resolver{}, value, k);
}

} // namespace tf::cpp::async
