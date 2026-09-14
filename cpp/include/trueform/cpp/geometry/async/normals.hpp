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
#include "trueform/cpp/geometry/normals.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

/// @brief Resolve runtime primitive normals on the common executor.
template <typename Resolver, typename Real>
auto normals(Resolver &&resolver, const primitive<Real, 3> &value) {
  auto owned_value = value;
  return submit<nd_array<Real>>(
      std::forward<Resolver>(resolver),
      [value = std::move(owned_value)] { return cpp::normals(value); });
}

template <typename Real>
auto normals(const primitive<Real, 3> &value) -> std::future<nd_array<Real>> {
  return async::normals(future_resolver{}, value);
}

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon, std::enable_if_t<Dims == 3, int> = 0>
auto normals(Resolver &&resolver,
             const cpp::mesh<Index, Real, Dims, Ngon> &value) {
  return submit<nd_array<Real>>(std::forward<Resolver>(resolver), [value] {
    return cpp::normals<Index, Real, Dims, Ngon>(value);
  });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto normals(const cpp::mesh<Index, Real, Dims, Ngon> &value)
    -> std::future<nd_array<Real>> {
  return async::normals(future_resolver{}, value);
}

} // namespace tf::cpp::async
