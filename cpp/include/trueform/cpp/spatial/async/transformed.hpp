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
#include "trueform/cpp/spatial/primitive.hpp"
#include "trueform/cpp/spatial/transformed.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

/// @brief Resolve a transformed runtime primitive asynchronously.
template <typename Resolver, typename Real, std::size_t Dims>
auto transformed(Resolver &&resolver, const primitive<Real, Dims> &value,
                 const nd_array<Real> &matrix) {
  auto owned_value = value;
  auto owned_matrix = matrix;
  return submit<primitive<Real, Dims>>(
      std::forward<Resolver>(resolver),
      [value = std::move(owned_value), matrix = std::move(owned_matrix)] {
        return cpp::transformed(value, matrix);
      });
}

/// @brief Transform a runtime primitive on the common executor.
template <typename Real, std::size_t Dims>
auto transformed(const primitive<Real, Dims> &value,
                 const nd_array<Real> &matrix)
    -> std::future<primitive<Real, Dims>> {
  return async::transformed(future_resolver{}, value, matrix);
}

} // namespace tf::cpp::async
