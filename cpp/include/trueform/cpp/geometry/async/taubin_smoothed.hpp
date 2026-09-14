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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/geometry/taubin_smoothed.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

/// @brief Resolve Taubin smoothing on the common executor.
template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon>
auto taubin_smoothed(Resolver &&resolver,
                     const cpp::mesh<Index, Real, Dims, Ngon> &value,
                     int iterations, Real lambda = Real{0.5},
                     Real kpb = Real{0.1}) {
  return submit<tf::polygons_buffer<Index, Real, Dims, Ngon>>(
      std::forward<Resolver>(resolver), [value, iterations, lambda, kpb] {
        return cpp::taubin_smoothed(value, iterations, lambda, kpb);
      });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto taubin_smoothed(const cpp::mesh<Index, Real, Dims, Ngon> &value,
                     int iterations, Real lambda = Real{0.5},
                     Real kpb = Real{0.1})
    -> std::future<tf::polygons_buffer<Index, Real, Dims, Ngon>> {
  return async::taubin_smoothed(future_resolver{}, value, iterations, lambda,
                                kpb);
}

} // namespace tf::cpp::async
