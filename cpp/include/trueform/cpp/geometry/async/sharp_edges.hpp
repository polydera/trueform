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

#include "trueform/core/angle.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/geometry/sharp_edges.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

#define TF_CPP_DEFINE_ASYNC_SHARP_EDGES(Angle)                                 \
  template <typename Resolver, typename Index, typename Real,                  \
            std::size_t Dims, std::size_t Ngon,                                \
            std::enable_if_t<Dims == 3, int> = 0>                              \
  auto sharp_edges(Resolver &&resolver,                                        \
                   const cpp::mesh<Index, Real, Dims, Ngon> &value,            \
                   tf::Angle<Real> angle_threshold) {                          \
    return submit<nd_array<Index>>(                                            \
        std::forward<Resolver>(resolver), [value, angle_threshold] {           \
          return cpp::sharp_edges(value, angle_threshold);                     \
        });                                                                    \
  }                                                                            \
                                                                               \
  template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon, \
            std::enable_if_t<Dims == 3, int> = 0>                              \
  auto sharp_edges(const cpp::mesh<Index, Real, Dims, Ngon> &value,            \
                   tf::Angle<Real> angle_threshold)                            \
      -> std::future<nd_array<Index>> {                                        \
    return async::sharp_edges(future_resolver{}, value, angle_threshold);      \
  }

TF_CPP_DEFINE_ASYNC_SHARP_EDGES(rad)
TF_CPP_DEFINE_ASYNC_SHARP_EDGES(deg)

#undef TF_CPP_DEFINE_ASYNC_SHARP_EDGES

} // namespace tf::cpp::async
