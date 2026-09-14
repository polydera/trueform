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
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/topology/neighborhoods.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {
template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon>
auto neighborhoods(Resolver &&resolver,
                   const cpp::mesh<Index, Real, Dims, Ngon> &value, Real radius,
                   bool inclusive = false)
    -> resolver_result_t<Resolver, offset_blocked_buffer<Index, Index>> {
  return submit<offset_blocked_buffer<Index, Index>>(
      std::forward<Resolver>(resolver), [value, radius, inclusive] {
        return cpp::neighborhoods(value, radius, inclusive);
      });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto neighborhoods(const cpp::mesh<Index, Real, Dims, Ngon> &value, Real radius,
                   bool inclusive = false)
    -> std::future<offset_blocked_buffer<Index, Index>> {
  return async::neighborhoods(future_resolver{}, value, radius, inclusive);
}

template <typename Index, typename Real, std::size_t Dims, typename Resolver,
          std::enable_if_t<
              is_supported_index_v<Index> && (Dims == 2 || Dims == 3), int> = 0>
auto neighborhoods(Resolver &&resolver,
                   const offset_blocked_buffer<Index, Index> &connectivity,
                   const nd_array<Real> &points, Real radius,
                   bool inclusive = false)
    -> resolver_result_t<Resolver, offset_blocked_buffer<Index, Index>> {
  return submit<offset_blocked_buffer<Index, Index>>(
      std::forward<Resolver>(resolver),
      [connectivity, points, radius, inclusive] {
        return cpp::neighborhoods<Index, Real, Dims>(connectivity, points,
                                                     radius, inclusive);
      });
}

template <typename Index, typename Real, std::size_t Dims,
          std::enable_if_t<
              is_supported_index_v<Index> && (Dims == 2 || Dims == 3), int> = 0>
auto neighborhoods(const offset_blocked_buffer<Index, Index> &connectivity,
                   const nd_array<Real> &points, Real radius,
                   bool inclusive = false)
    -> std::future<offset_blocked_buffer<Index, Index>> {
  return async::neighborhoods<Index, Real, Dims>(
      future_resolver{}, connectivity, points, radius, inclusive);
}

} // namespace tf::cpp::async
