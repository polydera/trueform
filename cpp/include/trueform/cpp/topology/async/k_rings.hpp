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
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/topology/k_rings.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {
template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon>
auto k_rings(Resolver &&resolver,
             const cpp::mesh<Index, Real, Dims, Ngon> &value, std::int32_t k,
             bool inclusive = false)
    -> resolver_result_t<Resolver, offset_blocked_buffer<Index, Index>> {
  return submit<offset_blocked_buffer<Index, Index>>(
      std::forward<Resolver>(resolver),
      [value, k, inclusive] { return cpp::k_rings(value, k, inclusive); });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto k_rings(const cpp::mesh<Index, Real, Dims, Ngon> &value, std::int32_t k,
             bool inclusive = false)
    -> std::future<offset_blocked_buffer<Index, Index>> {
  return async::k_rings(future_resolver{}, value, k, inclusive);
}

template <typename Resolver, typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto k_rings(Resolver &&resolver,
             const offset_blocked_buffer<Index, Index> &connectivity,
             std::int32_t k, bool inclusive = false)
    -> resolver_result_t<Resolver, offset_blocked_buffer<Index, Index>> {
  return submit<offset_blocked_buffer<Index, Index>>(
      std::forward<Resolver>(resolver), [connectivity, k, inclusive] {
        return cpp::k_rings(connectivity, k, inclusive);
      });
}

template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto k_rings(const offset_blocked_buffer<Index, Index> &connectivity,
             std::int32_t k, bool inclusive = false)
    -> std::future<offset_blocked_buffer<Index, Index>> {
  return async::k_rings(future_resolver{}, connectivity, k, inclusive);
}

} // namespace tf::cpp::async
