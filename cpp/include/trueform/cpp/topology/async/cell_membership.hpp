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
#include "trueform/cpp/core/detail/non_deduced.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/topology/cell_membership.hpp"

#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {
template <typename Resolver, typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto cell_membership(Resolver &&resolver, const nd_array<Index> &cells,
                     cpp::detail::non_deduced_t<Index> n_ids)
    -> resolver_result_t<Resolver, offset_blocked_buffer<Index, Index>> {
  return submit<offset_blocked_buffer<Index, Index>>(
      std::forward<Resolver>(resolver),
      [cells, n_ids] { return cpp::cell_membership(cells, n_ids); });
}

template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto cell_membership(const nd_array<Index> &cells,
                     cpp::detail::non_deduced_t<Index> n_ids)
    -> std::future<offset_blocked_buffer<Index, Index>> {
  return async::cell_membership(future_resolver{}, cells, n_ids);
}

template <typename Resolver, typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto cell_membership(Resolver &&resolver,
                     const offset_blocked_buffer<Index, Index> &cells,
                     cpp::detail::non_deduced_t<Index> n_ids)
    -> resolver_result_t<Resolver, offset_blocked_buffer<Index, Index>> {
  return submit<offset_blocked_buffer<Index, Index>>(
      std::forward<Resolver>(resolver),
      [cells, n_ids] { return cpp::cell_membership(cells, n_ids); });
}

template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto cell_membership(const offset_blocked_buffer<Index, Index> &cells,
                     cpp::detail::non_deduced_t<Index> n_ids)
    -> std::future<offset_blocked_buffer<Index, Index>> {
  return async::cell_membership(future_resolver{}, cells, n_ids);
}

} // namespace tf::cpp::async
