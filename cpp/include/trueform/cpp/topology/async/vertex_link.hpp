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
#include "trueform/cpp/topology/vertex_link.hpp"

#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {
template <typename Resolver, typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto vertex_link_edges(Resolver &&resolver, const nd_array<Index> &edges,
                       cpp::detail::non_deduced_t<Index> n_ids)
    -> resolver_result_t<Resolver, offset_blocked_buffer<Index, Index>> {
  return submit<offset_blocked_buffer<Index, Index>>(
      std::forward<Resolver>(resolver),
      [edges, n_ids] { return cpp::vertex_link_edges(edges, n_ids); });
}

template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto vertex_link_edges(const nd_array<Index> &edges,
                       cpp::detail::non_deduced_t<Index> n_ids)
    -> std::future<offset_blocked_buffer<Index, Index>> {
  return async::vertex_link_edges(future_resolver{}, edges, n_ids);
}

template <typename Resolver, typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto vertex_link_faces(
    Resolver &&resolver, const nd_array<Index> &faces,
    const offset_blocked_buffer<Index, Index> &cell_membership)
    -> resolver_result_t<Resolver, offset_blocked_buffer<Index, Index>> {
  return submit<offset_blocked_buffer<Index, Index>>(
      std::forward<Resolver>(resolver), [faces, cell_membership] {
        return cpp::vertex_link_faces(faces, cell_membership);
      });
}

template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto vertex_link_faces(
    const nd_array<Index> &faces,
    const offset_blocked_buffer<Index, Index> &cell_membership)
    -> std::future<offset_blocked_buffer<Index, Index>> {
  return async::vertex_link_faces(future_resolver{}, faces, cell_membership);
}

template <typename Resolver, typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto vertex_link_faces(
    Resolver &&resolver, const offset_blocked_buffer<Index, Index> &faces,
    const offset_blocked_buffer<Index, Index> &cell_membership)
    -> resolver_result_t<Resolver, offset_blocked_buffer<Index, Index>> {
  return submit<offset_blocked_buffer<Index, Index>>(
      std::forward<Resolver>(resolver), [faces, cell_membership] {
        return cpp::vertex_link_faces(faces, cell_membership);
      });
}

template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto vertex_link_faces(
    const offset_blocked_buffer<Index, Index> &faces,
    const offset_blocked_buffer<Index, Index> &cell_membership)
    -> std::future<offset_blocked_buffer<Index, Index>> {
  return async::vertex_link_faces(future_resolver{}, faces, cell_membership);
}

} // namespace tf::cpp::async
