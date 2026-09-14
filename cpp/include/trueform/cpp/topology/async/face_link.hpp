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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/topology/face_link.hpp"

#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {
template <typename Resolver, typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto face_link(Resolver &&resolver, const nd_array<Index> &faces,
               const offset_blocked_buffer<Index, Index> &face_membership) {
  return submit<offset_blocked_buffer<Index, Index>>(
      std::forward<Resolver>(resolver), [faces, face_membership] {
        return cpp::face_link(faces, face_membership);
      });
}

template <typename Resolver, typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto face_link(Resolver &&resolver,
               const offset_blocked_buffer<Index, Index> &faces,
               const offset_blocked_buffer<Index, Index> &face_membership) {
  return submit<offset_blocked_buffer<Index, Index>>(
      std::forward<Resolver>(resolver), [faces, face_membership] {
        return cpp::face_link(faces, face_membership);
      });
}

template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto face_link(const nd_array<Index> &faces,
               const offset_blocked_buffer<Index, Index> &face_membership)
    -> std::future<offset_blocked_buffer<Index, Index>> {
  return async::face_link(future_resolver{}, faces, face_membership);
}

template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto face_link(const offset_blocked_buffer<Index, Index> &faces,
               const offset_blocked_buffer<Index, Index> &face_membership)
    -> std::future<offset_blocked_buffer<Index, Index>> {
  return async::face_link(future_resolver{}, faces, face_membership);
}

} // namespace tf::cpp::async
