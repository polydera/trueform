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
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/topology/connect_edges_to_paths.hpp"

#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {

/// The synchronous entry states which widths this build answers for, and these
/// carry it: an index the matrix left out has no overload here either.
#define TF_CPP_DEFINE_ASYNC_CONNECT_EDGES_TO_PATHS(Index)                      \
  template <typename Resolver>                                                 \
  auto connect_edges_to_paths(Resolver &&resolver,                             \
                              const nd_array<Index> &edges)                    \
      ->resolver_result_t<Resolver, offset_blocked_buffer<Index, Index>> {     \
    return submit<offset_blocked_buffer<Index, Index>>(                        \
        std::forward<Resolver>(resolver),                                      \
        [edges] { return cpp::connect_edges_to_paths(edges); });               \
  }                                                                            \
                                                                               \
  inline auto connect_edges_to_paths(const nd_array<Index> &edges)             \
      ->std::future<offset_blocked_buffer<Index, Index>> {                     \
    return async::connect_edges_to_paths(future_resolver{}, edges);            \
  }

TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_DEFINE_ASYNC_CONNECT_EDGES_TO_PATHS)

#undef TF_CPP_DEFINE_ASYNC_CONNECT_EDGES_TO_PATHS

} // namespace tf::cpp::async
