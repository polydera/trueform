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
#include "trueform/cpp/topology/label_connected_components.hpp"

#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {

namespace detail {
/// The blocked connectivity named by its index alone, so the matrix can iterate
/// it: a type with a comma in it cannot be a macro argument.
template <typename Index>
using connected_blocks = offset_blocked_buffer<Index, Index>;
} // namespace detail

/// The synchronous entry states which widths this build answers for, and these
/// carry it: an index the matrix left out has no overload here either.
#define TF_CPP_DEFINE_ASYNC_LABEL_CONNECTED_COMPONENTS(Carrier, Index)         \
  template <typename Resolver>                                                 \
  auto label_connected_components(Resolver &&resolver,                         \
                                  const Carrier &connectivity)                 \
      ->resolver_result_t<Resolver, connected_components_result<Index>> {      \
    return submit<connected_components_result<Index>>(                         \
        std::forward<Resolver>(resolver), [connectivity] {                     \
          return cpp::label_connected_components(connectivity);                \
        });                                                                    \
  }                                                                            \
                                                                               \
  template <typename Resolver>                                                 \
  auto label_connected_components(Resolver &&resolver,                         \
                                  const Carrier &connectivity,                 \
                                  Index expected_number_of_components)         \
      ->resolver_result_t<Resolver, connected_components_result<Index>> {      \
    return submit<connected_components_result<Index>>(                         \
        std::forward<Resolver>(resolver),                                      \
        [connectivity, expected_number_of_components] {                        \
          return cpp::label_connected_components(                              \
              connectivity, expected_number_of_components);                    \
        });                                                                    \
  }                                                                            \
                                                                               \
  inline auto label_connected_components(const Carrier &connectivity)          \
      ->std::future<connected_components_result<Index>> {                      \
    return async::label_connected_components(future_resolver{}, connectivity); \
  }                                                                            \
                                                                               \
  inline auto label_connected_components(                                      \
      const Carrier &connectivity, Index expected_number_of_components)        \
      ->std::future<connected_components_result<Index>> {                      \
    return async::label_connected_components(future_resolver{}, connectivity,  \
                                             expected_number_of_components);   \
  }

#define TF_CPP_DEFINE_ASYNC_LABEL_BLOCKS(Index)                                \
  TF_CPP_DEFINE_ASYNC_LABEL_CONNECTED_COMPONENTS(                              \
      detail::connected_blocks<Index>, Index)

#define TF_CPP_DEFINE_ASYNC_LABEL_DENSE(Index)                                 \
  TF_CPP_DEFINE_ASYNC_LABEL_CONNECTED_COMPONENTS(nd_array<Index>, Index)

TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_DEFINE_ASYNC_LABEL_BLOCKS)
TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_DEFINE_ASYNC_LABEL_DENSE)

#undef TF_CPP_DEFINE_ASYNC_LABEL_DENSE
#undef TF_CPP_DEFINE_ASYNC_LABEL_BLOCKS
#undef TF_CPP_DEFINE_ASYNC_LABEL_CONNECTED_COMPONENTS

} // namespace tf::cpp::async
