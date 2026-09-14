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
#include "trueform/cpp/core/elementwise/logical.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver>
auto logical_not(Resolver &&resolver, const nd_array<std::int8_t> &a) {
  return submit<nd_array<std::int8_t>>(std::forward<Resolver>(resolver),
                                       [a] { return cpp::logical_not(a); });
}

inline auto logical_not(const nd_array<std::int8_t> &a)
    -> std::future<nd_array<std::int8_t>> {
  return async::logical_not(future_resolver{}, a);
}

template <typename Resolver>
auto logical_not_inplace(Resolver &&resolver, nd_array<std::int8_t> &a) {
  return submit<void>(std::forward<Resolver>(resolver),
                      [a]() mutable { cpp::logical_not_inplace(a); });
}

inline auto logical_not_inplace(nd_array<std::int8_t> &a) -> std::future<void> {
  return async::logical_not_inplace(future_resolver{}, a);
}

#define TF_CPP_ASYNC_LOGICAL(NAME)                                             \
  template <typename Resolver>                                                 \
  auto NAME(Resolver &&resolver, const nd_array<std::int8_t> &a,               \
            const nd_array<std::int8_t> &b) {                                  \
    return submit<nd_array<std::int8_t>>(std::forward<Resolver>(resolver),     \
                                         [a, b] { return cpp::NAME(a, b); });  \
  }                                                                            \
  inline auto NAME(const nd_array<std::int8_t> &a,                             \
                   const nd_array<std::int8_t> &b)                             \
      -> std::future<nd_array<std::int8_t>> {                                  \
    return async::NAME(future_resolver{}, a, b);                               \
  }

TF_CPP_ASYNC_LOGICAL(logical_and)
TF_CPP_ASYNC_LOGICAL(logical_or)

#undef TF_CPP_ASYNC_LOGICAL

} // namespace tf::cpp::async
