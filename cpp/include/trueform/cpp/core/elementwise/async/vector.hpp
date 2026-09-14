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
#include "trueform/cpp/core/elementwise/vector.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename T>
auto mat_mul(Resolver &&resolver, const nd_array<T> &a, const nd_array<T> &b) {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [a, b] { return cpp::mat_mul(a, b); });
}

template <typename T>
auto mat_mul(const nd_array<T> &a, const nd_array<T> &b)
    -> std::future<nd_array<T>> {
  return async::mat_mul(future_resolver{}, a, b);
}

template <typename Resolver, typename T>
auto inverted(Resolver &&resolver, const nd_array<T> &matrix) {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [matrix] { return cpp::inverted(matrix); });
}

template <typename T>
auto inverted(const nd_array<T> &matrix) -> std::future<nd_array<T>> {
  return async::inverted(future_resolver{}, matrix);
}

#define TF_CPP_ASYNC_VECTOR(NAME)                                              \
  template <typename Resolver, typename T>                                     \
  auto NAME(Resolver &&resolver, const nd_array<T> &a, const nd_array<T> &b) { \
    return submit<nd_array<T>>(std::forward<Resolver>(resolver),               \
                               [a, b] { return cpp::NAME(a, b); });            \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME(const nd_array<T> &a, const nd_array<T> &b)                        \
      -> std::future<nd_array<T>> {                                            \
    return async::NAME(future_resolver{}, a, b);                               \
  }

TF_CPP_ASYNC_VECTOR(dot)
TF_CPP_ASYNC_VECTOR(cross)

#undef TF_CPP_ASYNC_VECTOR

} // namespace tf::cpp::async
