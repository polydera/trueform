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
#include "trueform/cpp/core/elementwise/arithmetic.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <future>
#include <utility>

namespace tf::cpp::async {

#define TF_CPP_ASYNC_BINARY(NAME)                                              \
  template <typename Resolver, typename T>                                     \
  auto NAME(Resolver &&resolver, const nd_array<T> &a, const nd_array<T> &b) { \
    return submit<nd_array<T>>(std::forward<Resolver>(resolver),               \
                               [a, b] { return cpp::NAME(a, b); });            \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME(const nd_array<T> &a, const nd_array<T> &b)                        \
      -> std::future<nd_array<T>> {                                            \
    return async::NAME(future_resolver{}, a, b);                               \
  }                                                                            \
  template <typename Resolver, typename T>                                     \
  auto NAME##_inplace(Resolver &&resolver, nd_array<T> &a,                     \
                      const nd_array<T> &b) {                                  \
    return submit<void>(std::forward<Resolver>(resolver),                      \
                        [a, b]() mutable { cpp::NAME##_inplace(a, b); });      \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME##_inplace(nd_array<T> &a, const nd_array<T> &b)                    \
      -> std::future<void> {                                                   \
    return async::NAME##_inplace(future_resolver{}, a, b);                     \
  }                                                                            \
  template <typename Resolver, typename T>                                     \
  auto NAME##_scalar(Resolver &&resolver, const nd_array<T> &a, T scalar) {    \
    return submit<nd_array<T>>(std::forward<Resolver>(resolver), [a, scalar] { \
      return cpp::NAME##_scalar(a, scalar);                                    \
    });                                                                        \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME##_scalar(const nd_array<T> &a, T scalar)                           \
      -> std::future<nd_array<T>> {                                            \
    return async::NAME##_scalar(future_resolver{}, a, scalar);                 \
  }                                                                            \
  template <typename Resolver, typename T>                                     \
  auto NAME##_scalar_inplace(Resolver &&resolver, nd_array<T> &a, T scalar) {  \
    return submit<void>(                                                       \
        std::forward<Resolver>(resolver),                                      \
        [a, scalar]() mutable { cpp::NAME##_scalar_inplace(a, scalar); });     \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME##_scalar_inplace(nd_array<T> &a, T scalar) -> std::future<void> {  \
    return async::NAME##_scalar_inplace(future_resolver{}, a, scalar);         \
  }

TF_CPP_ASYNC_BINARY(add)
TF_CPP_ASYNC_BINARY(sub)
TF_CPP_ASYNC_BINARY(mul)
TF_CPP_ASYNC_BINARY(div)
TF_CPP_ASYNC_BINARY(mod)

#undef TF_CPP_ASYNC_BINARY

} // namespace tf::cpp::async
