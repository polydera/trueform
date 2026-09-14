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
#include "trueform/cpp/core/elementwise/comparison.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {

#define TF_CPP_ASYNC_COMPARISON(NAME)                                          \
  template <typename Resolver, typename T>                                     \
  auto NAME(Resolver &&resolver, const nd_array<T> &a, const nd_array<T> &b) { \
    return submit<nd_array<std::int8_t>>(std::forward<Resolver>(resolver),     \
                                         [a, b] { return cpp::NAME(a, b); });  \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME(const nd_array<T> &a, const nd_array<T> &b)                        \
      -> std::future<nd_array<std::int8_t>> {                                  \
    return async::NAME(future_resolver{}, a, b);                               \
  }                                                                            \
  template <typename Resolver, typename T>                                     \
  auto NAME##_scalar(Resolver &&resolver, const nd_array<T> &a, T scalar) {    \
    return submit<nd_array<std::int8_t>>(                                      \
        std::forward<Resolver>(resolver),                                      \
        [a, scalar] { return cpp::NAME##_scalar(a, scalar); });                \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME##_scalar(const nd_array<T> &a, T scalar)                           \
      -> std::future<nd_array<std::int8_t>> {                                  \
    return async::NAME##_scalar(future_resolver{}, a, scalar);                 \
  }

TF_CPP_ASYNC_COMPARISON(eq)
TF_CPP_ASYNC_COMPARISON(neq)
TF_CPP_ASYNC_COMPARISON(lt)
TF_CPP_ASYNC_COMPARISON(gt)
TF_CPP_ASYNC_COMPARISON(lte)
TF_CPP_ASYNC_COMPARISON(gte)

#undef TF_CPP_ASYNC_COMPARISON

} // namespace tf::cpp::async
