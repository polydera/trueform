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
#include "trueform/cpp/core/elementwise/unary.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cmath>
#include <future>
#include <utility>

namespace tf::cpp::async {

#define TF_CPP_ASYNC_UNARY(NAME)                                               \
  template <typename Resolver, typename T>                                     \
  auto NAME(Resolver &&resolver, const nd_array<T> &a) {                       \
    return submit<nd_array<T>>(std::forward<Resolver>(resolver),               \
                               [a] { return cpp::NAME(a); });                  \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME(const nd_array<T> &a) -> std::future<nd_array<T>> {                \
    return async::NAME(future_resolver{}, a);                                  \
  }                                                                            \
  template <typename Resolver, typename T>                                     \
  auto NAME##_inplace(Resolver &&resolver, nd_array<T> &a) {                   \
    return submit<void>(std::forward<Resolver>(resolver),                      \
                        [a]() mutable { cpp::NAME##_inplace(a); });            \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME##_inplace(nd_array<T> &a) -> std::future<void> {                   \
    return async::NAME##_inplace(future_resolver{}, a);                        \
  }

TF_CPP_ASYNC_UNARY(sqrt)
TF_CPP_ASYNC_UNARY(sin)
TF_CPP_ASYNC_UNARY(cos)
TF_CPP_ASYNC_UNARY(tan)
TF_CPP_ASYNC_UNARY(asin)
TF_CPP_ASYNC_UNARY(acos)
TF_CPP_ASYNC_UNARY(atan)
TF_CPP_ASYNC_UNARY(exp)
TF_CPP_ASYNC_UNARY(log)
TF_CPP_ASYNC_UNARY(log2)
TF_CPP_ASYNC_UNARY(log10)
TF_CPP_ASYNC_UNARY(floor)
TF_CPP_ASYNC_UNARY(ceil)
TF_CPP_ASYNC_UNARY(round)
TF_CPP_ASYNC_UNARY(abs)
TF_CPP_ASYNC_UNARY(neg)

#undef TF_CPP_ASYNC_UNARY

template <typename Resolver, typename T>
auto pow(Resolver &&resolver, const nd_array<T> &a, T exponent) {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [a, exponent] { return cpp::pow(a, exponent); });
}

template <typename T>
auto pow(const nd_array<T> &a, T exponent) -> std::future<nd_array<T>> {
  return async::pow(future_resolver{}, a, exponent);
}

template <typename Resolver, typename T>
auto pow_inplace(Resolver &&resolver, nd_array<T> &a, T exponent) {
  return submit<void>(
      std::forward<Resolver>(resolver),
      [a, exponent]() mutable { cpp::pow_inplace(a, exponent); });
}

template <typename T>
auto pow_inplace(nd_array<T> &a, T exponent) -> std::future<void> {
  return async::pow_inplace(future_resolver{}, a, exponent);
}

template <typename Resolver, typename T>
auto atan2(Resolver &&resolver, const nd_array<T> &y, const nd_array<T> &x) {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [y, x] { return cpp::atan2(y, x); });
}

template <typename T>
auto atan2(const nd_array<T> &y, const nd_array<T> &x)
    -> std::future<nd_array<T>> {
  return async::atan2(future_resolver{}, y, x);
}

template <typename Resolver, typename T>
auto clip(Resolver &&resolver, const nd_array<T> &a, T lower, T upper) {
  return submit<nd_array<T>>(
      std::forward<Resolver>(resolver),
      [a, lower, upper] { return cpp::clip(a, lower, upper); });
}

template <typename T>
auto clip(const nd_array<T> &a, T lower, T upper) -> std::future<nd_array<T>> {
  return async::clip(future_resolver{}, a, lower, upper);
}

template <typename Resolver, typename T>
auto clip_inplace(Resolver &&resolver, nd_array<T> &a, T lower, T upper) {
  return submit<void>(
      std::forward<Resolver>(resolver),
      [a, lower, upper]() mutable { cpp::clip_inplace(a, lower, upper); });
}

template <typename T>
auto clip_inplace(nd_array<T> &a, T lower, T upper) -> std::future<void> {
  return async::clip_inplace(future_resolver{}, a, lower, upper);
}

} // namespace tf::cpp::async
