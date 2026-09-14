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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/reductions.hpp"

#include <cstdint>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

#define TF_CPP_UNPAREN(...) __VA_ARGS__
#define TF_CPP_ASYNC_REDUCTION(NAME, SCALAR_RESULT, AXIS_RESULT)               \
  template <typename Resolver, typename T>                                     \
  auto NAME(Resolver &&resolver, const nd_array<T> &array)                     \
      -> resolver_result_t<Resolver, TF_CPP_UNPAREN SCALAR_RESULT> {           \
    return submit<TF_CPP_UNPAREN SCALAR_RESULT>(                               \
        std::forward<Resolver>(resolver),                                      \
        [array] { return cpp::NAME(array); });                                 \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME(const nd_array<T> &array)                                          \
      -> std::future<TF_CPP_UNPAREN SCALAR_RESULT> {                           \
    return async::NAME(future_resolver{}, array);                              \
  }                                                                            \
  template <typename Resolver, typename T>                                     \
  auto NAME(Resolver &&resolver, const nd_array<T> &array, int axis)           \
      -> resolver_result_t<Resolver, TF_CPP_UNPAREN AXIS_RESULT> {             \
    return submit<TF_CPP_UNPAREN AXIS_RESULT>(                                 \
        std::forward<Resolver>(resolver),                                      \
        [array, axis] { return cpp::NAME(array, axis); });                     \
  }                                                                            \
  template <typename T>                                                        \
  auto NAME(const nd_array<T> &array, int axis)                                \
      -> std::future<TF_CPP_UNPAREN AXIS_RESULT> {                             \
    return async::NAME(future_resolver{}, array, axis);                        \
  }

TF_CPP_ASYNC_REDUCTION(
    sum, (std::conditional_t<std::is_same_v<T, std::int8_t>, std::int32_t, T>),
    (nd_array<
        std::conditional_t<std::is_same_v<T, std::int8_t>, std::int32_t, T>>));
TF_CPP_ASYNC_REDUCTION(min, (T), (nd_array<T>));
TF_CPP_ASYNC_REDUCTION(max, (T), (nd_array<T>));
TF_CPP_ASYNC_REDUCTION(
    mean, (double),
    (nd_array<std::conditional_t<std::is_same_v<T, double>, double, float>>));
TF_CPP_ASYNC_REDUCTION(
    norm, (std::conditional_t<std::is_same_v<T, double>, double, float>),
    (nd_array<std::conditional_t<std::is_same_v<T, double>, double, float>>));
TF_CPP_ASYNC_REDUCTION(argmin, (std::int32_t), (nd_array<std::int32_t>));
TF_CPP_ASYNC_REDUCTION(argmax, (std::int32_t), (nd_array<std::int32_t>));

#undef TF_CPP_ASYNC_REDUCTION
#undef TF_CPP_UNPAREN

#define TF_CPP_ASYNC_BOOLEAN_REDUCTION(NAME)                                   \
  template <typename Resolver>                                                 \
  auto NAME(Resolver &&resolver, const nd_array<std::int8_t> &array)           \
      -> resolver_result_t<Resolver, int> {                                    \
    return submit<int>(std::forward<Resolver>(resolver),                       \
                       [array] { return cpp::NAME(array); });                  \
  }                                                                            \
  inline auto NAME(const nd_array<std::int8_t> &array) -> std::future<int> {   \
    return async::NAME(future_resolver{}, array);                              \
  }                                                                            \
  template <typename Resolver>                                                 \
  auto NAME(Resolver &&resolver, const nd_array<std::int8_t> &array, int axis) \
      -> resolver_result_t<Resolver, nd_array<std::int8_t>> {                  \
    return submit<nd_array<std::int8_t>>(                                      \
        std::forward<Resolver>(resolver),                                      \
        [array, axis] { return cpp::NAME(array, axis); });                     \
  }                                                                            \
  inline auto NAME(const nd_array<std::int8_t> &array, int axis)               \
      -> std::future<nd_array<std::int8_t>> {                                  \
    return async::NAME(future_resolver{}, array, axis);                        \
  }

TF_CPP_ASYNC_BOOLEAN_REDUCTION(any);
TF_CPP_ASYNC_BOOLEAN_REDUCTION(all);

#undef TF_CPP_ASYNC_BOOLEAN_REDUCTION

} // namespace tf::cpp::async
