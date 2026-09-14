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
#include "trueform/cpp/core/nd_array_sorting.hpp"

#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {

/// @brief Return a row-sorted copy on the common executor.
template <typename Resolver, typename T>
auto sort(Resolver &&resolver, const nd_array<T> &array)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [array] { return cpp::sort(array); });
}

template <typename T>
auto sort(const nd_array<T> &array) -> std::future<nd_array<T>> {
  return async::sort(future_resolver{}, array);
}

/// @brief Resolve the lexicographic row permutation on the common executor.
template <typename Resolver, typename T>
auto argsort(Resolver &&resolver, const nd_array<T> &array)
    -> resolver_result_t<Resolver, nd_array<std::int32_t>> {
  return submit<nd_array<std::int32_t>>(
      std::forward<Resolver>(resolver),
      [array] { return cpp::argsort(array); });
}

template <typename T>
auto argsort(const nd_array<T> &array) -> std::future<nd_array<std::int32_t>> {
  return async::argsort(future_resolver{}, array);
}

/// @brief Row-sort shared array storage on the common executor.
///
/// The worker retains shared storage ownership. Concurrent access to the same
/// storage remains the caller's responsibility until completion.
template <typename Resolver, typename T>
auto sort_inplace(Resolver &&resolver, nd_array<T> &array)
    -> resolver_result_t<Resolver, void> {
  return submit<void>(std::forward<Resolver>(resolver),
                      [array]() mutable { cpp::sort_inplace(array); });
}

template <typename T>
auto sort_inplace(nd_array<T> &array) -> std::future<void> {
  return async::sort_inplace(future_resolver{}, array);
}

/// @brief Remove adjacent duplicate rows on the common executor.
template <typename Resolver, typename T>
auto unique(Resolver &&resolver, const nd_array<T> &array)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [array] { return cpp::unique(array); });
}

template <typename T>
auto unique(const nd_array<T> &array) -> std::future<nd_array<T>> {
  return async::unique(future_resolver{}, array);
}

/// @brief Merge two sorted row sets on the common executor.
template <typename Resolver, typename T>
auto set_union(Resolver &&resolver, const nd_array<T> &a, const nd_array<T> &b)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [a, b] { return cpp::set_union(a, b); });
}

template <typename T>
auto set_union(const nd_array<T> &a, const nd_array<T> &b)
    -> std::future<nd_array<T>> {
  return async::set_union(future_resolver{}, a, b);
}

/// @brief Intersect two sorted row sets on the common executor.
template <typename Resolver, typename T>
auto set_intersection(Resolver &&resolver, const nd_array<T> &a,
                      const nd_array<T> &b)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [a, b] { return cpp::set_intersection(a, b); });
}

template <typename T>
auto set_intersection(const nd_array<T> &a, const nd_array<T> &b)
    -> std::future<nd_array<T>> {
  return async::set_intersection(future_resolver{}, a, b);
}

/// @brief Subtract two sorted row sets on the common executor.
template <typename Resolver, typename T>
auto set_difference(Resolver &&resolver, const nd_array<T> &a,
                    const nd_array<T> &b)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [a, b] { return cpp::set_difference(a, b); });
}

template <typename T>
auto set_difference(const nd_array<T> &a, const nd_array<T> &b)
    -> std::future<nd_array<T>> {
  return async::set_difference(future_resolver{}, a, b);
}

} // namespace tf::cpp::async
