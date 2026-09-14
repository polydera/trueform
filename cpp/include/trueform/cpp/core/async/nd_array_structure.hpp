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
#include "trueform/cpp/core/nd_array_structure.hpp"

#include <cstdint>
#include <future>
#include <utility>
#include <vector>

namespace tf::cpp::async {

template <typename Resolver, typename T>
auto stack(Resolver &&resolver, const std::vector<nd_array<T>> &arrays,
           int axis) -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver), [arrays, axis] {
    return cpp::stack(arrays, axis);
  });
}

template <typename T>
auto stack(const std::vector<nd_array<T>> &arrays, int axis)
    -> std::future<nd_array<T>> {
  return async::stack(future_resolver{}, arrays, axis);
}

template <typename Resolver, typename T>
auto concatenate(Resolver &&resolver, const std::vector<nd_array<T>> &arrays,
                 int axis) -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver), [arrays, axis] {
    return cpp::concatenate(arrays, axis);
  });
}

template <typename T>
auto concatenate(const std::vector<nd_array<T>> &arrays, int axis)
    -> std::future<nd_array<T>> {
  return async::concatenate(future_resolver{}, arrays, axis);
}

template <typename Resolver, typename T>
auto tile(Resolver &&resolver, const nd_array<T> &array,
          const std::vector<int> &repetitions)
    -> resolver_result_t<Resolver, nd_array<T>> {
  auto owned_repetitions = repetitions;
  return submit<nd_array<T>>(
      std::forward<Resolver>(resolver),
      [array, repetitions = std::move(owned_repetitions)] {
        return cpp::tile(array, repetitions);
      });
}

template <typename T>
auto tile(const nd_array<T> &array, const std::vector<int> &repetitions)
    -> std::future<nd_array<T>> {
  return async::tile(future_resolver{}, array, repetitions);
}

template <typename Resolver, typename T>
auto transpose(Resolver &&resolver, const nd_array<T> &array)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [array] { return cpp::transpose(array); });
}

template <typename T>
auto transpose(const nd_array<T> &array) -> std::future<nd_array<T>> {
  return async::transpose(future_resolver{}, array);
}

template <typename Resolver, typename T>
auto transpose(Resolver &&resolver, const nd_array<T> &array,
               const std::vector<int> &axes)
    -> resolver_result_t<Resolver, nd_array<T>> {
  auto owned_axes = axes;
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [array, axes = std::move(owned_axes)] {
                               return cpp::transpose(array, axes);
                             });
}

template <typename T>
auto transpose(const nd_array<T> &array, const std::vector<int> &axes)
    -> std::future<nd_array<T>> {
  return async::transpose(future_resolver{}, array, axes);
}

template <typename Resolver, typename T>
auto where(Resolver &&resolver, const nd_array<std::int8_t> &condition,
           const nd_array<T> &x, const nd_array<T> &y)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(
      std::forward<Resolver>(resolver),
      [condition, x, y] { return cpp::where(condition, x, y); });
}

template <typename T>
auto where(const nd_array<std::int8_t> &condition, const nd_array<T> &x,
           const nd_array<T> &y) -> std::future<nd_array<T>> {
  return async::where(future_resolver{}, condition, x, y);
}

template <typename Resolver, typename T>
auto clone(Resolver &&resolver, const nd_array<T> &array)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [array] { return cpp::clone(array); });
}

template <typename T>
auto clone(const nd_array<T> &array) -> std::future<nd_array<T>> {
  return async::clone(future_resolver{}, array);
}

} // namespace tf::cpp::async
