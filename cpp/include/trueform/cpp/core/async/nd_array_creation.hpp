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

#include "trueform/core/small_vector.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/nd_array_creation.hpp"

#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename T, typename Resolver>
auto zeros(Resolver &&resolver, tf::small_vector<int, 3> shape)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [shape = std::move(shape)]() mutable {
                               return cpp::zeros<T>(std::move(shape));
                             });
}

template <typename T>
auto zeros(tf::small_vector<int, 3> shape) -> std::future<nd_array<T>> {
  return async::zeros<T>(future_resolver{}, std::move(shape));
}

template <typename T, typename Resolver>
auto ones(Resolver &&resolver, tf::small_vector<int, 3> shape)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [shape = std::move(shape)]() mutable {
                               return cpp::ones<T>(std::move(shape));
                             });
}

template <typename T>
auto ones(tf::small_vector<int, 3> shape) -> std::future<nd_array<T>> {
  return async::ones<T>(future_resolver{}, std::move(shape));
}

template <typename T, typename Resolver>
auto full(Resolver &&resolver, tf::small_vector<int, 3> shape, T value)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [shape = std::move(shape), value]() mutable {
                               return cpp::full<T>(std::move(shape), value);
                             });
}

template <typename T>
auto full(tf::small_vector<int, 3> shape, T value) -> std::future<nd_array<T>> {
  return async::full<T>(future_resolver{}, std::move(shape), value);
}

template <typename T, typename Resolver>
auto eye(Resolver &&resolver, int size)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [size] { return cpp::eye<T>(size); });
}

template <typename T> auto eye(int size) -> std::future<nd_array<T>> {
  return async::eye<T>(future_resolver{}, size);
}

template <typename T, typename Resolver>
auto arange(Resolver &&resolver, T start, T stop, T step)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(
      std::forward<Resolver>(resolver),
      [start, stop, step] { return cpp::arange<T>(start, stop, step); });
}

template <typename T>
auto arange(T start, T stop, T step) -> std::future<nd_array<T>> {
  return async::arange<T>(future_resolver{}, start, stop, step);
}

template <typename T, typename Resolver>
auto linspace(Resolver &&resolver, T start, T stop, int count)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(
      std::forward<Resolver>(resolver),
      [start, stop, count] { return cpp::linspace<T>(start, stop, count); });
}

template <typename T>
auto linspace(T start, T stop, int count) -> std::future<nd_array<T>> {
  return async::linspace<T>(future_resolver{}, start, stop, count);
}

template <typename T, typename Resolver>
auto random(Resolver &&resolver, tf::small_vector<int, 3> shape, T lower,
            T upper) -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(
      std::forward<Resolver>(resolver),
      [shape = std::move(shape), lower, upper]() mutable {
        return cpp::random<T>(std::move(shape), lower, upper);
      });
}

template <typename T>
auto random(tf::small_vector<int, 3> shape, T lower, T upper)
    -> std::future<nd_array<T>> {
  return async::random<T>(future_resolver{}, std::move(shape), lower, upper);
}

} // namespace tf::cpp::async
