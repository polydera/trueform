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
#include "trueform/cpp/core/nd_array_indexing.hpp"

#include <cstdint>
#include <future>
#include <utility>
#include <vector>

namespace tf::cpp::async {

template <typename Resolver, typename T>
auto take(Resolver &&resolver, const nd_array<T> &array,
          const nd_array<std::int32_t> &indices, int axis = 0)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(
      std::forward<Resolver>(resolver),
      [array, indices, axis] { return cpp::take(array, indices, axis); });
}

template <typename T>
auto take(const nd_array<T> &array, const nd_array<std::int32_t> &indices,
          int axis = 0) -> std::future<nd_array<T>> {
  return async::take(future_resolver{}, array, indices, axis);
}

template <typename Resolver, typename T>
auto multi_take(Resolver &&resolver, const nd_array<T> &array,
                const std::vector<multi_take_index> &indices)
    -> resolver_result_t<Resolver, nd_array<T>> {
  auto owned_indices = indices;
  return submit<nd_array<T>>(std::forward<Resolver>(resolver),
                             [array, indices = std::move(owned_indices)] {
                               return cpp::multi_take(array, indices);
                             });
}

template <typename T>
auto multi_take(const nd_array<T> &array,
                const std::vector<multi_take_index> &indices)
    -> std::future<nd_array<T>> {
  return async::multi_take(future_resolver{}, array, indices);
}

template <typename Resolver, typename T>
auto take_along_axis(Resolver &&resolver, const nd_array<T> &array,
                     const nd_array<std::int32_t> &indices, int axis)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(
      std::forward<Resolver>(resolver), [array, indices, axis] {
        return cpp::take_along_axis(array, indices, axis);
      });
}

template <typename T>
auto take_along_axis(const nd_array<T> &array,
                     const nd_array<std::int32_t> &indices, int axis)
    -> std::future<nd_array<T>> {
  return async::take_along_axis(future_resolver{}, array, indices, axis);
}

template <typename Resolver, typename T>
auto boolean_index(Resolver &&resolver, const nd_array<T> &array,
                   const nd_array<std::int8_t> &mask)
    -> resolver_result_t<Resolver, nd_array<T>> {
  return submit<nd_array<T>>(std::forward<Resolver>(resolver), [array, mask] {
    return cpp::boolean_index(array, mask);
  });
}

template <typename T>
auto boolean_index(const nd_array<T> &array, const nd_array<std::int8_t> &mask)
    -> std::future<nd_array<T>> {
  return async::boolean_index(future_resolver{}, array, mask);
}

} // namespace tf::cpp::async
