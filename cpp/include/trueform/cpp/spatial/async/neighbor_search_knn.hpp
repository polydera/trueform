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
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/spatial/detail/spatial_carrier.hpp"
#include "trueform/cpp/spatial/neighbor_search_knn.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <future>
#include <limits>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <
    typename Resolver, typename Form, typename Real, std::size_t Dims,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto neighbor_search_knn(Resolver &&resolver, const Form &form,
                         const primitive<Real, Dims> &query, int k,
                         Real radius = std::numeric_limits<Real>::infinity()) {
  using result_type =
      decltype(cpp::neighbor_search_knn(form, query, k, radius));
  return submit<result_type>(
      std::forward<Resolver>(resolver), [form, query, k, radius] {
        return cpp::neighbor_search_knn(form, query, k, radius);
      });
}

template <
    typename Form, typename Real, std::size_t Dims,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto neighbor_search_knn(const Form &form, const primitive<Real, Dims> &query,
                         int k,
                         Real radius = std::numeric_limits<Real>::infinity())
    -> std::future<decltype(cpp::neighbor_search_knn(form, query, k, radius))> {
  return async::neighbor_search_knn(future_resolver{}, form, query, k, radius);
}

template <
    typename Resolver, typename Form, typename Real, std::size_t Dims,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto neighbor_search_knn_batch(
    Resolver &&resolver, const Form &form, const primitive<Real, Dims> &queries,
    int k, Real radius = std::numeric_limits<Real>::infinity()) {
  using result_type =
      decltype(cpp::neighbor_search_knn_batch(form, queries, k, radius));
  return submit<result_type>(
      std::forward<Resolver>(resolver), [form, queries, k, radius] {
        return cpp::neighbor_search_knn_batch(form, queries, k, radius);
      });
}

template <
    typename Form, typename Real, std::size_t Dims,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto neighbor_search_knn_batch(
    const Form &form, const primitive<Real, Dims> &queries, int k,
    Real radius = std::numeric_limits<Real>::infinity())
    -> std::future<decltype(cpp::neighbor_search_knn_batch(form, queries, k,
                                                           radius))> {
  return async::neighbor_search_knn_batch(future_resolver{}, form, queries, k,
                                          radius);
}

} // namespace tf::cpp::async
