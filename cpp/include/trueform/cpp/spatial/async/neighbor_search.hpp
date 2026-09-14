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
#include "trueform/cpp/spatial/neighbor_search.hpp"
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
auto neighbor_search(Resolver &&resolver, const Form &form,
                     const primitive<Real, Dims> &query,
                     Real radius = std::numeric_limits<Real>::infinity()) {
  using result_type = decltype(cpp::neighbor_search(form, query, radius));
  return submit<result_type>(std::forward<Resolver>(resolver),
                             [form, query, radius] {
                               return cpp::neighbor_search(form, query, radius);
                             });
}

template <
    typename Form, typename Real, std::size_t Dims,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto neighbor_search(const Form &form, const primitive<Real, Dims> &query,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> std::future<decltype(cpp::neighbor_search(form, query, radius))> {
  return async::neighbor_search(future_resolver{}, form, query, radius);
}

template <
    typename Resolver, typename Form, typename Real, std::size_t Dims,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto neighbor_search_batch(
    Resolver &&resolver, const Form &form, const primitive<Real, Dims> &queries,
    Real radius = std::numeric_limits<Real>::infinity()) {
  using result_type =
      decltype(cpp::neighbor_search_batch(form, queries, radius));
  return submit<result_type>(
      std::forward<Resolver>(resolver), [form, queries, radius] {
        return cpp::neighbor_search_batch(form, queries, radius);
      });
}

template <
    typename Form, typename Real, std::size_t Dims,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto neighbor_search_batch(const Form &form,
                           const primitive<Real, Dims> &queries,
                           Real radius = std::numeric_limits<Real>::infinity())
    -> std::future<decltype(cpp::neighbor_search_batch(form, queries,
                                                       radius))> {
  return async::neighbor_search_batch(future_resolver{}, form, queries, radius);
}

template <typename Resolver, typename Form0, typename Form1,
          std::enable_if_t<cpp::detail::is_spatial_carrier<Form0>::value &&
                               cpp::detail::is_spatial_carrier<Form1>::value,
                           int> = 0>
auto neighbor_search(
    Resolver &&resolver, const Form0 &a, const Form1 &b,
    typename Form0::real_type radius =
        std::numeric_limits<typename Form0::real_type>::infinity()) {
  using result_type = decltype(cpp::neighbor_search(a, b, radius));
  return submit<result_type>(std::forward<Resolver>(resolver), [a, b, radius] {
    return cpp::neighbor_search(a, b, radius);
  });
}

template <typename Form0, typename Form1,
          std::enable_if_t<cpp::detail::is_spatial_carrier<Form0>::value &&
                               cpp::detail::is_spatial_carrier<Form1>::value,
                           int> = 0>
auto neighbor_search(
    const Form0 &a, const Form1 &b,
    typename Form0::real_type radius =
        std::numeric_limits<typename Form0::real_type>::infinity())
    -> std::future<decltype(cpp::neighbor_search(a, b, radius))> {
  return async::neighbor_search(future_resolver{}, a, b, radius);
}

} // namespace tf::cpp::async
