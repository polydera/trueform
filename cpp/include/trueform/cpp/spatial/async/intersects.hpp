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
#include "trueform/cpp/spatial/intersection_result.hpp"
#include "trueform/cpp/spatial/intersects.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Real0, typename Real1, std::size_t Dims>
auto intersects(Resolver &&resolver, const primitive<Real0, Dims> &a,
                const primitive<Real1, Dims> &b) {
  return submit<intersection_result>(std::forward<Resolver>(resolver),
                                     [a, b] { return cpp::intersects(a, b); });
}

template <typename Real0, typename Real1, std::size_t Dims>
auto intersects(const primitive<Real0, Dims> &a,
                const primitive<Real1, Dims> &b)
    -> std::future<intersection_result> {
  return async::intersects(future_resolver{}, a, b);
}

template <
    typename Resolver, typename Form, typename Real, std::size_t Dims,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto intersects(Resolver &&resolver, const Form &form,
                const primitive<Real, Dims> &query) {
  return submit<intersection_result>(
      std::forward<Resolver>(resolver),
      [form, query] { return cpp::intersects(form, query); });
}

template <
    typename Form, typename Real, std::size_t Dims,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto intersects(const Form &form, const primitive<Real, Dims> &query)
    -> std::future<intersection_result> {
  return async::intersects(future_resolver{}, form, query);
}

template <
    typename Resolver, typename Real, std::size_t Dims, typename Form,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto intersects(Resolver &&resolver, const primitive<Real, Dims> &query,
                const Form &form) {
  return submit<intersection_result>(
      std::forward<Resolver>(resolver),
      [query, form] { return cpp::intersects(query, form); });
}

template <
    typename Real, std::size_t Dims, typename Form,
    std::enable_if_t<cpp::detail::is_spatial_carrier<Form>::value, int> = 0>
auto intersects(const primitive<Real, Dims> &query, const Form &form)
    -> std::future<intersection_result> {
  return async::intersects(future_resolver{}, query, form);
}

template <typename Resolver, typename Form0, typename Form1,
          std::enable_if_t<cpp::detail::is_spatial_carrier<Form0>::value &&
                               cpp::detail::is_spatial_carrier<Form1>::value,
                           int> = 0>
auto intersects(Resolver &&resolver, const Form0 &a, const Form1 &b) {
  return submit<intersection_result>(std::forward<Resolver>(resolver),
                                     [a, b] { return cpp::intersects(a, b); });
}

template <typename Form0, typename Form1,
          std::enable_if_t<cpp::detail::is_spatial_carrier<Form0>::value &&
                               cpp::detail::is_spatial_carrier<Form1>::value,
                           int> = 0>
auto intersects(const Form0 &a, const Form1 &b)
    -> std::future<intersection_result> {
  return async::intersects(future_resolver{}, a, b);
}

} // namespace tf::cpp::async
