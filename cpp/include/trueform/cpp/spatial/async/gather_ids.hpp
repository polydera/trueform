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
#include "trueform/cpp/core/detail/non_deduced.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/spatial/detail/spatial_carrier.hpp"
#include "trueform/cpp/spatial/gather_ids.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <
    typename Resolver, typename Form, typename Real, std::size_t Dims,
    std::enable_if_t<cpp::detail::is_spatial_carrier<std::decay_t<Form>>::value,
                     int> = 0>
auto gather_ids(Resolver &&resolver, const Form &form,
                const primitive<Real, Dims> &query) {
  using result_type = decltype(cpp::gather_ids(form, query));
  return submit<result_type>(std::forward<Resolver>(resolver), [form, query] {
    return cpp::gather_ids(form, query);
  });
}

template <
    typename Form, typename Real, std::size_t Dims,
    std::enable_if_t<cpp::detail::is_spatial_carrier<std::decay_t<Form>>::value,
                     int> = 0>
auto gather_ids(const Form &form, const primitive<Real, Dims> &query)
    -> std::future<decltype(cpp::gather_ids(form, query))> {
  return async::gather_ids(future_resolver{}, form, query);
}

template <typename Resolver, typename Form0, typename Form1,
          std::enable_if_t<
              cpp::detail::is_spatial_carrier<std::decay_t<Form0>>::value &&
                  cpp::detail::is_spatial_carrier<std::decay_t<Form1>>::value,
              int> = 0>
auto gather_ids(Resolver &&resolver, const Form0 &a, const Form1 &b) {
  using result_type = decltype(cpp::gather_ids(a, b));
  return submit<result_type>(std::forward<Resolver>(resolver),
                             [a, b] { return cpp::gather_ids(a, b); });
}

template <typename Form0, typename Form1,
          std::enable_if_t<
              cpp::detail::is_spatial_carrier<std::decay_t<Form0>>::value &&
                  cpp::detail::is_spatial_carrier<std::decay_t<Form1>>::value,
              int> = 0>
auto gather_ids(const Form0 &a, const Form1 &b)
    -> std::future<decltype(cpp::gather_ids(a, b))> {
  return async::gather_ids(future_resolver{}, a, b);
}

} // namespace tf::cpp::async
