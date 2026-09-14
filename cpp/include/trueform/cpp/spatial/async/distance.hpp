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
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/spatial/distance.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

#define TF_CPP_DEFINE_ASYNC_PRIMITIVE_DISTANCE(Operation)                      \
  template <typename Resolver, typename Real0, typename Real1,                 \
            std::size_t Dims>                                                  \
  auto Operation(Resolver &&resolver, const primitive<Real0, Dims> &a,         \
                 const primitive<Real1, Dims> &b) {                            \
    auto owned_a = a;                                                          \
    auto owned_b = b;                                                          \
    using result_type = distance_result<std::common_type_t<Real0, Real1>>;     \
    return submit<result_type>(                                                \
        std::forward<Resolver>(resolver),                                      \
        [a = std::move(owned_a), b = std::move(owned_b)] {                     \
          return cpp::Operation(a, b);                                         \
        });                                                                    \
  }                                                                            \
                                                                               \
  template <typename Real0, typename Real1, std::size_t Dims>                  \
  auto Operation(const primitive<Real0, Dims> &a,                              \
                 const primitive<Real1, Dims> &b)                              \
      -> std::future<distance_result<std::common_type_t<Real0, Real1>>> {      \
    return async::Operation(future_resolver{}, a, b);                          \
  }

TF_CPP_DEFINE_ASYNC_PRIMITIVE_DISTANCE(distance2)
TF_CPP_DEFINE_ASYNC_PRIMITIVE_DISTANCE(distance)

#undef TF_CPP_DEFINE_ASYNC_PRIMITIVE_DISTANCE

namespace detail {

/// Fewer than the three spatial carriers: `cpp::distance` is stated between a
/// mesh and a cloud only, so the edge mesh has no entry to reach.
template <typename T> struct is_distance_form : std::false_type {};
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
struct is_distance_form<cpp::mesh<Index, Real, Dims, Ngon>> : std::true_type {};
template <typename Real, std::size_t Dims>
struct is_distance_form<cpp::point_cloud<Real, Dims>> : std::true_type {};

} // namespace detail

#define TF_CPP_DEFINE_ASYNC_FORM_DISTANCE(Operation)                           \
  template <typename Resolver, typename Form, typename Real, std::size_t Dims, \
            std::enable_if_t<detail::is_distance_form<Form>::value, int> = 0>  \
  auto Operation(Resolver &&resolver, const Form &form,                        \
                 const primitive<Real, Dims> &query) {                         \
    using result_type = distance_result<typename Form::real_type>;             \
    return submit<result_type>(                                                \
        std::forward<Resolver>(resolver),                                      \
        [form, query] { return cpp::Operation(form, query); });                \
  }                                                                            \
                                                                               \
  template <typename Form, typename Real, std::size_t Dims,                    \
            std::enable_if_t<detail::is_distance_form<Form>::value, int> = 0>  \
  auto Operation(const Form &form, const primitive<Real, Dims> &query)         \
      -> std::future<distance_result<typename Form::real_type>> {              \
    return async::Operation(future_resolver{}, form, query);                   \
  }                                                                            \
                                                                               \
  template <typename Resolver, typename Form0, typename Form1,                 \
            std::enable_if_t<detail::is_distance_form<Form0>::value &&         \
                                 detail::is_distance_form<Form1>::value,       \
                             int> = 0>                                         \
  auto Operation(Resolver &&resolver, const Form0 &a, const Form1 &b) {        \
    using result_type = std::common_type_t<typename Form0::real_type,          \
                                           typename Form1::real_type>;         \
    return submit<result_type>(std::forward<Resolver>(resolver),               \
                               [a, b] { return cpp::Operation(a, b); });       \
  }                                                                            \
                                                                               \
  template <typename Form0, typename Form1,                                    \
            std::enable_if_t<detail::is_distance_form<Form0>::value &&         \
                                 detail::is_distance_form<Form1>::value,       \
                             int> = 0>                                         \
  auto Operation(const Form0 &a, const Form1 &b)                               \
      -> std::future<std::common_type_t<typename Form0::real_type,             \
                                        typename Form1::real_type>> {          \
    return async::Operation(future_resolver{}, a, b);                          \
  }

TF_CPP_DEFINE_ASYNC_FORM_DISTANCE(distance2)
TF_CPP_DEFINE_ASYNC_FORM_DISTANCE(distance)

#undef TF_CPP_DEFINE_ASYNC_FORM_DISTANCE

} // namespace tf::cpp::async
