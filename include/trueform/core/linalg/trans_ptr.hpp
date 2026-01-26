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
#include "../transformation_like.hpp"
#include <type_traits>
namespace tf::linalg {
template <std::size_t Dims, typename Policy> struct trans_ptr {
  using element_type = typename Policy::element_type;
  using value_type = typename Policy::value_type;
  using coordinate_type = typename Policy::coordinate_type;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;
  constexpr static std::size_t n_rows = Policy::n_rows;
  constexpr static std::size_t n_columns = Policy::n_columns;

  trans_ptr() = default;
  trans_ptr(Policy &policy) : _policy{&policy} {}
  trans_ptr(const trans_ptr &) = default;
  auto operator=(const trans_ptr &other) -> trans_ptr & {
    *_policy = *other._policy;
    return *this;
  }

  auto operator()(std::size_t i, std::size_t j) const -> decltype(auto) {
    return (*_policy)(i, j);
  }

  auto operator()(std::size_t i, std::size_t j) -> decltype(auto) {
    return (*_policy)(i, j);
  }

  Policy *_policy;
};

template <std::size_t Dims, typename Policy>
struct trans_ptr<Dims, const Policy> {
  using element_type = const typename Policy::element_type;
  using value_type = typename Policy::value_type;
  using coordinate_type = typename Policy::coordinate_type;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;
  constexpr static std::size_t n_rows = Policy::n_rows;
  constexpr static std::size_t n_columns = Policy::n_columns;

  trans_ptr() = default;
  trans_ptr(const Policy &policy) : _policy{&policy} {}
  trans_ptr(const trans_ptr &) = default;
  auto operator=(const trans_ptr &other) -> trans_ptr & = delete;

  auto operator()(std::size_t i, std::size_t j) const -> decltype(auto) {
    return (*_policy)(i, j);
  }

  const Policy *_policy;
};

namespace implementation {
template <std::size_t Dims, typename Policy>
auto is_trans_ptr(const tf::linalg::trans_ptr<Dims, Policy> *)
    -> std::true_type;
auto is_trans_ptr(const void *) -> std::false_type;
} // namespace implementation

template <typename T>
inline constexpr bool is_trans_ptr = decltype(implementation::is_trans_ptr(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

template <std::size_t Dims, typename Policy>
auto make_trans_ptr(const tf::transformation_like<Dims, Policy> &t) {
  if constexpr (is_trans_ptr<Policy>)
    return t;
  else
    return tf::transformation_like<Dims,
                                   tf::linalg::trans_ptr<Dims, const Policy>>{
        t};
}

template <std::size_t Dims, typename Policy>
auto make_trans_ptr(tf::transformation_like<Dims, Policy> &t) {
  if constexpr (is_trans_ptr<Policy>)
    return t;
  else
    return tf::transformation_like<Dims, tf::linalg::trans_ptr<Dims, Policy>>{
        t};
}

template <typename T, std::size_t Dims>
auto make_trans_ptr(tf::identity_transformation<T, Dims> id) {
  return id;
}

template <std::size_t Dims, typename Policy>
auto make_trans_ptr(tf::transformation_like<Dims, Policy> &&) = delete;
template <std::size_t Dims, typename Policy>
auto make_trans_ptr(const tf::transformation_like<Dims, Policy> &&) = delete;
} // namespace tf::linalg
