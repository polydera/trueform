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
#include "./trans_ptr.hpp"
#include <utility>
namespace tf::linalg {
template <std::size_t Dims, typename Policy> struct transpose : Policy {
  transpose() = default;
  transpose(const Policy &policy) : Policy{policy} {}
  transpose(Policy &&policy) : Policy{std::move(policy)} {}
  using Policy::Policy;
  using Policy::operator=;
  using Policy::operator();
  constexpr static std::size_t n_rows = Dims;
  constexpr static std::size_t n_columns = Dims;

  auto operator()(std::size_t i, std::size_t j) const -> decltype(auto) {
    return Policy::operator()(j, i);
  }

  auto operator()(std::size_t i, std::size_t j) -> decltype(auto) {
    return Policy::operator()(j, i);
  }
};

template <std::size_t Dims, typename Policy>
auto make_transpose(const tf::transformation_like<Dims, Policy> &t) {
  return tf::transformation_like<Dims, tf::linalg::transpose<Dims, Policy>>{t};
}

template <std::size_t Dims, typename Policy>
auto make_transpose_view(const tf::transformation_like<Dims, Policy> &t) {
  return make_transpose(make_trans_ptr(t));
}

template <std::size_t Dims, typename Policy>
auto make_transpose_view(tf::transformation_like<Dims, Policy> &t) {
  return make_transpose(make_trans_ptr(t));
}
} // namespace tf::linalg
