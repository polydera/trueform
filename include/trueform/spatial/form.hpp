/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/frame_like.hpp"
#include "../core/frame_ptr.hpp"
#include "./dyn_model.hpp"
#include "./model.hpp"

namespace tf {

template <std::size_t Dims, typename Policy> struct form : public Policy {
  form(Policy &&policy) : Policy{std::move(policy)} {}
  form(const Policy &policy) : Policy{policy} {}

  friend auto unwrap(const form &seg) -> decltype(auto) {
    return static_cast<const Policy &>(seg);
  }

  friend auto unwrap(form &seg) -> decltype(auto) {
    return static_cast<Policy &>(seg);
  }

  friend auto unwrap(form &&seg) -> decltype(auto) {
    return static_cast<Policy &&>(seg);
  }

  template <typename T> friend auto wrap_like(const form &, T &&t) {
    return form<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T> friend auto wrap_like(form &, T &&t) {
    return form<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T> friend auto wrap_like(form &&, T &&t) {
    return form<Dims, std::decay_t<T>>{static_cast<T &&>(t)};
  }
};

template <std::size_t Dims, typename FPolicy, typename Index, typename RealT,
          typename Policy>
auto make_form(const tf::frame_like<Dims, FPolicy> &_frame,
               tf::tree<Index, RealT, Dims> &_tree, Policy &&policy) {
  auto base =
      tf::spatial::make_dyn_model(tf::make_frame_ptr(_frame), _tree, policy);
  return form<Dims, decltype(base)>{std::move(base)};
}

template <std::size_t Dims, typename FPolicy, typename Index, typename RealT,
          typename Policy>
auto make_form(tf::frame_like<Dims, FPolicy> &&_frame,
               tf::tree<Index, RealT, Dims> &_tree, Policy &&policy) {
  auto base = tf::spatial::make_dyn_model(
      tf::make_frame_like(_frame.transformation(),
                          _frame.inverse_transformation()),
      _tree, policy);
  return form<Dims, decltype(base)>{std::move(base)};
}

template <typename Index, typename RealT, std::size_t Dims, typename Policy>
auto make_form(tf::tree<Index, RealT, Dims> &_tree, Policy &&policy) {
  auto base = tf::spatial::make_model(_tree, policy);
  return form<Dims, decltype(base)>{std::move(base)};
}

} // namespace tf
