/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./base/plane.hpp"
#include "./coordinate_type.hpp"
#include "./unit_vector_like.hpp"
#include <utility>

namespace tf {
template <std::size_t Dims, typename Policy> struct plane_like : Policy {
  plane_like() = default;
  plane_like(const Policy &policy) : Policy{policy} {}
  plane_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::operator=;
  using Policy::d;
  using Policy::normal;

  template <typename RealT>
  operator tf::plane_like<
      Dims, tf::core::plane<Dims, tf::core::vec<RealT, Dims>>>() const {
    return {normal, d};
  }
};

template <std::size_t V, typename Policy>
auto unwrap(const plane_like<V, Policy> &poly) -> decltype(auto) {
  return static_cast<const Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(plane_like<V, Policy> &poly) -> decltype(auto) {
  return static_cast<Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(plane_like<V, Policy> &&poly) -> decltype(auto) {
  return static_cast<Policy &&>(poly);
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const plane_like<V, Policy> &, T &&t) {
  return plane_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(plane_like<V, Policy> &, T &&t) {
  return plane_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(plane_like<V, Policy> &&, T &&t) {
  return plane_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const plane_like<V, Policy> &&, T &&t) {
  return plane_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t Dims, typename Policy>
auto make_plane_like(const unit_vector_like<Dims, Policy> &normal,
                     tf::coordinate_type<Policy> d) {
  return plane_like<Dims, tf::core::plane<Dims, Policy>>{normal, d};
}
} // namespace tf
