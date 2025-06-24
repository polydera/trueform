/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./base/ray.hpp"
#include <utility>

namespace tf {
template <std::size_t Dims, typename Policy> struct ray_like : Policy {
  ray_like() = default;
  ray_like(const Policy &policy) : Policy{policy} {}
  ray_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::operator=;
  using Policy::direction;
  using Policy::origin;
  using coordinate_type = typename Policy::coordinate_type;

  auto operator()(coordinate_type t) const { return origin + t * direction; }

  template <typename RealT>
  operator tf::ray_like<Dims, tf::core::ray<Dims, tf::core::pt<RealT, Dims>,
                                            tf::core::vec<RealT, Dims>>>()
      const {
    return {origin, direction};
  }
};

template <std::size_t V, typename Policy>
auto unwrap(const ray_like<V, Policy> &poly) -> decltype(auto) {
  return static_cast<const Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(ray_like<V, Policy> &poly) -> decltype(auto) {
  return static_cast<Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(ray_like<V, Policy> &&poly) -> decltype(auto) {
  return static_cast<Policy &&>(poly);
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const ray_like<V, Policy> &, T &&t) {
  return ray_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(ray_like<V, Policy> &, T &&t) {
  return ray_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(ray_like<V, Policy> &&, T &&t) {
  return ray_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const ray_like<V, Policy> &&, T &&t) {
  return ray_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_ray_like(const point_like<Dims, Policy0> &origin,
                   const vector_like<Dims, Policy1> &direction) {
  return tf::ray_like<Dims, tf::core::ray<Dims, Policy0, Policy1>>{origin,
                                                                   direction};
}
} // namespace tf
