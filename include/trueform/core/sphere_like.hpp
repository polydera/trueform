/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./base/sphere.hpp"
#include "./coordinate_type.hpp"
#include "./point_like.hpp"
#include <utility>

namespace tf {
template <std::size_t Dims, typename Policy> struct sphere_like : Policy {
  sphere_like() = default;
  sphere_like(const Policy &policy) : Policy{policy} {}
  sphere_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::operator=;
  using Policy::origin;
  using Policy::r;

  template <typename RealT>
  operator tf::sphere_like<
      Dims, tf::core::sphere<Dims, tf::core::pt<RealT, Dims>>>() const {
    return {origin, r};
  }
};

template <std::size_t V, typename Policy>
auto unwrap(const sphere_like<V, Policy> &poly) -> decltype(auto) {
  return static_cast<const Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(sphere_like<V, Policy> &poly) -> decltype(auto) {
  return static_cast<Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(sphere_like<V, Policy> &&poly) -> decltype(auto) {
  return static_cast<Policy &&>(poly);
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const sphere_like<V, Policy> &, T &&t) {
  return sphere_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(sphere_like<V, Policy> &, T &&t) {
  return sphere_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(sphere_like<V, Policy> &&, T &&t) {
  return sphere_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const sphere_like<V, Policy> &&, T &&t) {
  return sphere_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t Dims, typename Policy>
auto make_sphere_like(const point_like<Dims, Policy> &origin,
                      tf::coordinate_type<Policy> r) {
  return sphere_like<Dims, Policy>{origin, r};
}
} // namespace tf
