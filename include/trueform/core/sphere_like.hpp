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
#include "./base/sphere.hpp"
#include "./coordinate_type.hpp"
#include "./point_like.hpp"
#include <utility>

namespace tf {

/// @ingroup core_primitives
/// @brief Base template for sphere types.
///
/// Provides the common interface for spheres, defined by a center point
/// `origin` and a radius `r`.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy The storage policy.
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

/// @ingroup core_primitives
/// @brief Create a sphere view from a point view and radius.
///
/// Returns a view when inputs are views, preserving zero-copy semantics.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy The center point policy.
/// @param origin The center of the sphere.
/// @param r The radius of the sphere.
/// @return A @ref tf::sphere_like instance.
template <std::size_t Dims, typename Policy>
auto make_sphere_like(const point_like<Dims, Policy> &origin,
                      tf::coordinate_type<Policy> r) {
  return sphere_like<Dims, Policy>{origin, r};
}
} // namespace tf
