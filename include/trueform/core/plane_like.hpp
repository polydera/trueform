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
#include "./base/plane.hpp"
#include "./coordinate_type.hpp"
#include "./unit_vector_like.hpp"
#include <utility>

namespace tf {

/// @ingroup core_primitives
/// @brief Base template for plane types.
///
/// Provides the common interface for infinite planes. A plane is defined by
/// a unit normal vector and a signed distance `d` from the origin. Points on
/// the plane satisfy: `dot(normal, point) + d = 0`.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy The storage policy.
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

/// @ingroup core_primitives
/// @brief Create a plane view from a unit normal view and offset.
///
/// Returns a view when inputs are views, preserving zero-copy semantics.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy The unit normal policy.
/// @param normal A unit-length normal vector.
/// @param d Signed offset from the origin.
/// @return A @ref tf::plane_like instance.
template <std::size_t Dims, typename Policy>
auto make_plane_like(const unit_vector_like<Dims, Policy> &normal,
                     tf::coordinate_type<Policy> d) {
  return plane_like<Dims, tf::core::plane<Dims, Policy>>{normal, d};
}
} // namespace tf
