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
#include "./base/line.hpp"
#include <utility>

namespace tf {

/// @ingroup core_primitives
/// @brief Base template for line types.
///
/// Provides the common interface for infinite lines, including parametric
/// evaluation via `operator()(t)` which returns `origin + t * direction`.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy The storage policy.
template <std::size_t Dims, typename Policy> struct line_like : Policy {
  line_like() = default;
  line_like(const Policy &policy) : Policy{policy} {}
  line_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::operator=;
  using Policy::direction;
  using Policy::origin;
  using coordinate_type = typename Policy::coordinate_type;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  auto operator()(coordinate_type t) const { return origin + t * direction; }

  template <typename RealT>
  operator tf::line_like<Dims, tf::core::line<Dims, tf::core::pt<RealT, Dims>,
                                              tf::core::vec<RealT, Dims>>>()
      const {
    return {origin, direction};
  }
};

template <std::size_t V, typename Policy>
auto unwrap(const line_like<V, Policy> &poly) -> decltype(auto) {
  return static_cast<const Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(line_like<V, Policy> &poly) -> decltype(auto) {
  return static_cast<Policy &>(poly);
}

template <std::size_t V, typename Policy>
auto unwrap(line_like<V, Policy> &&poly) -> decltype(auto) {
  return static_cast<Policy &&>(poly);
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const line_like<V, Policy> &, T &&t) {
  return line_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(line_like<V, Policy> &, T &&t) {
  return line_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(line_like<V, Policy> &&, T &&t) {
  return line_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const line_like<V, Policy> &&, T &&t) {
  return line_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_primitives
/// @brief Create a line view from point and vector views.
///
/// Returns a view when inputs are views, preserving zero-copy semantics.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy0 The origin point policy.
/// @tparam Policy1 The direction vector policy.
/// @param origin A point on the line.
/// @param direction The direction of the line.
/// @return A @ref tf::line_like instance.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_line_like(const point_like<Dims, Policy0> &origin,
                    const vector_like<Dims, Policy1> &direction) {
  return tf::line_like<Dims, tf::core::line<Dims, Policy0, Policy1>>{origin,
                                                                     direction};
}
} // namespace tf
