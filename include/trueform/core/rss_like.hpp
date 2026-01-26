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
#include "./base/pt.hpp"
#include "./base/rss_impl.hpp"
#include "./base/vec.hpp"
#include "./point.hpp"
#include "./unit_vector.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Base template for rectangle swept sphere types.
///
/// Provides the common interface for RSS bounding volumes, defined by an
/// origin, orthonormal axes, rectangle lengths, and a sphere radius.
/// An RSS is the Minkowski sum of a rectangle and a sphere.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy The storage policy.
template <std::size_t Dims, typename Policy> struct rss_like : Policy {
  rss_like() = default;
  rss_like(const Policy &policy) : Policy{policy} {}
  rss_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::operator=;
  using Policy::axes;
  using Policy::length;
  using Policy::origin;
  using Policy::radius;
  using coordinate_type = typename Policy::coordinate_type;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  /// @brief Compute the center point of the RSS rectangle.
  ///
  /// The origin is the local (0,0,0) corner, so center is
  /// origin + sum(axes[i] * length[i] * 0.5) for i in [0, Dims-1).
  ///
  /// @return A `point<T, N>` representing the center of the rectangle.
  auto center() const -> point<coordinate_type, Dims> {
    auto out = origin;
    auto c = out.as_vector_view();
    for (std::size_t i = 0; i < Dims - 1; ++i)
      c = c + axes[i] * (length[i] * coordinate_type(0.5));
    return out;
  }

  /// @brief Return the dimensionality of the RSS.
  ///
  /// Provided as a compile-time constant.
  ///
  /// @return The number of spatial dimensions (N).
  constexpr auto size() -> std::size_t { return Dims; }

  template <typename RealT>
  operator tf::rss_like<Dims, tf::core::rss<Dims, tf::core::pt<RealT, Dims>,
                                            tf::core::vec<RealT, Dims>>>()
      const {
    std::array<tf::unit_vector<RealT, Dims>, Dims> converted_axes;
    for (std::size_t i = 0; i < Dims; ++i)
      converted_axes[i] = tf::make_unit_vector(tf::unsafe, axes[i]);
    std::array<RealT, Dims - 1> converted_length;
    for (std::size_t i = 0; i < Dims - 1; ++i)
      converted_length[i] = static_cast<RealT>(length[i]);
    return {origin, converted_axes, converted_length,
            static_cast<RealT>(radius)};
  }
};

template <std::size_t V, typename Policy>
auto unwrap(const rss_like<V, Policy> &obj) -> decltype(auto) {
  return static_cast<const Policy &>(obj);
}

template <std::size_t V, typename Policy>
auto unwrap(rss_like<V, Policy> &obj) -> decltype(auto) {
  return static_cast<Policy &>(obj);
}

template <std::size_t V, typename Policy>
auto unwrap(rss_like<V, Policy> &&obj) -> decltype(auto) {
  return static_cast<Policy &&>(obj);
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const rss_like<V, Policy> &, T &&t) {
  return rss_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(rss_like<V, Policy> &, T &&t) {
  return rss_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(rss_like<V, Policy> &&, T &&t) {
  return rss_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

template <std::size_t V, typename Policy, typename T>
auto wrap_like(const rss_like<V, Policy> &&, T &&t) {
  return rss_like<V, std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_primitives
/// @brief Create an RSS view from point and vector views.
///
/// Returns a view when inputs are views, preserving zero-copy semantics.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy0 The origin point policy.
/// @tparam Policy1 The axes unit vector policy.
/// @param origin The corner point of the inner rectangle.
/// @param axes The orthonormal axes (Dims-1 span the rectangle, last is normal).
/// @param length The lengths along the first Dims-1 axes.
/// @param radius The sphere radius for the Minkowski sum.
/// @return A @ref tf::rss_like instance.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_rss_like(const point_like<Dims, Policy0> &origin,
                   const std::array<unit_vector_like<Dims, Policy1>, Dims> &axes,
                   const std::array<tf::coordinate_type<Policy0>, Dims - 1> &length,
                   tf::coordinate_type<Policy0> radius) {
  return tf::rss_like<Dims, tf::core::rss<Dims, Policy0, Policy1>>{
      origin, axes, length, radius};
}

} // namespace tf
